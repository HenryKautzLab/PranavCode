"""
Video analysis pipeline: LLaVA-NeXT + Whisper + EasyOCR.

Supports two layouts:
  flat   — DATA_DIR/*.mp4
  nested — DATA_DIR/{post_id}/video.mp4   (Instagram / tiktok_download_videos.py output)

Output per video: {OUTPUT_DIR}/{video_id}/analysis.json
Transcripts cached in: {OUTPUT_DIR}/{video_id}/transcript.json
"""

import argparse
import json
import os
import subprocess
import time
import cv2
import whisper
import easyocr
from pathlib import Path
from PIL import Image
import torch
import numpy as np
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import BitsAndBytesConfig

# ── Config ────────────────────────────────────────────────────────────────────
NUM_FRAMES   = 10
# evenly spaced fractions 0% → 95% (avoid EOF edge)
FRAME_POSITIONS = [i / (NUM_FRAMES - 1) * 0.95 for i in range(NUM_FRAMES)]
MODEL_ID     = "llava-hf/llava-v1.6-mistral-7b-hf"
WHISPER_SIZE = "base"

QUESTIONS = {
    "content_summary":       "Describe in detail what is happening in this video. What are the main visual elements, people, objects, and actions?",
    "emotional_tone":        "What emotions does this content evoke? Describe the mood and emotional impact on viewers.",
    "persuasion_techniques": "What persuasive or rhetorical techniques are being used? Consider visual appeal, emotional manipulation, social proof, authority, or urgency tactics.",
    "target_audience":       "Who is the intended audience? What age group, demographic, or interest group would this appeal to?",
    "credibility_assessment":"Does this content appear credible and trustworthy? Are there any red flags, misleading elements, or signs of manipulation?",
    "misinformation_risk":   "Could this content spread misinformation or false claims? What are the potential risks to viewers?",
    "behavioral_impact":     "How might this video influence viewer behavior, beliefs, or actions? What specific behaviors might it encourage?",
    "content_category":      "Category: entertainment, education, health, political, product promotion, or other? Choose one.",
    "key_message":           "What is the main message or takeaway viewers are supposed to get from this video?",
    "risk_level":            "On a scale of low/medium/high, what is the potential risk to viewers? Explain briefly.",
    "audio_visual_alignment":"Does the spoken audio and on-screen text align with what is shown visually? Note any mismatches.",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",   default="../../data",
                   help="Directory containing videos (flat *.mp4 or nested {id}/video.mp4)")
    p.add_argument("--output-dir", default="../../data/outputs",
                   help="Where to write per-video analysis.json and transcript.json")
    p.add_argument("--layout",     choices=["flat", "nested"], default="nested",
                   help="'nested' = {id}/video.mp4 (default), 'flat' = *.mp4")
    p.add_argument("--limit",      type=int, default=None,
                   help="Max videos to process (useful for quick tests)")
    p.add_argument("--video-id",   default=None,
                   help="Process a single video by ID (nested: {id}/video.mp4, flat: {id}.mp4)")
    return p.parse_args()


# ── Video helpers ─────────────────────────────────────────────────────────────

def get_duration(video_path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(video_path)],
        capture_output=True, text=True, timeout=15)
    try:
        return float(json.loads(r.stdout)["format"]["duration"])
    except Exception:
        return 0.0


def extract_frames(video_path: Path, num_frames: int = NUM_FRAMES,
                   positions: list = FRAME_POSITIONS) -> list[Image.Image]:
    """Extract frames at fractional positions using ffmpeg timestamps (accurate)."""
    duration = get_duration(video_path)
    if duration <= 0:
        return []
    frames = []
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    for frac in positions:
        ts = max(0.0, min(duration * frac, duration - 0.1))
        frame_num = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_num, total - 1))
        ok, frame = cap.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def has_audio(video_path: Path) -> bool:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_name",
         "-of", "default=noprint_wrappers=1", str(video_path)],
        capture_output=True, text=True)
    return bool(r.stdout.strip())


def transcribe(video_path: Path, whisper_model) -> dict:
    if not has_audio(video_path):
        print(f"   ⚠  No audio in {video_path.name}")
        return {"transcript": "", "language": "unknown", "segments": []}
    result = whisper_model.transcribe(str(video_path))
    return {
        "transcript": result["text"].strip(),
        "language":   result["language"],
        "segments":   [{"start": s["start"], "end": s["end"], "text": s["text"]}
                       for s in result["segments"]],
    }


# ── OCR ───────────────────────────────────────────────────────────────────────

def extract_ocr(frames: list[Image.Image], ocr_reader) -> list[str]:
    seen, out = set(), []
    for frame in frames:
        for text in ocr_reader.readtext(np.array(frame), detail=0):
            c = text.strip()
            if c and c.lower() not in seen:
                seen.add(c.lower())
                out.append(c)
    return out


# ── LLaVA ────────────────────────────────────────────────────────────────────

def ask_llava(image: Image.Image, question: str, processor, model, device: str,
              transcript: str = "", ocr_text: list = None) -> str:
    context = []
    if transcript:
        context.append(f'SPOKEN AUDIO (Whisper): "{transcript}"')
    if ocr_text:
        context.append(f'ON-SCREEN TEXT (OCR): "{" | ".join(ocr_text)}"')
    prompt_text = ("\n".join(context) + "\n\nUsing what you see AND the above: " + question
                   if context else question)

    conversation = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompt_text},
    ]}]
    prompt  = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs  = processor(images=image, text=prompt, return_tensors="pt").to(device)
    output  = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    decoded = processor.decode(output[0], skip_special_tokens=True)
    return decoded.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in decoded else decoded.strip()


def get_segment_near(segments: list, t: float, window: float = 5.0) -> str:
    return " ".join(s["text"] for s in segments if abs(s["start"] - t) < window).strip()


# ── Per-video analysis ────────────────────────────────────────────────────────

def analyze_video(video_path: Path, audio_data: dict,
                  ocr_reader, processor, model, device: str) -> dict:
    timings: dict = {}

    # ── Frame extraction ──────────────────────────────────────────────────────
    print(f"   Extracting {NUM_FRAMES} frames...")
    t0 = time.perf_counter()
    frames = extract_frames(video_path)
    timings["frame_extraction_s"] = round(time.perf_counter() - t0, 3)
    if not frames:
        raise RuntimeError("No frames extracted — corrupt or unreadable video")

    # ── EasyOCR ───────────────────────────────────────────────────────────────
    print(f"   Running EasyOCR on {len(frames)} frames...")
    t0 = time.perf_counter()
    ocr_text = extract_ocr(frames, ocr_reader)
    timings["ocr_s"] = round(time.perf_counter() - t0, 3)

    transcript = audio_data.get("transcript", "")
    segments   = audio_data.get("segments", [])
    duration   = get_duration(video_path)

    # Whisper time was recorded during the pre-pass and stored in audio_data.
    if "whisper_s" in audio_data:
        timings["whisper_s"] = audio_data["whisper_s"]

    if ocr_text:
        print(f"   OCR: {' | '.join(ocr_text[:5])}{'...' if len(ocr_text)>5 else ''}")
    if transcript:
        print(f"   Audio: {transcript[:80]}...")

    mid_frame = frames[len(frames) // 2]

    # ── Main questions (on middle frame, full pipeline: visual + OCR + audio) ─
    print(f"   LLaVA: answering {len(QUESTIONS)} questions...")
    t0 = time.perf_counter()
    analysis: dict = {}
    for i, (key, q) in enumerate(QUESTIONS.items(), 1):
        print(f"     [{i}/{len(QUESTIONS)}] {key}")
        analysis[key] = ask_llava(mid_frame, q, processor, model, device,
                                  transcript=transcript, ocr_text=ocr_text)
    timings["llava_main_questions_s"] = round(time.perf_counter() - t0, 3)
    timings["llava_per_question_avg_s"] = round(
        timings["llava_main_questions_s"] / len(QUESTIONS), 3)

    # ── Temporal progression (all frames, full pipeline) ─────────────────────
    print(f"   LLaVA: temporal progression across {len(frames)} frames...")
    t0 = time.perf_counter()
    temporal = []
    for idx, frame in enumerate(frames):
        frac = idx / max(len(frames) - 1, 1)
        t    = duration * frac if duration else 0
        seg  = get_segment_near(segments, t) if segments else ""
        desc = ask_llava(frame, f"Describe what is shown at the {int(frac*100)}% point of this video.",
                         processor, model, device,
                         transcript=seg or transcript, ocr_text=ocr_text)
        temporal.append({"frame_index": idx, "position_pct": round(frac * 100),
                         "description": desc, "audio_segment": seg})
    timings["llava_temporal_s"] = round(time.perf_counter() - t0, 3)
    analysis["temporal_progression"] = temporal

    timings["total_s"] = round(
        timings.get("whisper_s", 0)
        + timings["frame_extraction_s"]
        + timings["ocr_s"]
        + timings["llava_main_questions_s"]
        + timings["llava_temporal_s"],
        3,
    )

    analysis.update({
        "whisper_transcript":   transcript,
        "whisper_segments":     segments,
        "detected_language":    audio_data.get("language", "unknown"),
        "ocr_onscreen_text":    ocr_text,
        "num_frames_analyzed":  len(frames),
        "analysis_method":      f"LLaVA-NeXT ({MODEL_ID}) + Whisper-{WHISPER_SIZE} + EasyOCR",
        "pipeline_timings":     timings,
    })
    return analysis


# ── Main ──────────────────────────────────────────────────────────────────────

def collect_videos(data_dir: Path, layout: str, video_id: str = None) -> list[tuple[str, Path]]:
    """Returns list of (video_id, video_path)."""
    if video_id:
        if layout == "nested":
            for name in ("video.mp4", "video.mp3"):
                vp = data_dir / video_id / name
                if vp.exists():
                    return [(video_id, vp)]
        else:
            for ext in (".mp4", ".mp3"):
                vp = data_dir / (video_id + ext)
                if vp.exists():
                    return [(video_id, vp)]
        return []
    videos = []
    if layout == "nested":
        for d in sorted(data_dir.iterdir()):
            if d.is_dir():
                for name in ("video.mp4", "video.mp3"):
                    vp = d / name
                    if vp.exists():
                        videos.append((d.name, vp))
                        break
    else:
        for vp in sorted(data_dir.glob("*.mp4")) + sorted(data_dir.glob("*.mp3")):
            videos.append((vp.stem, vp))
    return videos


def main():
    args = parse_args()
    data_dir   = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Data  : {data_dir}  [{args.layout}]")
    print(f"Output: {output_dir}\n")

    videos = collect_videos(data_dir, args.layout, args.video_id)
    if args.limit:
        videos = videos[:args.limit]
    print(f"{len(videos)} video(s) found\n")

    # ── Step 1: transcribe all videos (Whisper), then unload ──
    print("Loading Whisper...")
    whisper_model = whisper.load_model(WHISPER_SIZE, device=device)
    print(f"Transcribing {len(videos)} video(s)...\n")
    transcripts = {}
    for vid_id, vp in videos:
        cache = output_dir / vid_id / "transcript.json"
        cache.parent.mkdir(parents=True, exist_ok=True)
        if cache.exists():
            print(f"  [cached] {vid_id}")
            with open(cache) as f:
                transcripts[vid_id] = json.load(f)
            # whisper_s absent for cached entries — that's fine
        else:
            print(f"  Transcribing {vid_id}...")
            t0_whisper = time.perf_counter()
            data = transcribe(vp, whisper_model)
            data["whisper_s"] = round(time.perf_counter() - t0_whisper, 3)
            with open(cache, "w") as f:
                json.dump(data, f, indent=2)
            transcripts[vid_id] = data
            lang = data["language"]
            prev = data["transcript"][:80]
            print(f"    lang={lang}  whisper={data['whisper_s']:.1f}s  preview: {prev}...")
    del whisper_model
    torch.cuda.empty_cache()
    print("\nWhisper unloaded.\n")

    # ── Step 2: EasyOCR ──
    print("Loading EasyOCR...")
    ocr_reader = easyocr.Reader(["en"], gpu=(device == "cuda"))
    print("EasyOCR ready.\n")

    # ── Step 3: LLaVA ──
    print(f"Loading LLaVA-NeXT ({MODEL_ID})...")
    quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    processor = LlavaNextProcessor.from_pretrained(MODEL_ID)
    model     = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=quant_cfg, low_cpu_mem_usage=True)
    print("LLaVA ready.\n")

    # ── Step 4: analyze ──
    results = []
    for i, (vid_id, vp) in enumerate(videos, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(videos)}]  {vid_id}  ({vp})")
        print(f"{'='*70}")

        out_file = output_dir / vid_id / "analysis.json"
        if out_file.exists():
            print("  Already analyzed — skipping (delete analysis.json to re-run)")
            with open(out_file) as f:
                results.append({"video_id": vid_id, "video_path": str(vp),
                                 "llava_analysis": json.load(f)})
            continue

        try:
            analysis = analyze_video(vp, transcripts[vid_id],
                                     ocr_reader, processor, model, device)
            record = {"video_id": vid_id, "video_path": str(vp), "llava_analysis": analysis}
            results.append(record)
            with open(out_file, "w") as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            print(f"  Saved → {out_file}")
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()

    # ── Summary ──
    summary_path = output_dir / "all_results.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"Done. {len(results)}/{len(videos)} videos analyzed.")
    print(f"Results: {summary_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
