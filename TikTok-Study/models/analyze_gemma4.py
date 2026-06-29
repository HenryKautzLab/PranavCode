"""
Video analysis pipeline: Gemma 4 12B (QAT) — native multimodal.

Unlike the LLaVA pipeline, Gemma 4 handles video frames and audio in a single
model call — no separate Whisper or EasyOCR step needed.

Modes (--mode):
  video_only        — 10 extracted frames, no audio, no metadata
  all_modality      — frames + native audio (librosa → 16 kHz mono array)
  metadata_only     — text-only prompt: title, hashtags, description, engagement
  all_plus_metadata — frames + audio + metadata text prefix

Output per video: {OUTPUT_DIR}/{video_id}/analysis_gemma4_{mode}.json
Shared transcript cache NOT written (audio handled natively by model).

Usage:
  python analyze_gemma4.py --data-dir ../../data/classification --mode all_modality
  python analyze_gemma4.py --data-dir ../../data/classification --mode metadata_only \\
      --meta-csv ../../data/classification/sample_manifest.csv
  python analyze_gemma4.py --video-id C6Zf0eHOzBi --mode all_plus_metadata \\
      --meta-csv ../../data/classification/sample_manifest.csv

NOTE: Model class names may vary by transformers version. If AutoModelForCausalLM
      fails to load vision/audio inputs, update to Gemma4ForConditionalGeneration
      and Gemma4Processor from a newer transformers release.
"""

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
# QAT-trained weights in BF16 — better quality at 4-bit than standard PTQ.
# Loaded with bitsandbytes 4bit below (~8 GB VRAM).
# Alternative: "google/gemma-4-12B-it" + standard bitsandbytes PTQ.
MODEL_ID        = "google/gemma-4-12B-it-qat-q4_0-unquantized"
NUM_FRAMES      = 10
FRAME_POSITIONS = [i / (NUM_FRAMES - 1) * 0.95 for i in range(NUM_FRAMES)]
AUDIO_SR        = 16_000   # Gemma 4 audio encoder expects 16 kHz mono

QUESTIONS = {
    "transcript": "Transcribe all spoken words, dialogue, narration, or lyrics in the audio exactly as heard. If there is no speech or audio, write 'No speech detected'.",
    "video_description":        "Describe each segment of the video. What is happening in this video? What are the main visual elements, people, objects, and actions?",
    "emotional_tone":         "What emotions does this content evoke? Determine one or more emotions from either positive or negative axis."
    "Negative: aggression, anger, disgust, dominant personality, hate, kill, negative emotion, nervousness, pain, rage, sadness, suffering, swearing terms, terrorism, violence. Or"
    "Positive: joy, love, optimist, politeness, positive emotion.",
    # "persuasion_techniques":  "What persuasive or rhetorical techniques are being used? Consider visual appeal, emotional manipulation, social proof, authority, or urgency tactics.",
    "target_audience":        "Who is the intended audience? What age group, demographic, or interest group would this appeal to?",
    # "credibility_assessment": "Does this content appear credible and trustworthy? Are there any red flags, misleading elements, or signs of manipulation?",
    # "misinformation_risk":    "Could this content spread misinformation or false claims? What are the potential risks to viewers?",
    # "behavioral_impact":      "How might this video influence viewer behavior, beliefs, or actions? What specific behaviors might it encourage?",
    "content_category":       "Category: entertainment, education, health, political, product promotion, or other? Choose one.",
    # "key_message":            "What is the main message or takeaway viewers are supposed to get from this video?",
    # "risk_level":             "On a scale of low/medium/high, what is the potential risk to viewers? Explain briefly.",
    "audio_visual_alignment": "Does the spoken audio and on-screen text align with what is shown visually? Note any mismatches.",
    "location": "Where is this video being filmed? Describe the location and setting (i.e., outside, inside, gym, home, office, etc.).",
    "gender": "What is the gender of the person(s) in the video? If multiple, describe each.",
    "action": "What is the main action or activity taking place in the video?",
    "gesture": "What gestures are being made by the person(s) in the video? Describe the hand movements and body language.",
    "body image": "Does this video show a positive or negative body image? Describe the portrayal of the human body.",
    "clothing": "What are the person(s) in the video wearing? Describe their clothing and accessories."
}

MODES = ["video_only", "all_modality", "metadata_only", "all_plus_metadata"]


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


def extract_frames(video_path: Path) -> list[Image.Image]:
    duration = get_duration(video_path)
    if duration <= 0:
        return []
    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = []
    for frac in FRAME_POSITIONS:
        ts       = max(0.0, min(duration * frac, duration - 0.1))
        frame_no = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_no, total - 1))
        ok, frame = cap.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def extract_audio(video_path: Path) -> "np.ndarray | None":
    """Load video audio track as float32 mono array at AUDIO_SR Hz."""
    try:
        import librosa
        arr, _ = librosa.load(str(video_path), sr=AUDIO_SR, mono=True)
        return arr.astype(np.float32)
    except Exception as e:
        print(f"   WARNING: audio extraction failed ({e})")
        return None


# ── Metadata ──────────────────────────────────────────────────────────────────

def load_metadata(meta_csv: "Path | None") -> dict[str, dict]:
    if not meta_csv or not meta_csv.exists():
        return {}
    result = {}
    with open(meta_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # key by row_id (folder name e.g. "001") AND by actual video_id
            if row.get("row_id"):
                result[row["row_id"]] = row
            if row.get("video_id"):
                result[row["video_id"]] = row
    return result


def build_metadata_text(meta: dict) -> str:
    parts = []
    for label, key in [
        ("Title",         "title"),
        ("Description",   "description"),
        ("Hashtags",      "hashtags"),
        ("Duration (s)",  "duration"),
        ("Views",         "view_count"),
        ("Likes",         "like_count"),
        ("Comments",      "comment_count"),
        ("Platform",      "platform"),
        ("Uploader",      "uploader"),
        ("Upload date",   "upload_date"),
    ]:
        val = meta.get(key, "")
        if val:
            parts.append(f"{label}: {val}")
    return "\n".join(parts)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model():
    from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.bfloat16,  # makes weight.dtype → bfloat16, not uint8
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_cfg,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()
    return processor, model


# ── Inference ─────────────────────────────────────────────────────────────────

def build_questions_block(meta_prefix: str = "") -> str:
    q_lines = "\n".join(f"{k}: {q}" for k, q in QUESTIONS.items())
    block = (
        "Answer ALL of the following questions about this video. "
        "Start each answer on a new line with the exact label and a colon.\n\n"
        + q_lines
    )
    if meta_prefix:
        block = f"VIDEO METADATA:\n{meta_prefix}\n\n" + block
    return block


def run_inference(content_parts: list, prompt_text: str, processor, model) -> str:
    messages = [
        {
            "role": "user",
            "content": content_parts + [{"type": "text", "text": prompt_text}],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=2000, do_sample=False)

    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


def parse_response(raw: str) -> dict:
    results = {}
    keys = list(QUESTIONS.keys())
    for i, key in enumerate(keys):
        marker = key + ":"
        start = raw.find(marker)
        if start == -1:
            results[key] = ""
            continue
        start += len(marker)
        end = len(raw)
        for nk in keys[i + 1:]:
            pos = raw.find(nk + ":", start)
            if pos != -1:
                end = pos
                break
        results[key] = raw[start:end].strip()
    return results


# ── Per-video analysis ────────────────────────────────────────────────────────

def analyze_video(video_path: Path, mode: str, meta: dict,
                  processor, model) -> dict:
    timings: dict = {}
    content_parts: list = []
    used_audio = False

    # ── Frames ──
    if mode in ("video_only", "all_modality", "all_plus_metadata"):
        print(f"   Extracting {NUM_FRAMES} frames...")
        t0 = time.perf_counter()
        frames = extract_frames(video_path)
        timings["frame_extraction_s"] = round(time.perf_counter() - t0, 3)
        if not frames:
            raise RuntimeError("No frames extracted — corrupt or unreadable video")
        # Pass frames as a sequence of images labeled with their temporal position.
        # Gemma 4 processors that support {"type":"video"} can receive a list here;
        # for older transformers versions each frame is passed as a separate image.
        for i, frame in enumerate(frames):
            content_parts.append({"type": "image", "image": frame})
        position_labels = ", ".join(f"{int(p*100)}%" for p in FRAME_POSITIONS[:len(frames)])
        content_parts.append({
            "type": "text",
            "text": (f"The above {len(frames)} images are frames extracted at "
                     f"{position_labels} through the video (temporal order, left-to-right).")
        })
        print(f"   Frames extracted in {timings['frame_extraction_s']:.2f}s")

    # ── Audio ──
    if mode in ("all_modality", "all_plus_metadata"):
        print("   Extracting audio...")
        t0 = time.perf_counter()
        audio = extract_audio(video_path)
        timings["audio_extraction_s"] = round(time.perf_counter() - t0, 3)
        if audio is not None:
            content_parts.append({"type": "audio", "audio": audio})
            used_audio = True
            print(f"   Audio extracted in {timings['audio_extraction_s']:.2f}s  "
                  f"({len(audio)/AUDIO_SR:.1f}s of audio)")
        else:
            print("   Audio unavailable — proceeding without audio modality")

    # ── Metadata prefix ──
    meta_text = ""
    if mode in ("metadata_only", "all_plus_metadata") and meta:
        meta_text = build_metadata_text(meta)

    prompt = build_questions_block(meta_prefix=meta_text)

    # ── Single Gemma 4 call ──
    print(f"   Gemma 4: 1 call for all {len(QUESTIONS)} questions...")
    t0 = time.perf_counter()
    raw = run_inference(content_parts, prompt, processor, model)
    timings["gemma4_s"] = round(time.perf_counter() - t0, 3)

    analysis = parse_response(raw)
    analysis["gemma4_raw_response"] = raw
    timings["total_s"] = round(sum(timings.values()), 3)
    analysis.update({
        "analysis_method":  f"Gemma4 ({MODEL_ID}) mode={mode}",
        "mode":             mode,
        "used_audio":       used_audio,
        "used_metadata":    bool(meta_text),
        "metadata_used":    meta_text,
        "pipeline_timings": timings,
    })
    return analysis


# ── Video collection ──────────────────────────────────────────────────────────

def collect_videos(data_dir: Path, layout: str,
                   video_id: str = None) -> list[tuple[str, Path]]:
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
        for ext in ("*.mp4", "*.mp3"):
            for vp in sorted(data_dir.glob(ext)):
                videos.append((vp.stem, vp))
    return videos


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",   default="../../data/classification",
                   help="Root dir with videos (nested: {id}/video.mp4 or flat: {id}.mp4)")
    p.add_argument("--output-dir", default="../../data/outputs/gemma4",
                   help="Where to write per-video analysis_gemma4_{mode}.json")
    p.add_argument("--layout",     choices=["flat", "nested"], default="nested")
    p.add_argument("--mode",       choices=MODES, default="all_modality")
    p.add_argument("--meta-csv",   default=None,
                   help="Path to sample_manifest.csv; required for metadata_only / all_plus_metadata")
    p.add_argument("--limit",      type=int, default=None)
    p.add_argument("--video-id",   default=None, help="Process a single video by ID")
    return p.parse_args()


def main():
    args       = parse_args()
    data_dir   = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_csv   = Path(args.meta_csv).resolve() if args.meta_csv else None

    print(f"Model : {MODEL_ID}")
    print(f"Mode  : {args.mode}")
    print(f"Data  : {data_dir}  [{args.layout}]")
    print(f"Output: {output_dir}\n")

    if args.mode in ("metadata_only", "all_plus_metadata") and not meta_csv:
        print("WARNING: metadata modes require --meta-csv; prompts will have no metadata context.\n")

    videos   = collect_videos(data_dir, args.layout, args.video_id)
    if args.limit:
        videos = videos[:args.limit]
    metadata = load_metadata(meta_csv)
    print(f"{len(videos)} video(s) found\n")

    print(f"Loading Gemma 4 ({MODEL_ID})...")
    processor, model = load_model()
    print("Gemma 4 ready.\n")

    out_key = f"analysis_gemma4_{args.mode}.json"
    results = []

    for i, (vid_id, vp) in enumerate(videos, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(videos)}]  {vid_id}  [{args.mode}]")
        print(f"{'='*70}")

        out_file = output_dir / vid_id / out_key
        if out_file.exists():
            print(f"  Already done — skipping (delete {out_key} to re-run)")
            with open(out_file) as f:
                results.append({"video_id": vid_id, "analysis": json.load(f)})
            continue

        try:
            meta     = metadata.get(vid_id, {})
            analysis = analyze_video(vp, args.mode, meta, processor, model)
            (output_dir / vid_id).mkdir(parents=True, exist_ok=True)
            with open(out_file, "w") as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            t = analysis["pipeline_timings"]
            print(f"  Saved → {out_file}")
            print(f"  Timings: gemma4={t.get('gemma4_s','?')}s  total={t.get('total_s','?')}s")
            results.append({"video_id": vid_id, "analysis": analysis})
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()

    summary_path = output_dir / f"all_results_gemma4_{args.mode}.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"Done. {len(results)}/{len(videos)} videos analyzed.")
    print(f"Results: {summary_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
