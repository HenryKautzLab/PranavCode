"""
Modality ablation: run the same LLaVA questions under 4 input conditions,
plus raw outputs for audio-only and OCR-only, then compare all 6 via
embedding cosine similarity against the full (visual+audio+OCR) pipeline.

Modalities:
  visual_only       — LLaVA frames, no audio, no OCR
  visual_ocr        — LLaVA frames + OCR text
  visual_audio      — LLaVA frames + Whisper transcript
  visual_audio_ocr  — LLaVA frames + Whisper + OCR  (full pipeline / reference)
  audio_only        — raw Whisper transcript (no LLM)
  ocr_only          — raw EasyOCR text (no LLM)

Output: {OUTPUT_DIR}/{video_id}/modality_comparison.json
        {OUTPUT_DIR}/{video_id}/modality_scores.json   (cosine sim vs full)

Usage:
  python3 compare_modalities.py --data-dir ../../data --layout nested
  python3 compare_modalities.py --data-dir ../../data --video-id DUN5Eo8AX9y
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import cv2
import easyocr
import numpy as np
import torch
import whisper
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (BitsAndBytesConfig,
                          LlavaNextForConditionalGeneration,
                          LlavaNextProcessor)

# ── Config ────────────────────────────────────────────────────────────────────
NUM_FRAMES      = 10
FRAME_POSITIONS = [i / (NUM_FRAMES - 1) * 0.95 for i in range(NUM_FRAMES)]
MODEL_ID        = "llava-hf/llava-v1.6-mistral-7b-hf"
WHISPER_SIZE    = "base"
EMBED_MODEL_ID  = "all-MiniLM-L6-v2"

QUESTIONS = {
    "content_summary":       "Describe what is happening in this video. What are the main visual elements, people, objects, and actions?",
    "emotional_tone":        "What emotions does this content evoke? Describe the mood and emotional impact.",
    "persuasion_techniques": "What persuasive or rhetorical techniques are being used? (visual appeal, emotional manipulation, social proof, authority, urgency)",
    "target_audience":       "Who is the intended audience? What age group, demographic, or interest group?",
    "credibility_assessment":"Does this content appear credible? Are there any red flags or signs of manipulation?",
    "misinformation_risk":   "Could this content spread misinformation? What are the risks to viewers?",
    "behavioral_impact":     "How might this video influence viewer behavior, beliefs, or actions?",
    "content_category":      "Category: entertainment, education, health, political, product promotion, or other?",
    "key_message":           "What is the main message or takeaway viewers get from this video?",
    "risk_level":            "Risk level (low/medium/high) and brief reason.",
}

# Fields used for embedding-similarity comparison
TEXT_FIELDS  = ["content_summary", "emotional_tone", "persuasion_techniques",
                "target_audience", "credibility_assessment", "misinformation_risk",
                "behavioral_impact", "key_message"]
LABEL_FIELDS = ["content_category", "risk_level"]


# ── Video helpers ─────────────────────────────────────────────────────────────

def get_duration(vp: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(vp)],
        capture_output=True, text=True, timeout=15)
    try:
        return float(json.loads(r.stdout)["format"]["duration"])
    except Exception:
        return 0.0


def extract_frames(vp: Path) -> list[Image.Image]:
    duration = get_duration(vp)
    if duration <= 0:
        return []
    cap   = cv2.VideoCapture(str(vp))
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


def has_audio(vp: Path) -> bool:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_name",
         "-of", "default=noprint_wrappers=1", str(vp)],
        capture_output=True, text=True)
    return bool(r.stdout.strip())


def transcribe(vp: Path, wm) -> dict:
    if not has_audio(vp):
        return {"transcript": "", "language": "unknown", "segments": []}
    r = wm.transcribe(str(vp))
    return {
        "transcript": r["text"].strip(),
        "language":   r["language"],
        "segments":   [{"start": s["start"], "end": s["end"], "text": s["text"]}
                       for s in r["segments"]],
    }


def extract_ocr(frames: list[Image.Image], reader) -> list[str]:
    seen, out = set(), []
    for frame in frames:
        for t in reader.readtext(np.array(frame), detail=0):
            c = t.strip()
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
    full_q = ("\n".join(context) + "\n\nUsing what you see AND the above: " + question
              if context else question)

    conv   = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": full_q}]}]
    prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    out    = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    dec    = processor.decode(out[0], skip_special_tokens=True)
    return dec.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in dec else dec.strip()


def run_llava_modality(frames: list[Image.Image], transcript: str, ocr_text: list,
                       processor, model, device: str,
                       use_audio: bool, use_ocr: bool) -> dict:
    mid  = frames[len(frames) // 2]
    t    = transcript if use_audio else ""
    ocr  = ocr_text   if use_ocr   else None
    out  = {}
    for key, q in QUESTIONS.items():
        out[key] = ask_llava(mid, q, processor, model, device, transcript=t, ocr_text=ocr)
    return out


# ── Comparison ────────────────────────────────────────────────────────────────

def compare_to_reference(results: dict, reference_key: str,
                          embed_model) -> dict:
    """
    Compute per-field cosine similarity of each modality vs the reference (full pipeline).
    Returns {modality: {field: score, ..., "overall_avg": score}}.
    """
    ref  = results[reference_key]
    out  = {}
    for mod_key, mod_data in results.items():
        if mod_key in ("audio_only", "ocr_only"):
            # text-only outputs — skip field comparison (no question answers)
            continue
        if mod_key == reference_key:
            continue
        scores = {}
        for field in TEXT_FIELDS:
            t1 = ref.get(field, "")
            t2 = mod_data.get(field, "")
            if t1 and t2:
                e1 = embed_model.encode([t1])
                e2 = embed_model.encode([t2])
                scores[field] = float(cosine_similarity(e1, e2)[0][0])
        label_matches = {}
        for field in LABEL_FIELDS:
            t1 = ref.get(field, "").lower()
            t2 = mod_data.get(field, "").lower()
            label_matches[field] = (t1.split()[0] == t2.split()[0]) if t1 and t2 else False
        avg = sum(scores.values()) / len(scores) if scores else 0.0
        out[mod_key] = {"field_similarity": scores,
                        "label_agreement":  label_matches,
                        "overall_avg_sim":  round(avg, 4)}
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def collect_videos(data_dir: Path, layout: str, video_id: str = None):
    if video_id:
        vp = data_dir / video_id / "video.mp4" if layout == "nested" else data_dir / f"{video_id}.mp4"
        return [(video_id, vp)] if vp.exists() else []
    if layout == "nested":
        return [(d.name, d / "video.mp4") for d in sorted(data_dir.iterdir())
                if d.is_dir() and (d / "video.mp4").exists()]
    return [(vp.stem, vp) for vp in sorted(data_dir.glob("*.mp4"))]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",   default="../../data")
    p.add_argument("--output-dir", default="../../data/outputs")
    p.add_argument("--layout",     choices=["flat", "nested"], default="nested")
    p.add_argument("--video-id",   default=None, help="Process a single video by ID")
    p.add_argument("--limit",      type=int, default=None)
    return p.parse_args()


def main():
    args       = parse_args()
    data_dir   = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device     = "cuda" if torch.cuda.is_available() else "cpu"

    videos = collect_videos(data_dir, args.layout, args.video_id)
    if args.limit:
        videos = videos[:args.limit]
    print(f"Device : {device}")
    print(f"Videos : {len(videos)}\n")

    # ── Load models ──
    print("Loading Whisper...")
    wm = whisper.load_model(WHISPER_SIZE, device=device)

    print("Loading EasyOCR...")
    ocr_reader = easyocr.Reader(["en"], gpu=(device == "cuda"))

    print(f"Loading LLaVA-NeXT ({MODEL_ID})...")
    qcfg      = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    processor = LlavaNextProcessor.from_pretrained(MODEL_ID)
    llava     = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=qcfg, low_cpu_mem_usage=True)

    print(f"Loading embedding model ({EMBED_MODEL_ID})...")
    embed_model = SentenceTransformer(EMBED_MODEL_ID)
    print("All models loaded.\n")

    for vid_id, vp in videos:
        print(f"\n{'='*70}\n{vid_id}\n{'='*70}")
        vid_out = output_dir / vid_id
        vid_out.mkdir(parents=True, exist_ok=True)

        comp_path = vid_out / "modality_comparison.json"
        if comp_path.exists():
            print("  Already compared — skipping (delete modality_comparison.json to re-run)")
            continue

        timings: dict = {}

        # ── Extract base inputs ──
        print("  Extracting frames...")
        t0 = time.perf_counter()
        frames = extract_frames(vp)
        timings["frame_extraction_s"] = round(time.perf_counter() - t0, 3)
        if not frames:
            print("  ERROR: no frames extracted")
            continue

        print("  Transcribing audio...")
        t0 = time.perf_counter()
        audio_data = transcribe(vp, wm)
        timings["whisper_s"] = round(time.perf_counter() - t0, 3)
        transcript = audio_data["transcript"]

        print("  Running OCR...")
        t0 = time.perf_counter()
        ocr_text = extract_ocr(frames, ocr_reader)
        timings["ocr_s"] = round(time.perf_counter() - t0, 3)

        print(f"  Frames={len(frames)}  whisper={timings['whisper_s']:.1f}s"
              f"  ocr={timings['ocr_s']:.1f}s  Audio='{transcript[:60]}...'  OCR={ocr_text[:3]}")

        # ── Run each modality ──
        modalities = {}
        pipeline_timings: dict = {}
        configs = [
            ("visual_only",      False, False),
            ("visual_ocr",       False, True),
            ("visual_audio",     True,  False),
            ("visual_audio_ocr", True,  True),   # full — used as reference
        ]
        for mod_key, use_audio, use_ocr in configs:
            print(f"\n  [{mod_key}]")
            t0 = time.perf_counter()
            modalities[mod_key] = run_llava_modality(
                frames, transcript, ocr_text, processor, llava, device,
                use_audio=use_audio, use_ocr=use_ocr)
            pipeline_timings[mod_key + "_s"] = round(time.perf_counter() - t0, 3)
            print(f"    done in {pipeline_timings[mod_key + '_s']:.1f}s")

        # raw text-only outputs (no LLM)
        modalities["audio_only"] = {
            "raw_transcript": transcript,
            "language":       audio_data["language"],
            "segments":       audio_data["segments"],
            "note":           "Raw Whisper output — no LLM involved",
        }
        modalities["ocr_only"] = {
            "raw_ocr_text": ocr_text,
            "note":         "Raw EasyOCR output — no LLM involved",
        }

        timings["pipelines"] = pipeline_timings
        timings["total_s"] = round(
            timings["frame_extraction_s"]
            + timings["whisper_s"]
            + timings["ocr_s"]
            + sum(pipeline_timings.values()),
            3,
        )

        # ── Compute similarity scores ──
        print("\n  Computing embedding similarity vs full pipeline...")
        scores = compare_to_reference(modalities, "visual_audio_ocr", embed_model)

        # ── Save ──
        comparison_record = {
            "video_id":           vid_id,
            "video_path":         str(vp),
            "modalities":         modalities,
            "whisper_transcript": transcript,
            "whisper_language":   audio_data["language"],
            "ocr_text":           ocr_text,
            "num_frames":         len(frames),
            "pipeline_timings":   timings,
        }
        with open(comp_path, "w") as f:
            json.dump(comparison_record, f, indent=2, ensure_ascii=False)

        scores_path = vid_out / "modality_scores.json"
        with open(scores_path, "w") as f:
            json.dump(scores, f, indent=2)

        # ── Print summary ──
        print(f"\n  Pipeline timings:")
        print(f"  {'Stage':<25} {'Time (s)':>10}")
        print(f"  {'-'*37}")
        print(f"  {'frame_extraction':<25} {timings['frame_extraction_s']:>10.2f}")
        print(f"  {'whisper':<25} {timings['whisper_s']:>10.2f}")
        print(f"  {'ocr':<25} {timings['ocr_s']:>10.2f}")
        for mod_key, *_ in configs:
            print(f"  {mod_key:<25} {pipeline_timings[mod_key + '_s']:>10.2f}")
        print(f"  {'-'*37}")
        print(f"  {'TOTAL':<25} {timings['total_s']:>10.2f}")

        print(f"\n  Similarity vs full pipeline (visual_audio_ocr):")
        print(f"  {'Modality':<25} {'Avg sim':>8}  {'Label agreement'}")
        print(f"  {'-'*60}")
        for mk, s in scores.items():
            avg  = s["overall_avg_sim"]
            labs = "  ".join(f"{k}={'✓' if v else '✗'}"
                             for k, v in s["label_agreement"].items())
            print(f"  {mk:<25} {avg:>8.3f}  {labs}")

        print(f"\n  Saved → {comp_path}")
        print(f"  Saved → {scores_path}")

    print(f"\n{'='*70}\nDone.\n{'='*70}")


if __name__ == "__main__":
    main()
