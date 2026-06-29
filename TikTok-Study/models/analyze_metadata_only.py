"""
Metadata-only classification baseline.

Uses a text LLM (Gemma 4 by default, no video/audio input at all) to answer
the same classification questions as the video-based pipelines, using only
platform metadata: title, description, hashtags, duration, engagement stats.

This is the "text baseline" in the modality ablation — it tells us how much
is learnable purely from metadata without ever looking at the video.

Output per video: {OUTPUT_DIR}/{video_id}/analysis_metadata_only.json
Summary:          {OUTPUT_DIR}/all_results_metadata_only.json

Usage:
  python analyze_metadata_only.py \\
      --meta-csv ../../data/classification/sample_manifest.csv

  # Use a different model:
  python analyze_metadata_only.py \\
      --meta-csv ../../data/classification/sample_manifest.csv \\
      --model google/gemma-4-12B-it-qat-q4_0-unquantized

  # Skip model loading entirely — output the prompts only (for manual inspection):
  python analyze_metadata_only.py --meta-csv sample_manifest.csv --dry-run
"""

import argparse
import csv
import json
import time
from pathlib import Path

import torch

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_MODEL_ID = "google/gemma-4-12B-it-qat-q4_0-unquantized"

QUESTIONS = {
    "content_summary":        "Describe in detail what you would expect this video to be about based on the metadata. What are the likely main visual elements, topics, and actions?",
    "emotional_tone":         "What emotions does this content likely evoke? Describe the probable mood and emotional impact on viewers.",
    "persuasion_techniques":  "What persuasive or rhetorical techniques are likely being used? Consider visual appeal, emotional manipulation, social proof, authority, or urgency tactics.",
    "target_audience":        "Who is the intended audience? What age group, demographic, or interest group would this appeal to?",
    "credibility_assessment": "Based on the metadata alone, does this content appear credible and trustworthy? Are there any red flags?",
    "misinformation_risk":    "Could this content spread misinformation or false claims? What are the potential risks based on the metadata?",
    "behavioral_impact":      "How might this video influence viewer behavior, beliefs, or actions based on what is described in the metadata?",
    "content_category":       "Category: entertainment, education, health, political, product promotion, or other? Choose one.",
    "key_message":            "What is the main message or takeaway the creator likely intends viewers to get?",
    "risk_level":             "On a scale of low/medium/high, what is the potential risk to viewers? Explain briefly.",
    "audio_visual_alignment": "Based on the title and description, do you expect the audio and visuals to align? Note any potential mismatches.",
}


# ── Metadata loading ──────────────────────────────────────────────────────────

def load_manifest(meta_csv: Path) -> list[dict]:
    with open(meta_csv, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_metadata_prompt(row: dict) -> str:
    """Format metadata row as a structured context block."""
    fields = [
        ("Platform",      row.get("platform", "")),
        ("Title",         row.get("title", "")),
        ("Description",   row.get("description", "")),
        ("Hashtags",      row.get("hashtags", "")),
        ("Uploader",      row.get("uploader", "")),
        ("Upload date",   row.get("upload_date", "")),
        ("Duration (s)",  row.get("duration", "")),
        ("View count",    row.get("view_count", "")),
        ("Like count",    row.get("like_count", "")),
        ("Comment count", row.get("comment_count", "")),
        ("Repost count",  row.get("repost_count", "")),
    ]
    lines = [f"{label}: {val}" for label, val in fields if val]
    if not lines:
        return "[No metadata available]"
    return "\n".join(lines)


def build_full_prompt(meta_block: str) -> str:
    q_block = "\n".join(f"{k}: {q}" for k, q in QUESTIONS.items())
    return (
        "You are analyzing a social media video using only its metadata "
        "(no actual video or audio is available to you).\n\n"
        f"METADATA:\n{meta_block}\n\n"
        "Based solely on the above metadata, answer ALL of the following questions. "
        "Start each answer on a new line with the exact label and a colon.\n\n"
        + q_block
    )


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(model_id: str):
    from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_cfg,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()
    return processor, model


def run_inference(prompt: str, processor, model) -> tuple[str, float]:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=2000, do_sample=False)
    elapsed = round(time.perf_counter() - t0, 3)

    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    response = processor.decode(new_tokens, skip_special_tokens=True).strip()
    return response, elapsed


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


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--meta-csv",   required=True,
                   help="Path to sample_manifest.csv from sample_videos.py")
    p.add_argument("--output-dir", default="../../data/outputs/metadata_only",
                   help="Where to write per-video analysis_metadata_only.json")
    p.add_argument("--model",      default=DEFAULT_MODEL_ID,
                   help="HuggingFace model ID for text inference")
    p.add_argument("--dry-run",    action="store_true",
                   help="Print prompts only, no model loading or inference")
    p.add_argument("--limit",      type=int, default=None)
    return p.parse_args()


def main():
    args       = parse_args()
    meta_csv   = Path(args.meta_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model    : {args.model}")
    print(f"Meta CSV : {meta_csv}")
    print(f"Output   : {output_dir}")
    print(f"Dry run  : {args.dry_run}\n")

    rows = load_manifest(meta_csv)
    if args.limit:
        rows = rows[:args.limit]
    print(f"{len(rows)} video(s) to process\n")

    processor = model = None
    if not args.dry_run:
        print(f"Loading {args.model}...")
        processor, model = load_model(args.model)
        print("Model ready.\n")

    results = []

    for i, row in enumerate(rows, 1):
        vid_id   = row["video_id"]
        platform = row.get("platform", "unknown")
        print(f"\n{'='*70}")
        print(f"[{i}/{len(rows)}]  {vid_id}  [{platform}]")
        print(f"{'='*70}")

        out_file = output_dir / vid_id / "analysis_metadata_only.json"

        if out_file.exists() and not args.dry_run:
            print("  Already done — skipping (delete analysis_metadata_only.json to re-run)")
            with open(out_file) as f:
                results.append({"video_id": vid_id, "analysis": json.load(f)})
            continue

        meta_block = build_metadata_prompt(row)
        prompt     = build_full_prompt(meta_block)

        if args.dry_run:
            print(f"\n--- PROMPT ---\n{prompt[:500]}...\n")
            continue

        try:
            t_start = time.perf_counter()
            raw, inference_s = run_inference(prompt, processor, model)
            total_s = round(time.perf_counter() - t_start, 3)

            analysis = parse_response(raw)
            analysis["raw_response"]      = raw
            analysis["metadata_used"]     = meta_block
            analysis["analysis_method"]   = f"metadata_only ({args.model})"
            analysis["pipeline_timings"]  = {"inference_s": inference_s, "total_s": total_s}
            analysis["video_id"]          = vid_id
            analysis["platform"]          = platform

            (output_dir / vid_id).mkdir(parents=True, exist_ok=True)
            with open(out_file, "w") as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)

            print(f"  Saved → {out_file}  ({inference_s}s)")
            results.append({"video_id": vid_id, "platform": platform, "analysis": analysis})

        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()

    if not args.dry_run:
        summary_path = output_dir / "all_results_metadata_only.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n{'='*70}")
        print(f"Done. {len(results)}/{len(rows)} processed.")
        print(f"Results: {summary_path}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
