"""
Benchmark evaluation: compare all model × modality combinations against
human ground truth annotations.

Metrics:
  - Semantic similarity (cosine) of content_summary vs gt_summary
  - Semantic similarity of key_message vs gt_key_message
  - Label accuracy for content_category and risk_level (exact first-word match)

Model × modality combinations tracked:
  llava   / visual_only          (from compare_modalities.py → modality_comparison.json)
  llava   / visual_audio_ocr     (from compare_modalities.py → modality_comparison.json)
  gemma4  / video_only           (from analyze_gemma4.py → analysis_gemma4_video_only.json)
  gemma4  / all_modality         (from analyze_gemma4.py → analysis_gemma4_all_modality.json)
  gemma4  / all_plus_metadata    (from analyze_gemma4.py → analysis_gemma4_all_plus_metadata.json)
  gemma4  / metadata_only        (from analyze_gemma4.py → analysis_gemma4_metadata_only.json)
  metadata_only / text_baseline  (from analyze_metadata_only.py → analysis_metadata_only.json)

Usage:
  python evaluate_benchmark.py \\
      --gt ../../data/classification/ground_truth.csv \\
      --llava-dir ../../data/outputs/llava \\
      --gemma4-dir ../../data/outputs/gemma4 \\
      --metadata-dir ../../data/outputs/metadata_only \\
      --out ../../data/classification/benchmark_results.csv

  # Skip models you haven't run yet:
  python evaluate_benchmark.py --gt ground_truth.csv --gemma4-dir outputs/gemma4
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

# ── Label fields used for accuracy, text fields used for semantic similarity ──
TEXT_FIELDS  = ["content_summary", "key_message"]
LABEL_FIELDS = ["content_category", "risk_level"]

GT_TEXT_MAP  = {"content_summary": "gt_summary", "key_message": "gt_key_message"}
GT_LABEL_MAP = {"content_category": "gt_content_category", "risk_level": "gt_risk_level"}


# ── Ground truth loading ──────────────────────────────────────────────────────

def load_ground_truth(gt_csv: Path) -> dict[str, dict]:
    with open(gt_csv, newline="", encoding="utf-8") as f:
        return {row["video_id"]: row for row in csv.DictReader(f)}


def gt_is_annotated(gt_row: dict) -> bool:
    return any(gt_row.get(c, "").strip()
               for c in ("gt_summary", "gt_key_message",
                         "gt_content_category", "gt_risk_level"))


# ── Output loading ────────────────────────────────────────────────────────────

def load_llava_outputs(llava_dir: Path,
                       video_ids: list[str]) -> dict[str, dict]:
    """
    Load modality_comparison.json written by compare_modalities.py.
    Returns {video_id: {modality_key: {field: answer, ...}, ...}}
    """
    out = {}
    for vid_id in video_ids:
        p = llava_dir / vid_id / "modality_comparison.json"
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            out[vid_id] = data.get("modalities", {})
    return out


def load_gemma4_outputs(gemma4_dir: Path, mode: str,
                         video_ids: list[str]) -> dict[str, dict]:
    """Load analysis_gemma4_{mode}.json written by analyze_gemma4.py."""
    out = {}
    fname = f"analysis_gemma4_{mode}.json"
    for vid_id in video_ids:
        p = gemma4_dir / vid_id / fname
        if p.exists():
            with open(p) as f:
                out[vid_id] = json.load(f)
    return out


def load_metadata_outputs(meta_dir: Path,
                           video_ids: list[str]) -> dict[str, dict]:
    """Load analysis_metadata_only.json written by analyze_metadata_only.py."""
    out = {}
    for vid_id in video_ids:
        p = meta_dir / vid_id / "analysis_metadata_only.json"
        if p.exists():
            with open(p) as f:
                out[vid_id] = json.load(f)
    return out


# ── Embedding model ───────────────────────────────────────────────────────────

_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def cosine_sim(t1: str, t2: str) -> Optional[float]:
    if not t1.strip() or not t2.strip():
        return None
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    em = get_embed_model()
    e1 = em.encode([t1])
    e2 = em.encode([t2])
    return float(cosine_similarity(e1, e2)[0][0])


def label_match(pred: str, gt: str) -> Optional[bool]:
    if not pred.strip() or not gt.strip():
        return None
    return pred.strip().lower().split()[0] == gt.strip().lower().split()[0]


# ── Per-video scoring ─────────────────────────────────────────────────────────

def score_prediction(pred: dict, gt_row: dict) -> dict:
    """Return per-field similarity and label accuracy for one video."""
    scores = {}
    for field in TEXT_FIELDS:
        gt_key = GT_TEXT_MAP[field]
        sim = cosine_sim(pred.get(field, ""), gt_row.get(gt_key, ""))
        scores[f"sim_{field}"] = round(sim, 4) if sim is not None else None

    for field in LABEL_FIELDS:
        gt_key = GT_LABEL_MAP[field]
        match  = label_match(pred.get(field, ""), gt_row.get(gt_key, ""))
        scores[f"acc_{field}"] = int(match) if match is not None else None

    # Avg similarity (over available pairs only)
    sims = [v for k, v in scores.items() if k.startswith("sim_") and v is not None]
    scores["avg_text_sim"] = round(sum(sims) / len(sims), 4) if sims else None

    accs = [v for k, v in scores.items() if k.startswith("acc_") and v is not None]
    scores["avg_label_acc"] = round(sum(accs) / len(accs), 4) if accs else None

    return scores


# ── Aggregate across all videos ───────────────────────────────────────────────

def aggregate(per_video_scores: list[dict]) -> dict:
    agg = {}
    all_keys = set(k for s in per_video_scores for k in s)
    for k in sorted(all_keys):
        vals = [s[k] for s in per_video_scores if s.get(k) is not None]
        agg[k] = round(sum(vals) / len(vals), 4) if vals else None
        agg[k + "_n"] = len(vals)
    return agg


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gt",           required=True,
                   help="Path to ground_truth.csv (with gt_* columns filled in)")
    p.add_argument("--llava-dir",    default=None,
                   help="Output dir from compare_modalities.py")
    p.add_argument("--gemma4-dir",   default=None,
                   help="Output dir from analyze_gemma4.py")
    p.add_argument("--metadata-dir", default=None,
                   help="Output dir from analyze_metadata_only.py")
    p.add_argument("--out",          default="../../data/classification/benchmark_results.csv",
                   help="Where to write the per-video results CSV")
    p.add_argument("--no-embed",     action="store_true",
                   help="Skip semantic similarity (label accuracy only, faster)")
    return p.parse_args()


def main():
    args   = parse_args()
    gt     = load_ground_truth(Path(args.gt))
    videos = list(gt.keys())

    annotated = [v for v in videos if gt_is_annotated(gt[v])]
    print(f"Ground truth: {len(videos)} total, {len(annotated)} annotated\n")
    if not annotated:
        print("No annotated rows found in ground_truth.csv. "
              "Fill in gt_summary, gt_key_message, gt_content_category, "
              "gt_risk_level for at least a few videos and re-run.")
        return

    eval_videos = annotated

    # ── Collect model outputs ──
    conditions: dict[str, dict[str, dict]] = {}   # {condition_label: {video_id: pred_dict}}

    if args.llava_dir:
        llava_dir  = Path(args.llava_dir)
        llava_outs = load_llava_outputs(llava_dir, eval_videos)
        for mod_key in ("visual_only", "visual_audio_ocr", "visual_audio", "visual_ocr"):
            label = f"llava/{mod_key}"
            preds = {vid: outs[mod_key]
                     for vid, outs in llava_outs.items()
                     if mod_key in outs}
            if preds:
                conditions[label] = preds
                print(f"  Loaded {label}: {len(preds)} videos")

    if args.gemma4_dir:
        gemma4_dir = Path(args.gemma4_dir)
        for mode in ("video_only", "all_modality", "all_plus_metadata", "metadata_only"):
            label  = f"gemma4/{mode}"
            preds  = load_gemma4_outputs(gemma4_dir, mode, eval_videos)
            if preds:
                conditions[label] = preds
                print(f"  Loaded {label}: {len(preds)} videos")

    if args.metadata_dir:
        meta_dir = Path(args.metadata_dir)
        preds    = load_metadata_outputs(meta_dir, eval_videos)
        if preds:
            conditions["metadata_only/text_baseline"] = preds
            print(f"  Loaded metadata_only/text_baseline: {len(preds)} videos")

    if not conditions:
        print("\nNo model outputs found. Run the analysis scripts first, "
              "then re-run evaluate_benchmark.py with the output directories.")
        return

    print()

    # ── Score each condition ──
    all_rows   = []
    agg_rows   = []

    for cond_label, preds in conditions.items():
        per_video = []
        for vid_id in eval_videos:
            if vid_id not in preds:
                continue
            pred   = preds[vid_id]
            gt_row = gt[vid_id]

            if args.no_embed:
                scores = {}
                for field in LABEL_FIELDS:
                    gt_key = GT_LABEL_MAP[field]
                    m = label_match(pred.get(field, ""), gt_row.get(gt_key, ""))
                    scores[f"acc_{field}"] = int(m) if m is not None else None
                accs = [v for v in scores.values() if v is not None]
                scores["avg_label_acc"] = round(sum(accs)/len(accs), 4) if accs else None
                scores["avg_text_sim"]  = None
            else:
                scores = score_prediction(pred, gt_row)

            per_video.append(scores)
            all_rows.append({
                "condition":   cond_label,
                "video_id":    vid_id,
                "platform":    gt_row.get("platform", ""),
                **scores,
            })

        agg = aggregate(per_video)
        agg_rows.append({"condition": cond_label, "n_videos": len(per_video), **agg})
        print(f"  {cond_label:<35}  n={len(per_video):<4}  "
              f"avg_sim={agg.get('avg_text_sim','—')}  "
              f"avg_acc={agg.get('avg_label_acc','—')}")

    # ── Write per-video CSV ──
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if all_rows:
        all_cols = list(dict.fromkeys(k for r in all_rows for k in r))
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nPer-video results → {out_path}")

    # ── Write aggregate CSV ──
    agg_path = out_path.with_name(out_path.stem + "_aggregate.csv")
    if agg_rows:
        agg_cols = list(dict.fromkeys(k for r in agg_rows for k in r))
        with open(agg_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=agg_cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(agg_rows)
        print(f"Aggregate results  → {agg_path}")

    # ── Print summary table ──
    print(f"\n{'Condition':<35}  {'N':>4}  {'Avg Sem Sim':>12}  {'Avg Label Acc':>14}")
    print("-" * 72)
    for r in sorted(agg_rows, key=lambda x: (x.get("avg_text_sim") or 0), reverse=True):
        sim = f"{r['avg_text_sim']:.4f}" if r.get("avg_text_sim") is not None else "   N/A  "
        acc = f"{r['avg_label_acc']:.4f}" if r.get("avg_label_acc") is not None else "   N/A  "
        print(f"  {r['condition']:<33}  {r['n_videos']:>4}  {sim:>12}  {acc:>14}")


if __name__ == "__main__":
    main()
