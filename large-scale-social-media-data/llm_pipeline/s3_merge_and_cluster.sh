#!/usr/bin/env bash
set -euo pipefail

VENV=/p/zenodo/code/reddit/arctic_shift/venv/bin/activate
source "$VENV"

SCRIPT_DIR=/p/zenodo/code/reddit/arctic_shift/scripts/llm_pipeline
cd "$SCRIPT_DIR"

OUT_ROOT=/p/zenodo/code/reddit/output/llm_pipeline
LABELED_DIR=$OUT_ROOT/sample
CLUSTER_DIR=$OUT_ROOT/cluster

mkdir -p "$CLUSTER_DIR"

MERGED=$OUT_ROOT/labeled_all.jsonl
cat "$LABELED_DIR"/*_labeled.jsonl > "$MERGED"

python s5_cluster_and_sample.py \
  --input-jsonl "$MERGED" \
  --out-dir "$CLUSTER_DIR" \
  --max-items 8000 \
  --min-conf 0.60 \
  --knn-k 30 \
  --sample-n 100 \
  --embed-model sentence-transformers/all-MiniLM-L6-v2

echo "WROTE:"
echo "  $CLUSTER_DIR/cluster_summary.csv"
echo "  $CLUSTER_DIR/sample_for_annotation.csv"
