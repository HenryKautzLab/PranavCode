#!/usr/bin/env bash
set -euo pipefail

VENV=/p/zenodo/code/reddit/arctic_shift/venv/bin/activate
source "$VENV"

SCRIPT_DIR=/p/zenodo/code/reddit/arctic_shift/scripts/llm_pipeline
cd "$SCRIPT_DIR"

python s7_make_eval_sample.py \
  --input-jsonl /p/zenodo/code/reddit/output/llm_pipeline/labeled_all.jsonl \
  --out-csv /p/zenodo/code/reddit/output/llm_pipeline/eval/eval_sample_100.csv \
  --n 100 \
  --pos-frac 0.5
