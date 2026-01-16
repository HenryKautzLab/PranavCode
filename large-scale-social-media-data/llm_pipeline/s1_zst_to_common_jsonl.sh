#!/usr/bin/env bash
set -euo pipefail

VENV=/p/zenodo/code/reddit/arctic_shift/venv/bin/activate
source "$VENV"

SCRIPT_DIR=/p/zenodo/code/reddit/arctic_shift/scripts/llm_pipeline
cd "$SCRIPT_DIR"

OUT_ROOT=/p/zenodo/code/reddit/output/llm_pipeline
MAP_DIR=$OUT_ROOT/file_map
JSONL_DIR=$OUT_ROOT/jsonl

mkdir -p "$JSONL_DIR"

FILES="$MAP_DIR/files_all.txt"
test -f "$FILES"

while read -r ZST; do
  # collision-proof: encode full path into filename
  BN=$(echo "$ZST" | sed 's|^/||; s|/|__|g; s|\.zst$||')

  python s1_zst_to_common_jsonl.py \
    --input-zst "$ZST" \
    --output-jsonl "$JSONL_DIR/${BN}.jsonl"

done < "$FILES"

echo "DONE: wrote jsonl files into $JSONL_DIR"
