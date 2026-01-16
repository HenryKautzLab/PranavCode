#!/bin/bash
#SBATCH --job-name=reddit_maha_icl
#SBATCH --output=logs/reddit_maha_icl_%A_%a.out
#SBATCH --error=logs/reddit_maha_icl_%A_%a.err
#SBATCH --partition=nolim    
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00

source /p/zenodo/code/reddit/arctic_shift/venv/bin/activate
cd /p/zenodo/code/reddit/arctic_shift/scripts/llm_pipeline

python s4_icl_maha_vllm.py \
  --input-jsonl /p/zenodo/code/reddit/output/llm_pipeline/sample/sample_1000.jsonl \
  --output-jsonl /p/zenodo/code/reddit/output/llm_pipeline/sample/sample_1000_labeled.jsonl \
  --base-url http://jaguar02:8000 \
  --model Qwen/Qwen2.5-3B-Instruct \
  --concurrency 16
