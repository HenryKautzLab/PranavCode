#!/usr/bin/env bash
#SBATCH --job-name=maha-qlora
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/p/zenodo/code/reddit/output/llm_pipeline/logs/qlora_%j.out
#SBATCH --error=/p/zenodo/code/reddit/output/llm_pipeline/logs/qlora_%j.err

module load cuda
source /p/zenodo/code/reddit/arctic_shift/venv/bin/activate

# Some RTX cards don't support bf16 well; fp16 is safer
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
TRAIN_JSONL=/p/zenodo/code/reddit/output/llm_pipeline/sft/sft_maha_theme.jsonl
OUT_DIR=/p/zenodo/code/reddit/output/llm_pipeline/sft/qlora_qwen25_3b_$(date +%Y%m%d_%H%M)

python /p/zenodo/code/reddit/arctic_shift/scripts/llm_pipeline/train_qlora_sft.py \
  --model "$BASE_MODEL" \
  --train_jsonl "$TRAIN_JSONL" \
  --output_dir "$OUT_DIR" \
  --epochs 3 \
  --lr 2e-4 \
  --batch_size 1 \
  --grad_accum 16 \
  --max_seq_len 2048 \
  --save_steps 100 \
  --logging_steps 10 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05
