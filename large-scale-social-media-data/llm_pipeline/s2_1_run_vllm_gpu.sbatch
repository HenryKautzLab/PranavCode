#!/usr/bin/env bash
#SBATCH --job-name=vllm-maha
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/p/zenodo/code/reddit/output/llm_pipeline/logs/vllm_%j.out
#SBATCH --error=/p/zenodo/code/reddit/output/llm_pipeline/logs/vllm_%j.err

module load cuda
source /p/zenodo/code/reddit/arctic_shift/venv/bin/activate

MODEL=Qwen/Qwen2.5-3B-Instruct
PORT=8000

python -m vllm.entrypoints.openai.api_server \
  --model $MODEL \
  --host 0.0.0.0 \
  --port $PORT \
  --max-model-len 8192
