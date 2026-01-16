#!/bin/bash
#SBATCH --job-name=reddit_count_subs
#SBATCH --output=logs/count_subs_%A_%a.out
#SBATCH --error=logs/count_subs_%A_%a.err
#SBATCH --array=0-1023
#SBATCH --partition=nolim    
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=120:00:00

source /p/zenodo/code/reddit/arctic_shift/venv/bin/activate
cd /p/zenodo/code/reddit/arctic_shift/scripts

python s5_reddit_maha_search_zst.py \
  --input-dir /p/zenodo/code/reddit/output/2020_2025_zst \
  --output-dir /p/zenodo/code/reddit/output/maha_themes_2020_2025_zst \
  --shard-id $SLURM_ARRAY_TASK_ID \
  --num-shards $SLURM_ARRAY_TASK_COUNT
