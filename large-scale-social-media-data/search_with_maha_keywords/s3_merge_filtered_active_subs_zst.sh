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

python s2_filter_active_subs_zst.py \
  --stage merge_counts \
  --counts-dir /p/zenodo/code/reddit/output/submission_counts_2020_2025 \
  --min-posts 100 \
  --active-subs-pkl /p/zenodo/code/reddit/output/submission_counts_2020_2025_merged/active_subs_min100_2020_2025.pkl