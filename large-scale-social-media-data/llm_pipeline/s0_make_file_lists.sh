find /p/zenodo/code/reddit/output/maha_themes_2020_2025_zst \
  -type f \
  \( -name "*.zst" \) \
  | sort > /p/zenodo/code/reddit/output/llm_pipeline/file_map/files_all.txt
