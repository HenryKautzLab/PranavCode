# Plan: Merge frames into a grid for single LLaVA call (temporal progression)

## Context
`analyze_tiktok_LLaVA.py` currently makes 10 separate LLaVA calls for temporal progression — one per frame. This is the most expensive part of the pipeline. The goal is to stitch all 10 frames into a single labeled grid image and replace those 10 calls with 1, reducing inference time significantly at the cost of some per-frame detail.

## File to modify
`released_data/code/video_classification/PranavCode/TikTok-Study/models/analyze_tiktok_LLaVA.py`

---

## Implementation

### 1. Add `build_frame_grid(frames, cols=5)` helper
After `extract_ocr`, add a new function that:
- Resizes each frame to a fixed thumbnail size (e.g. 336×336 to match LLaVA-NeXT's tile size)
- Arranges them in a grid (2 rows × 5 cols for 10 frames)
- Draws a label on each cell: `"0%"`, `"11%"`, ..., `"95%"` using `PIL.ImageDraw`
- Returns a single `PIL.Image`

```python
from PIL import ImageDraw
import math

def build_frame_grid(frames: list[Image.Image], positions: list,
                     cols: int = 5, thumb: int = 336) -> Image.Image:
    rows = math.ceil(len(frames) / cols)
    grid = Image.new("RGB", (cols * thumb, rows * thumb), (0, 0, 0))
    draw = ImageDraw.Draw(grid)
    for i, (frame, frac) in enumerate(zip(frames, positions)):
        r, c = divmod(i, cols)
        cell = frame.resize((thumb, thumb), Image.LANCZOS)
        grid.paste(cell, (c * thumb, r * thumb))
        draw.text((c * thumb + 4, r * thumb + 4), f"{int(frac*100)}%", fill="white")
    return grid
```

### 2. Replace the temporal progression loop in `analyze_video`

**Remove** the 10-call loop and **replace with:**
```python
print(f"   Building frame grid ({len(frames)} frames → 1 LLaVA call)...")
grid_image = build_frame_grid(frames, FRAME_POSITIONS)
grid_prompt = (
    "This image is a grid of video frames in reading order (left→right, top→bottom), "
    f"labeled with their position in the video (0% to {int(FRAME_POSITIONS[-1]*100)}%). "
    "For each labeled frame, briefly describe what is shown."
)
t0 = time.perf_counter()
grid_description = ask_llava(grid_image, grid_prompt, processor, model, device,
                              transcript=transcript, ocr_text=ocr_text)
timings["llava_temporal_s"] = round(time.perf_counter() - t0, 3)
analysis["temporal_progression"] = grid_description  # single string instead of list
```

### 3. Update `analysis_method` string
```python
"analysis_method": f"LLaVA-NeXT ({MODEL_ID}) + Whisper-{WHISPER_SIZE} + EasyOCR (grid temporal)"
```

---

## Output schema change
`temporal_progression` changes from a **list of dicts** to a **single string**.

---

## Verification
```bash
python analyze_tiktok_LLaVA.py \
  --data-dir ".../Sabit/data" \
  --output-dir ".../Sabit/output" \
  --video-id DBZuWe6OyQW
```
Check that `analysis.json` has `temporal_progression` as a string and `pipeline_timings.llava_temporal_s` is lower.

---

# LLM Model Research: SOTA Video Understanding Models (June 2026)

## Current model in use

**`llava-hf/llava-v1.6-mistral-7b-hf`** — LLaVA-NeXT v1.6, Mistral-7B variant.

**Precision:** INT4 (post-training quantized via BitsAndBytesConfig), compute dtype FP16.
```python
quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
```
Base weights on HuggingFace are BF16 but loaded as INT4. This is PTQ (post-training quantization) — quality is degraded after-the-fact.

**Pipeline:** 3 separate components:
1. LLaVA-NeXT (visual QA on frames)
2. Whisper-base (audio transcription)
3. EasyOCR (on-screen text)

---

## SOTA alternatives as of mid-2026

### Models that replace all 3 pipeline components natively (video + audio + OCR in one call)

| Model | Video | Audio | OCR | VRAM | Availability |
|---|---|---|---|---|---|
| **Qwen3.5-Omni** | ✅ native | ✅ native | ✅ native | ~40GB | ⚠️ Proprietary — Alibaba Cloud API only, weights not freely downloadable |
| **Gemma 4 12B** | ✅ native | ✅ native | ✅ native | 16GB (Q4) | ✅ Open-source, Apache 2.0 |
| **Llama 4 Scout** | ✅ native (20h) | ❌ | ✅ native | 24GB+ | ✅ Open-weight (restricted license) |
| **Gemini 3 Flash** | ✅ native | ✅ native | ✅ native | N/A (API) | API only |

---

## Performance comparison

| Task | Qwen3.5-Omni | Gemma 4 12B | Gemini 3 Flash |
|---|---|---|---|
| MMMU Pro | ~67% | ~60% | ~79% |
| Audio-visual benchmarks | SOTA (215 tasks, beats Gemini 3.1 Pro on audio) | Strong for size | Near Pro-level |
| Video input | 400s @ 720p + 10h audio | Native (no published limit) | Native |

---

## Inference speed & execution time

| Model | VRAM | Output tok/s | Est. time per 30s TikTok video (11 questions) |
|---|---|---|---|
| **LLaVA-v1.6 7B INT4** (current) | ~8GB | ~30-50 tps | ~5-10 min (10 frame calls + Whisper + OCR separately) |
| **Gemma 4 12B QAT INT4** | 16GB | ~21 tps (RTX 4060) / ~58 tps (A100, optimized); 3x speedup with MTP drafters | ~30-60s on A100 |
| **Qwen3.5-Omni BF16** | ~40GB | ~50-100 tps (A100, estimated) | ~3-6 min |
| **Gemini 3 Flash** (API) | N/A | Not GPU-bound | ~15-30s per video |

---

## Pricing (Gemini 3 Flash API — for ~10K video dataset)

- Video input: 258 tokens/second → 30s video ≈ 7,700 tokens @ $0.50/1M
- Audio input: ~$1/1M tokens
- Output: ~2,000 tokens/video @ $3/1M
- **Estimate: ~$0.01–0.015 per video → ~$100–150 total for 10K videos**

For SLURM self-hosted (Gemma 4 / Qwen3.5-Omni): free compute but GPU-hours cost.
At ~30-60s/video on A100: 10K videos = ~83–166 GPU-hours.

---

## Quantization: PTQ vs QAT

| Model | Quantization type | Quality impact |
|---|---|---|
| **LLaVA-v1.6 (current)** | PTQ INT4 (BitsAndBytesConfig) | Noticeable quality loss |
| **Gemma 4 12B** | QAT INT4 (`gemma-4-12B-it-qat-q4_0`) — trained to be quantized | Much better quality than PTQ at same memory |
| **Qwen3.5-Omni** | BF16 base; INT4/INT8/FP8/GGUF available (PTQ) | BF16 = full quality; PTQ variants degrade |
| **Gemini 3 Flash** | Managed by Google | N/A |

**Key insight:** Gemma 4 QAT INT4 would give better quality than current LLaVA PTQ INT4, at similar or lower VRAM, while replacing all three pipeline components.

---

## Recommendation for this project

| Priority | Pick |
|---|---|
| **Best local/SLURM + data stays private** | Gemma 4 12B (QAT INT4) — Apache 2.0, 16GB VRAM, replaces Whisper+OCR+LLaVA |
| **Best quality open-source** | Qwen3.5-Omni — SOTA on 215 AV benchmarks, but proprietary (API only) and heavy |
| **Fastest turnaround, data can leave infra** | Gemini 3 Flash — ~15-30s/video, ~$150 for 10K videos, one API call per video |

---

# Benchmark Study: Modality Ablation (June 2026)

## Goal

Compare four model × modality conditions to quantify how much each input modality (video, audio, OCR, metadata) contributes to classification quality, and to determine whether Gemma 4 outperforms the current LLaVA pipeline.

## Conditions evaluated

| Condition label | Model | Modalities used |
|---|---|---|
| `llava/visual_only` | LLaVA-NeXT v1.6 7B INT4 | Frames only |
| `llava/visual_audio` | LLaVA-NeXT v1.6 7B INT4 | Frames + Whisper audio |
| `llava/visual_ocr` | LLaVA-NeXT v1.6 7B INT4 | Frames + EasyOCR |
| `llava/visual_audio_ocr` | LLaVA-NeXT v1.6 7B INT4 | Frames + Whisper + EasyOCR |
| `gemma4/video_only` | Gemma 4 12B QAT INT4 | Frames only |
| `gemma4/all_modality` | Gemma 4 12B QAT INT4 | Frames + audio |
| `gemma4/all_plus_metadata` | Gemma 4 12B QAT INT4 | Frames + audio + metadata |
| `gemma4/metadata_only` | Gemma 4 12B QAT INT4 | Metadata text only |
| `metadata_only/text_baseline` | Gemma 4 12B QAT INT4 | Metadata text only (standalone script) |

## Metrics

| Metric | How computed |
|---|---|
| `sim_content_summary` | Cosine similarity between predicted `content_summary` and human `gt_summary` (sentence-transformers `all-MiniLM-L6-v2`) |
| `sim_key_message` | Cosine similarity between predicted `key_message` and human `gt_key_message` |
| `avg_text_sim` | Mean of the two similarity scores |
| `acc_content_category` | Exact first-word match vs `gt_content_category` |
| `acc_risk_level` | Exact first-word match vs `gt_risk_level` |
| `avg_label_acc` | Mean of the two accuracy scores |

## Sample

100 videos: 50 TikTok (from `released_data/video_frames/`) + 50 Instagram (from `released_data/instagram_videos/`).
Sorted by `video_id` alphabetically → staged as `data/sampled_videos/001/video.mp4` … `100/video.mp4`.
Annotators fill in `data/classification/ground_truth.csv` before running `evaluate_benchmark.py`.

---

## Scripts

### `sample_videos.py`

**Purpose:** Sample 50 TikTok + 50 Instagram videos, copy them into numbered folders, and write two CSVs.

**What it does:**
1. Loads TikTok metadata from `video_cache/tiktok/tiktok_video_cache_*.csv` and Instagram from `video_cache/instagram/instagram_media_cache_*.csv` (newest-first, `enrich_status == "ok"` only)
2. Finds all `video_frames/{id}/video.mp4` (TikTok) and `instagram_videos/{id}/video.mp4` (Instagram)
3. Samples `N` per platform (default 50) using a fixed seed
4. Sorts combined list by `video_id` alphabetically
5. Copies (or symlinks with `--symlink`) each video into `data/sampled_videos/{row_id:03d}/video.mp4`
6. Writes `data/classification/sample_manifest.csv` — full metadata per video
7. Writes `data/classification/ground_truth.csv` — scaffold for human annotation (gt_* columns blank)

**Key columns in `sample_manifest.csv`:**
`row_id`, `video_id`, `platform`, `video_path`, `source_path`, `title`, `description`, `hashtags`, `uploader`, `upload_date`, `duration`, `view_count`, `like_count`, `comment_count`, `repost_count`, `webpage_url`

**Key columns in `ground_truth.csv`:**
`row_id`, `video_id`, `platform`, `video_path`, `gt_content_category`, `gt_risk_level`, `gt_summary`, `gt_key_message`, `gt_notes`

**Usage:**
```bash
python sample_videos.py                          # 50 per platform, copy
python sample_videos.py --n 25 --seed 99         # 25 per platform
python sample_videos.py --symlink                # symlink instead of copy
```

**Output structure:**
```
data/
  sampled_videos/
    001/video.mp4    ← row 1 in both CSVs (sorted by video_id)
    002/video.mp4
    ...
    100/video.mp4
  classification/
    sample_manifest.csv
    ground_truth.csv
```

---

### `analyze_gemma4.py`

**Purpose:** Run Gemma 4 12B (QAT INT4) on sampled videos in 4 modality modes.

**Model:** `google/gemma-4-12B-it-qat-q4_0-unquantized`
**Quantization:** BitsAndBytesConfig — INT4 QAT (quantization-aware trained, better quality than PTQ at same memory), BF16 compute, double quant. ~8–16 GB VRAM.

**Modes (`--mode`):**
- `video_only` — 10 frames extracted via cv2, passed as PIL images; no audio, no metadata
- `all_modality` — frames + audio (librosa at 16 kHz mono, passed as `{"type": "audio", "audio": array}`)
- `all_plus_metadata` — frames + audio + metadata text prefix in prompt
- `metadata_only` — metadata text only (no video or audio loaded)

**Frame extraction:** 10 frames at `FRAME_POSITIONS = [0%, 11%, 22%, …, 95%]`; passed as individual `{"type": "image", "image": PIL}` content parts with a temporal position label string appended.

**Single model call** for all 11 questions per video.

**Output:** `{output_dir}/{video_id}/analysis_gemma4_{mode}.json`

**11 questions:** `content_summary`, `emotional_tone`, `persuasion_techniques`, `target_audience`, `credibility_assessment`, `misinformation_risk`, `behavioral_impact`, `content_category`, `key_message`, `risk_level`, `audio_visual_alignment`

**Usage:**
```bash
python analyze_gemma4.py --data-dir ../../data/sampled_videos --mode all_modality
python analyze_gemma4.py --data-dir ../../data/sampled_videos --mode all_plus_metadata \
    --meta-csv ../../data/classification/sample_manifest.csv
python analyze_gemma4.py --video-id 001 --mode video_only --data-dir ../../data/sampled_videos
```

---

### `analyze_metadata_only.py`

**Purpose:** Text-only baseline — answer the same 11 classification questions using only platform metadata, without loading any video.

**Model:** Same Gemma 4 12B QAT INT4 (text-only path; `AutoProcessor` + `AutoModelForCausalLM`).

**Input:** `sample_manifest.csv` (via `--meta-csv`). Formats: platform, title, description, hashtags, uploader, upload_date, duration, view/like/comment/repost counts into a structured text block.

**Questions:** Same 11 questions as above, rephrased to say "based on the metadata."

**Flags:**
- `--dry-run` — print prompts without loading model (useful for prompt inspection)
- `--limit N` — process only first N rows
- `--model` — override model ID

**Output:** `{output_dir}/{video_id}/analysis_metadata_only.json` + `all_results_metadata_only.json`

**Usage:**
```bash
python analyze_metadata_only.py \
    --meta-csv ../../data/classification/sample_manifest.csv
python analyze_metadata_only.py --meta-csv sample_manifest.csv --dry-run
```

---

### `evaluate_benchmark.py`

**Purpose:** Score all model × modality conditions against human-annotated ground truth. Produces per-video and aggregate results.

**Inputs:**
- `ground_truth.csv` — filled-in `gt_*` columns from annotators
- `--llava-dir` — output dir from `compare_modalities.py` (contains `{video_id}/modality_comparison.json`)
- `--gemma4-dir` — output dir from `analyze_gemma4.py` (contains `{video_id}/analysis_gemma4_{mode}.json`)
- `--metadata-dir` — output dir from `analyze_metadata_only.py`

**How scoring works:**
1. Loads `sentence-transformers` `all-MiniLM-L6-v2` once
2. For each `(condition, video_id)`: cosine similarity for text fields, exact first-word match for label fields
3. Aggregates per condition across all annotated videos

**Flags:**
- `--no-embed` — skip semantic similarity (label accuracy only, faster, no GPU needed)

**Outputs:**
- `benchmark_results.csv` — one row per (condition, video_id)
- `benchmark_results_aggregate.csv` — one row per condition (means)
- Console summary table sorted by `avg_text_sim`

**Usage:**
```bash
python evaluate_benchmark.py \
    --gt ../../data/classification/ground_truth.csv \
    --llava-dir ../../data/outputs/llava \
    --gemma4-dir ../../data/outputs/gemma4 \
    --metadata-dir ../../data/outputs/metadata_only

# Label accuracy only (no sentence-transformers required):
python evaluate_benchmark.py --gt ground_truth.csv --gemma4-dir outputs/gemma4 --no-embed
```

---

## End-to-end workflow

```bash
# Step 1 — Sample videos and write CSVs
python sample_videos.py --symlink    # or omit --symlink to copy

# Step 2 — Human annotation (fill in ground_truth.csv)
# Open data/classification/ground_truth.csv; fill gt_content_category,
# gt_risk_level, gt_summary, gt_key_message for each row.

# Step 3 — Run LLaVA ablation (all 4 modality conditions)
python compare_modalities.py --data-dir ../../data/sampled_videos

# Step 4 — Run Gemma 4 (one mode at a time; each ~30-60s/video on A100)
for MODE in video_only all_modality all_plus_metadata metadata_only; do
  python analyze_gemma4.py \
    --data-dir ../../data/sampled_videos \
    --mode $MODE \
    --meta-csv ../../data/classification/sample_manifest.csv \
    --output-dir ../../data/outputs/gemma4
done

# Step 5 — Metadata-only baseline
python analyze_metadata_only.py \
    --meta-csv ../../data/classification/sample_manifest.csv \
    --output-dir ../../data/outputs/metadata_only

# Step 6 — Evaluate all conditions
python evaluate_benchmark.py \
    --gt ../../data/classification/ground_truth.csv \
    --llava-dir ../../data/outputs/llava \
    --gemma4-dir ../../data/outputs/gemma4 \
    --metadata-dir ../../data/outputs/metadata_only \
    --out ../../data/classification/benchmark_results.csv
```

---

# TikTok Video Download Strategy: yt-dlp + Bright Data (IP Rotation)

## Background

UVA cluster IPs are range-blocked by TikTok — yt-dlp is currently disabled in `tiktok_fetch_metadata.py` (oEmbed-only fallback). Bright Data is a commercial proxy service that rotates residential IPs, making requests appear to come from real home users rather than a datacenter.

Reference: https://medium.com/@dataforAI/scrape-youtube-videos-with-yt-dlp-and-bright-data-f9e8843c38b3

## Integration

```bash
yt-dlp \
  --proxy "http://USERNAME:PASSWORD@brd.superproxy.io:22225" \
  --sleep-interval 2 --max-sleep-interval 5 \
  --format "worstvideo" \
  -o "%(id)s.%(ext)s" \
  "https://www.tiktok.com/@x/video/VIDEO_ID"
```

Bright Data's super-proxy endpoint auto-rotates residential IPs per request. Combined with `--cookies-from-browser chrome` (logged-in TikTok session), success rate improves significantly over anonymous requests.

## Cost Estimate at 300k Videos

| Scenario | Avg size/video | Total bandwidth | Residential (~$8–15/GB) | Datacenter (~$0.60/GB) |
|----------|---------------|-----------------|--------------------------|------------------------|
| Full quality | ~25 MB | ~7.5 TB | $60k–$112k | $4,500 |
| Lowest quality (`worstvideo`) | ~5 MB | ~1.5 TB | $12,000 | ~$900 |

**Datacenter IPs are cheaper but more detectable by TikTok. Residential is safer but expensive at this scale.**

## Key Blockers Beyond IP Rotation

- **JavaScript challenges**: yt-dlp's TikTok extractor handles some but this is an active arms race
- **Session/cookie requirements**: anonymous requests increasingly return 403 — need a valid logged-in cookie
- **Device fingerprinting**: IP rotation alone may not be sufficient

## Recommended Approach Before Committing

1. **Audit live videos first**: Run oEmbed on all 300k IDs — many will be deleted/private. Actual downloadable set may be 30–40% (~90–120k), which cuts costs proportionally.
2. **Small pilot**: Test 100 IDs with Bright Data residential + cookies, measure success rate and actual bandwidth.
3. **SLURM integration**: Each worker uses a different Bright Data `session-ID` for a sticky IP per worker batch.
4. **Force lowest quality**: `--format "worstvideo"` — sufficient for visual classification, ~5x smaller files.
