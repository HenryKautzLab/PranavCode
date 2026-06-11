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
