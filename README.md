## Multimodal Pipeline — LLaVA + Whisper + EasyOCR

This branch extends the base LLaVA pipeline with audio and text extraction to give the model fuller context about each video.

### What Changed

- **OpenAI Whisper** — transcribes audio track and injects transcription into LLaVA prompt
- **EasyOCR** — extracts on-screen text (subtitles, captions) and injects into LLaVA prompt
- LLaVA alone misses audio context and can overlook on-screen text — Whisper and EasyOCR address both gaps directly

Whisper proved especially useful on news report videos where audio carries most of the meaning. EasyOCR adds critical context for subtitle-heavy content.

### Reliability Validation

Used **all-MiniLM-L6-v2** to measure semantic similarity between single-model (LLaVA only) and three-model outputs:

- Consistent **0.6-range similarity scores** across all tested videos
- Moderate similarity confirms added modalities increase output richness without sacrificing coherence
- Score stability across tested videos suggests consistent behavior across diverse content types

See `/results` for side-by-side output comparisons across all tested videos.
