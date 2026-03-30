## Model Benchmarking — Gemini vs. BLIP vs. LLaVA-NeXT

This branch documents the model selection process for image analysis, benchmarking three vision-language models against the core requirement: generating interpretive, emotionally aware captions useful for misinformation and persuasion analysis.

### Models Evaluated

| Model | Findings |
|---|---|
| Gemini | Strong interpretive output but hit free-tier rate limits — not viable at scale |
| BLIP | Lightweight and fast but captions were too generic for meaningful interpretation |
| LLaVA-NeXT | Best balance of interpretive depth and local deployment viability — selected as primary model |

### Key Finding

Caption specificity was the critical differentiator. BLIP's outputs described what was in an image; LLaVA-NeXT's outputs interpreted what it meant and how it would make viewers feel — which is exactly what this research requires.

Side-by-side output comparisons are available in the `/results` folder. The difference in output quality is significant.

### Takeaway

High-quality free models exist but require evaluation to find. LLaVA-NeXT proved that open-source alternatives can match or exceed proprietary models for domain-specific tasks when selected through evidence-based benchmarking rather than assumption.

This branch feeds directly into the multimodal pipeline in subsequent branches.
