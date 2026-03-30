## Initial Experimentation — BLIP and Gemini Baselines

This branch documents the initial exploration phase: testing vision-language models on images before building the full video pipeline.

### What Was Tested

**Flickr dataset + BLIP** (`/flickr` folder)
- Used BLIP to generate captions for images
- Found captions too generic for meaningful interpretation — insufficient detail for persuasion or misinformation analysis
- Identified need for a model with stronger prompt-following and interpretive capability

**TikTok screenshots + Gemini** (`/TikTokSS` folder)
- Switched to Gemini with a structured, specification-driven prompt
- Output was significantly more specific, accurate, and useful for content interpretation
- Validated that prompt design is as important as model selection

### Key Takeaway

BLIP's caption quality was a ceiling on downstream analysis quality. Moving to prompt-driven models (Gemini, then LLaVA-NeXT) unlocked the interpretive depth needed for this research.

This branch is the starting point — see subsequent branches for the full multimodal pipeline.

### Technical Document

Full research notes and methodology: [Technical Document](https://docs.google.com/document/d/15UdZD-YeCTmvvoogqotaDxWpPrW_yz2_DM5TWk2ND6A/edit?tab=t.0)
