import json
import cv2
import whisper
import easyocr
from pathlib import Path
from PIL import Image
import torch
import numpy as np
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import BitsAndBytesConfig

print("=" * 80)
print("TikTok Video Analysis with LLaVA-NeXT + Whisper + EasyOCR")
print("=" * 80)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")

videos_dir = Path("tiktok_downloads/videos")

def has_audio(video_path):
    import subprocess
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_name",
         "-of", "default=noprint_wrappers=1", str(video_path)],
        capture_output=True, text=True
    )
    return bool(result.stdout.strip())

def transcribe_video(video_path, whisper_model):
    if not has_audio(video_path):
        print(f"   ⚠️  No audio stream found in {video_path.name}")
        return {"transcript": "", "language": "unknown", "segments": []}
    result = whisper_model.transcribe(str(video_path))
    return {
        "transcript": result["text"].strip(),
        "language": result["language"],
        "segments": [
            {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
            for seg in result["segments"]
        ]
    }

# ── Step 1: Transcribe all videos with Whisper then unload ──
transcripts_cache = Path("tiktok_downloads/transcripts")
transcripts_cache.mkdir(parents=True, exist_ok=True)

video_files = sorted(videos_dir.glob("*.mp4"))
print(f"\n🎙️  Loading Whisper (base) model...")
whisper_model = whisper.load_model("base", device=device)
print(f"🎙️  Transcribing {len(video_files)} videos...\n")

for video_file in video_files:
    cache_file = transcripts_cache / f"{video_file.stem}_transcript.json"
    if cache_file.exists():
        print(f"   ✅ Cached transcript found for {video_file.name}")
        continue
    print(f"   🎙️  Transcribing {video_file.name}...")
    audio_data = transcribe_video(video_file, whisper_model)
    with open(cache_file, "w") as f:
        json.dump(audio_data, f, indent=2)
    print(f"   ✅ Done — detected language: {audio_data['language']}")
    if audio_data["transcript"]:
        print(f"   📝 Preview: {audio_data['transcript'][:80]}...")

del whisper_model
torch.cuda.empty_cache()
print("\n🗑️  Whisper unloaded from GPU memory.\n")

# ── Step 2: Load EasyOCR ──
print("🔤 Loading EasyOCR...")
ocr_reader = easyocr.Reader(['en'], gpu=False)
print("✅ EasyOCR ready.\n")

# ── Step 3: Load LLaVA-NeXT ──
print("🤖 Loading LLaVA-NeXT model (this may take a few minutes)...")
model_id = "llava-hf/llava-v1.6-mistral-7b-hf"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

processor = LlavaNextProcessor.from_pretrained(model_id)
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    low_cpu_mem_usage=True
)
print(f"✅ LLaVA-NeXT loaded on device: {device}\n")


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def extract_frames(video_path, num_frames=3):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(num_frames):
        frame_num = int(total_frames * i / (num_frames - 1)) if num_frames > 1 else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        success, frame = cap.read()
        if success:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames


def extract_onscreen_text(frames):
    seen = set()
    all_text = []
    for frame in frames:
        frame_np = np.array(frame)
        results = ocr_reader.readtext(frame_np, detail=0)
        for text in results:
            cleaned = text.strip()
            if cleaned and cleaned.lower() not in seen:
                seen.add(cleaned.lower())
                all_text.append(cleaned)
    return all_text


def get_segment_text_near(segments, time_point, window=5):
    return " ".join([
        seg["text"] for seg in segments
        if abs(seg["start"] - time_point) < window
    ]).strip()


def ask_llava(image, question, transcript=None, ocr_text=None):
    context_parts = []
    if transcript:
        context_parts.append(f"SPOKEN AUDIO (from Whisper): \"{transcript}\"")
    if ocr_text:
        formatted_ocr = " | ".join(ocr_text)
        context_parts.append(f"ON-SCREEN TEXT (from OCR): \"{formatted_ocr}\"")

    if context_parts:
        context_block = "\n".join(context_parts)
        augmented_question = (
            f"{context_block}\n\n"
            f"Using both what you see visually AND the above context: {question}"
        )
    else:
        augmented_question = question

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": augmented_question}
            ]
        }
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    response = processor.decode(output[0], skip_special_tokens=True)

    if "ASSISTANT:" in response:
        return response.split("ASSISTANT:")[-1].strip()
    return response.strip()


def parse_filename(stem):
    """
    Parse what we can from filename format:
    20250114_145227_7430278927710571806_xavier61__
    → date, time, video_id, author
    """
    parts = stem.split('_')
    return {
        "date": parts[0] if len(parts) > 0 else "unknown",
        "time": parts[1] if len(parts) > 1 else "unknown",
        "video_id": parts[2] if len(parts) > 2 else stem,
        "author": parts[3] if len(parts) > 3 else "unknown"
    }


def analyze_video(video_path, audio_data):
    print(f"   🎬 Extracting frames...")
    frames = extract_frames(video_path, num_frames=3)
    main_frame = frames[1]

    print(f"   🔤 Extracting on-screen text with EasyOCR...")
    ocr_text = extract_onscreen_text(frames)
    if ocr_text:
        print(f"   📋 OCR found: {' | '.join(ocr_text[:5])}{'...' if len(ocr_text) > 5 else ''}")
    else:
        print(f"   📋 OCR found no on-screen text")

    transcript = audio_data.get("transcript", "")
    segments = audio_data.get("segments", [])

    print(f"   🤖 Running LLaVA-NeXT analysis (audio + OCR grounded)...")

    questions = {
        "content_summary": "Describe in detail what is happening in this TikTok video. What are the main visual elements, people, objects, and actions?",
        "emotional_tone": "What emotions does this content evoke? Describe the mood and emotional impact on viewers.",
        "persuasion_techniques": "What persuasive or rhetorical techniques are being used in this video? Consider visual appeal, emotional manipulation, social proof, authority, or urgency tactics.",
        "target_audience": "Who is the intended audience for this content? What age group, demographic, or interest group would this appeal to?",
        "credibility_assessment": "Does this content appear credible and trustworthy? Are there any red flags, misleading elements, or signs of manipulation?",
        "misinformation_risk": "Could this content spread misinformation or false claims? What are the potential risks to viewers?",
        "behavioral_impact": "How might this video influence viewer behavior, beliefs, or actions? What specific behaviors might it encourage?",
        "content_category": "What category does this content belong to: entertainment, education, health information, political content, product promotion, or something else?",
        "key_message": "What is the main message or takeaway that viewers are supposed to get from this video?",
        "risk_level": "On a scale of low, medium, or high, what is the potential risk level of this content in terms of negative impact on viewers? Explain why.",
        "audio_visual_alignment": "Does the spoken audio and on-screen text align with what is shown visually, or is there a mismatch? Mismatches can indicate misleading or deceptive content."
    }

    analysis = {}
    for i, (key, question) in enumerate(questions.items(), 1):
        print(f"   📊 Question {i}/{len(questions)}: {key}...")
        answer = ask_llava(main_frame, question, transcript=transcript, ocr_text=ocr_text)
        analysis[key] = answer

    print(f"   🎞️  Analyzing video progression...")
    temporal_analysis = []
    frame_labels = ["Beginning", "Middle", "End"]
    total_duration = segments[-1]["end"] if segments else 0

    for idx, (frame, label) in enumerate(zip(frames, frame_labels)):
        time_point = total_duration * idx / (len(frames) - 1) if len(frames) > 1 and total_duration else 0
        segment_text = get_segment_text_near(segments, time_point) if segments else ""

        desc = ask_llava(
            frame,
            f"Describe what is shown in the {label.lower()} of this video.",
            transcript=segment_text if segment_text else transcript,
            ocr_text=ocr_text
        )
        temporal_analysis.append({
            "timestamp": label,
            "description": desc,
            "audio_at_this_point": segment_text
        })

    analysis["temporal_progression"] = temporal_analysis
    analysis["whisper_transcript"] = transcript
    analysis["whisper_segments"] = segments
    analysis["detected_language"] = audio_data.get("language", "unknown")
    analysis["ocr_onscreen_text"] = ocr_text
    analysis["analysis_method"] = "LLaVA-NeXT-Mistral-7B + Whisper-base + EasyOCR (Colab CUDA)"
    analysis["num_frames_analyzed"] = len(frames)

    return analysis


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

def main():
    results = []

    for video_file in sorted(videos_dir.glob("*.mp4"))[:3]:
        file_info = parse_filename(video_file.stem)

        transcript_file = transcripts_cache / f"{video_file.stem}_transcript.json"
        if not transcript_file.exists():
            print(f"⚠️  No transcript for {video_file.name} — skipping")
            continue

        with open(transcript_file) as f:
            audio_data = json.load(f)

        print(f"\n{'='*80}")
        print(f"🎬 Video {len(results)+1}/{len(list(videos_dir.glob('*.mp4')))}")
        print(f"File: {video_file.name}")
        print(f"Author: {file_info['author']} | Date: {file_info['date']}")
        print(f"{'='*80}")

        try:
            analysis = analyze_video(video_file, audio_data)

            result = {
                "video_id": file_info["video_id"],
                "author": file_info["author"],
                "date": file_info["date"],
                "filename": video_file.name,
                "llava_analysis": analysis
            }
            results.append(result)
            print(f"   ✅ Analysis complete!")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    output_file = "llava_next_analysis.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 80}")
    print(f"✅ ANALYSIS COMPLETE!")
    print(f"📊 Videos analyzed: {len(results)}/{len(list(videos_dir.glob('*.mp4')))}")
    print(f"💾 Results saved to: llava_next_analysis.json")
    print(f"{'=' * 80}")


main()