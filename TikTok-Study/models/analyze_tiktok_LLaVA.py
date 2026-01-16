#!/usr/bin/env python3
import json
import cv2
from pathlib import Path
from PIL import Image
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

# Load LLaVA-NeXT model
print("Loading LLaVA-NeXT model (this may take a few minutes)...")
model_id = "llava-hf/llava-v1.6-mistral-7b-hf"  # Or use "llava-hf/llava-v1.6-vicuna-7b-hf"
processor = LlavaNextProcessor.from_pretrained(model_id)
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

# Use MPS (Apple Silicon) if available
device = "mps" if torch.backends.mps.is_available() else "cpu"
if device == "cpu":
    model = model.to(torch.float32)  # CPU needs float32
else:
    model = model.to(device)
    
print(f"Using device: {device}\n")

def extract_frames(video_path, num_frames=3):
    """Extract key frames from video"""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    # Extract beginning, middle, end
    for i in range(num_frames):
        frame_num = int(total_frames * i / (num_frames - 1)) if num_frames > 1 else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        success, frame = cap.read()
        if success:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    
    cap.release()
    return frames

def ask_llava(image, question):
    """Ask LLaVA-NeXT a specific question about the image"""
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]
        }
    ]
    
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    
    # Generate response
    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    response = processor.decode(output[0], skip_special_tokens=True)
    
    # Extract answer (remove the prompt part)
    if "ASSISTANT:" in response:
        answer = response.split("ASSISTANT:")[-1].strip()
    else:
        answer = response.strip()
    
    return answer

def analyze_video_with_llava(video_path, metadata):
    """Comprehensive video analysis using LLaVA-NeXT"""
    
    print(f"   🎬 Extracting frames...")
    frames = extract_frames(video_path, num_frames=3)
    main_frame = frames[1]  # Use middle frame for detailed analysis
    
    print(f"   🤖 Analyzing with LLaVA-NeXT...")
    
    # Detailed questions for comprehensive analysis
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
        
        "risk_level": "On a scale of low, medium, or high, what is the potential risk level of this content in terms of negative impact on viewers? Explain why."
    }
    
    analysis = {}
    for i, (key, question) in enumerate(questions.items(), 1):
        print(f"   📊 Question {i}/{len(questions)}: {key}...")
        answer = ask_llava(main_frame, question)
        analysis[key] = answer
    
    # Analyze temporal progression (beginning → middle → end)
    print(f"   🎞️  Analyzing video progression...")
    temporal_analysis = []
    frame_labels = ["Beginning", "Middle", "End"]
    
    for frame, label in zip(frames, frame_labels):
        desc = ask_llava(frame, f"Describe what is shown in the {label.lower()} of this video.")
        temporal_analysis.append({
            "timestamp": label,
            "description": desc
        })
    
    analysis["temporal_progression"] = temporal_analysis
    analysis["analysis_method"] = "LLaVA-NeXT-Mistral-7B (local)"
    analysis["num_frames_analyzed"] = len(frames)
    
    return analysis

def main():
    videos_dir = Path("tiktok_downloads/videos")
    metadata_dir = Path("tiktok_downloads/metadata")
    results = []
    
    print("=" * 80)
    print("TikTok Video Analysis with LLaVA-NeXT")
    print("=" * 80)
    print()
    
    for video_file in sorted(videos_dir.glob("*.mp4")):
        parts = video_file.stem.split('_')
        if len(parts) >= 3:
            video_id = parts[2]
        else:
            print(f"⚠️  Skipping {video_file.name}")
            continue
        
        meta_file = metadata_dir / f"{video_id}_metadata.json"
        
        if not meta_file.exists():
            print(f"⚠️  No metadata for {video_file.name}")
            continue
        
        with open(meta_file) as f:
            metadata = json.load(f)
        
        print(f"\n{'='*80}")
        print(f"🎬 Video {len(results)+1}/{len(list(videos_dir.glob('*.mp4')))}")
        print(f"Title: {metadata['title']}")
        print(f"Author: {metadata['author']} | Views: {metadata['view_count']:,}")
        print(f"{'='*80}")
        
        try:
            analysis = analyze_video_with_llava(video_file, metadata)
            
            result = {
                "video_id": video_id,
                "title": metadata['title'],
                "author": metadata['author'],
                "views": metadata['view_count'],
                "duration": metadata['duration'],
                "watched_date": metadata['watched_date'],
                "llava_analysis": analysis,
                "filename": video_file.name
            }
            results.append(result)
            
            print(f"   ✅ Analysis complete!")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    output_file = "llava_next_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'=' * 80}")
    print(f"✅ ANALYSIS COMPLETE!")
    print(f"📊 Videos analyzed: {len(results)}/{len(list(videos_dir.glob('*.mp4')))}")
    print(f"💾 Results saved to: {output_file}")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    main()