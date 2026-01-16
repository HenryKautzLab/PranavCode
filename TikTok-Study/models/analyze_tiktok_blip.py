#!/usr/bin/env python3
import json
import cv2
from pathlib import Path
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Load BLIP model (runs locally on your Mac)
print("Loading BLIP model...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Use GPU if available (M-series Macs have MPS)
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
print(f"Using device: {device}\n")

def extract_frames(video_path, num_frames=5):
    """Extract frames from video"""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    # Extract frames at different points
    for i in range(num_frames):
        frame_num = int(total_frames * i / (num_frames - 1)) if num_frames > 1 else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        success, frame = cap.read()
        if success:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    
    cap.release()
    return frames

def analyze_frame(image, prompt=None):
    """Analyze a single frame with BLIP"""
    if prompt:
        inputs = processor(image, prompt, return_tensors="pt").to(device)
    else:
        inputs = processor(image, return_tensors="pt").to(device)
    
    out = model.generate(**inputs, max_new_tokens=100)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def analyze_video_with_blip(video_path, metadata):
    """Analyze video using BLIP locally"""
    
    print(f"   🎬 Extracting frames...")
    frames = extract_frames(video_path, num_frames=5)
    
    print(f"   🤖 Analyzing {len(frames)} frames...")
    
    # Analyze each frame
    frame_descriptions = []
    for i, frame in enumerate(frames):
        caption = analyze_frame(frame)
        frame_descriptions.append(f"Frame {i+1}: {caption}")
    
    # Get specific insights
    print(f"   📊 Generating insights...")
    
    # Analyze middle frame for main content
    main_frame = frames[len(frames)//2]
    
    emotional_analysis = analyze_frame(
        main_frame, 
        prompt="What is the emotional tone of this image?"
    )
    
    content_analysis = analyze_frame(
        main_frame,
        prompt="What is happening in this image?"
    )
    
    # Compile analysis
    analysis = {
        "content_summary": content_analysis,
        "frame_by_frame": frame_descriptions,
        "emotional_assessment": emotional_analysis,
        "primary_topic": "Analyzed from visual content",
        "num_frames_analyzed": len(frames),
        "analysis_method": "BLIP (local, offline)"
    }
    
    return analysis

def main():
    videos_dir = Path("tiktok_downloads/videos")
    metadata_dir = Path("tiktok_downloads/metadata")
    results = []
    
    print("Analyzing TikTok videos with BLIP (Local Analysis)...\n")
    print("=" * 80)
    
    for video_file in sorted(videos_dir.glob("*.mp4")):
        # Extract video ID
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
        
        print(f"\n🎬 Video {len(results)+1}: {metadata['title'][:60]}")
        print(f"   Author: {metadata['author']} | Views: {metadata['view_count']:,}")
        
        try:
            analysis = analyze_video_with_blip(video_file, metadata)
            
            result = {
                "video_id": video_id,
                "title": metadata['title'],
                "author": metadata['author'],
                "views": metadata['view_count'],
                "duration": metadata['duration'],
                "watched_date": metadata['watched_date'],
                "blip_analysis": analysis,
                "filename": video_file.name
            }
            results.append(result)
            
            print(f"   ✅ Complete!")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    # Save results
    output_file = "blip_video_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'=' * 80}")
    print(f"✅ Analysis complete!")
    print(f"📊 Videos analyzed: {len(results)}")
    print(f"💾 Results saved to: {output_file}")

if __name__ == "__main__":
    main()