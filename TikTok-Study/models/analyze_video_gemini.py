#!/usr/bin/env python3
import json
import time
import os
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini - REPLACE WITH YOUR API KEY
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def analyze_video_with_gemini(video_path, metadata):
    """Analyze full video with Gemini"""
    
    print(f"   📤 Uploading video...")
    
    # Upload file with NEW API - specify mime_type for .mp4 files
    with open(video_path, 'rb') as f:
        upload_file = client.files.upload(
            file=f,
            config=types.UploadFileConfig(
                mime_type='video/mp4',
                display_name=video_path.name
            )
        )
    
    print(f"   🤖 Analyzing content...")
    
    prompt = f"""
Analyze this TikTok video for its potential impact on viewers.

VIDEO METADATA:
- Title: {metadata['title']}
- Author: {metadata['author']}
- Views: {metadata['view_count']:,}
- Duration: {metadata['duration']} seconds

Please provide analysis in JSON format:

{{
  "content_summary": "What happens in the video",
  "primary_topic": "Main topic/theme",
  "emotional_tone": ["list of emotions evoked"],
  "persuasion_techniques": ["techniques used if any"],
  "target_audience": "Who this targets",
  "credibility_assessment": "High/Medium/Low and why",
  "misinformation_risk": "High/Medium/Low and why",
  "potential_behavioral_impact": "How might this influence viewers",
  "content_category": "Entertainment/Education/Health/Political/etc",
  "risk_level": "High/Medium/Low",
  "key_messages": ["main takeaways"]
}}
"""
    
    # Generate content with NEW API
    response = client.models.generate_content(
        model='gemini-2.0-flash-exp',
        contents=[
            types.Part.from_uri(
                file_uri=upload_file.uri,
                mime_type=upload_file.mime_type
            ),
            prompt
        ]
    )
    
    # Clean up
    client.files.delete(name=upload_file.name)
    
    return response.text

def main():
    videos_dir = Path("tiktok_downloads/videos")
    metadata_dir = Path("tiktok_downloads/metadata")
    results = []
    
    print("Analyzing TikTok videos with Gemini...\n")
    print("=" * 80)
    
    for video_file in sorted(videos_dir.glob("*.mp4")):
        # Extract video ID from filename
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
            analysis_text = analyze_video_with_gemini(video_file, metadata)
            
            result = {
                "video_id": video_id,
                "title": metadata['title'],
                "author": metadata['author'],
                "views": metadata['view_count'],
                "duration": metadata['duration'],
                "watched_date": metadata['watched_date'],
                "gemini_analysis": analysis_text,
                "filename": video_file.name
            }
            results.append(result)
            
            print(f"   ✅ Complete!")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    # Save results
    output_file = "gemini_video_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'=' * 80}")
    print(f"✅ Analysis complete!")
    print(f"📊 Videos analyzed: {len(results)}")
    print(f"💾 Results saved to: {output_file}")

if __name__ == "__main__":
    main()