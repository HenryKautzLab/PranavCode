import google.generativeai as genai
import os
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

def analyze_screenshot_emotion(image_path: str) -> str:
    # Analyze a TikTok screenshot and determine what emotions/feelings it might evoke.
    
    # Args are image_path: Path to the screenshot image
        
    # Returns a string describing the potential emotional impact

    # Configure Gemini
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    
    # Initialize the model (using gemini-2.5-flash)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # Open the image
    img = Image.open(image_path)
    
    # Create the prompt
    prompt = """Look at this TikTok screenshot and describe what you literally see and what simple, direct emotions it might trigger.

Focus on:
1. What is literally in the image (describe what you see - people, food items, products, actions, settings)
2. Basic emotional responses these would cause:
   - Food/drinks → hunger, thirst, cravings
   - People doing activities → curiosity, relatability
   - Products/shopping → desire to buy, comparison
   - Aesthetic settings → relaxation, envy, inspiration
   - Funny/unusual moments → amusement, surprise

Be factual and straightforward. Don't over-interpret or make dramatic conclusions. Just describe what's there and the obvious feelings it would create."""

    # Generate response
    response = model.generate_content([prompt, img])
    
    return response.text

def analyze_multiple_screenshots(image_paths: list) -> str:

    # Analyze multiple TikTok screenshots together and provide an overall emotional impact.
    
    # Args are image_paths: List of paths to screenshot images
        
    # Returns a string describing the overall emotional impact across all images

    # Collect individual analyses
    individual_analyses = []
    
    for path in image_paths:
        analysis = analyze_screenshot_emotion(path)
        individual_analyses.append(f"Image {path}:\n{analysis}")
    
    # Configure Gemini
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    
    # Initialize the model
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # Combine all analyses
    combined_analysis = "\n\n".join(individual_analyses)
    
    # Create prompt for overall analysis
    prompt = f"""Here are analyses of {len(image_paths)} TikTok screenshots:

{combined_analysis}

Based on these individual analyses, provide a brief overall summary of:
1. Common themes or patterns across all images
2. The cumulative emotional impact of viewing these together
3. What this collection of content might say about the viewer's interests or mood

Keep it concise and factual."""

    # Generate overall response
    response = model.generate_content(prompt)
    
    return response.text


def main():
    screenshot_paths = [
        "FightOne.jpg",
        "FightTwo.jpg",
        "FightThree.jpg",
        "FightFour.jpg"
    ]
    
    # Check all files exist
    missing_files = [path for path in screenshot_paths if not os.path.exists(path)]
    if missing_files:
        print(f"Error: Missing files: {', '.join(missing_files)}")
        return
    
    print(f"\nAnalyzing {len(screenshot_paths)} screenshots together...\n")
    
    try:
        overall_analysis = analyze_multiple_screenshots(screenshot_paths)
        print("=== OVERALL EMOTIONAL IMPACT ANALYSIS ===")
        print(overall_analysis)
        print("\n" + "="*50 + "\n")
    except Exception as e:
        print(f"Error analyzing images: {str(e)}")


if __name__ == "__main__":
    main()