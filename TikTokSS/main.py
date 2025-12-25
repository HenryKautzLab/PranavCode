import google.generativeai as genai
import os
from pathlib import Path
from PIL import Image

def analyze_screenshot_emotion(image_path: str) -> str:
    """
    Analyze a TikTok screenshot and determine what emotions/feelings it might evoke.
    
    Args:
        image_path: Path to the screenshot image
        
    Returns:
        String describing the potential emotional impact
    """
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


def main():
    # Directly use FoodTest.jpg
    screenshot_path = "FoodTest.jpg"
    
    if not os.path.exists(screenshot_path):
        print(f"Error: File not found at {screenshot_path}")
        return
    
    print("\nAnalyzing screenshot...\n")
    
    try:
        analysis = analyze_screenshot_emotion(screenshot_path)
        print("=== EMOTIONAL IMPACT ANALYSIS ===")
        print(analysis)
    except Exception as e:
        print(f"Error analyzing image: {str(e)}")


if __name__ == "__main__":
    main()