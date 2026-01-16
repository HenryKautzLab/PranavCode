#!/usr/bin/env python3
import json
import re
from pathlib import Path
from typing import List, Dict

# ===== MAHA THEME PATTERNS (from s5_reddit_maha_search_zst.py) =====

# 1) MAHA / MAKE AMERICA HEALTHY AGAIN
MAHA_PATTERNS = [
    r"\bmake\s+america\s+healthy\s+again\b",
    r"\bMAHA\b",
    r"#maha\b",
    r"#makeamericahealthyagain\b",
]

# 2) VACCINES
VACCINE_PATTERNS = [
    r"\bvaccine(s)?\b",
    r"\bvaccination(s)?\b",
    r"\banti[-\s]?vax(x)?\b",
    r"\bvax\b",
    r"\bvaxxed\b",
    r"\bbooster(s)?\b",
    r"\bmrna\b",
    r"\bPfizer\b",
    r"\bModerna\b",
    r"#antivax\b",
]

# 3) MASKS / COVID NPIs
MASK_PATTERNS = [
    r"\bmask(s)?\b",
    r"\bmasking\b",
    r"\bmask\s+mandate(s)?\b",
    r"\banti[-\s]?mask\b",
    r"\bN95\b",
    r"#nomask\b",
]

# 4) UPF / JUNK FOOD
UPF_PATTERNS = [
    r"\bultra[-\s]?processed\s+food(s)?\b",
    r"\bUPF(s)?\b",
    r"\bjunk\s+food(s)?\b",
    r"\bfast\s+food(s)?\b",
    r"\bseed\s+oil(s)?\b",
    r"\bprocessed\s+food(s)?\b",
    r"#ultraprocessed\b",
    r"#junkfood\b",
]

# 5) BIG FOOD
BIG_FOOD_PATTERNS = [
    r"\bBig\s+Food\b",
    r"\bfood\s+industry\b",
    r"\bfood\s+lobby\b",
]

# 6) PESTICIDES
PESTICIDE_PATTERNS = [
    r"\bpesticide(s)?\b",
    r"\bglyphosate\b",
    r"\bRoundup\b",
]

# 7) MICROPLASTICS
MICROPLASTICS_PATTERNS = [
    r"\bmicroplastic(s)?\b",
    r"\bplastic\s+pollution\b",
]

# 8) PFAS
PFAS_PATTERNS = [
    r"\bPFAS\b",
    r"\bforever\s+chemical(s)?\b",
]

THEME_PATTERNS = {
    "maha": MAHA_PATTERNS,
    "vaccines": VACCINE_PATTERNS,
    "masks": MASK_PATTERNS,
    "upf": UPF_PATTERNS,
    "big_food": BIG_FOOD_PATTERNS,
    "pesticides": PESTICIDE_PATTERNS,
    "microplastics": MICROPLASTICS_PATTERNS,
    "pfas": PFAS_PATTERNS,
}

# Compile all patterns
COMPILED_THEMES = {
    theme: re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)
    for theme, patterns in THEME_PATTERNS.items()
}


def extract_text(metadata: Dict) -> str:
    """Extract searchable text from TikTok metadata"""
    parts = []
    for key in ("title", "author"):
        v = metadata.get(key)
        if isinstance(v, str):
            parts.append(v)
    return " ".join(parts)


def find_themes(text: str) -> List[str]:
    """Find matching MAHA themes in text"""
    matched = []
    for theme, compiled_regex in COMPILED_THEMES.items():
        if compiled_regex.search(text):
            matched.append(theme)
    return matched


def analyze_tiktok_metadata():
    metadata_dir = Path("tiktok_downloads/metadata")
    results = []
    
    print("Analyzing TikTok metadata for MAHA themes...\n")
    
    for meta_file in sorted(metadata_dir.glob("*_metadata.json")):
        # Skip already-analyzed files
        if "_analyzed" in meta_file.name:
            continue
            
        with open(meta_file) as f:
            metadata = json.load(f)
        
        # Extract text from title and author
        text = extract_text(metadata)
        
        # Find matching themes
        themes = find_themes(text)
        
        # Store result
        result = {
            "video_id": metadata["video_id"],
            "title": metadata["title"],
            "author": metadata["author"],
            "views": metadata["view_count"],
            "watched_date": metadata["watched_date"],
            "maha_related": len(themes) > 0,
            "themes": themes,
            "filename": metadata["filename"]
        }
        results.append(result)
        
        # Print result
        status = "✓ MAHA" if themes else "  GENERIC"
        print(f"{status} | {metadata['title'][:50]:<50} | {metadata['author']:<15}")
        if themes:
            print(f"         Themes: {', '.join(themes)}")
        
        # Save enriched metadata
        metadata["__maha_themes"] = themes
        metadata["__maha_related"] = len(themes) > 0
        
        analyzed_file = meta_file.parent / f"{meta_file.stem}_analyzed.json"
        with open(analyzed_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY:")
    print(f"  Total videos: {len(results)}")
    print(f"  MAHA-related: {sum(1 for r in results if r['maha_related'])}")
    print(f"  Generic: {sum(1 for r in results if not r['maha_related'])}")
    
    # Save full results
    with open("maha_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nFull results saved to: maha_analysis_results.json")
    print(f"Analyzed metadata saved to: tiktok_downloads/metadata/*_analyzed.json")
    
    # Theme breakdown
    if any(r['maha_related'] for r in results):
        print(f"\nTHEME BREAKDOWN:")
        theme_counts = {}
        for result in results:
            for theme in result['themes']:
                theme_counts[theme] = theme_counts.get(theme, 0) + 1
        
        for theme, count in sorted(theme_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {theme}: {count}")


if __name__ == "__main__":
    analyze_tiktok_metadata()