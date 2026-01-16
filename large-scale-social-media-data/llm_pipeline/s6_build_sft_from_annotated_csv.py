#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import orjson

THEME_LABELS = ["VACCINE","MASK","FOOD_LIFESTYLE","ENV_TOXINS","AGENCIES","PHARMA","POLICY_IDEOLOGY","OTHER"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--out-jsonl", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)

    outp = Path(args.out_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with outp.open("wb") as f:
        for _, r in df.iterrows():
            text = str(r.get("text","")).strip()
            if not text:
                continue

            gold_maha = str(r.get("gold_maha","")).strip().upper()
            gold_theme = str(r.get("gold_theme","")).strip().upper()

            # Task A
            if gold_maha in ("MAHA","GENERIC"):
                rec = {
                    "messages": [
                        {"role": "user", "content":
                            "Classify whether the following text is MAHA-related discourse or GENERIC. "
                            "Output only: MAHA or GENERIC.\n\n"
                            f"Text:\n{text}"
                        },
                        {"role": "assistant", "content": gold_maha}
                    ]
                }
                f.write(orjson.dumps(rec) + b"\n")
                n += 1

            # Task B (single-label theme)
            if gold_maha == "MAHA" and gold_theme in THEME_LABELS:
                rec = {
                    "messages": [
                        {"role": "user", "content":
                            "Assign exactly ONE dominant theme label to the MAHA-related text. "
                            f"Valid labels: {', '.join(THEME_LABELS)}. "
                            "Output only the label.\n\n"
                            f"Text:\n{text}"
                        },
                        {"role": "assistant", "content": gold_theme}
                    ]
                }
                f.write(orjson.dumps(rec) + b"\n")
                n += 1

    print(f"WROTE {n} examples -> {outp}")

if __name__ == "__main__":
    main()
