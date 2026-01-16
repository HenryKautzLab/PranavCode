#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import orjson
import requests
from tqdm import tqdm
import asyncio
import aiohttp

SYSTEM = (
    "You are a careful social science research assistant. "
    "You label Reddit text for whether it participates in Make America Healthy Again (MAHA)-adjacent health discourse."
)

# FEW-SHOT EXAMPLES (EDIT THESE AFTER YOU READ THE MAHA COMMISSION REPORT + YOUR SEARCH RESULTS)
FEWSHOT = [
    # MAHA (broad; includes distrust, mandates, agencies, lifestyle toxins etc.)
    ("The CDC and FDA are captured by Big Pharma. Stop trusting their vaccine schedules.", "MAHA"),
    ("I watched the video but am still curious to know how this will work. \n\nIf natural immunity "
     "won’t work how will a vaccine?\nLike from what I read, the thought is getting the virus doesnt "
     "make you immune because your body won’t remember the antibody for a mild case\nBut that’s exactly"
     " how a vaccine works- isn’t it?\nFrom Googling: A vaccine works by training the immune system to "
     "recognize and combat pathogens, either viruses or bacteria. To do this, certain molecules from the "
     "pathogen must be introduced into the body to trigger an immune response.\n\nIf anyone could educate"
     " me on this I would greatly appreciate it- thank you!!", "MAHA"),
    ("Seed oils and ultra-processed foods are making people sick. We need real food again.", "MAHA"),
    ("Mask mandates were political theater. Public health officials lied.", "MAHA"),
    ("Caleb Wallace, anti-mask organizer and co-founder of the San Angelo Freedom Defenders, "
     "dies of COVID-19", "MAHA"),
    ("Is fluoride in water linked to lower IQ? Why is this still allowed?", "MAHA"),
    # GENERIC
    ("I’m selling my used GPU. DM me if interested.", "GENERIC"),
    ("Here’s my recipe for chili and meal prep tips.", "GENERIC"),
    ("The game last night was insane. What a comeback.", "GENERIC"),
    ("My internet has been down all day, any fixes?", "GENERIC"),
    ("Popped 2 big pots of popcorn. Boiled up 2 c brown sugar, 1/2 cup cane syrup, 2 sticks butter(it ain’t cheap"
     " now people. Use margarine if u must), 1/2 tsp salt. Boil up 5 min. Stir. Stir. Find a big container. Pour"
     " 1tsp vanilla and 1/2 tsp baking soda into to sugar mix. Pour on popcorn, on cookie sheets…..time to bake."
     , "GENERIC"),
]

def parse_json_loose(s: str):
    s = s.strip()
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 2:
            s = parts[1]
        s = s.replace("json", "", 1).strip()
    try:
        return json.loads(s)
    except Exception:
        return None

def build_prompt(text: str) -> str:
    ex = "\n".join([f'Example: "{t}"\nLabel: {y}\n' for t, y in FEWSHOT])
    return (
        "Task: Classify the text as MAHA or GENERIC.\n"
        "MAHA = MAHA-adjacent health discourse (any stance; debate/sarcasm/questions OK).\n"
        "GENERIC = not MAHA-adjacent.\n\n"
        f"{ex}\n"
        "Return ONLY JSON:\n"
        "{\"label\":\"MAHA\"|\"GENERIC\",\"confidence\":0.0}\n\n"
        f'Text: """{text}"""\n'
    )

async def one_call(session, url, model, temperature, text):
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_prompt(text)},
        ],
    }
    async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=180)) as resp:
        resp.raise_for_status()
        data = await resp.json()
        out = data["choices"][0]["message"]["content"]
        pj = parse_json_loose(out) or {"label":"UNKNOWN","confidence":0.0}
        return pj

async def main_async(args):
    in_recs = []
    with open(args.input_jsonl, "rb") as f:
        for line in f:
            if line.strip():
                in_recs.append(orjson.loads(line))

    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    sem = asyncio.Semaphore(args.concurrency)

    async with aiohttp.ClientSession(connector=connector) as session:
        async def worker(rec):
            txt = (rec.get("text") or "")[:args.max_chars]
            async with sem:
                pj = await one_call(session, url, args.model, args.temperature, txt)
            rec["maha_label"] = str(pj.get("label","UNKNOWN")).upper()
            rec["maha_confidence"] = float(pj.get("confidence",0.0) or 0.0)
            return rec

        tasks = [asyncio.create_task(worker(r)) for r in in_recs]
        outp = open(args.output_jsonl, "wb")
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="ICL"):
            rec = await fut
            outp.write(orjson.dumps(rec) + b"\n")
        outp.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--concurrency", type=int, default=16)  # increase if GPU can handle
    ap.add_argument("--max-chars", type=int, default=2000)
    args = ap.parse_args()
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()
