#!/usr/bin/env python3
import argparse, random
from pathlib import Path
import orjson
from tqdm import tqdm

def reservoir_sample(paths, k, seed=13):
    rng = random.Random(seed)
    sample = []
    n = 0
    for p in paths:
        with open(p, "rb") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = orjson.loads(line)
                except Exception:
                    continue
                # must have text
                txt = (rec.get("text") or "").strip()
                if not txt:
                    continue

                n += 1
                if len(sample) < k:
                    sample.append(rec)
                else:
                    j = rng.randrange(n)
                    if j < k:
                        sample[j] = rec
    return sample, n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl-dir", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--k", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    indir = Path(args.jsonl_dir)
    paths = sorted(indir.glob("*.jsonl"))
    if not paths:
        raise SystemExit(f"No jsonl files found in {indir}")

    sample, seen = reservoir_sample(paths, args.k, args.seed)

    outp = Path(args.out_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "wb") as out:
        for rec in sample:
            out.write(orjson.dumps(rec) + b"\n")

    print(f"Seen={seen}  Sampled={len(sample)} -> {outp}")

if __name__ == "__main__":
    main()
