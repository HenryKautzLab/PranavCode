#!/usr/bin/env python3
import argparse
import random
from pathlib import Path
import orjson
import pandas as pd

def reservoir_stratified(jsonl_path: Path, n_pos: int, n_neg: int, seed: int = 13):
    rng = random.Random(seed)

    pos = []
    neg = []
    seen_pos = 0
    seen_neg = 0

    with jsonl_path.open("rb") as f:
        for line in f:
            if not line.strip():
                continue
            r = orjson.loads(line)

            lab = str(r.get("maha_label", "")).upper()
            txt = (r.get("text") or "").strip()
            if not txt:
                continue

            if lab == "MAHA":
                seen_pos += 1
                if len(pos) < n_pos:
                    pos.append(r)
                else:
                    j = rng.randrange(seen_pos)
                    if j < n_pos:
                        pos[j] = r
            else:
                # treat everything else as GENERIC for eval strat
                seen_neg += 1
                if len(neg) < n_neg:
                    neg.append(r)
                else:
                    j = rng.randrange(seen_neg)
                    if j < n_neg:
                        neg[j] = r

    return pos, neg, seen_pos, seen_neg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", required=True, help="ICL labeled jsonl (merged or single)")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--n", type=int, default=100, help="total eval sample size")
    ap.add_argument("--pos-frac", type=float, default=0.5, help="fraction of MAHA in eval sample")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    n_pos = int(round(args.n * args.pos_frac))
    n_neg = args.n - n_pos

    pos, neg, seen_pos, seen_neg = reservoir_stratified(
        Path(args.input_jsonl), n_pos=n_pos, n_neg=n_neg, seed=args.seed
    )

    rows = pos + neg
    random.Random(args.seed).shuffle(rows)

    df = pd.DataFrame(rows)
    # add gold columns
    df["gold_maha"] = ""
    df["gold_theme"] = ""
    df["gold_stance"] = ""
    df["gold_sarcasm"] = ""
    df["gold_notes"] = ""

    outp = Path(args.out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outp, index=False)

    print(f"Seen predicted MAHA={seen_pos}, predicted non-MAHA={seen_neg}")
    print(f"Wrote eval sample: {outp} (MAHA={len(pos)}, non-MAHA={len(neg)})")

if __name__ == "__main__":
    main()
