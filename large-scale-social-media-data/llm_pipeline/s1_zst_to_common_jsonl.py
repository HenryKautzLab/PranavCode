#!/usr/bin/env python3
import argparse
from pathlib import Path
import zstandard as zstd
import orjson
from tqdm import tqdm
import hashlib

def iter_zst_lines(zst_path: Path):
    with zst_path.open("rb") as fh:
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        with dctx.stream_reader(fh) as reader:
            buf = b""
            while True:
                chunk = reader.read(1 << 20)
                if not chunk:
                    break
                buf += chunk
                while True:
                    nl = buf.find(b"\n")
                    if nl < 0:
                        break
                    line = buf[:nl]
                    buf = buf[nl+1:]
                    if line:
                        yield line
            if buf.strip():
                yield buf

def infer_kind_from_file(p: Path) -> str:
    name = p.name.lower()
    s = str(p).lower()
    if name.startswith("rc_") or "/comments/" in s:
        return "comment"
    if name.startswith("rs_") or "/submissions/" in s:
        return "submission"
    if "comment" in name:
        return "comment"
    if "submission" in name or "post" in name:
        return "submission"
    return "unknown"

def infer_kind_from_obj(obj: dict) -> str:
    if isinstance(obj.get("body"), str):
        return "comment"
    if isinstance(obj.get("title"), str) or isinstance(obj.get("selftext"), str):
        return "submission"
    return "unknown"

def extract_text(obj: dict, kind: str) -> str:
    if kind == "comment":
        body = obj.get("body")
        if isinstance(body, str):
            t = body.strip()
            if t and t not in ("[deleted]", "[removed]"):
                return t
        return ""
    if kind == "submission":
        title = obj.get("title")
        selftext = obj.get("selftext")
        title = title.strip() if isinstance(title, str) else ""
        selftext = selftext.strip() if isinstance(selftext, str) else ""
        if selftext in ("[deleted]", "[removed]"):
            selftext = ""
        if title and selftext:
            return f"{title}\n\n{selftext}".strip()
        return (title or selftext).strip()
    return ""

def h16(s: str) -> str:
    return hashlib.blake2b(s.encode("utf-8","ignore"), digest_size=16).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-zst", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--max", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    inp = Path(args.input_zst)
    outp = Path(args.output_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)

    kind_hint = infer_kind_from_file(inp)
    wrote = 0

    with outp.open("wb") as f:
        for raw in tqdm(iter_zst_lines(inp), desc=inp.name):
            try:
                obj = orjson.loads(raw)
            except Exception:
                continue

            kind = kind_hint if kind_hint != "unknown" else infer_kind_from_obj(obj)
            if kind == "unknown":
                continue

            text = extract_text(obj, kind)
            if not text:
                continue

            rec = {
                "kind": kind,  # comment or submission
                "id": obj.get("id"),
                "subreddit": obj.get("subreddit"),
                "author": obj.get("author"),
                "created_utc": obj.get("created_utc"),
                "permalink": obj.get("permalink"),
                "text": text[:4000],
                "text_hash": h16(text[:4000]),
                "source_file": str(inp),
            }
            f.write(orjson.dumps(rec) + b"\n")
            wrote += 1
            if args.max and wrote >= args.max:
                break

    print(f"WROTE {wrote} -> {outp}")

if __name__ == "__main__":
    main()
