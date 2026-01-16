#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import zstandard as zstd
import re

# =========================
#  MAHA / THEME PATTERNS
# =========================

# 1) MAHA / MAKE AMERICA HEALTHY AGAIN
CORE_SLOGAN_PATTERNS = [
    r"\bmake\s+america\s+healthy\s+again\b",
    r"\bmake\s+america\s+healthy\s+again!\b",
    r"\bmake\s+america\s+healthy\s+again!!+\b",
    r"\bmake\s+america\s+more\s+healthy\s+again\b",
    r"\bmake\s+the\s+america\s+healthy\s+again\b",
    r"\bmake\s+the\s+u\.?s\.?a?\.?\s+healthy\s+again\b",
    r"\bmake\s+us\s+healthy\s+again\b",
]

SLOGAN_MOVEMENT_PATTERNS = [
    r"\bmake\s+america\s+healthy\s+again\s+movement\b",
    r"\bmake\s+america\s+healthy\s+again\s+campaign\b",
    r"\bmake\s+america\s+healthy\s+again\s+plan\b",
    r"\bmake\s+america\s+healthy\s+again\s+agenda\b",
    r"\bmake\s+america\s+healthy\s+again\s+platform\b",
    r"\bthe\s+make\s+america\s+healthy\s+again\s+movement\b",
]

HASHTAG_MAHA_PATTERNS = [
    r"#makeamericahealthyagain\b",
    r"#makeamericahealthy\b",
    r"#makeamericahealthieragain\b",
    r"#makeamericahealthyagain20(1[6-9]|2[0-9])\b",
    r"#makeamericahealthy20(1[6-9]|2[0-9])\b",
    r"#maha\b",
    r"#mahamovement\b",
    r"#maha202[0-9]\b",
    r"#maha20(1[6-9]|2[0-9])\b",
]

ACRONYM_MAHA_PATTERNS = [
    r"\bMAHA\b.*\bhealth(y|ier|care|care\s+system)?\b",
    r"\bhealth(y|ier|care|care\s+system)?\b.*\bMAHA\b",
    r"\bMAHA\s+(movement|campaign|plan|agenda|platform|supporters|crowd|rally|coalition|initiative)\b",
    r"\b(pro|anti)?-?MAHA\s+(movement|supporters|crowd|rally|coalition)\b",
    r"\bM\.A\.H\.A\.\b",
    r"\bM\.A\.H\.A\.\s+(movement|campaign|plan|agenda|platform)\b",
    r"\bmaha\s+(movement|campaign|plan|agenda|platform|supporters|crowd|rally|coalition)\b",
]

LONGFORM_MAHA_PATTERNS = [
    r"\bthe\s+MAHA\s+movement\b",
    r"\bthe\s+pro-?MAHA\s+movement\b",
    r"\bMAHA\s+supporters\b",
    r"\bMAHA\s+rally\b",
    r"\bMAHA\s+event\b",
    r"\bMAHA\s+summit\b",
    r"\bMAHA\s+conference\b",
    r"\bMAHA\s+tour\b",
    r"\"make\s+america\s+healthy\s+again\"",
    r"'make\s+america\s+healthy\s+again'",
    r"\bMake\s+America\s+Healthy\s+Again\s+\(MAHA\)\b",
    r"\bMAHA\s+\(Make\s+America\s+Healthy\s+Again\)\b",
]

SOFT_MAHA_VARIANTS = [
    r"\bmaking\s+america\s+healthy\s+again\b",
    r"\btrying\s+to\s+make\s+america\s+healthy\s+again\b",
    r"\bkeep\s+america\s+healthy\s+again\b",
    r"\bkeep\s+america\s+healthy\b",
    r"\bmake\s+america\s+healthier\s+again\b",
]

MAHA_PATTERNS = (
    CORE_SLOGAN_PATTERNS
    + SLOGAN_MOVEMENT_PATTERNS
    + HASHTAG_MAHA_PATTERNS
    + ACRONYM_MAHA_PATTERNS
    + LONGFORM_MAHA_PATTERNS
    + SOFT_MAHA_VARIANTS
)

# 2) VACCINES
VACCINE_PATTERNS = [
    r"\bvaccine(s)?\b",
    r"\bvaccination(s)?\b",
    r"\bvaccinated\b",
    r"\bunvaccinated\b",
    r"\banti[-\s]?vax(x)?(er(s)?|xing|xed)?\b",
    r"\bpro[-\s]?vax(x)?(er(s)?|xing|xed)?\b",
    r"\banti[-\s]?vaccination\b",
    r"\bvax\b",
    r"\bvaxxed\b",
    r"\bget\s+the\s+shot\b",
    r"\bthe\s+shot\b",
    r"\bbooster(s)?\b",
    r"\bbooster\s+shot(s)?\b",
    r"\bmrna\b",
    r"\bmrna\s+vaccine(s)?\b",
    r"\bPfizer\b",
    r"\bModerna\b",
    r"\bAstraZeneca\b",
    r"\bJ&J\b",
    r"\bJohnson\s*&\s*Johnson\b",
    r"\bCovid\s+shot(s)?\b",
    r"\bcovid[-\s]?vaccine(s)?\b",
    r"\bMMR\b",
    r"\bmeasles\s+mumps\s+rubella\b",
    r"\bHPV\s+vaccine\b",
    r"#antivax\b",
    r"#antivaxx\b",
    r"#provax\b",
    r"#getvaccinated\b",
    r"#vaccineinjur(y|ies)\b",
]

# 3) MASKS / COVID NPIs
MASK_PATTERNS = [
    r"\bface\s+mask(s)?\b",
    r"\bmask(s)?\b",
    r"\bmasking\b",
    r"\bwear\s+your\s+mask\b",
    r"\bmasked\b",
    r"\bmask\s+mandate(s)?\b",
    r"\bno[-\s]?mask\b",
    r"\banti[-\s]?mask(er(s)?)?\b",
    r"\bpro[-\s]?mask(er(s)?)?\b",
    r"\bN95\b",
    r"\bKN95\b",
    r"\bFFP2\b",
    r"#nomask\b",
    r"#maskup\b",
]

# 4) UPF / JUNK FOOD
UPF_PATTERNS = [
    r"\bultra[-\s]?processed\s+food(s)?\b",
    r"\bultraprocessed\s+food(s)?\b",
    r"\bUPF(s)?\b",
    r"\bjunk\s+food(s)?\b",
    r"\bfast\s+food(s)?\b",
    r"\bhighly\s+processed\s+food(s)?\b",
    r"\bprocessed\s+meat(s)?\b",
    r"\bprocessed\s+snack(s)?\b",
    r"\bsoda(s)?\b",
    r"\bsugary\s+drink(s)?\b",
    r"\bsugar[-\s]?sweetened\s+beverage(s)?\b",
    r"\bSSB(s)?\b",
    r"\bhigh[-\s]?fructose\s+corn\s+syrup\b",
    r"\bHFCS\b",
    r"\benergy\s+drink(s)?\b",
    r"\bsnack\s+food(s)?\b",
    r"#ultraprocessed\b",
    r"#junkfood\b",
    r"#fastfood\b",
]

# 5) BIG FOOD
BIG_FOOD_PATTERNS = [
    r"\bBig\s+Food\b",
    r"\bfood\s+industry\b",
    r"\bfood\s+lobby\b",
    r"\bagribusiness\b",
    r"\bprocessed\s+food\s+industry\b",
    r"\bCoca[-\s]?Cola\b",
    r"\bPepsi(Co)?\b",
    r"\bNestl[eé]\b",
    r"\bMondelez\b",
    r"\bMcDonald'?s\b",
    r"\bBurger\s+King\b",
    r"\bKFC\b",
    r"#bigfood\b",
]

# 6) PESTICIDES
PESTICIDE_PATTERNS = [
    r"\bpesticide(s)?\b",
    r"\bherbicide(s)?\b",
    r"\binsecticide(s)?\b",
    r"\bfungicide(s)?\b",
    r"\bglyphosate\b",
    r"\bRoundup\b",
    r"\bneonicotinoid(s)?\b",
    r"\bneonics\b",
    r"\borganophosphate(s)?\b",
    r"\bpesticide\s+exposure\b",
]

# 7) MICROPLASTICS
MICROPLASTICS_PATTERNS = [
    r"\bmicroplastic(s)?\b",
    r"\bnano[-\s]?plastic(s)?\b",
    r"\bplastic\s+pollution\b",
    r"\bplastic\s+in\s+the\s+ocean\b",
    r"\bplastic\s+in\s+our\s+blood\b",
    r"#microplastics\b",
]

# 8) PFAS
PFAS_PATTERNS = [
    r"\bPFAS\b",
    r"\bPFOA\b",
    r"\bPFOS\b",
    r"\bper[-\s]?fluoro[-\s]?alkyl\b",
    r"\bpoly[-\s]?fluoro[-\s]?alkyl\b",
    r"\bforever\s+chemical(s)?\b",
    r"#pfas\b",
    r"#foreverchemicals\b",
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

COMPILED_THEMES = {
    theme: re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)
    for theme, patterns in THEME_PATTERNS.items()
}

# =========================
#  UTILS
# =========================

def parse_args():
    p = argparse.ArgumentParser(
        description="Search Reddit .zst (already year-filtered) for MAHA-related themes and write matches (zst)."
    )
    p.add_argument("--input-dir", type=Path, required=True,
                   help="Root of year-filtered .zst files (e.g. /p/zenodo/reddit/2020_2025_zst).")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Root for theme-filtered outputs (compressed .jsonl.zst).")
    p.add_argument("--created-key", default="created_utc")

    # Sharding for SLURM arrays
    p.add_argument("--shard-id", type=int, default=0,
                   help="Shard index for this job (0-based).")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Total number of shards/jobs.")
    return p.parse_args()


def iter_zst_lines(path: Path):
    dctx = zstd.ZstdDecompressor()
    with path.open("rb") as fh:
        stream = dctx.stream_reader(fh)
        buffer = b""
        while True:
            chunk = stream.read(2 ** 20)
            if not chunk:
                break
            buffer += chunk
            *lines, buffer = buffer.split(b"\n")
            for line in lines:
                if line:
                    yield line
        if buffer:
            yield buffer


def extract_text(obj: Dict) -> str:
    parts: List[str] = []
    for key in ("title", "selftext", "body"):
        v = obj.get(key)
        if isinstance(v, str):
            parts.append(v)
    return " ".join(parts)


def find_themes(text: str):
    matched = []
    for theme, cre in COMPILED_THEMES.items():
        if cre.search(text):
            matched.append(theme)
    return matched


def search_file_themes(path: Path,
                       input_root: Path,
                       output_root: Path) -> tuple[int, int]:
    rel = path.relative_to(input_root)
    # keep same relative path, but ensure .jsonl.zst suffix
    out_path = (output_root / rel).with_suffix(".jsonl.zst")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_lines = 0
    kept_lines = 0

    cctx = zstd.ZstdCompressor(level=3)
    with out_path.open("wb") as fh_out:
        writer = cctx.stream_writer(fh_out)

        for raw_line in iter_zst_lines(path):
            total_lines += 1
            line_strip = raw_line.strip()
            if not line_strip:
                continue
            try:
                obj = json.loads(line_strip)
            except json.JSONDecodeError:
                continue

            text = extract_text(obj)
            if not text:
                continue

            themes = find_themes(text)
            if not themes:
                continue

            obj["__maha_themes"] = themes
            out_line = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
            writer.write(out_line)
            kept_lines += 1

            if total_lines % 100_000 == 0:
                print(
                    f"[search {path.name}] processed={total_lines:,} kept={kept_lines:,}",
                    file=sys.stderr,
                    flush=True,
                )

        writer.flush(zstd.FLUSH_FRAME)

    print(
        f"[search {path.name}] DONE total={total_lines:,} kept={kept_lines:,}",
        file=sys.stderr,
        flush=True,
    )
    return total_lines, kept_lines


def main():
    args = parse_args()
    input_root = args.input_dir.resolve()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    all_files = sorted(input_root.rglob("*.zst"))
    n_total = len(all_files)
    print(f"[search] Found {n_total} .zst files under {input_root}")

    if args.num_shards > 1:
        if not (0 <= args.shard_id < args.num_shards):
            print(f"ERROR: shard-id {args.shard_id} must be in [0, {args.num_shards - 1}]", file=sys.stderr)
            sys.exit(1)
        files = all_files[args.shard_id::args.num_shards]
        print(f"[search] Shard {args.shard_id}/{args.num_shards} → {len(files)} files")
    else:
        files = all_files
        print(f"[search] Single shard → {len(files)} files")

    grand_total = 0
    grand_kept = 0
    for i, p in enumerate(files, 1):
        t, k = search_file_themes(
            p,
            input_root=input_root,
            output_root=output_root,
        )
        grand_total += t
        grand_kept += k
        if i % 20 == 0 or i == len(files):
            print(
                f"[search] [{i}/{len(files)}] files; total={grand_total:,} kept={grand_kept:,}",
                file=sys.stderr,
                flush=True,
            )


if __name__ == "__main__":
    main()
