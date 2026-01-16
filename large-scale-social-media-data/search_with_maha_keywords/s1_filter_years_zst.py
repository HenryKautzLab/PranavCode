#!/usr/bin/env python3
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import zstandard as zstd


def parse_args():
    p = argparse.ArgumentParser(
        description="Stream .zst Reddit dumps and keep only rows with created_utc in [start_year, end_year]."
    )
    p.add_argument("--input-dir", type=Path, required=True,
                   help="Root directory of ORIGINAL .zst dumps (RC_*.zst, RS_*.zst, etc).")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Root directory where year-filtered dumps will be written (still .zst).")
    p.add_argument("--start-year", type=int, default=2020)
    p.add_argument("--end-year", type=int, default=2025)
    p.add_argument("--created-key", default="created_utc")

    # Sharding for SLURM arrays
    p.add_argument("--shard-id", type=int, default=0,
                   help="Shard index for this job (0-based).")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Total number of shards/jobs.")
    return p.parse_args()


def iter_zst_lines(path: Path):
    """Stream lines from .zst file without full decompression."""
    dctx = zstd.ZstdDecompressor()
    with path.open("rb") as fh:
        stream = dctx.stream_reader(fh)
        buffer = b""
        while True:
            chunk = stream.read(2 ** 20)  # 1 MB
            if not chunk:
                break
            buffer += chunk
            *lines, buffer = buffer.split(b"\n")
            for line in lines:
                if line:
                    yield line
        if buffer:
            yield buffer


def get_year(ts):
    try:
        ts = float(ts)
    except Exception:
        return None
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.year


def filter_file_years(path: Path,
                      input_root: Path,
                      output_root: Path,
                      start_year: int,
                      end_year: int,
                      created_key: str) -> tuple[int, int]:
    valid_years = set(range(start_year, end_year + 1))

    rel = path.relative_to(input_root)
    out_path = (output_root / rel)  # same structure + filename, still .zst
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0

    cctx = zstd.ZstdCompressor(level=3)
    with out_path.open("wb") as fh_out:
        writer = cctx.stream_writer(fh_out)

        for raw_line in iter_zst_lines(path):
            total += 1
            line_strip = raw_line.strip()
            if not line_strip:
                continue
            try:
                obj = json.loads(line_strip)
            except json.JSONDecodeError:
                continue

            year = get_year(obj.get(created_key))
            if year not in valid_years:
                continue

            out_line = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
            writer.write(out_line)
            kept += 1

            if total % 100_000 == 0:
                print(
                    f"[year_filter {path.name}] processed={total:,} kept={kept:,}",
                    file=sys.stderr,
                    flush=True,
                )

        # close the zstd frame
        writer.flush(zstd.FLUSH_FRAME)

    print(
        f"[year_filter {path.name}] DONE total={total:,} kept={kept:,}",
        file=sys.stderr,
        flush=True,
    )
    return total, kept


def main():
    args = parse_args()
    input_root = args.input_dir.resolve()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    all_files = sorted(input_root.rglob("*.zst"))
    n_total = len(all_files)
    print(f"[year_filter] Found {n_total} .zst files under {input_root}")

    if args.num_shards > 1:
        if not (0 <= args.shard_id < args.num_shards):
            print(f"ERROR: shard-id {args.shard_id} must be in [0, {args.num_shards - 1}]", file=sys.stderr)
            sys.exit(1)
        files = all_files[args.shard_id::args.num_shards]
        print(f"[year_filter] Shard {args.shard_id}/{args.num_shards} → {len(files)} files")
    else:
        files = all_files
        print(f"[year_filter] Single shard → {len(files)} files")

    grand_total = 0
    grand_kept = 0
    for i, p in enumerate(files, 1):
        t, k = filter_file_years(
            p,
            input_root=input_root,
            output_root=output_root,
            start_year=args.start_year,
            end_year=args.end_year,
            created_key=args.created_key,
        )
        grand_total += t
        grand_kept += k
        if i % 20 == 0 or i == len(files):
            print(
                f"[year_filter] [{i}/{len(files)}] files; total={grand_total:,} kept={grand_kept:,}",
                file=sys.stderr,
                flush=True,
            )


if __name__ == "__main__":
    main()
