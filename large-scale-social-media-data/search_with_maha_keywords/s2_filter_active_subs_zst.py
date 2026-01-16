#!/usr/bin/env python3
import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import zstandard as zstd


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Pipeline on YEAR-FILTERED Reddit zst data:\n"
            "  1) count_submissions: count submissions per subreddit using ONLY submissions files.\n"
            "  2) merge_counts: merge shard counts -> active_subs.pkl (subs with >= min-posts submissions).\n"
            "  3) filter_all: keep ALL rows (subs+comments) whose subreddit is in active_subs.\n"
        )
    )

    p.add_argument(
        "--stage",
        required=True,
        choices=["count_submissions", "merge_counts", "filter_all"],
    )

    # common-ish args
    p.add_argument("--input-dir", type=Path,
                   help="Root of YEAR-FILTERED .zst files (e.g. /p/zenodo/reddit/2020_2025_zst).")
    p.add_argument("--output-dir", type=Path,
                   help="Root for active-subs-only outputs (still .zst).")

    p.add_argument("--counts-dir", type=Path,
                   help="Directory to store shard submission_counts_shard*.pkl for count_submissions/merge_counts.")
    p.add_argument("--active-subs-pkl", type=Path,
                   help="Path to save/load active_subs.pkl (list of active subreddit names).")

    p.add_argument("--min-posts", type=int, default=100,
                   help="Minimum number of SUBMISSIONS for a subreddit to be considered active.")
    p.add_argument("--subreddit-key", default="subreddit",
                   help="JSON key for subreddit (default: 'subreddit').")

    # SHARDING
    p.add_argument("--shard-id", type=int, default=0,
                   help="Shard index for this job (0-based).")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Total number of shards/jobs.")
    return p.parse_args()


def iter_zst(path: Path):
    dctx = zstd.ZstdDecompressor()
    with path.open("rb") as f:
        stream = dctx.stream_reader(f)
        buf = b""
        while True:
            chunk = stream.read(2**20)
            if not chunk:
                break
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for line in lines:
                if line:
                    yield line
        if buf:
            yield buf


def is_submission_file(path: Path):
    """
    Decide if this .zst file is submissions-only.

    Matches your layout:

      - 2005_2024/subreddits24/*_submissions.zst
      - 2025_0X/reddit/submissions/RS_2025-0X.zst

    Also supports classic RS_*.zst pattern.
    """
    name = path.name
    if name.endswith("_submissions.zst"):
        return True
    if name.startswith("RS_"):
        return True
    return False


# ---------------------------------------
# STAGE 1: SHARDED submission counting
# ---------------------------------------
def stage_count_submissions(args):
    if not args.input_dir or not args.counts_dir:
        print("--input-dir and --counts-dir are required for count_submissions", file=sys.stderr)
        sys.exit(1)

    input_root = args.input_dir.resolve()
    counts_dir = args.counts_dir.resolve()
    counts_dir.mkdir(parents=True, exist_ok=True)

    all_files = sorted(input_root.rglob("*.zst"))
    sub_files = [p for p in all_files if is_submission_file(p)]

    # shard by index
    sub_files = sub_files[args.shard_id::args.num_shards]

    print(
        f"[count_submissions] Shard {args.shard_id}/{args.num_shards}: {len(sub_files)} submissions files",
        file=sys.stderr,
    )

    counts = Counter()

    for idx, path in enumerate(sub_files, 1):
        file_total = 0

        for raw in iter_zst(path):
            file_total += 1
            try:
                obj = json.loads(raw)
            except Exception:
                continue

            sub = obj.get(args.subreddit_key)
            if isinstance(sub, str):
                counts[sub] += 1

            if file_total % 100_000 == 0:
                print(
                    f"[count {path.name}] {file_total:,} submissions processed",
                    file=sys.stderr,
                    flush=True,
                )

        print(
            f"[count {path.name}] DONE {file_total:,} submissions",
            file=sys.stderr,
            flush=True,
        )

    # save shard counts
    out_path = counts_dir / f"submission_counts_shard{args.shard_id}.pkl"
    with out_path.open("wb") as f:
        pickle.dump(counts, f)

    print(f"[count_submissions] Saved shard counts to {out_path}", file=sys.stderr)


# ---------------------------------------
# STAGE 2: merge submission counts
# ---------------------------------------
def stage_merge_counts(args):
    if not args.counts_dir or not args.active_subs_pkl:
        print("--counts-dir and --active-subs-pkl are required for merge_counts", file=sys.stderr)
        sys.exit(1)

    counts_dir = args.counts_dir.resolve()
    pkl_files = sorted(counts_dir.glob("submission_counts_shard*.pkl"))
    if not pkl_files:
        print(f"[merge_counts] No submission_counts_shard*.pkl found in {counts_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[merge_counts] Found {len(pkl_files)} shard PKLs", file=sys.stderr)

    global_counts = Counter()

    for pkl in pkl_files:
        print(f"[merge_counts] Loading {pkl}", file=sys.stderr)
        with pkl.open("rb") as f:
            c = pickle.load(f)
        global_counts.update(c)

    print(f"[merge_counts] Total unique subs (from submissions): {len(global_counts):,}", file=sys.stderr)

    active_subs = {s for s, c in global_counts.items() if c >= args.min_posts}
    print(f"[merge_counts] Active subs ≥ {args.min_posts} submissions: {len(active_subs):,}", file=sys.stderr)

    args.active_subs_pkl.parent.mkdir(parents=True, exist_ok=True)
    with args.active_subs_pkl.open("wb") as f:
        pickle.dump(sorted(active_subs), f)

    print(f"[merge_counts] Saved active_subs to {args.active_subs_pkl}", file=sys.stderr)


# ---------------------------------------
# STAGE 3: SHARDED filtering using active_subs.pkl
# ---------------------------------------
def stage_filter_all(args):
    if not args.input_dir or not args.output_dir or not args.active_subs_pkl:
        print("--input-dir, --output-dir and --active-subs-pkl are required for filter_all", file=sys.stderr)
        sys.exit(1)

    input_root = args.input_dir.resolve()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # load active subs
    with args.active_subs_pkl.open("rb") as f:
        active_subs = set(pickle.load(f))

    print(f"[filter_all] Loaded {len(active_subs):,} active subs from {args.active_subs_pkl}", file=sys.stderr)

    all_files = sorted(input_root.rglob("*.zst"))
    shard_files = all_files[args.shard_id::args.num_shards]

    print(
        f"[filter_all] Shard {args.shard_id}/{args.num_shards}: {len(shard_files)} files",
        file=sys.stderr,
    )

    for path in shard_files:
        rel = path.relative_to(input_root)
        out_path = output_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cctx = zstd.ZstdCompressor(level=3)
        file_total = 0
        file_kept = 0

        with out_path.open("wb") as f_out:
            w = cctx.stream_writer(f_out)

            for raw in iter_zst(path):
                file_total += 1
                try:
                    obj = json.loads(raw)
                except Exception:
                    continue

                sub = obj.get(args.subreddit_key)
                if not isinstance(sub, str):
                    continue

                if sub in active_subs:
                    # raw is a single JSON line; ensure newline when writing
                    w.write(raw + b"\n")
                    file_kept += 1

                if file_total % 200_000 == 0:
                    print(
                        f"[filter {path.name}] processed={file_total:,} kept={file_kept:,}",
                        file=sys.stderr,
                        flush=True,
                    )

            w.flush(zstd.FLUSH_FRAME)

        print(
            f"[filter {path.name}] DONE total={file_total:,} kept={file_kept:,}",
            file=sys.stderr,
            flush=True,
        )


# ---------------------------------------
def main():
    args = parse_args()

    if args.stage == "count_submissions":
        stage_count_submissions(args)

    elif args.stage == "merge_counts":
        stage_merge_counts(args)

    elif args.stage == "filter_all":
        stage_filter_all(args)

    else:
        print(f"Unknown stage: {args.stage}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
