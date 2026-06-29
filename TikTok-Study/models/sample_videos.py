"""
Sample videos for the classification benchmark.

Two modes:

  global (default)
      Sample --n videos per platform (50 TikTok + 50 Instagram) from the full
      available pool.  Videos sorted by video_id → staged as
      data/sampled_videos/001/video.mp4 … 100/video.mp4.

  per-subject  (--per-subject)
      Sample up to 10 videos per participant.  If a subject has both platforms,
      take 5 from each.  If only one platform, take up to 10 from it.  If a
      subject has no downloaded videos, list and skip them.  TikTok source is
      tiktok_videos/random_3_by_subjects/ (per-subject download dir).  Videos
      sorted by (subject_id, video_id) → staged as
      data/sampled_by_subject/001/video.mp4 … NNN/video.mp4.

Usage:
  python sample_videos.py                          # global, 50 per platform
  python sample_videos.py --seed 99 --n 25         # 25 per platform
  python sample_videos.py --symlink                # symlink instead of copy
  python sample_videos.py --per-subject            # 10 per subject (5+5)
  python sample_videos.py --per-subject --seed 99
"""

import argparse
import csv
import glob
import os
import re
import random
import shutil
from pathlib import Path

# ── Default data paths (relative to this script's location) ──────────────────
_SCRIPT_DIR   = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent.parent.parent  # .../released_data

# Global mode
TIKTOK_VIDEO_DIR     = _PROJECT_ROOT / "video_frames"
INSTAGRAM_VIDEO_DIR  = _PROJECT_ROOT / "instagram_videos"
TIKTOK_CACHE_GLOB    = str(_PROJECT_ROOT / "video_cache" / "tiktok" / "tiktok_video_cache_*.csv")
INSTAGRAM_CACHE_GLOB = str(_PROJECT_ROOT / "video_cache" / "instagram" / "instagram_media_cache_*.csv")
OUT_DIR              = _SCRIPT_DIR.parent / "data" / "classification"
SAMPLED_VIDEOS_DIR   = _SCRIPT_DIR.parent / "data" / "sampled_videos"

# Per-subject mode
PER_SUBJECT_TIKTOK_DIR  = _PROJECT_ROOT / "tiktok_videos" / "random_3_by_subjects"
PER_SUBJECT_OUT_DIR     = _SCRIPT_DIR.parent / "data" / "classification_by_subject"
PER_SUBJECT_SAMPLED_DIR = _SCRIPT_DIR.parent / "data" / "sampled_by_subject"

RANDOM_SEED      = 42
N_PER_PLATFORM   = 50
N_PER_SUBJECT    = 10   # per-subject mode total per participant

# ── URL parsers ───────────────────────────────────────────────────────────────

_TIKTOK_ID_RE    = re.compile(r"/video/(\d+)|/share/video/(\d+)|/v/(\d+)", re.IGNORECASE)
_INSTAGRAM_ID_RE = re.compile(r"/(?:p|reel|tv)/([^/?#]+)", re.IGNORECASE)


def _tiktok_video_id(url: str) -> "str | None":
    if not url:
        return None
    m = _TIKTOK_ID_RE.search(url)
    return (m.group(1) or m.group(2) or m.group(3)) if m else None


def _instagram_post_id(url: str) -> "str | None":
    if not url:
        return None
    m = _INSTAGRAM_ID_RE.search(url)
    return m.group(1) if m else None


# ── Metadata loading ──────────────────────────────────────────────────────────

def load_tiktok_metadata(cache_glob: str) -> dict[str, dict]:
    """Load all TikTok cache CSVs, keep newest non-empty enrich_status per video_id."""
    meta: dict[str, dict] = {}
    for fpath in sorted(glob.glob(cache_glob), reverse=True):
        with open(fpath, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                vid = row.get("video_id", "").strip()
                if not vid:
                    continue
                if vid not in meta and row.get("enrich_status") == "ok":
                    meta[vid] = {
                        "video_id":      vid,
                        "platform":      "tiktok",
                        "title":         row.get("enrich_title", ""),
                        "description":   row.get("enrich_description", ""),
                        "hashtags":      row.get("enrich_hashtags", ""),
                        "uploader":      row.get("enrich_artist", ""),
                        "upload_date":   row.get("enrich_upload_date", ""),
                        "duration":      row.get("enrich_duration", ""),
                        "view_count":    row.get("enrich_view_count", ""),
                        "like_count":    row.get("enrich_like_count", ""),
                        "comment_count": row.get("enrich_comment_count", ""),
                        "repost_count":  row.get("enrich_repost_count", ""),
                        "webpage_url":   row.get("enrich_webpage_url", ""),
                    }
    return meta


def load_instagram_metadata(cache_glob: str) -> dict[str, dict]:
    """Load all Instagram cache CSVs, keep newest non-empty enrich_status per post_id."""
    meta: dict[str, dict] = {}
    for fpath in sorted(glob.glob(cache_glob), reverse=True):
        with open(fpath, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                vid = row.get("post_id", "").strip()
                if not vid:
                    continue
                if vid not in meta and row.get("enrich_status") == "ok":
                    meta[vid] = {
                        "video_id":      vid,
                        "platform":      "instagram",
                        "title":         row.get("enrich_title", ""),
                        "description":   row.get("enrich_description", ""),
                        "hashtags":      row.get("enrich_hashtags", ""),
                        "uploader":      row.get("enrich_uploader", ""),
                        "upload_date":   row.get("enrich_upload_date", ""),
                        "duration":      row.get("enrich_duration", ""),
                        "view_count":    row.get("enrich_view_count", ""),
                        "like_count":    row.get("enrich_like_count", ""),
                        "comment_count": row.get("enrich_comment_count", ""),
                        "repost_count":  "",
                        "webpage_url":   row.get("enrich_webpage_url", ""),
                    }
    return meta


# ── Video discovery (global mode) ─────────────────────────────────────────────

def find_videos_with_mp4(video_dir: Path) -> list[tuple[str, Path]]:
    """Return [(video_id, path_to_video.mp4)] for all nested dirs with video.mp4."""
    results = []
    if not video_dir.exists():
        print(f"  WARNING: {video_dir} does not exist — skipping")
        return results
    for d in sorted(video_dir.iterdir()):
        if d.is_dir():
            vp = d / "video.mp4"
            if vp.exists():
                results.append((d.name, vp))
    return results


# ── Copy / symlink videos ─────────────────────────────────────────────────────

def stage_video(src: Path, row_id: int, sampled_dir: Path, symlink: bool) -> Path:
    """Copy (or symlink) src into sampled_dir/{row_id:03d}/video.mp4."""
    dest_dir = sampled_dir / f"{row_id:03d}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "video.mp4"
    if dest.exists() or dest.is_symlink():
        return dest
    if symlink:
        dest.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dest)
    return dest


# ── Global mode sampling ──────────────────────────────────────────────────────

def sample_videos(video_dir: Path, metadata: dict[str, dict],
                  n: int, seed: int, platform: str) -> list[dict]:
    all_videos = find_videos_with_mp4(video_dir)
    print(f"  {platform}: {len(all_videos)} videos with video.mp4 found")

    with_meta    = [(vid, vp) for vid, vp in all_videos if vid in metadata]
    without_meta = [(vid, vp) for vid, vp in all_videos if vid not in metadata]
    print(f"  {platform}: {len(with_meta)} with metadata, "
          f"{len(without_meta)} without")

    rng     = random.Random(seed)
    pool    = with_meta + without_meta
    sampled = rng.sample(pool, min(n, len(pool)))
    if len(sampled) < n:
        print(f"  WARNING: only {len(sampled)}/{n} videos available for {platform}")

    rows = []
    for vid_id, src_vp in sampled:
        meta = metadata.get(vid_id, {})
        rows.append({
            "subject_id":    "",
            "video_id":      vid_id,
            "platform":      platform,
            "source_path":   str(src_vp),
            "title":         meta.get("title", ""),
            "description":   meta.get("description", ""),
            "hashtags":      meta.get("hashtags", ""),
            "uploader":      meta.get("uploader", ""),
            "upload_date":   meta.get("upload_date", ""),
            "duration":      meta.get("duration", ""),
            "view_count":    meta.get("view_count", ""),
            "like_count":    meta.get("like_count", ""),
            "comment_count": meta.get("comment_count", ""),
            "repost_count":  meta.get("repost_count", ""),
            "webpage_url":   meta.get("webpage_url", ""),
        })
    return rows


# ── Per-subject mode helpers ──────────────────────────────────────────────────

def discover_subjects(base: Path) -> dict[str, dict[str, Path]]:
    """
    Scan released CSVs and return {subj_dir: {platform: latest_csv_path}}.
    subj_dir is the anon_id directory (parent of tiktok/ and instagram/ subdirs).
    """
    subjects: dict[str, dict[str, Path]] = {}
    for platform in ("tiktok", "instagram"):
        # */* matches {study_dir}/{anon_id}
        for p in sorted((base).glob(f"*/*/{platform}/released-*.csv")):
            if "_old" in str(p):
                continue
            subj_dir = str(p.parent.parent)   # .../anon_id/
            if subj_dir not in subjects:
                subjects[subj_dir] = {}
            subjects[subj_dir][platform] = p  # sorted → latest filename wins
    return subjects


def _meta_fields(meta: dict) -> dict:
    return {k: meta.get(k, "") for k in (
        "title", "description", "hashtags", "uploader", "upload_date",
        "duration", "view_count", "like_count", "comment_count",
        "repost_count", "webpage_url",
    )}


def get_subject_tiktok_available(
    csv_path: Path, tiktok_dir: Path, meta: dict[str, dict]
) -> list[dict]:
    """Watch-history video IDs from subject's CSV that exist in tiktok_dir."""
    available = []
    seen: set[str] = set()
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("category") == "activity" and row.get("type") == "watch_history":
                    vid = _tiktok_video_id(row.get("uri", ""))
                    if not vid or vid in seen:
                        continue
                    seen.add(vid)
                    vp = tiktok_dir / vid / "video.mp4"
                    if vp.exists():
                        available.append({
                            "video_id":    vid,
                            "platform":    "tiktok",
                            "source_path": str(vp),
                            **_meta_fields(meta.get(vid, {})),
                        })
    except Exception:
        pass
    return available


def get_subject_instagram_available(
    csv_path: Path, ig_dir: Path, meta: dict[str, dict]
) -> list[dict]:
    """Instagram post IDs from subject's CSV that exist in ig_dir."""
    available = []
    seen: set[str] = set()
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                post_id = _instagram_post_id(row.get("uri", ""))
                if not post_id or post_id in seen:
                    continue
                seen.add(post_id)
                for fname in ("video.mp4", "image.jpg"):
                    vp = ig_dir / post_id / fname
                    if vp.exists():
                        available.append({
                            "video_id":    post_id,
                            "platform":    "instagram",
                            "source_path": str(vp),
                            **_meta_fields(meta.get(post_id, {})),
                        })
                        break
    except Exception:
        pass
    return available


def run_per_subject(
    subjects: dict[str, dict[str, Path]],
    tiktok_dir: Path,
    ig_dir: Path,
    tiktok_meta: dict[str, dict],
    ig_meta: dict[str, dict],
    n_total: int,
    seed: int,
) -> tuple[list[dict], list[str]]:
    """
    Sample up to n_total videos per subject.
    - Both platforms available: n_total//2 from each.
    - One platform only: up to n_total from it.
    - No videos at all: skip and record name.

    Returns (rows, skipped_anon_ids).
    """
    rng         = random.Random(seed)
    n_each      = n_total // 2
    all_rows:   list[dict] = []
    skipped:    list[str]  = []

    for subj_dir in sorted(subjects):
        platforms = subjects[subj_dir]
        anon_id   = os.path.basename(subj_dir)

        tk_avail: list[dict] = []
        ig_avail: list[dict] = []

        if "tiktok" in platforms:
            tk_avail = get_subject_tiktok_available(
                platforms["tiktok"], tiktok_dir, tiktok_meta)
        if "instagram" in platforms:
            ig_avail = get_subject_instagram_available(
                platforms["instagram"], ig_dir, ig_meta)

        if not tk_avail and not ig_avail:
            skipped.append(anon_id)
            continue

        has_both = bool(tk_avail) and bool(ig_avail)

        if has_both:
            tk_sample = rng.sample(tk_avail, min(n_each, len(tk_avail)))
            ig_sample = rng.sample(ig_avail, min(n_each, len(ig_avail)))
        elif tk_avail:
            tk_sample = rng.sample(tk_avail, min(n_total, len(tk_avail)))
            ig_sample = []
        else:
            tk_sample = []
            ig_sample = rng.sample(ig_avail, min(n_total, len(ig_avail)))

        n_tk = len(tk_sample)
        n_ig = len(ig_sample)
        print(f"  {anon_id[:20]:<20}  TikTok: {n_tk:>3}  Instagram: {n_ig:>3}  "
              f"total: {n_tk+n_ig}")

        for row in tk_sample + ig_sample:
            row["subject_id"] = anon_id
            all_rows.append(row)

    return all_rows, skipped


# ── CSV schema ────────────────────────────────────────────────────────────────

MANIFEST_COLS = [
    "row_id", "subject_id", "video_id", "platform", "video_path", "source_path",
    "title", "description", "hashtags", "uploader", "upload_date",
    "duration", "view_count", "like_count", "comment_count",
    "repost_count", "webpage_url",
]

GT_COLS = [
    "row_id", "subject_id", "video_id", "platform", "video_path",
    "gt_content_category",
    "gt_risk_level",
    "gt_summary",
    "gt_key_message",
    "gt_notes",
]


def write_csv(path: Path, rows: list[dict], cols: list[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in cols})


# ── Stage + assign row IDs ────────────────────────────────────────────────────

def stage_all(rows: list[dict], sampled_dir: Path, symlink: bool) -> list[dict]:
    """Assign row_id (1-indexed) and stage videos; mutates rows in place."""
    action = "Symlinking" if symlink else "Copying"
    print(f"\n{action} {len(rows)} videos → {sampled_dir}/")
    for row_id, row in enumerate(rows, 1):
        dest = stage_video(Path(row["source_path"]), row_id, sampled_dir, symlink)
        row["row_id"]     = f"{row_id:03d}"
        row["video_path"] = str(dest)
        subj = f"[{row['subject_id'][:12]}]" if row.get("subject_id") else ""
        print(f"  [{row_id:03d}] {row['platform']:<10} {row['video_id']}  {subj}")
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--per-subject",    type=int, nargs="?", const=N_PER_SUBJECT,
                   metavar="N",
                   help=f"Per-subject mode: sample N videos per participant "
                        f"(default {N_PER_SUBJECT}, half TikTok + half Instagram if both available). "
                        f"TikTok source: tiktok_videos/random_3_by_subjects/")
    p.add_argument("--n",              type=int, default=N_PER_PLATFORM,
                   help="(global) Number of videos per platform (default 50)")
    p.add_argument("--seed",           type=int, default=RANDOM_SEED)
    p.add_argument("--tiktok-dir",     default=str(TIKTOK_VIDEO_DIR))
    p.add_argument("--instagram-dir",  default=str(INSTAGRAM_VIDEO_DIR))
    p.add_argument("--tiktok-cache",   default=TIKTOK_CACHE_GLOB)
    p.add_argument("--ig-cache",       default=INSTAGRAM_CACHE_GLOB)
    p.add_argument("--out-dir",        default=None,
                   help="Where to write CSVs (default depends on mode)")
    p.add_argument("--sampled-dir",    default=None,
                   help="Where to stage video files (default depends on mode)")
    p.add_argument("--symlink",        action="store_true",
                   help="Symlink videos instead of copying (saves disk space)")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve output dirs based on mode
    if args.per_subject is not None:
        out_dir     = Path(args.out_dir)     if args.out_dir     else PER_SUBJECT_OUT_DIR
        sampled_dir = Path(args.sampled_dir) if args.sampled_dir else PER_SUBJECT_SAMPLED_DIR
        tiktok_dir  = PER_SUBJECT_TIKTOK_DIR
        ig_dir      = Path(args.instagram_dir)
    else:
        out_dir     = Path(args.out_dir)     if args.out_dir     else OUT_DIR
        sampled_dir = Path(args.sampled_dir) if args.sampled_dir else SAMPLED_VIDEOS_DIR
        tiktok_dir  = Path(args.tiktok_dir)
        ig_dir      = Path(args.instagram_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    sampled_dir.mkdir(parents=True, exist_ok=True)

    mode_label = "per-subject" if args.per_subject is not None else "global"
    print(f"Mode            : {mode_label}")
    print(f"Seed            : {args.seed}")
    print(f"TikTok dir      : {tiktok_dir}")
    print(f"Instagram dir   : {ig_dir}")
    print(f"Sampled videos  : {sampled_dir}  ({'symlink' if args.symlink else 'copy'})")
    print(f"Classification  : {out_dir}\n")

    print("Loading TikTok metadata...")
    tk_meta = load_tiktok_metadata(args.tiktok_cache)
    print(f"  {len(tk_meta)} TikTok videos with enrichment metadata")

    print("Loading Instagram metadata...")
    ig_meta = load_instagram_metadata(args.ig_cache)
    print(f"  {len(ig_meta)} Instagram videos with enrichment metadata\n")

    # ── Per-subject mode ──────────────────────────────────────────────────────
    if args.per_subject is not None:
        print("Discovering subjects from released CSVs...")
        subjects = discover_subjects(_PROJECT_ROOT)
        print(f"  {len(subjects)} subjects found\n")
        print("Sampling per subject:")
        all_rows, skipped = run_per_subject(
            subjects, tiktok_dir, ig_dir, tk_meta, ig_meta,
            args.per_subject, args.seed,
        )

        if skipped:
            print(f"\n  {len(skipped)} subject(s) with NO downloaded videos — omitted:")
            for s in skipped:
                print(f"    {s}")

        # Sort: group by subject, then by video_id within each subject
        all_rows.sort(key=lambda r: (r["subject_id"], r["video_id"]))

    # ── Global mode ───────────────────────────────────────────────────────────
    else:
        print("Sampling TikTok...")
        tk_rows = sample_videos(tiktok_dir, tk_meta, args.n, args.seed, "tiktok")

        print("\nSampling Instagram...")
        ig_rows = sample_videos(ig_dir, ig_meta, args.n, args.seed + 1, "instagram")

        all_rows = sorted(tk_rows + ig_rows, key=lambda r: r["video_id"])

    # ── Stage videos and write CSVs ───────────────────────────────────────────
    all_rows = stage_all(all_rows, sampled_dir, args.symlink)

    tk_count = sum(1 for r in all_rows if r["platform"] == "tiktok")
    ig_count = sum(1 for r in all_rows if r["platform"] == "instagram")
    print(f"\nTotal sampled: {len(all_rows)} videos  "
          f"({tk_count} TikTok + {ig_count} Instagram)")
    if args.per_subject is not None:
        subj_count = len({r["subject_id"] for r in all_rows})
        print(f"Subjects with videos: {subj_count}")
    print(f"Videos staged : {sampled_dir}/001 … {len(all_rows):03d}\n")

    manifest_path = out_dir / "sample_manifest.csv"
    gt_path       = out_dir / "ground_truth.csv"
    write_csv(manifest_path, all_rows, MANIFEST_COLS)
    write_csv(gt_path,       all_rows, GT_COLS)

    print(f"Written: {manifest_path}")
    print(f"Written: {gt_path}")
    print()
    print("Next steps:")
    print("  1. Have annotators open ground_truth.csv and fill in the gt_* columns")
    print("  2. Run analyze_gemma4.py --data-dir <sampled-dir> --layout nested")
    print("  3. Run analyze_metadata_only.py --meta-csv <out-dir>/sample_manifest.csv")
    print("  4. Run compare_modalities.py --data-dir <sampled-dir> for LLaVA ablation")
    print("  5. Run evaluate_benchmark.py --gt <out-dir>/ground_truth.csv")


if __name__ == "__main__":
    main()
