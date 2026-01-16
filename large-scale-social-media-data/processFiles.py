import sys
version = sys.version_info
if version.major < 3 or (version.major == 3 and version.minor < 10):
    raise RuntimeError("This script requires Python 3.10 or higher")

import os
from typing import Iterable
from pathlib import Path
from datetime import datetime
import csv

from fileStreams import getFileJsonStream
from utils import FileProgressLog

# ========= CONFIG =========
# Folder that contains ALL your downloaded dumps:
#  - 2005-2024 torrents
#  - 2025_01, 2025_02, ...
fileOrFolderPath = r"/p/zenodo/reddit"

# Process subfolders recursively (True = walk nested year/month dirs)
recursive = True

# Only keep rows from these years:
YEAR_WHITELIST = {2020, 2020}

# Where to write per-file user–subreddit pairs
OUTPUT_DIR = Path("/p/zenodo/code/reddit/output/2020/user_sub_pairs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# ==========================


def year_from_timestamp(ts: int) -> int:
    return datetime.utcfromtimestamp(int(ts)).year


def is_valid_user(author: str) -> bool:
    if not author:
        return False
    author = str(author)
    if author in ("[deleted]", "AutoModerator"):
        return False
    return True


def processFile(path: str):
    print(f"Processing file {path}")
    basename = os.path.basename(path)
    out_path = OUTPUT_DIR / f"user_sub_pairs_{basename}.csv"

    # If you want to skip already-processed:
    # if out_path.exists():
    #     print(f"  -> Skipping, output exists: {out_path}")
    #     return

    with open(path, "rb") as f:
        jsonStream = getFileJsonStream(path, f)
        if jsonStream is None:
            print(f"Skipping unknown file {path}")
            return

        progressLog = FileProgressLog(path, f)

        # write a small CSV per input file
        with open(out_path, "w", newline="", encoding="utf-8") as out_f:
            writer = csv.writer(out_f)
            writer.writerow(["author", "subreddit"])  # header

            for row in jsonStream:
                progressLog.onRow()

                try:
                    created = row["created_utc"]
                    year = year_from_timestamp(created)
                except Exception:
                    # if created_utc missing/corrupt, skip row
                    continue

                if year not in YEAR_WHITELIST:
                    continue

                author = row.get("author")
                subreddit = row.get("subreddit")

                if not is_valid_user(author) or subreddit is None:
                    continue

                writer.writerow([author, subreddit])

            progressLog.logProgress("\n")

    print(f"  -> Wrote {out_path}")


def processFolder(path: str):
    if recursive:
        def recursiveFileIterator():
            for root, dirs, files in os.walk(path):
                for file in files:
                    yield os.path.join(root, file)
        fileIterator: Iterable[str] = recursiveFileIterator()
    else:
        fileIterator = os.listdir(path)
        fileIterator = (os.path.join(path, file) for file in fileIterator)

    for i, file in enumerate(fileIterator):
        print(f"Processing file {i + 1:3} {file}")
        processFile(file)


def main():
    if os.path.isdir(fileOrFolderPath):
        processFolder(fileOrFolderPath)
    else:
        processFile(fileOrFolderPath)
    print("Done :>")


if __name__ == "__main__":
    main()
