#!/usr/bin/env python3
"""
Download driving video with object tracking (MOT) dataset.
Source : Kaggle — robikscube/driving-video-with-object-tracking
Dest   : data/raw/bdd100k_mot/

Authentication: kagglehub reads KAGGLE_USERNAME + KAGGLE_KEY from environment
or from ~/.kaggle/kaggle.json.

Parallel copy:
- Uses ThreadPoolExecutor
- Defaults to 32 copy workers
- Can be overridden with BDD_COPY_WORKERS=16, etc.
"""

import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from alert.utils.paths import data_raw_root


DST = data_raw_root() / "bdd100k_mot"
DATASET = "robikscube/driving-video-with-object-tracking"


def get_worker_count() -> int:
    env_value = os.getenv("BDD_COPY_WORKERS")

    if env_value:
        try:
            workers = int(env_value)
            if workers < 1:
                raise ValueError
            return workers
        except ValueError:
            print(f"WARNING: Invalid BDD_COPY_WORKERS={env_value!r}. Falling back to default.")

    return min(16, os.cpu_count() or 4)


def copy_file(src_file: Path, dst_file: Path) -> None:
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_file, dst_file)


def copy_directory_parallel(src: Path, dst: Path, max_workers: int) -> None:
    dst.mkdir(parents=True, exist_ok=True)

    # Create directories first, including empty directories.
    for src_dir in src.rglob("*"):
        if src_dir.is_dir():
            relative_dir = src_dir.relative_to(src)
            (dst / relative_dir).mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        src_file
        for src_file in src.rglob("*")
        if src_file.is_file()
    ]

    total_files = len(files_to_copy)

    if total_files == 0:
        print("No files found to copy.")
        return

    print(f"Copying {total_files} files using {max_workers} workers...")

    copied = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        for src_file in files_to_copy:
            relative_path = src_file.relative_to(src)
            dst_file = dst / relative_path
            futures.append(executor.submit(copy_file, src_file, dst_file))

        for future in as_completed(futures):
            future.result()
            copied += 1

            if copied % 1000 == 0 or copied == total_files:
                print(f"Copied {copied}/{total_files} files")


def main() -> None:
    try:
        import kagglehub
    except ImportError:
        print("ERROR: kagglehub not installed. Run: uv add kagglehub")
        sys.exit(1)

    print(f"Downloading Kaggle dataset: {DATASET}")

    try:
        cache_path = kagglehub.dataset_download(DATASET)
    except Exception as exc:
        print(f"\nERROR: {exc}")
        print(
            "\nCommon fixes:\n"
            "  - Accept the dataset license on Kaggle:\n"
            f"    https://www.kaggle.com/datasets/{DATASET}\n"
            "  - Verify KAGGLE_USERNAME and KAGGLE_KEY in ~/.kaggle/kaggle.json\n"
        )
        sys.exit(1)

    src = Path(cache_path)
    max_workers = get_worker_count()

    print(f"Downloaded to cache: {src}")
    print(f"Copying to: {DST}")

    try:
        copy_directory_parallel(src, DST, max_workers=max_workers)
    except Exception as exc:
        print(f"\nERROR while copying files: {exc}")
        sys.exit(1)

    print(f"\nDone. Data saved to:\n  {DST}")


if __name__ == "__main__":
    main()