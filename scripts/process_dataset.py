#!/usr/bin/env python3
"""End-to-end ALERT data processing.

Sources (raw):
    data/raw/bdd100k_alvaromalfaro/bdd100k/labels/{train,val}/*.json
    data/raw/bdd100k_alvaromalfaro/bdd100k/images/100k/{train,val}/*.jpg
    data/raw/bdd100k_mot/mot_labels.parquet
    data/raw/bdd100k_mot/bdd100k_videos_train_00/bdd100k/videos/train/*.mov

Outputs (processed):
    data/processed/bdd100k_frames/
        images/{train,val,test}/*.jpg            (symlinks to raw)
        labels/{train,val,test}/*.txt            (YOLO format)
        instances_{train,val,test}.json          (COCO format for pycocotools)
        drivable/polygons/{train,val,test}/*.json   (source-of-truth polygons)
        drivable/masks/{train,val,test}/*.png       (class-ID PNG: 0=direct,1=alt,2=bg)
        drivable/colormaps/{train,val,test}/*.png   (RGB visualization)
        bdd100k.yaml                             (Ultralytics dataset config)
    data/processed/bdd100k_videos/
        videos/{train,val,test}/*.mov            (symlinks to raw)
        labels/{train,val,test}.parquet
    data/processed/splits/
        detection_split.json   {stem: split}
        mot_split.json         {videoName: split}

Splits:
    Detection: original BDD100K split — 70k train / 10k val / 20k test from alvaromalfaro.
    MOT:       761/100/100 from the 961 videos with footage.
               Stratified by timeofday (joined via alvaromalfaro train labels).

Modes:
    (default)  Full dataset (70k/10k/20k det, 761/100/100 mot).  ~15 min, server GPU recommended.
    --local    2000/500/500 det train/val/test, 50/8/8 mot.  ~30 s, fits 8 GB VRAM, local laptop dev.
    --tiny     80/10/10 det train/val/test, 12/3/3 mot.  ~5 s, smoke test only — too small to train.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from alert.data_processing.detection import (
    process_detection_split,
    write_yolo_dataset_yaml,
)
from alert.data_processing.drivable import process_drivable_split
from alert.data_processing.mot import process_mot
from alert.data_processing.splits import (
    make_detection_split,
    make_mot_split,
    split_counts,
    stratum_counts,
    write_split,
)
from alert.utils.paths import data_processed_root, data_raw_root

RAW = data_raw_root()
PROC = data_processed_root()

ALV_ROOT = RAW / "bdd100k_alvaromalfaro" / "bdd100k"
ALV_LABELS = ALV_ROOT / "labels"
ALV_IMAGES = ALV_ROOT / "images" / "100k"

MOT_ROOT = RAW / "bdd100k_mot"
MOT_PARQUET = MOT_ROOT / "mot_labels.parquet"
MOT_VIDEO_DIR = MOT_ROOT / "bdd100k_videos_train_00" / "bdd100k" / "videos" / "train"

FRAMES_OUT = PROC / "bdd100k_frames"
VIDEOS_OUT = PROC / "bdd100k_videos"
SPLITS_OUT = PROC / "splits"


def _step(msg: str) -> None:
    print(f"\n=== {msg} ===")


def _read_timeofday(label_dir: Path, stems: list[str]) -> dict[str, str]:
    """Read timeofday attribute from each per-image alvaromalfaro JSON."""
    out: dict[str, str] = {}
    for stem in stems:
        try:
            data = json.loads((label_dir / f"{stem}.json").read_text())
            out[stem] = (data.get("attributes") or {}).get("timeofday") or "undefined"
        except FileNotFoundError:
            out[stem] = "undefined"
    return out


def _list_stems(label_dir: Path) -> list[str]:
    return sorted(p.stem for p in label_dir.glob("*.json"))


def build_splits(
    *,
    n_det_train: int | None,
    n_det_val: int | None,
    n_det_test: int | None,
    n_mot_train: int,
    n_mot_val: int,
    n_mot_test: int,
) -> tuple[dict[str, str], dict[str, str]]:
    """Build the detection and MOT split assignments."""
    _step("Building splits")

    train_stems = _list_stems(ALV_LABELS / "train")
    val_stems = _list_stems(ALV_LABELS / "val")
    test_stems = _list_stems(ALV_LABELS / "test") if (ALV_LABELS / "test").exists() else []
    print(f"  raw alvaromalfaro: train={len(train_stems)}, val={len(val_stems)}, test={len(test_stems)}")

    if n_det_train is not None:
        train_stems = train_stems[:n_det_train]
    if n_det_val is not None:
        val_stems = val_stems[:n_det_val]
    if n_det_test is not None:
        test_stems = test_stems[:n_det_test]
    if any(x is not None for x in (n_det_train, n_det_val, n_det_test)):
        print(f"  subset mode: train={len(train_stems)}, val={len(val_stems)}, test={len(test_stems)}")

    train_strata = _read_timeofday(ALV_LABELS / "train", train_stems)
    val_strata = _read_timeofday(ALV_LABELS / "val", val_stems)
    test_strata = _read_timeofday(ALV_LABELS / "test", test_stems) if test_stems else {}
    det_assignment = make_detection_split(train_stems, val_stems, test_stems)
    print(f"  detection split counts: {split_counts(det_assignment)}")
    print(f"  detection stratification:")
    full_strata = {**train_strata, **val_strata, **test_strata}
    for split, dist in stratum_counts(det_assignment, full_strata).items():
        print(f"    {split}: {dist}")

    write_split(det_assignment, SPLITS_OUT / "detection_split.json")
    print(f"  → wrote {SPLITS_OUT / 'detection_split.json'}")

    # MOT: filter haveVideo=True, join timeofday from alvaromalfaro train (perfect 1:1)
    import pandas as pd

    df = pd.read_parquet(MOT_PARQUET)
    have_video_videos = sorted(set(df[df["haveVideo"] == True]["videoName"].unique()))  # noqa: E712
    print(f"  raw MOT: {len(have_video_videos)} videos with footage")

    total_mot = n_mot_train + n_mot_val + n_mot_test
    if len(have_video_videos) > total_mot:
        have_video_videos = have_video_videos[:total_mot]
        print(f"  subset mode: limited videos={len(have_video_videos)}")

    mot_strata = _read_timeofday(ALV_LABELS / "train", have_video_videos)
    mot_assignment = make_mot_split(
        have_video_videos, mot_strata,
        n_train=n_mot_train, n_val=n_mot_val, n_test=n_mot_test,
    )
    print(f"  MOT split counts: {split_counts(mot_assignment)}")
    print(f"  MOT stratification:")
    for split, dist in stratum_counts(mot_assignment, mot_strata).items():
        print(f"    {split}: {dist}")

    write_split(mot_assignment, SPLITS_OUT / "mot_split.json")
    print(f"  → wrote {SPLITS_OUT / 'mot_split.json'}")

    return det_assignment, mot_assignment


def _build_path_maps(det_assignment: dict[str, str]) -> tuple[dict[str, Path], dict[str, Path]]:
    """For each stem, return its source label dir and image dir (same as its split name)."""
    label_map: dict[str, Path] = {}
    image_map: dict[str, Path] = {}
    for stem, split in det_assignment.items():
        label_map[stem] = ALV_LABELS / split
        image_map[stem] = ALV_IMAGES / split
    return label_map, image_map


def run_detection(det_assignment: dict[str, str], workers: int) -> None:
    _step("Processing detection (YOLO + COCO + image symlinks)")
    label_map, image_map = _build_path_maps(det_assignment)

    by_split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for stem, split in det_assignment.items():
        by_split[split].append(stem)

    for split, stems in by_split.items():
        t0 = time.time()
        summary = process_detection_split(
            split=split,
            stems=stems,
            src_label_dir_for_stem=label_map,
            src_image_dir_for_stem=image_map,
            out_root=FRAMES_OUT,
            workers=workers,
        )
        dt = time.time() - t0
        print(f"  {split}: {summary['n_images']} images, "
              f"{summary['n_boxes']} boxes  [{dt:.1f}s]")

    yaml_path = write_yolo_dataset_yaml(FRAMES_OUT)
    print(f"  → wrote {yaml_path}")


def run_drivable(det_assignment: dict[str, str], workers: int) -> None:
    _step("Processing drivable polygons + colormaps")
    label_map, _ = _build_path_maps(det_assignment)

    by_split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for stem, split in det_assignment.items():
        by_split[split].append(stem)

    for split, stems in by_split.items():
        t0 = time.time()
        summary = process_drivable_split(
            split=split,
            stems=stems,
            src_label_dir_for_stem=label_map,
            out_root=FRAMES_OUT,
            workers=workers,
        )
        dt = time.time() - t0
        print(f"  {split}: {summary['n_images']} images, "
              f"{summary['n_polygons']} drivable polygons  [{dt:.1f}s]")


def run_mot(mot_assignment: dict[str, str]) -> None:
    _step("Processing MOT (parquet split + video symlinks)")
    t0 = time.time()
    summary = process_mot(
        src_parquet=MOT_PARQUET,
        src_video_dir=MOT_VIDEO_DIR,
        assignment=mot_assignment,
        out_root=VIDEOS_OUT,
    )
    dt = time.time() - t0
    for split, info in summary.items():
        print(f"  {split}: {info['n_videos']} videos, "
              f"{info['n_rows']} annotation rows, "
              f"{info['n_video_files']} symlinks  → {Path(info['path']).name}")
    print(f"  done [{dt:.1f}s]")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--tiny", action="store_true",
                      help="Smoke-test subset (80/10/10 det train/val/test, 12/3/3 mot). "
                           "Verifies the pipeline runs end-to-end; too small to train.")
    mode.add_argument("--local", action="store_true",
                      help="Local-dev subset (2000/500/500 det train/val/test, 50/8/8 mot). "
                           "Fits in 8 GB VRAM; enough for a full training loop on a laptop GPU.")
    p.add_argument("--workers", type=int, default=8, help="Parallel workers for I/O")
    p.add_argument("--steps", default="splits,detection,drivable,mot",
                   help="Comma-separated subset of steps to run")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    steps = set(args.steps.split(","))

    if args.tiny:
        n_det_train, n_det_val, n_det_test = 80, 10, 10
        n_mot_train, n_mot_val, n_mot_test = 12, 3, 3
    elif args.local:
        n_det_train, n_det_val, n_det_test = 2_000, 500, 500
        n_mot_train, n_mot_val, n_mot_test = 50, 8, 8
    else:
        n_det_train, n_det_val, n_det_test = None, None, None
        n_mot_train, n_mot_val, n_mot_test = 761, 100, 100

    det_assignment, mot_assignment = build_splits(
        n_det_train=n_det_train, n_det_val=n_det_val, n_det_test=n_det_test,
        n_mot_train=n_mot_train, n_mot_val=n_mot_val, n_mot_test=n_mot_test,
    )

    if "detection" in steps:
        run_detection(det_assignment, workers=args.workers)
    if "drivable" in steps:
        run_drivable(det_assignment, workers=args.workers)
    if "mot" in steps:
        run_mot(mot_assignment)

    _step("Done")
    print(f"  outputs under {PROC}")


if __name__ == "__main__":
    main()
