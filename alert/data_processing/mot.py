"""Convert raw bdd100k_mot data into per-split parquet labels + symlinked videos."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def filter_have_video(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["haveVideo"] == True].reset_index(drop=True)  # noqa: E712


def split_mot_labels(
    df: pd.DataFrame, assignment: dict[str, str]
) -> dict[str, pd.DataFrame]:
    split_col = df["videoName"].map(assignment)
    out: dict[str, pd.DataFrame] = {}
    for split in ["train", "val", "test"]:
        sub = df[split_col == split].reset_index(drop=True)
        out[split] = sub
    return out


def symlink_videos(
    assignment: dict[str, str],
    src_video_dir: Path,
    dst_video_root: Path,
) -> dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    for video_name, split in assignment.items():
        src = src_video_dir / f"{video_name}.mov"
        if not src.exists():
            raise FileNotFoundError(f"Missing video: {src}")
        dst_dir = dst_video_root / split
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{video_name}.mov"
        if not dst.exists():
            dst.symlink_to(src.resolve())
        counts[split] += 1
    return counts


def process_mot(
    *,
    src_parquet: Path,
    src_video_dir: Path,
    assignment: dict[str, str],
    out_root: Path,
) -> dict:
    df = pd.read_parquet(src_parquet)
    df = filter_have_video(df)

    label_videos = set(df["videoName"].unique())
    miss = set(assignment) - label_videos
    if miss:
        raise ValueError(f"{len(miss)} assigned videos missing from labels parquet")

    parts = split_mot_labels(df, assignment)
    labels_out = out_root / "labels"
    labels_out.mkdir(parents=True, exist_ok=True)
    summary: dict = {}
    for split, part in parts.items():
        part_path = labels_out / f"{split}.parquet"
        part.to_parquet(part_path, index=False)
        summary[split] = {
            "n_rows": len(part),
            "n_videos": part["videoName"].nunique(),
            "path": str(part_path),
        }

    sym_counts = symlink_videos(
        assignment, src_video_dir, out_root / "videos"
    )
    for split, n in sym_counts.items():
        summary[split]["n_video_files"] = n

    return summary
