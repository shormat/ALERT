"""Deterministic, stratified train/val/test splits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from sklearn.model_selection import train_test_split

SEED = 42


def _coalesce_rare_strata(strata: list[str], min_count: int = 2) -> list[str]:
    # sklearn stratified split requires >= min_count members per class; fold rare strata
    # into the largest existing one to avoid introducing a singleton class.
    counts: dict[str, int] = {}
    for s in strata:
        counts[s] = counts.get(s, 0) + 1
    most_common = max(counts, key=lambda k: counts[k])
    return [s if counts[s] >= min_count else most_common for s in strata]



def make_detection_split(
    train_stems: Iterable[str],
    val_stems: Iterable[str],
    test_stems: Iterable[str] = (),
) -> dict[str, str]:
    assignment: dict[str, str] = {}
    for s in sorted(train_stems):
        assignment[s] = "train"
    for s in sorted(val_stems):
        assignment[s] = "val"
    for s in sorted(test_stems):
        assignment[s] = "test"
    return assignment


def make_mot_split(
    video_names: Iterable[str],
    strata: dict[str, str],
    n_train: int = 761,
    n_val: int = 100,
    n_test: int = 100,
    seed: int = SEED,
) -> dict[str, str]:
    videos = sorted(video_names)
    if len(videos) != n_train + n_val + n_test:
        raise ValueError(
            f"Got {len(videos)} videos, need exactly {n_train + n_val + n_test}."
        )

    strat = [strata.get(v, "undefined") for v in videos]
    remainder, test = train_test_split(
        videos,
        test_size=n_test,
        stratify=_coalesce_rare_strata(strat),
        random_state=seed,
    )
    remainder_strat = [strata.get(v, "undefined") for v in remainder]
    train, val = train_test_split(
        remainder,
        train_size=n_train,
        test_size=n_val,
        stratify=_coalesce_rare_strata(remainder_strat),
        random_state=seed,
    )

    assignment: dict[str, str] = {}
    for v in sorted(train):
        assignment[v] = "train"
    for v in sorted(val):
        assignment[v] = "val"
    for v in sorted(test):
        assignment[v] = "test"
    return assignment


def write_split(assignment: dict[str, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_assignment = {k: assignment[k] for k in sorted(assignment)}
    path.write_text(json.dumps(sorted_assignment, indent=2))


def read_split(path: Path) -> dict[str, str]:
    return json.loads(path.read_text())


def split_counts(assignment: dict[str, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in assignment.values():
        counts[split] = counts.get(split, 0) + 1
    return counts


def stratum_counts(
    assignment: dict[str, str], strata: dict[str, str]
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for key, split in assignment.items():
        stratum = strata.get(key, "undefined")
        out.setdefault(split, {}).setdefault(stratum, 0)
        out[split][stratum] += 1
    return out
