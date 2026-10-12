"""Predefined preference profiles for cross-task ranking."""
from __future__ import annotations

from alert.evaluation.preference_ranking import rank_pipelines

_PROFILES: dict[str, dict[str, float]] = {
    "Balanced": {
        "detection_score":    0.33,
        "tracking_score":     0.33,
        "segmentation_score": 0.34,
    },
    "Detection-first": {
        "detection_score":    0.60,
        "tracking_score":     0.20,
        "segmentation_score": 0.20,
    },
    "Tracking-first": {
        "detection_score":    0.20,
        "tracking_score":     0.60,
        "segmentation_score": 0.20,
    },
    "Segmentation-first": {
        "detection_score":    0.20,
        "tracking_score":     0.20,
        "segmentation_score": 0.60,
    },
}


def get_preference_profiles() -> dict[str, dict[str, float]]:
    return {name: dict(weights) for name, weights in _PROFILES.items()}


def rank_pipelines_across_profiles(
    results: list[dict],
    profiles: dict[str, dict[str, float]] | None = None,
    thresholds: dict[str, float] | None = None,
    mode: str = "continuous",
) -> dict[str, list[dict]]:
    if profiles is None:
        profiles = get_preference_profiles()
    if not profiles:
        raise ValueError("profiles dict is empty")

    return {
        name: rank_pipelines(results, weights=weights, thresholds=thresholds, mode=mode)
        for name, weights in profiles.items()
    }
