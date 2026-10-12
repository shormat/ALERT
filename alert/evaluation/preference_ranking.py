"""Preference-based ranking score for pipeline configurations."""
from __future__ import annotations

from typing import Iterable

_METRIC_KEYS = ("detection_score", "tracking_score", "segmentation_score")


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    if not weights:
        raise ValueError("weights dict is empty")
    for k, v in weights.items():
        if v < 0:
            raise ValueError(f"weight for '{k}' is negative ({v})")
    total = float(sum(weights.values()))
    if total == 0.0:
        raise ValueError("all weights are zero; cannot normalize")
    return {k: float(v) / total for k, v in weights.items()}


def _validate_metric_keys(metrics: dict, weights: dict) -> None:
    missing = [k for k in weights if k not in metrics]
    if missing:
        raise KeyError(f"metric(s) missing for weighted keys: {missing}")
    for k in weights:
        v = metrics[k]
        if not isinstance(v, (int, float)):
            raise TypeError(f"metric '{k}' must be numeric, got {type(v).__name__}")
        if v < 0.0 or v > 1.0:
            pass


def compute_continuous_ranking_score(
    metrics: dict[str, float],
    weights: dict[str, float],
) -> float:
    norm = normalize_weights(weights)
    _validate_metric_keys(metrics, norm)
    return float(sum(norm[k] * float(metrics[k]) for k in norm))


def compute_threshold_satisfaction_score(
    metrics: dict[str, float],
    weights: dict[str, float],
    thresholds: dict[str, float],
) -> float:
    norm = normalize_weights(weights)
    _validate_metric_keys(metrics, norm)
    missing_t = [k for k in norm if k not in thresholds]
    if missing_t:
        raise KeyError(f"threshold(s) missing for weighted keys: {missing_t}")

    score = 0.0
    for k, w in norm.items():
        s_k = 1.0 if float(metrics[k]) >= float(thresholds[k]) else 0.0
        score += w * s_k
    return float(score)


def rank_pipelines(
    results: Iterable[dict],
    weights: dict[str, float],
    thresholds: dict[str, float] | None = None,
    mode: str = "continuous",
) -> list[dict]:
    if mode not in ("continuous", "threshold"):
        raise ValueError(f"unknown mode '{mode}', expected 'continuous' or 'threshold'")
    if mode == "threshold" and thresholds is None:
        raise ValueError("threshold mode requires a thresholds dict")

    ranked: list[dict] = []
    for r in results:
        if "name" not in r:
            raise KeyError("each pipeline result must have a 'name' field")
        out = dict(r)
        if mode == "continuous":
            out["final_score"] = compute_continuous_ranking_score(r, weights)
        else:
            out["final_score"] = compute_threshold_satisfaction_score(r, weights, thresholds)
        ranked.append(out)
    ranked.sort(key=lambda x: x["final_score"], reverse=True)
    return ranked
