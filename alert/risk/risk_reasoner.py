"""Per-object and scene-level risk scoring."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from alert.tracking.base import TrackedObject
    from alert.segmentation.base import SegmentationResult

_CLASS_WEIGHTS: dict[str, float] = {
    "pedestrian": 1.0,
    "rider":      0.9,
    "motorcycle": 0.85,
    "bicycle":    0.85,
    "car":        0.7,
    "truck":      0.7,
    "bus":        0.7,
    "train":      0.7,
    "traffic light": 0.0,
    "traffic sign":  0.0,
}
_DEFAULT_WEIGHT = 0.7
_TTC_LAMBDA = 3.0
_HISTORY_WINDOW = 10
_AREA_GAMMA = 0.35
_AREA_SCALE = 4.0
_TTC_FLOOR = 0.15


@dataclass
class RiskResult:
    object_risks: dict[int, float] = field(default_factory=dict)
    ttc_estimates: dict[int, float] = field(default_factory=dict)
    scene_risk: float = 0.0

    @property
    def alert_level(self) -> str:
        if self.scene_risk >= 0.6:
            return "CRITICAL"
        if self.scene_risk >= 0.3:
            return "WARNING"
        return "INFO"


class RiskReasoner:
    def __init__(self, config: dict | None = None) -> None:
        cfg = (config or {}).get("risk", {})
        self._ttc_lambda = float(cfg.get("ttc_lambda", _TTC_LAMBDA))
        self._history_window = int(cfg.get("history_window", _HISTORY_WINDOW))
        self._fps = float(cfg.get("fps", 30.0))
        self._area_history: dict[int, list[float]] = {}

    def reset(self) -> None:
        self._area_history.clear()

    def update(
        self,
        tracks: list,
        seg_result=None,
        frame_shape: tuple[int, int] = (720, 1280),
    ) -> RiskResult:
        if not tracks:
            return RiskResult()

        H, W = frame_shape[:2]
        drivable = seg_result.drivable_mask if seg_result is not None else None

        object_risks: dict[int, float] = {}
        ttc_estimates: dict[int, float] = {}

        frame_area = float(max(1, H * W))

        for t in tracks:
            tid = t.track_id
            area = t.area

            if tid not in self._area_history:
                self._area_history[tid] = []
            self._area_history[tid].append(area)
            if len(self._area_history[tid]) > self._history_window:
                self._area_history[tid].pop(0)

            ttc = self._estimate_ttc(tid)
            ttc_estimates[tid] = ttc

            wc = _CLASS_WEIGHTS.get(t.label, _DEFAULT_WEIGHT)
            q  = float(t.confidence)
            p  = self._in_path_likelihood(t, drivable, H, W)

            area_norm = float(np.clip(area / frame_area, 0.0, 1.0))
            area_factor = float(np.clip(_AREA_SCALE * (area_norm ** _AREA_GAMMA), 0.0, 1.0))

            # Floor unknown TTC so static near-objects still register.
            if math.isfinite(ttc):
                ttc_factor = math.exp(-ttc / self._ttc_lambda)
            else:
                ttc_factor = _TTC_FLOOR

            r  = wc * p * q * area_factor * ttc_factor
            object_risks[tid] = float(np.clip(r, 0.0, 1.0))

        scene = 1.0 - math.prod(1.0 - r for r in object_risks.values())

        live_ids = {t.track_id for t in tracks}
        stale = [k for k in self._area_history if k not in live_ids]
        for k in stale:
            self._area_history.pop(k)

        return RiskResult(
            object_risks=object_risks,
            ttc_estimates=ttc_estimates,
            scene_risk=float(np.clip(scene, 0.0, 1.0)),
        )

    def _estimate_ttc(self, tid: int) -> float:
        history = self._area_history.get(tid, [])
        if len(history) < 3:
            return float("inf")

        n = len(history)
        xs = np.arange(n, dtype=np.float64)
        ys = np.array(history, dtype=np.float64)

        slope = float(np.polyfit(xs, ys, 1)[0])

        if slope <= 0:
            return float("inf")

        current_area = history[-1]
        if current_area <= 0:
            return float("inf")

        rate_per_sec = slope * self._fps
        ttc = current_area / rate_per_sec
        return max(0.0, float(ttc))

    def _in_path_likelihood(
        self,
        track,
        drivable,
        H: int,
        W: int,
    ) -> float:
        if drivable is None:
            x1, y1, x2, y2 = track.bbox
            cx = (x1 + x2) / 2 / W
            cy = y2 / H
            return float(np.clip(cy * (1.0 - abs(cx - 0.5) * 2), 0.0, 1.0))

        x1, y1, x2, y2 = track.bbox
        ix1 = max(0, int(x1))
        ix2 = min(W - 1, int(x2))
        iy2 = min(H - 1, int(y2))
        if ix2 <= ix1:
            return 0.0
        # Sample a thin band just below the bbox bottom edge — drivable pixels inside
        # the bbox are occluded by the object itself.
        band_h = max(4, (iy2 - max(0, int(y1))) // 6)
        by1 = max(0, iy2 - 2)
        by2 = min(H - 1, iy2 + band_h)
        roi = drivable[by1:by2, ix1:ix2]
        if roi.size == 0:
            return 0.0
        return float(roi.sum()) / roi.size
