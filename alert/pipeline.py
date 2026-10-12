from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from alert.detection.base import Detection
    from alert.tracking.base import TrackedObject
    from alert.segmentation.base import SegmentationResult
    from alert.risk.risk_reasoner import RiskResult


@dataclass
class PerceptionResult:
    frame: np.ndarray
    detections: list = field(default_factory=list)
    tracks: list = field(default_factory=list)
    _seg_result: object = field(default=None, repr=False)
    risk_result: object = field(default=None)
    alert_message: str | None = None
    fps: float = 0.0

    @property
    def seg_result(self):
        return self._seg_result

    @property
    def drivable_mask(self):
        if self._seg_result is None:
            return None
        return self._seg_result.drivable_mask


class ALERTPipeline:
    def __init__(self, config: dict) -> None:
        self._cfg = config
        pipe_cfg = config.get("pipeline", {})

        self._enable_seg    = bool(pipe_cfg.get("enable_segmentation", True))
        self._enable_risk   = bool(pipe_cfg.get("enable_risk", True))
        self._enable_alerts = bool(pipe_cfg.get("enable_alerts", True))

        from alert.detection.factory import build_detector
        self._detector = build_detector(config)

        from alert.tracking.factory import build_tracker
        self._tracker = build_tracker(config)

        self._segmentor = None
        if self._enable_seg:
            from alert.segmentation.factory import build_segmentor
            self._segmentor = build_segmentor(config)

        self._risk_reasoner = None
        if self._enable_risk:
            from alert.risk.risk_reasoner import RiskReasoner
            self._risk_reasoner = RiskReasoner(config)

        self._alerts = None
        if self._enable_alerts:
            from alert.alerts.voice_alerts import VoiceAlertSystem
            self._alerts = VoiceAlertSystem(config)

        from alert.utils.timer import FPSCounter
        self._fps_counter = FPSCounter()

    def process_frame(self, frame: np.ndarray) -> PerceptionResult:
        H, W = frame.shape[:2]

        detections = self._detector.detect(frame)

        tracks = self._tracker.update(detections=detections, frame=frame)

        seg_result = None
        if self._segmentor is not None:
            seg_result = self._segmentor.segment(frame)

        risk_result = None
        if self._risk_reasoner is not None:
            risk_result = self._risk_reasoner.update(
                tracks=tracks,
                seg_result=seg_result,
                frame_shape=(H, W),
            )

        alert_msg = None
        if self._alerts is not None:
            alert_msg = self._alerts.update(
                tracks=tracks,
                seg_result=seg_result,
                frame_shape=(H, W),
                risk_result=risk_result,
            )

        fps = self._fps_counter.tick()

        return PerceptionResult(
            frame=frame,
            detections=detections,
            tracks=tracks,
            _seg_result=seg_result,
            risk_result=risk_result,
            alert_message=alert_msg,
            fps=fps,
        )

    def reset(self) -> None:
        self._tracker.reset()
        if self._risk_reasoner is not None:
            self._risk_reasoner.reset()

    def warmup(self) -> None:
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._detector.warmup()
        if self._segmentor is not None:
            self._segmentor.warmup()
