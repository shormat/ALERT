from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from alert.detection.base import Detection
    from alert.tracking.base import TrackedObject
    from alert.segmentation.base import SegmentationResult
    from alert.risk.risk_reasoner import RiskResult

_PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10),  (146, 204, 23), (61, 219, 134),
    (26, 147, 52),  (0, 212, 187),  (44, 153, 168), (0, 194, 255),
    (52, 69, 147),  (100, 115, 255),(0, 24, 236),   (132, 56, 255),
    (82, 0, 133),   (203, 56, 255), (255, 149, 200),(255, 55, 199),
]

_SEG_DIRECT_BGR = (255, 0, 0)
_SEG_ALT_BGR    = (0, 255, 255)
_SEG_ALPHA      = 0.35

_RISK_WARN_BGR  = (0, 165, 255)
_RISK_CRIT_BGR  = (0, 0, 255)


def _track_color(track_id: int) -> tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]


class Visualizer:
    def __init__(self, config: dict | None = None) -> None:
        pass

    def compose_frame(
        self,
        result_or_frame,
        detections: list["Detection"] | None = None,
        tracked_objects: list["TrackedObject"] | None = None,
        seg_result: "SegmentationResult | None" = None,
        risk_result: "RiskResult | None" = None,
        alert_text: str | None = None,
        fps: float = 0.0,
    ) -> np.ndarray:
        if hasattr(result_or_frame, "frame"):
            r = result_or_frame
            frame = r.frame
            detections = r.detections
            tracked_objects = r.tracks
            seg_result = r.seg_result
            risk_result = r.risk_result
            alert_text = r.alert_message
            fps = r.fps
        else:
            frame = result_or_frame
        out = frame.copy()

        if seg_result is not None:
            overlay = out.copy()
            mask = seg_result.mask
            overlay[mask == 0] = _SEG_DIRECT_BGR
            overlay[mask == 1] = _SEG_ALT_BGR
            cv2.addWeighted(overlay, _SEG_ALPHA, out, 1 - _SEG_ALPHA, 0, out)

        if tracked_objects:
            for t in tracked_objects:
                x1, y1, x2, y2 = (int(c) for c in t.bbox)
                color = _track_color(t.track_id)
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{t.track_id} {t.label}"
                cv2.putText(out, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        elif detections:
            for d in detections:
                x1, y1, x2, y2 = (int(c) for c in d.bbox)
                cv2.rectangle(out, (x1, y1), (x2, y2), (200, 200, 200), 1)
                cv2.putText(out, d.label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        if risk_result and tracked_objects:
            for t in tracked_objects:
                ri = risk_result.object_risks.get(t.track_id, 0.0)
                if ri >= 0.6:
                    x1, y1, x2, y2 = (int(c) for c in t.bbox)
                    cv2.rectangle(out, (x1, y1), (x2, y2), _RISK_CRIT_BGR, 3)
                    cx = (x1 + x2) // 2
                    cv2.putText(out, "(!)", (cx - 12, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _RISK_CRIT_BGR, 2)
                elif ri >= 0.3:
                    x1, y1, x2, y2 = (int(c) for c in t.bbox)
                    cv2.rectangle(out, (x1, y1), (x2, y2), _RISK_WARN_BGR, 2)

            if risk_result:
                level = risk_result.alert_level
                badge_color = _RISK_CRIT_BGR if level == "CRITICAL" else (_RISK_WARN_BGR if level == "WARNING" else (0, 180, 0))
                cv2.rectangle(out, (8, 8), (180, 36), badge_color, -1)
                cv2.putText(out, f"RISK {level} {risk_result.scene_risk:.2f}", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        if alert_text:
            cv2.putText(out, alert_text, (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        fps_text = f"{fps:.1f} FPS"
        (tw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(out, fps_text, (out.shape[1] - tw - 8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 1)

        return out
