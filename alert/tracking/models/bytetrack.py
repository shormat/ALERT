"""ByteTrack wrapper using vendor/ByteTrack."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from alert.tracking.base import Tracker, TrackedObject

_VENDOR = Path(__file__).resolve().parents[3] / "vendor" / "ByteTrack"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))


class ByteTrackTracker(Tracker):
    def __init__(self, config: dict) -> None:
        from yolox.tracker.byte_tracker import BYTETracker

        trk_cfg = config.get("tracking", {})
        self._bt_args = SimpleNamespace(
            track_thresh=trk_cfg.get("track_high_thresh", 0.5),
            track_buffer=trk_cfg.get("track_buffer", 30),
            match_thresh=trk_cfg.get("match_thresh", 0.8),
            mot20=False,
        )
        self._frame_rate = trk_cfg.get("frame_rate", 30)
        self._tracker = BYTETracker(self._bt_args, frame_rate=self._frame_rate)
        self._label_map: dict[int, str] = {}
        self._conf_map: dict[int, float] = {}
        self._history: dict[int, list[tuple]] = {}

    def update(self, detections=None, frame=None) -> list[TrackedObject]:
        # Empty-detection frames must still be forwarded so the Kalman filter advances;
        # returning early freezes track state and inflates ID switches when detections resume.
        if frame is None:
            return []
        detections = detections or []

        H, W = frame.shape[:2]
        if detections:
            arr = np.array([
                [d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.confidence]
                for d in detections
            ], dtype=np.float32)
        else:
            arr = np.empty((0, 5), dtype=np.float32)

        tracks = self._tracker.update(arr, [H, W], [H, W])

        # IoU-match tracks back to source detections to propagate the correct class label.
        det_boxes = np.array([d.bbox for d in detections], dtype=np.float32) \
            if detections else np.zeros((0, 4), dtype=np.float32)

        results: list[TrackedObject] = []
        for strack in tracks:
            x1, y1, x2, y2 = strack.tlbr
            tid = int(strack.track_id)
            if tid not in self._history:
                self._history[tid] = []
            self._history[tid].append((x1, y1, x2, y2))
            if len(self._history[tid]) > 30:
                self._history[tid].pop(0)

            label = self._label_map.get(tid, "car")
            if len(detections):
                tb = np.array([x1, y1, x2, y2], dtype=np.float32)
                ix1 = np.maximum(det_boxes[:, 0], tb[0])
                iy1 = np.maximum(det_boxes[:, 1], tb[1])
                ix2 = np.minimum(det_boxes[:, 2], tb[2])
                iy2 = np.minimum(det_boxes[:, 3], tb[3])
                iw = np.maximum(0.0, ix2 - ix1)
                ih = np.maximum(0.0, iy2 - iy1)
                inter = iw * ih
                area_d = (det_boxes[:, 2] - det_boxes[:, 0]) * (det_boxes[:, 3] - det_boxes[:, 1])
                area_t = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                ious = inter / np.maximum(area_d + area_t - inter, 1e-9)
                if ious.size and ious.max() > 0.3:
                    j = int(ious.argmax())
                    label = detections[j].label
                    self._label_map[tid] = label
                    self._conf_map[tid] = float(detections[j].confidence)
            conf = self._conf_map.get(tid, float(strack.score))

            # Guard against Kalman-drift producing degenerate boxes.
            x1f, x2f = float(min(x1, x2)), float(max(x1, x2))
            y1f, y2f = float(min(y1, y2)), float(max(y1, y2))
            results.append(TrackedObject(
                track_id=tid,
                bbox=(x1f, y1f, x2f, y2f),
                label=label,
                confidence=conf,
                box_history=list(self._history[tid]),
            ))
        return results

    def reset(self) -> None:
        from yolox.tracker.byte_tracker import BYTETracker
        self._tracker = BYTETracker(self._bt_args, frame_rate=self._frame_rate)
        self._label_map.clear()
        self._conf_map.clear()
        self._history.clear()
