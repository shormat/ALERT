"""RT-DETR detector via Ultralytics."""
from __future__ import annotations

import numpy as np
from ultralytics import RTDETR

from alert.detection.base import BDD100K_CLASSES, COCO_TO_BDD100K, Detection, Detector


class RTDETRDetector(Detector):
    def __init__(self, cfg: dict) -> None:
        det_cfg = cfg.get("detection", cfg)
        weights = det_cfg.get("weights") or "rtdetr-l.pt"
        self._conf = float(det_cfg.get("confidence_threshold", 0.35))
        self._iou  = float(det_cfg.get("nms_iou_threshold", 0.45))
        self._device = cfg.get("device", "cpu")
        self._model = RTDETR(weights)
        names = self._model.names
        self._is_bdd = set(names.values()) <= BDD100K_CLASSES

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self._model.predict(
            frame,
            conf=self._conf,
            iou=self._iou,
            verbose=False,
            device=self._device,
        )[0]
        detections: list[Detection] = []
        for box in results.boxes:
            cls_name = results.names[int(box.cls)]
            if self._is_bdd:
                label = cls_name
            else:
                label = COCO_TO_BDD100K.get(cls_name)
            if label is None or label not in BDD100K_CLASSES:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                label=label,
                confidence=float(box.conf),
            ))
        return detections
