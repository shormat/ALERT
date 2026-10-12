from __future__ import annotations

from alert.detection.base import Detector


def build_detector(cfg: dict) -> Detector:
    det_cfg = cfg.get("detection", {})
    model_key = str(det_cfg.get("model", "")).lower()

    if model_key in ("rtdetr-l", "rtdetr-x", "rtdetr"):
        from alert.detection.models.rtdetr import RTDETRDetector
        return RTDETRDetector(cfg)

    if model_key in ("yolov8m", "yolov8n", "yolov8s", "yolov8l", "yolov8x", "yolov8"):
        from alert.detection.models.yolov8 import YOLOv8Detector
        return YOLOv8Detector(cfg)

    raise ValueError(
        f"Unknown detection model: '{model_key}'. "
        "Expected one of: rtdetr-l, rtdetr-x, yolov8m, yolov8n, yolov8s, yolov8l, yolov8x"
    )
