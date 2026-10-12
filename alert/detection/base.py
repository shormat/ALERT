from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

BDD100K_CLASSES = {
    "pedestrian", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle", "traffic light", "traffic sign",
}

# "rider" and "traffic sign" have no COCO equivalent
COCO_TO_BDD100K: dict[str, str] = {
    "person":        "pedestrian",
    "bicycle":       "bicycle",
    "car":           "car",
    "motorcycle":    "motorcycle",
    "bus":           "bus",
    "truck":         "truck",
    "train":         "train",
    "traffic light": "traffic light",
}


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]
    label: str
    confidence: float

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])


class Detector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]: ...

    def warmup(self) -> None:
        self.detect(np.zeros((640, 640, 3), dtype=np.uint8))
