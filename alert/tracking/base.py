from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrackedObject:
    track_id: int
    bbox: tuple[float, float, float, float]
    label: str
    confidence: float
    box_history: list[tuple] = field(default_factory=list)

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


class Tracker(ABC):
    @abstractmethod
    def update(
        self,
        detections=None,
        frame=None,
    ) -> list[TrackedObject]: ...

    def reset(self) -> None:
        pass
