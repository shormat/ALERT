from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

SEG_DIRECT = 0
SEG_ALT = 1
SEG_BG = 2


@dataclass
class SegmentationResult:
    mask: np.ndarray

    @property
    def drivable_mask(self) -> np.ndarray:
        return self.mask < SEG_BG

    @property
    def direct_mask(self) -> np.ndarray:
        return self.mask == SEG_DIRECT

    @property
    def height(self) -> int:
        return self.mask.shape[0]

    @property
    def width(self) -> int:
        return self.mask.shape[1]


class Segmentor(ABC):
    @abstractmethod
    def segment(self, frame: np.ndarray) -> SegmentationResult: ...

    def warmup(self) -> None:
        self.segment(np.zeros((512, 1024, 3), dtype=np.uint8))
