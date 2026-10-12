from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


class VideoReader:
    def __init__(self, source: str | Path | int) -> None:
        if isinstance(source, str) and source == "webcam":
            source = 0
        self._cap = cv2.VideoCapture(str(source) if not isinstance(source, int) else source)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video source: {source}")

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def frame_count(self) -> int:
        # Unreliable for H.264 .mov — treat as a hint, not loop control.
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            yield frame

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *_) -> None:
        self.release()

    def release(self) -> None:
        self._cap.release()


class VideoWriter:
    def __init__(
        self,
        path: str | Path,
        fps: float,
        width: int,
        height: int,
        codec: str = "mp4v",
    ) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if not self._writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            self._writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *_) -> None:
        self.release()
