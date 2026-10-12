"""BDD100K detection dataset reading processed YOLO labels."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

from alert.detection.base import BDD100K_CLASSES, Detection
from alert.utils.paths import data_processed_root

_CLASS_NAMES = [
    "pedestrian", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle", "traffic light", "traffic sign",
]


class BDD100KDetDataset(Dataset):
    def __init__(self, split: str = "val") -> None:
        root = data_processed_root() / "bdd100k_frames"
        self._img_dir = root / "images" / split
        self._lbl_dir = root / "labels" / split
        self._images = sorted(self._img_dir.glob("*.jpg"))
        if not self._images:
            raise FileNotFoundError(f"No images found in {self._img_dir}")

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, list[Detection]]:
        img_path = self._images[idx]
        frame = cv2.imread(str(img_path))
        H, W = frame.shape[:2]

        lbl_path = self._lbl_dir / (img_path.stem + ".txt")
        detections: list[Detection] = []
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
                x1 = (cx - bw / 2) * W
                y1 = (cy - bh / 2) * H
                x2 = (cx + bw / 2) * W
                y2 = (cy + bh / 2) * H
                label = _CLASS_NAMES[cls_id] if cls_id < len(_CLASS_NAMES) else "unknown"
                detections.append(Detection(bbox=(x1, y1, x2, y2), label=label, confidence=1.0))
        return frame, detections

    @staticmethod
    def dataset_yaml_path() -> Path:
        return data_processed_root() / "bdd100k_frames" / "bdd100k.yaml"
