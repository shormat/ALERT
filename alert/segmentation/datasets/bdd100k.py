"""BDD100K drivable-area segmentation dataset."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

from alert.utils.paths import data_processed_root


class BDD100KSegDataset(Dataset):
    def __init__(self, split: str = "train") -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")
        self.split = split
        _data = data_processed_root() / "bdd100k_frames"
        img_dir = _data / "images" / split
        msk_dir = _data / "drivable" / "masks" / split

        if not img_dir.exists():
            raise FileNotFoundError(f"Image dir not found: {img_dir}")
        if not msk_dir.exists():
            raise FileNotFoundError(f"Mask dir not found: {msk_dir}")

        mask_stems = {p.stem: p for p in sorted(msk_dir.glob("*.png"))}
        self._pairs: list[tuple[Path, Path]] = []
        for img_path in sorted(img_dir.glob("*.jpg")):
            if img_path.stem in mask_stems:
                self._pairs.append((img_path, mask_stems[img_path.stem]))

        if not self._pairs:
            raise RuntimeError(f"No matched image/mask pairs found in {img_dir}")

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        img_path, msk_path = self._pairs[idx]
        frame = cv2.imread(str(img_path))
        mask = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)
        return frame, mask
