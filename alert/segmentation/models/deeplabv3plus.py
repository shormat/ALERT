"""DeepLabV3+ segmentor via segmentation-models-pytorch."""
from __future__ import annotations

import cv2
import numpy as np
import torch
from pathlib import Path

from alert.segmentation.base import SegmentationResult, Segmentor
from alert.utils.device import resolve_device

_REPO = Path(__file__).resolve().parents[3]


class DeepLabV3PlusSegmentor(Segmentor):
    def __init__(self, config: dict) -> None:
        import segmentation_models_pytorch as smp

        seg_cfg = config.get("segmentation", {})
        encoder = seg_cfg.get("encoder", "resnet101")
        num_classes = seg_cfg.get("num_classes", 3)
        img_size = seg_cfg.get("img_size", [512, 1024])
        self._h, self._w = (img_size[0], img_size[1]) if isinstance(img_size, list) else (img_size, img_size)
        weights_path = seg_cfg.get("weights")
        if weights_path and not Path(weights_path).is_absolute():
            weights_path = str(_REPO / weights_path)

        self.device = torch.device(resolve_device(config))
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=None if weights_path else "imagenet",
            in_channels=3,
            classes=num_classes,
        )

        if weights_path:
            state = torch.load(weights_path, map_location=self.device, weights_only=True)
            if "state_dict" in state:
                state = state["state_dict"]
            self.model.load_state_dict(state)

        self.model.to(self.device).eval()

    def segment(self, frame: np.ndarray) -> SegmentationResult:
        orig_h, orig_w = frame.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(frame, (self._w, self._h)), cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        if pred.shape != (orig_h, orig_w):
            pred = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        return SegmentationResult(mask=pred)
