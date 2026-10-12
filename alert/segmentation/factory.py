from __future__ import annotations

from alert.segmentation.base import Segmentor


def build_segmentor(config: dict) -> Segmentor:
    model_key = str(config.get("segmentation", {}).get("model", "")).lower()

    if model_key in ("deeplabv3plus", "deeplabv3+"):
        from alert.segmentation.models.deeplabv3plus import DeepLabV3PlusSegmentor
        return DeepLabV3PlusSegmentor(config)

    if model_key in ("segformer", "segformer_b2", "segformer-b2"):
        from alert.segmentation.models.segformer import SegFormerSegmentor
        return SegFormerSegmentor(config)

    raise ValueError(
        f"Unknown segmentation model: '{model_key}'. "
        "Expected one of: deeplabv3plus, segformer."
    )
