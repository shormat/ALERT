"""Detection training entry point."""
from __future__ import annotations

import os
import socket
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]

import torch
import wandb

from alert.detection.datasets.bdd100k import BDD100KDetDataset
from alert.utils.device import resolve_device


def _ultralytics_device(config: dict):
    # Ultralytics treats device="cuda" as a single GPU. Expand to a list of indices to
    # trigger DDP across all visible devices.
    raw = resolve_device(config)
    if raw == "cuda" and torch.cuda.is_available():
        n = torch.cuda.device_count()
        return list(range(n)) if n > 1 else 0
    return raw


def train_detection(config: dict, args) -> Path:
    from ultralytics import RTDETR, YOLO

    det_cfg = config.get("detection", {})
    model_key = str(det_cfg.get("model", "")).lower()
    weights = det_cfg.get("weights") or (
        "rtdetr-l.pt" if "rtdetr" in model_key else "yolov8m.pt"
    )
    project = det_cfg.get("project", "checkpoints/detection")
    if not Path(project).is_absolute():
        project = str(_REPO / project)
    name    = det_cfg.get("name", f"{model_key}_bdd100k")

    epochs = getattr(args, "epochs", 50)
    batch  = getattr(args, "batch", 16)
    imgsz  = getattr(args, "imgsz", 640)

    wandb.init(
        project="alert-detection",
        entity=os.environ.get("WANDB_ENTITY") or None,
        name=f"{model_key}-ep{epochs}-bs{batch}",
        tags=[socket.gethostname(), model_key],
        config={"detection": det_cfg, "epochs": epochs, "batch": batch, "imgsz": imgsz},
    )

    model = RTDETR(weights) if "rtdetr" in model_key else YOLO(weights)
    results = model.train(
        data=str(BDD100KDetDataset.dataset_yaml_path()),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=_ultralytics_device(config),
        project=project,
        name=name,
        exist_ok=True,
        amp=getattr(args, "amp", True),
    )
    save_dir = getattr(results, "save_dir", None) \
        or getattr(getattr(model, "trainer", None), "save_dir", None) \
        or Path(project) / name
    best_path = Path(save_dir) / "weights" / "best.pt"
    if best_path.exists():
        artifact = wandb.Artifact(
            name=f"{model_key}_bdd100k", type="model",
            metadata={"epochs": epochs, "batch": batch, "imgsz": imgsz},
        )
        artifact.add_file(str(best_path))
        wandb.log_artifact(artifact)
    wandb.finish()
    return best_path
