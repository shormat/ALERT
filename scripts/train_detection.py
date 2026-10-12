"""CLI wrapper for detection training."""
from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune RT-DETR or YOLOv8 on BDD100K.")
    p.add_argument("--detection-config", required=True, help="Path to detection YAML config")
    p.add_argument("--base-config", default=str(REPO / "config/base.yaml"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch",  type=int, default=16)
    p.add_argument("--imgsz",  type=int, default=640)
    p.add_argument("--amp",    action=argparse.BooleanOptionalAction, default=True,
                   help="Enable mixed precision. Use --no-amp on Pascal/older GPUs.")
    return p.parse_args()


def main():
    args = parse_args()
    from alert.utils.config import load_config
    from alert.detection.train import train_detection

    cfg = load_config(args.base_config, args.detection_config)
    best = train_detection(cfg, args)
    print(f"Best checkpoint: {best}")


if __name__ == "__main__":
    main()
