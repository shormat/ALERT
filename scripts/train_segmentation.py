"""CLI wrapper for segmentation training."""
from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description="Train a drivable-area segmentation model on BDD100K.")
    p.add_argument("--segmentation-config", required=True, help="Path to segmentation YAML config")
    p.add_argument("--base-config", default=str(REPO / "config/base.yaml"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                   help="Override segmentation.amp from the YAML. Use --no-amp on Pascal/older GPUs.")
    p.add_argument("--workers", type=int, default=4, help="DataLoader worker processes per GPU.")
    return p.parse_args()


def main():
    args = parse_args()
    from alert.utils.config import load_config
    from alert.segmentation.train import train_segmentation

    cfg = load_config(args.base_config, args.segmentation_config)
    best = train_segmentation(cfg, args)
    print(f"Best checkpoint: {best}")


if __name__ == "__main__":
    main()
