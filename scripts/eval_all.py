"""Run all evaluation scripts sequentially."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UV = "uv"


def _run(script: str, extra_args: list[str] = []) -> None:
    cmd = [UV, "run", "python", str(REPO / "scripts" / script)] + extra_args
    print(f"\n{'='*60}\nRunning: {' '.join(cmd)}\n{'='*60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[WARN] {script} exited with code {result.returncode}")


def parse_args():
    p = argparse.ArgumentParser(description="Run all evaluations.")
    p.add_argument("--detection-config",    default=str(REPO / "config/detection/yolov8m.yaml"))
    p.add_argument("--segmentation-config", default=str(REPO / "config/segmentation/deeplabv3plus_resnet101.yaml"))
    p.add_argument("--tracking-config",     default=str(REPO / "config/tracking/bytetrack.yaml"))
    p.add_argument("--base-config",         default=str(REPO / "config/base.yaml"))
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--skip-detection",  action="store_true")
    p.add_argument("--skip-segmentation", action="store_true")
    p.add_argument("--skip-tracking",  action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.skip_detection:
        _run("eval_detection.py", ["--detection-config", args.detection_config, "--base-config", args.base_config, "--split", args.split])

    if not args.skip_segmentation:
        _run("eval_segmentation.py", ["--segmentation-config", args.segmentation_config, "--base-config", args.base_config, "--split", args.split])

    if not args.skip_tracking:
        _run("eval_tracking.py", [
            "--detection-config", args.detection_config,
            "--tracking-config", args.tracking_config,
            "--base-config", args.base_config,
            "--split", args.split,
        ])

    print("\nAll evaluations complete.")


if __name__ == "__main__":
    main()
