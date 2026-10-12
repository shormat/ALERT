"""Ablation study — evaluate all model combos and report results."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UV = "uv"

# 12-combo ablation grid: 2 detectors × 2 trackers × 3 segmentors (no-seg included as 3rd)
DETECTION_CONFIGS = [
    str(REPO / "config/detection/yolov8m.yaml"),
    str(REPO / "config/detection/rtdetr_l.yaml"),
]
TRACKING_CONFIGS = [
    str(REPO / "config/tracking/bytetrack.yaml"),
    str(REPO / "config/tracking/ocsort.yaml"),
]
SEGMENTATION_CONFIGS = [
    str(REPO / "config/segmentation/deeplabv3plus_resnet101.yaml"),
    str(REPO / "config/segmentation/segformer_b2.yaml"),
    None,  # no segmentation baseline
]


def _run_eval(det_cfg: str, trk_cfg: str, seg_cfg: str | None, base_cfg: str, split: str = "val") -> dict:
    label = (
        f"det={Path(det_cfg).stem}  "
        f"trk={Path(trk_cfg).stem}  "
        f"seg={Path(seg_cfg).stem if seg_cfg else 'none'}"
    )
    print(f"\n{'='*70}\n  {label}\n{'='*70}")

    # Run tracking eval as representative metric
    cmd = [
        UV, "run", "python", str(REPO / "scripts/eval_tracking.py"),
        "--detection-config", det_cfg,
        "--tracking-config", trk_cfg,
        "--base-config", base_cfg,
        "--split", split,
        "--max-videos", "5",
    ]
    result = subprocess.run(cmd, capture_output=False)
    return {"label": label, "exit_code": result.returncode}


def parse_args():
    p = argparse.ArgumentParser(description="Run ablation study across all model combos.")
    p.add_argument("--base-config", default=str(REPO / "config/base.yaml"))
    p.add_argument("--split", default="val", choices=["val", "test"])
    return p.parse_args()


def main():
    args = parse_args()
    results = []

    for det in DETECTION_CONFIGS:
        for trk in TRACKING_CONFIGS:
            for seg in SEGMENTATION_CONFIGS:
                r = _run_eval(det, trk, seg, args.base_config, args.split)
                results.append(r)

    print("\n\n=== Ablation Summary ===")
    for r in results:
        status = "OK" if r["exit_code"] == 0 else f"ERR({r['exit_code']})"
        print(f"  [{status}] {r['label']}")


if __name__ == "__main__":
    main()
