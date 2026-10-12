"""CLI wrapper for segmentation evaluation — all metrics on BDD100K drivable masks."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["direct", "alt", "background"]
NUM_CLASSES = 3


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate segmentation on BDD100K val/test split.")
    p.add_argument("--segmentation-config", required=True)
    p.add_argument("--base-config", default=str(REPO / "config/base.yaml"))
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--device", default="0",
                   help="Physical GPU index, e.g. '0', '1'. Sets CUDA_VISIBLE_DEVICES before "
                        "torch initialises so the right GPU is used.")
    return p.parse_args()


def _per_sample_stats(pred: np.ndarray, gt: np.ndarray):
    """
    Returns arrays of shape (NUM_CLASSES,) for:
        intersection, union, pred_count, gt_count, correct_pixels, total_pixels
    """
    intersection = np.array(
        [((pred == c) & (gt == c)).sum() for c in range(NUM_CLASSES)], dtype=np.float64
    )
    union = np.array(
        [((pred == c) | (gt == c)).sum() for c in range(NUM_CLASSES)], dtype=np.float64
    )
    pred_count = np.array([(pred == c).sum() for c in range(NUM_CLASSES)], dtype=np.float64)
    gt_count   = np.array([(gt == c).sum()   for c in range(NUM_CLASSES)], dtype=np.float64)
    correct_pixels = (pred == gt).sum()
    total_pixels   = gt.size
    return intersection, union, pred_count, gt_count, correct_pixels, total_pixels


def main():
    args = parse_args()
    # Must be set before torch initialises CUDA — maps cuda:0 to the requested physical GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    import cv2
    from alert.utils.config import load_config
    from alert.segmentation.factory import build_segmentor
    from alert.segmentation.datasets.bdd100k import BDD100KSegDataset

    cfg = load_config(args.base_config, args.segmentation_config)
    segmentor = build_segmentor(cfg)
    ds = BDD100KSegDataset(split=args.split)

    n = len(ds) if args.max_samples is None else min(args.max_samples, len(ds))

    # Accumulators
    sum_intersection = np.zeros(NUM_CLASSES, dtype=np.float64)
    sum_union        = np.zeros(NUM_CLASSES, dtype=np.float64)
    sum_pred_count   = np.zeros(NUM_CLASSES, dtype=np.float64)
    sum_gt_count     = np.zeros(NUM_CLASSES, dtype=np.float64)
    sum_correct = 0
    sum_pixels  = 0

    for i in range(n):
        frame, gt_mask = ds[i]
        result = segmentor.segment(frame)
        pred = result.mask
        if pred.shape != gt_mask.shape:
            pred = cv2.resize(
                pred, (gt_mask.shape[1], gt_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        inter, union, p_cnt, g_cnt, correct, total = _per_sample_stats(pred, gt_mask)
        sum_intersection += inter
        sum_union        += union
        sum_pred_count   += p_cnt
        sum_gt_count     += g_cnt
        sum_correct      += correct
        sum_pixels       += total
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n}]")

    # --- IoU per class ---
    iou_per_class = np.where(sum_union > 0, sum_intersection / sum_union, np.nan)
    miou = float(np.nanmean(iou_per_class))

    # --- Pixel accuracy (overall) ---
    pixel_acc = sum_correct / max(sum_pixels, 1)

    # --- Mean pixel accuracy (per-class PA, then average) ---
    per_class_pa = np.where(sum_gt_count > 0, sum_intersection / sum_gt_count, np.nan)
    mean_pixel_acc = float(np.nanmean(per_class_pa))

    # --- Frequency-weighted IoU ---
    class_freq = sum_gt_count / max(sum_gt_count.sum(), 1)
    fw_iou = float(np.nansum(class_freq * np.where(np.isnan(iou_per_class), 0, iou_per_class)))

    print(f"\n=== Segmentation Evaluation ({args.split}, n={n}) ===")
    print(f"\n  --- Per-class IoU ---")
    for name, iou in zip(CLASS_NAMES, iou_per_class):
        pa = per_class_pa[CLASS_NAMES.index(name)]
        freq = class_freq[CLASS_NAMES.index(name)]
        print(f"    {name:<12s}: IoU={iou:.4f}  PA={pa:.4f}  freq={freq:.3f}")

    print(f"\n  --- Overall ---")
    print(f"  mIoU              : {miou:.4f}   (primary: BDD100K standard)")
    print(f"  Pixel Accuracy    : {pixel_acc:.4f}   (secondary: biased toward background)")
    print(f"  Mean Pixel Acc    : {mean_pixel_acc:.4f}   (secondary: per-class PA averaged)")
    print(f"  Freq-weighted IoU : {fw_iou:.4f}   (secondary: FW-IoU)")

    from alert.utils.paths import output_root
    model_tag = Path(args.segmentation_config).stem
    out_path = output_root() / "eval" / f"segmentation_{model_tag}_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "stage": "segmentation",
            "model": model_tag,
            "split": args.split,
            "segmentation_config": args.segmentation_config,
            "num_samples": n,
            "device": args.device,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        "metrics": {
            "mIoU": miou,
            "pixel_accuracy": float(pixel_acc),
            "mean_pixel_accuracy": float(mean_pixel_acc),
            "fw_iou": float(fw_iou),
            "per_class": {
                name: {
                    "iou": float(iou_per_class[i]),
                    "pixel_accuracy": float(per_class_pa[i]),
                    "freq": float(class_freq[i]),
                }
                for i, name in enumerate(CLASS_NAMES)
            },
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
