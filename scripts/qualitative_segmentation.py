"""Qualitative segmentation visualisation — sample grid + worst-IoU failure grid.

Each row shows one frame as three panels:
  [raw frame] [GT mask overlay] [predicted mask overlay]

Mask colour coding (semi-transparent overlay on frame):
  blue  (BGR 255,0,0) — direct drivable  (class 0)
  cyan  (BGR 255,255,0) — alt drivable   (class 1)
  no overlay            — background     (class 2)

Outputs:
  outputs/qualitative/segmentation/{model}_{split}_samples.png   (representative)
  outputs/qualitative/segmentation/{model}_{split}_failures.png  (worst mIoU)
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]

_SEG_DIRECT_BGR = (255,   0,   0)
_SEG_ALT_BGR    = (255, 255,   0)
_SEG_ALPHA      = 0.45
_CLASS_NAMES    = ["direct", "alt", "background"]


def parse_args():
    p = argparse.ArgumentParser(description="Qualitative segmentation output grids.")
    p.add_argument("--segmentation-config", required=True)
    p.add_argument("--base-config", default=str(REPO / "config/base.yaml"))
    p.add_argument("--split",      default="val", choices=["val", "test"])
    p.add_argument("--n-pool",     type=int, default=200,
                   help="Frames to run inference on")
    p.add_argument("--n-samples",  type=int, default=6,
                   help="Rows in the representative grid")
    p.add_argument("--n-failures", type=int, default=6,
                   help="Rows in the failure grid")
    p.add_argument("--cell-w",     type=int, default=480, help="Panel width px")
    p.add_argument("--cell-h",     type=int, default=270, help="Panel height px")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--device",     default="0")
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────

def _mask_overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return frame with semi-transparent drivable-area colour overlay."""
    overlay = frame.copy()
    overlay[mask == 0] = _SEG_DIRECT_BGR
    overlay[mask == 1] = _SEG_ALT_BGR
    return cv2.addWeighted(overlay, _SEG_ALPHA, frame, 1 - _SEG_ALPHA, 0)


def _per_sample_miou(pred: np.ndarray, gt: np.ndarray, n_classes: int = 3) -> float:
    ious = []
    for c in range(n_classes):
        inter = ((pred == c) & (gt == c)).sum()
        union = ((pred == c) | (gt == c)).sum()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def _thumb(img: np.ndarray, w: int, h: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_NEAREST
                         if img.ndim == 2 else cv2.INTER_LINEAR)
    out = np.zeros((h, w, 3) if img.ndim == 3 else (h, w), dtype=img.dtype)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = resized
    return out


def _make_triplet(frame, gt_mask, pred_mask, cell_w, cell_h) -> np.ndarray:
    raw   = _thumb(frame,    cell_w, cell_h)
    gt_ov = _mask_overlay(_thumb(frame, cell_w, cell_h), _thumb(gt_mask,   cell_w, cell_h))
    pr_ov = _mask_overlay(_thumb(frame, cell_w, cell_h), _thumb(pred_mask, cell_w, cell_h))
    return np.hstack([raw, gt_ov, pr_ov])


def _header_bar(total_w: int, titles: list[str], cell_w: int) -> np.ndarray:
    bar = np.zeros((22, total_w, 3), dtype=np.uint8)
    for i, title in enumerate(titles):
        x = i * cell_w + 4
        cv2.putText(bar, title, (x, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return bar


def _legend_bar(total_w: int) -> np.ndarray:
    bar = np.zeros((22, total_w, 3), dtype=np.uint8)
    items = [(_SEG_DIRECT_BGR, "direct"), (_SEG_ALT_BGR, "alt"), ((80, 80, 80), "background")]
    for i, (col, txt) in enumerate(items):
        x = 10 + i * 140
        cv2.rectangle(bar, (x, 4), (x + 14, 18), col, -1)
        cv2.putText(bar, txt, (x + 18, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
    return bar


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    from alert.utils.config import load_config
    from alert.segmentation.factory import build_segmentor
    from alert.segmentation.datasets.bdd100k import BDD100KSegDataset
    from alert.utils.paths import output_root

    cfg = load_config(args.base_config, args.segmentation_config)
    segmentor = build_segmentor(cfg)

    ds = BDD100KSegDataset(split=args.split)
    n_pool = min(args.n_pool, len(ds))
    rng = random.Random(args.seed)
    indices = sorted(rng.sample(range(len(ds)), n_pool))

    print(f"Running segmentor on {n_pool} frames ({args.split})...")
    scored: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []

    for i, idx in enumerate(indices):
        frame, gt_mask = ds[idx]
        result = segmentor.segment(frame)
        pred = result.mask
        if pred.shape != gt_mask.shape:
            pred = cv2.resize(pred, (gt_mask.shape[1], gt_mask.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        miou = _per_sample_miou(pred, gt_mask)
        scored.append((miou, frame, gt_mask, pred))
        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{n_pool}]")

    scored.sort(key=lambda x: x[0])  # ascending: worst first

    n_s = min(args.n_samples,  len(scored))
    n_f = min(args.n_failures, len(scored))
    # representative = highest mIoU (best), sorted ascending so tail = best
    sample_items  = scored[-n_s:]
    failure_items = scored[:n_f]

    model_tag = Path(args.segmentation_config).stem
    out_dir = output_root() / "qualitative" / "segmentation"
    out_dir.mkdir(parents=True, exist_ok=True)

    hdr   = _header_bar(args.cell_w * 3, ["Raw frame", "GT mask", "Pred mask"], args.cell_w)
    leg   = _legend_bar(args.cell_w * 3)
    prefix = np.vstack([hdr, leg])

    for tag, items in [("samples", sample_items), ("failures", failure_items)]:
        rows = [_make_triplet(frame, gt, pred, args.cell_w, args.cell_h)
                for _, frame, gt, pred in items]
        grid = np.vstack([prefix] + rows)
        out_path = out_dir / f"{model_tag}_{args.split}_{tag}.png"
        cv2.imwrite(str(out_path), grid)
        print(f"  saved → {out_path.name}")

    print(f"\nDone. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
