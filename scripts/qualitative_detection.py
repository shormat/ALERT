"""Qualitative detection visualisation — sample grid + worst-failure grid.

Legend (box colours in each annotated frame):
  green  — GT box that was matched       (TP from GT side)
  orange — GT box with no prediction     (FN)
  blue   — prediction that matched GT    (TP from pred side)
  red    — prediction with no GT match   (FP)

Outputs:
  outputs/qualitative/detection/{model}_{split}_samples.png   (representative frames)
  outputs/qualitative/detection/{model}_{split}_failures.png  (most FP+FN)
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]

# BGR colours
_GT_TP   = (30,  200,  30)   # green  — matched GT
_GT_FN   = (0,   100, 255)   # orange — missed GT
_PRED_TP = (200,  80,  30)   # blue   — matched prediction
_PRED_FP = (30,   30, 220)   # red    — false-positive prediction


def parse_args():
    p = argparse.ArgumentParser(description="Qualitative detection output grids.")
    p.add_argument("--detection-config", required=True)
    p.add_argument("--base-config", default=str(REPO / "config/base.yaml"))
    p.add_argument("--split",     default="val", choices=["val", "test"])
    p.add_argument("--n-pool",    type=int, default=200,
                   help="Frames to run inference on (candidates for both grids)")
    p.add_argument("--n-samples", type=int, default=8,
                   help="Frames in the representative grid (lowest FP+FN)")
    p.add_argument("--n-failures",type=int, default=8,
                   help="Frames in the failure grid (highest FP+FN)")
    p.add_argument("--n-cols",    type=int, default=4,  help="Grid columns")
    p.add_argument("--cell-w",    type=int, default=640, help="Grid cell width px")
    p.add_argument("--cell-h",    type=int, default=360, help="Grid cell height px")
    p.add_argument("--iou-thresh",type=float, default=0.5)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--device",    default="0")
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────

def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def _match(gt_dets, pred_dets, iou_thresh: float):
    """Greedy IoU matching. Returns (matched_gt_set, matched_pred_set)."""
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    for i, g in enumerate(gt_dets):
        best_j, best_v = -1, iou_thresh
        for j, p in enumerate(pred_dets):
            if j in matched_pred:
                continue
            v = _iou(g.bbox, p.bbox)
            if v > best_v:
                best_v, best_j = v, j
        if best_j >= 0:
            matched_gt.add(i)
            matched_pred.add(best_j)
    return matched_gt, matched_pred


def _annotate(frame, gt_dets, pred_dets, iou_thresh: float) -> np.ndarray:
    matched_gt, matched_pred = _match(gt_dets, pred_dets, iou_thresh)
    ann = frame.copy()

    for i, d in enumerate(gt_dets):
        x1, y1, x2, y2 = (int(c) for c in d.bbox)
        col = _GT_TP if i in matched_gt else _GT_FN
        cv2.rectangle(ann, (x1, y1), (x2, y2), col, 1)
        cv2.putText(ann, d.label, (x1, max(y1 - 3, 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)

    for j, d in enumerate(pred_dets):
        x1, y1, x2, y2 = (int(c) for c in d.bbox)
        col = _PRED_TP if j in matched_pred else _PRED_FP
        cv2.rectangle(ann, (x1, y1), (x2, y2), col, 2)
        cv2.putText(ann, f"{d.label} {d.confidence:.2f}", (x1, min(y2 + 13, ann.shape[0] - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)

    return ann


def _thumb(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Letterbox-resize to w×h."""
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img, (nw, nh))
    out = np.zeros((h, w, 3), dtype=np.uint8)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = resized
    return out


def _make_grid(frames, n_cols, cell_w, cell_h) -> np.ndarray:
    cells = [_thumb(f, cell_w, cell_h) for f in frames]
    while len(cells) % n_cols:
        cells.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))
    rows = [np.hstack(cells[r * n_cols:(r + 1) * n_cols])
            for r in range(len(cells) // n_cols)]
    return np.vstack(rows)


def _legend(cell_w: int) -> np.ndarray:
    bar = np.zeros((24, cell_w, 3), dtype=np.uint8)
    items = [(_GT_TP, "GT-TP"), (_GT_FN, "GT-FN"), (_PRED_TP, "Pred-TP"), (_PRED_FP, "Pred-FP")]
    for i, (col, txt) in enumerate(items):
        x = 10 + i * (cell_w // 4)
        cv2.rectangle(bar, (x, 5), (x + 14, 19), col, -1)
        cv2.putText(bar, txt, (x + 18, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
    return bar


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    from alert.utils.config import load_config
    from alert.detection.factory import build_detector
    from alert.detection.datasets.bdd100k import BDD100KDetDataset
    from alert.utils.paths import output_root

    cfg = load_config(args.base_config, args.detection_config)
    cfg["device"] = args.device
    detector = build_detector(cfg)

    ds = BDD100KDetDataset(split=args.split)
    n_pool = min(args.n_pool, len(ds))
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(ds)), n_pool)

    print(f"Running detector on {n_pool} frames ({args.split})...")
    scored: list[tuple[int, np.ndarray]] = []

    for i, idx in enumerate(indices):
        frame, gt_dets = ds[idx]
        pred_dets = detector.detect(frame)
        matched_gt, matched_pred = _match(gt_dets, pred_dets, args.iou_thresh)
        fp = len(pred_dets) - len(matched_pred)
        fn = len(gt_dets)   - len(matched_gt)
        ann = _annotate(frame, gt_dets, pred_dets, args.iou_thresh)
        scored.append((fp + fn, ann))
        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{n_pool}]")

    scored.sort(key=lambda x: x[0])

    n_s = min(args.n_samples,  len(scored))
    n_f = min(args.n_failures, len(scored))
    sample_frames  = [a for _, a in scored[:n_s]]
    failure_frames = [a for _, a in scored[-n_f:]]

    model_tag = Path(args.detection_config).stem
    out_dir = output_root() / "qualitative" / "detection"
    out_dir.mkdir(parents=True, exist_ok=True)

    leg = _legend(args.cell_w)

    for tag, frames in [("samples", sample_frames), ("failures", failure_frames)]:
        grid = _make_grid(frames, args.n_cols, args.cell_w, args.cell_h)
        # prepend legend bar (tiled to full grid width)
        n_leg_tiles = grid.shape[1] // args.cell_w
        legend_row  = np.hstack([leg] * n_leg_tiles)
        if legend_row.shape[1] < grid.shape[1]:
            pad = np.zeros((24, grid.shape[1] - legend_row.shape[1], 3), dtype=np.uint8)
            legend_row = np.hstack([legend_row, pad])
        out_img = np.vstack([legend_row, grid])
        out_path = out_dir / f"{model_tag}_{args.split}_{tag}.png"
        cv2.imwrite(str(out_path), out_img)
        print(f"  saved → {out_path.name}")

    print(f"\nDone. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
