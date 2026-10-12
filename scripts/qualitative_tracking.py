"""Qualitative tracking visualisation — annotated video + identity-switch grid.

Renders bounding boxes with persistent track IDs and trajectory tails (last
--tail-len frames of each track's centre-point history).

When GT is available (parquet loaded), identity-switch (IDSW) events are
detected by tracking the GT-ID → pred-ID assignment over time.  Frames at each
IDSW are saved to a separate grid image.

Outputs:
  outputs/qualitative/tracking/{det}_{trk}_{video}_{split}.mp4
  outputs/qualitative/tracking/{det}_{trk}_{split}_idsw.png   (if IDSW found)
"""
from __future__ import annotations

import argparse
import os
from collections import deque
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]

# BGR colour palette — same 20-colour ring as Visualizer
_PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10),  (146, 204, 23), (61, 219, 134),
    (26, 147, 52),  (0, 212, 187),  (44, 153, 168), (0, 194, 255),
    (52, 69, 147),  (100, 115, 255),(0, 24, 236),   (132, 56, 255),
    (82, 0, 133),   (203, 56, 255), (255, 149, 200),(255, 55, 199),
]
_GT_COL = (30, 220, 30)   # thin green for GT boxes


def parse_args():
    p = argparse.ArgumentParser(description="Qualitative tracking output: annotated video + IDSW grid.")
    p.add_argument("--detection-config",  required=True)
    p.add_argument("--tracking-config",   required=True)
    p.add_argument("--base-config", default=str(REPO / "config/base.yaml"))
    p.add_argument("--split",    default="val", choices=["val", "test"])
    p.add_argument("--video",    default=None,
                   help="Specific video stem to process (default: first video in split)")
    p.add_argument("--n-frames", type=int, default=None,
                   help="Limit to first N frames (default: whole video)")
    p.add_argument("--tail-len", type=int, default=30,
                   help="Trajectory tail length in frames")
    p.add_argument("--show-gt",  action="store_true",
                   help="Overlay thin GT boxes (requires parquet GT)")
    p.add_argument("--iou-thresh", type=float, default=0.5,
                   help="IoU threshold for GT↔pred matching (IDSW detection)")
    p.add_argument("--max-idsw-frames", type=int, default=12,
                   help="Max IDSW frames saved to grid")
    p.add_argument("--device", default="0")
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────

def _track_color(track_id: int):
    return _PALETTE[track_id % len(_PALETTE)]


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def _detect_idsw(
    gt_frame:     list[tuple],            # [(gt_id, x1, y1, x2, y2)]
    tracks:       list,                   # list[TrackedObject]
    gt_to_pred:   dict[str, int],         # mutable — updated in-place
    iou_thresh:   float,
) -> int:
    """Return number of new IDSW events detected this frame."""
    if not gt_frame or not tracks:
        return 0

    pred_boxes = [(t.track_id, t.bbox[0], t.bbox[1], t.bbox[2], t.bbox[3]) for t in tracks]
    n_idsw = 0

    for gt_id, gx1, gy1, gx2, gy2 in gt_frame:
        best_j, best_v = -1, iou_thresh
        for j, (_, px1, py1, px2, py2) in enumerate(pred_boxes):
            v = _iou((gx1, gy1, gx2, gy2), (px1, py1, px2, py2))
            if v > best_v:
                best_v, best_j = v, j
        if best_j >= 0:
            pred_id = pred_boxes[best_j][0]
            prev = gt_to_pred.get(str(gt_id))
            if prev is not None and prev != pred_id:
                n_idsw += 1
            gt_to_pred[str(gt_id)] = pred_id

    return n_idsw


def _render_frame(
    frame:        np.ndarray,
    tracks:       list,
    tails:        dict[int, deque],
    gt_frame:     list[tuple] | None,
    show_gt:      bool,
    frame_idx:    int,
    idsw_count:   int,
) -> np.ndarray:
    ann = frame.copy()

    # Draw trajectory tails (oldest→newest, fading from thin to thick)
    for t in tracks:
        pts = list(tails.get(t.track_id, []))
        col = _track_color(t.track_id)
        for k in range(1, len(pts)):
            p1 = (int(pts[k - 1][0]), int(pts[k - 1][1]))
            p2 = (int(pts[k][0]),     int(pts[k][1]))
            thickness = max(1, int(2 * k / max(len(pts), 1)))
            cv2.line(ann, p1, p2, col, thickness)

    # Draw prediction boxes
    for t in tracks:
        x1, y1, x2, y2 = (int(c) for c in t.bbox)
        col = _track_color(t.track_id)
        cv2.rectangle(ann, (x1, y1), (x2, y2), col, 2)
        cv2.putText(ann, f"ID:{t.track_id} {t.label}",
                    (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

    # Optionally draw GT boxes (thin green)
    if show_gt and gt_frame:
        for _, gx1, gy1, gx2, gy2 in gt_frame:
            cv2.rectangle(ann, (int(gx1), int(gy1)), (int(gx2), int(gy2)), _GT_COL, 1)

    # HUD: frame index + cumulative IDSW
    cv2.putText(ann, f"f={frame_idx}  IDSW={idsw_count}",
                (8, ann.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return ann


def _load_gt(split: str) -> dict:
    import pandas as pd
    from alert.utils.paths import data_processed_root
    parquet = data_processed_root() / "bdd100k_videos" / "labels" / f"{split}.parquet"
    if not parquet.exists():
        return {}
    df = pd.read_parquet(parquet)
    df = df[df["haveVideo"] == True]  # noqa: E712
    gt: dict = {}
    for _, row in df.iterrows():
        vid   = row["videoName"]
        fidx  = int(row["frameIndex"])
        gt_id = str(row["id"])
        x1, y1 = float(row["box2d.x1"]), float(row["box2d.y1"])
        x2, y2 = float(row["box2d.x2"]), float(row["box2d.y2"])
        gt.setdefault(vid, {}).setdefault(fidx, []).append((gt_id, x1, y1, x2, y2))
    return gt


def _thumb(img: np.ndarray, w: int, h: int) -> np.ndarray:
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


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    from alert.utils.config import load_config
    from alert.detection.factory import build_detector
    from alert.tracking.factory import build_tracker
    from alert.utils.paths import data_processed_root, output_root

    cfg = load_config(args.base_config, args.detection_config, args.tracking_config)
    cfg["device"] = args.device
    detector = build_detector(cfg)
    tracker  = build_tracker(cfg)

    video_dir = data_processed_root() / "bdd100k_videos" / "videos" / args.split
    if not video_dir.exists():
        print(f"Video directory not found: {video_dir}")
        return

    if args.video:
        candidates = [video_dir / (args.video if args.video.endswith(".mov") else args.video + ".mov")]
    else:
        candidates = sorted(video_dir.glob("*.mov"))

    if not candidates or not candidates[0].exists():
        print(f"No video found in {video_dir}")
        return
    video_path = candidates[0]
    seq_name = video_path.stem
    print(f"Processing: {video_path.name}  ({args.split})")

    # Load GT (best-effort — skip IDSW detection if unavailable)
    print("Loading GT parquet (for IDSW detection)...")
    all_gt = _load_gt(args.split)
    seq_gt  = all_gt.get(seq_name, {})
    if not seq_gt:
        print("  GT not found for this sequence — IDSW detection disabled.")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    det_tag = Path(args.detection_config).stem
    trk_tag = Path(args.tracking_config).stem
    out_dir = output_root() / "qualitative" / "tracking"
    out_dir.mkdir(parents=True, exist_ok=True)

    video_out_path = out_dir / f"{det_tag}_{trk_tag}_{seq_name}_{args.split}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_out_path), fourcc, fps, (W, H))

    tails: dict[int, deque] = {}
    gt_to_pred: dict[str, int] = {}
    idsw_total = 0
    idsw_frames: list[np.ndarray] = []  # annotated frames at IDSW events
    frame_idx = 0

    print("Rendering...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if args.n_frames and frame_idx >= args.n_frames:
            break

        dets   = detector.detect(frame)
        tracks = tracker.update(detections=dets, frame=frame)

        # Update tails
        for t in tracks:
            cx = (t.bbox[0] + t.bbox[2]) / 2
            cy = (t.bbox[1] + t.bbox[3]) / 2
            if t.track_id not in tails:
                tails[t.track_id] = deque(maxlen=args.tail_len)
            tails[t.track_id].append((cx, cy))

        # Detect IDSW
        gt_frame = seq_gt.get(frame_idx)
        n_new_idsw = _detect_idsw(gt_frame or [], tracks, gt_to_pred, args.iou_thresh)
        idsw_total += n_new_idsw

        ann = _render_frame(frame, tracks, tails, gt_frame, args.show_gt, frame_idx, idsw_total)
        writer.write(ann)

        if n_new_idsw > 0 and len(idsw_frames) < args.max_idsw_frames:
            idsw_frames.append(ann.copy())

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  frame {frame_idx}  tracks={len(tracks)}  IDSW={idsw_total}")

    cap.release()
    writer.release()
    print(f"  saved → {video_out_path.name}  ({frame_idx} frames, {idsw_total} IDSW)")

    # IDSW grid
    if idsw_frames:
        cell_w, cell_h = min(W, 640), min(H, 360)
        grid = _make_grid(idsw_frames, n_cols=4, cell_w=cell_w, cell_h=cell_h)
        hdr = np.zeros((22, grid.shape[1], 3), dtype=np.uint8)
        cv2.putText(hdr, f"IDSW events — {det_tag} + {trk_tag}  ({args.split})",
                    (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        grid_path = out_dir / f"{det_tag}_{trk_tag}_{args.split}_idsw.png"
        cv2.imwrite(str(grid_path), np.vstack([hdr, grid]))
        print(f"  saved → {grid_path.name}")
    else:
        print("  no IDSW events detected — grid skipped")

    print(f"\nDone. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
