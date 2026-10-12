"""
Tracking evaluation — all MOT metrics (MOTA, MOTP, IDF1, MT/ML, HOTA/DetA/AssA).

GT is loaded from the processed parquet (bdd100k_videos/labels/<split>.parquet).
Predictions come from running the configured detector + tracker on the .mov videos.

Evaluated categories (intersection of BDD100K MOT and our detector):
    pedestrian, rider, car, truck, bus, motorcycle, bicycle
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# GT categories we evaluate against (must be detectable by our models)
EVAL_CATEGORIES = {
    "pedestrian", "rider", "car", "truck", "bus", "motorcycle", "bicycle",
}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate tracker on BDD100K MOT.")
    p.add_argument("--detection-config", required=True)
    p.add_argument("--tracking-config",  required=True)
    p.add_argument("--base-config", default=str(REPO / "config/base.yaml"))
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--max-videos", type=int, default=None)
    p.add_argument("--iou-thresh", type=float, default=0.5,
                   help="IoU threshold for MOTA/IDF1 matching (default 0.5)")
    p.add_argument("--conf-threshold", type=float, default=0.5,
                   help="Override detection confidence_threshold for tracking eval. "
                        "Default 0.5 (suppresses FPs and matches the threshold used "
                        "for the test-split runs). Pass an explicit value to change it.")
    p.add_argument("--device", default="0",
                   help="Physical GPU index for the detector, e.g. '0', '1'. "
                        "Passed directly to Ultralytics (which sets CUDA_VISIBLE_DEVICES "
                        "internally). Trackers are CPU-only.")
    return p.parse_args()


def _load_gt(split: str) -> dict:
    """
    Load GT from parquet into {video_name: {frame_idx: [(gt_id, x1, y1, x2, y2)]}}.
    Filters to EVAL_CATEGORIES only.
    """
    import pandas as pd
    from alert.utils.paths import data_processed_root

    parquet = data_processed_root() / "bdd100k_videos" / "labels" / f"{split}.parquet"
    if not parquet.exists():
        raise FileNotFoundError(f"GT parquet not found: {parquet}")

    df = pd.read_parquet(parquet)
    df = df[df["category"].isin(EVAL_CATEGORIES)].copy()
    df = df[df["haveVideo"] == True]  # noqa: E712

    gt: dict = {}
    for _, row in df.iterrows():
        vid   = row["videoName"]
        fidx  = int(row["frameIndex"])
        gt_id = str(row["id"])
        x1    = float(row["box2d.x1"])
        y1    = float(row["box2d.y1"])
        x2    = float(row["box2d.x2"])
        y2    = float(row["box2d.y2"])
        if vid not in gt:
            gt[vid] = {}
        if fidx not in gt[vid]:
            gt[vid][fidx] = []
        gt[vid][fidx].append((gt_id, x1, y1, x2, y2))

    return gt


def main():
    args = parse_args()
    # Override cfg device with the physical GPU index so Ultralytics routes to the
    # correct GPU. Trackers (ByteTrack / OC-SORT) are CPU-only Kalman filters —
    # they ignore cfg["device"] entirely.
    # Note: do NOT set CUDA_VISIBLE_DEVICES here — Ultralytics sets it itself when
    # it receives a numeric index, and that is the correct behaviour.

    import cv2
    from alert.utils.config import load_config
    from alert.detection.factory import build_detector
    from alert.tracking.factory import build_tracker
    from alert.tracking.metrics import compute_tracking_metrics
    from alert.utils.paths import data_processed_root

    cfg = load_config(args.base_config, args.detection_config, args.tracking_config)
    cfg["device"] = args.device  # numeric GPU index, e.g. "1" — passed to Ultralytics predict()
    cfg.setdefault("detection", {})["confidence_threshold"] = args.conf_threshold

    detector = build_detector(cfg)
    tracker  = build_tracker(cfg)

    video_dir = data_processed_root() / "bdd100k_videos" / "videos" / args.split
    if not video_dir.exists():
        print(f"MOT data not found: {video_dir}")
        return

    video_paths = sorted(video_dir.glob("*.mov"))
    if not video_paths:
        print(f"No .mov files found in {video_dir}")
        return

    if args.max_videos:
        video_paths = video_paths[: args.max_videos]

    print(f"Loading GT ({args.split} split)...")
    sequences_gt = _load_gt(args.split)

    # keep only sequences that have a corresponding video file
    available_names = {vp.stem for vp in video_paths}
    sequences_gt = {k: v for k, v in sequences_gt.items() if k in available_names}

    print(f"Running tracker on {len(video_paths)} sequences...")
    sequences_pred: dict = {}

    for video_path in video_paths:
        seq_name = video_path.stem

        # BDD100K MOT videos run at 30 FPS but GT is annotated at 5 FPS
        # (every 6th raw frame is labeled). The parquet's `frameIndex` column
        # numbers labeled frames sequentially (0..~199 for a 40 s video), NOT
        # raw video frames. We therefore:
        #   1. run detector + tracker on EVERY raw frame (Kalman needs the full
        #      30 Hz signal to stay smooth), and
        #   2. only store predictions at raw_idx % stride == 0, keyed by
        #      labeled_idx = raw_idx // stride, so pred keys align with GT keys.
        gt_seq_for_video = sequences_gt.get(seq_name)
        if not gt_seq_for_video:
            print(f"  [WARN] No GT for {video_path.name} — skipping")
            continue

        tracker.reset()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  [WARN] Cannot open {video_path.name} — skipping")
            continue

        total_raw_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # `len(gt_seq_for_video)` only counts labeled frames that contain at
        # least one box of an EVAL_CATEGORY — empty labeled frames are absent.
        # `max(frameIndex)+1` is the proper labeled-frame count.
        n_labeled = max(gt_seq_for_video) + 1 if gt_seq_for_video else 0
        if total_raw_frames <= 0 or n_labeled <= 0:
            print(f"  [WARN] {video_path.name}: cannot compute stride "
                  f"(raw={total_raw_frames}, labeled={n_labeled}) — skipping")
            cap.release()
            continue

        # round() handles off-by-one (e.g. raw=1199 / labeled=200 → 6, not 5)
        stride = max(1, round(total_raw_frames / n_labeled))
        print(f"  {video_path.name}: raw={total_raw_frames} "
              f"labeled={n_labeled} stride={stride}")

        raw_idx = 0
        seq_pred: dict = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            dets   = detector.detect(frame)
            # Filter detector output to the same taxonomy used by GT before
            # passing to the tracker. Without this, traffic_light / traffic_sign /
            # train predictions would be tracked, leak through, and inflate FP
            # at every frame (BDD100K scenes are saturated with them).
            dets = [d for d in dets if d.label in EVAL_CATEGORIES]
            tracks = tracker.update(detections=dets, frame=frame)
            # Qualify pred track IDs with seq_name so two sequences cannot
            # collide on the same numeric tracker ID in the global metrics
            # accumulator (gt_track_len / pred_track_len in metrics.py).
            if raw_idx % stride == 0:
                labeled_idx = raw_idx // stride
                # Cap labeled_idx so we never produce a key beyond what GT has;
                # videos can have a few extra raw frames at the tail.
                if labeled_idx < n_labeled:
                    seq_pred[labeled_idx] = [
                        (f"{seq_name}__{t.track_id}",
                         t.bbox[0], t.bbox[1], t.bbox[2], t.bbox[3])
                        for t in tracks
                    ]
            raw_idx += 1

        cap.release()
        sequences_pred[seq_name] = seq_pred
        print(f"    -> {raw_idx} raw frames processed, "
              f"{sum(len(v) for v in seq_pred.values())} track observations "
              f"on {len(seq_pred)} labeled frames")

    print(f"\nComputing metrics (IoU thresh={args.iou_thresh})...")
    m = compute_tracking_metrics(sequences_gt, sequences_pred, iou_thresh=args.iou_thresh)

    print(f"\n=== Tracking Evaluation ({args.split}) ===")

    print(f"\n  --- CLEAR MOT ---")
    print(f"  MOTA  : {m['MOTA']:+.4f}   (primary: combined det+ID accuracy; 1.0 = perfect)")
    print(f"  MOTP  : {m['MOTP']:.4f}   (mean IoU of matched pairs; higher = better localisation)")
    print(f"  IDSW  : {m['IDSW']}       (identity switches; lower = better)")
    print(f"  FP    : {m['FP']}         (false positive detections)")
    print(f"  FN    : {m['FN']}         (missed detections)")

    print(f"\n  --- Identity ---")
    print(f"  IDF1  : {m['IDF1']:.4f}   (primary: ID-aware F1; rewards long consistent tracks)")

    print(f"\n  --- Track Coverage (of {m['num_gt_tracks']} GT tracks) ---")
    print(f"  MT    : {m['MT']:.3f}  ({m['MT_count']} tracks ≥80% covered)")
    print(f"  PT    : {m['PT']:.3f}  ({m['PT_count']} tracks 20-80% covered)")
    print(f"  ML    : {m['ML']:.3f}  ({m['ML_count']} tracks ≤20% covered)")

    print(f"\n  --- HOTA (mean over α=0.05…0.95) ---")
    print(f"  HOTA  : {m['HOTA']:.4f}   (primary: geometric mean of DetA × AssA)")
    print(f"  DetA  : {m['DetA']:.4f}   (detection sub-score)")
    print(f"  AssA  : {m['AssA']:.4f}   (association / re-ID sub-score)")

    print(f"\n  --- HOTA @ selected α ---")
    for a in [0.5, 0.75]:
        if a in m["HOTA_per_alpha"]:
            print(f"    HOTA@{a:.2f}: {m['HOTA_per_alpha'][a]:.4f}"
                  f"  DetA={m['DetA_per_alpha'][a]:.4f}"
                  f"  AssA={m['AssA_per_alpha'][a]:.4f}")

    from alert.utils.paths import output_root
    det_tag = Path(args.detection_config).stem
    trk_tag = Path(args.tracking_config).stem
    out_path = output_root() / "eval" / f"tracking_{det_tag}_{trk_tag}_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # JSON requires string keys — convert float alphas
    payload = {
        "meta": {
            "stage": "tracking",
            "detection_model": det_tag,
            "tracking_model": trk_tag,
            "split": args.split,
            "detection_config": args.detection_config,
            "tracking_config": args.tracking_config,
            "iou_thresh": args.iou_thresh,
            "conf_threshold": cfg.get("detection", {}).get("confidence_threshold"),
            "num_sequences": len(sequences_pred),
            "device": args.device,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        "metrics": {
            "MOTA":  m["MOTA"],
            "MOTP":  m["MOTP"],
            "IDF1":  m["IDF1"],
            "IDSW":  m["IDSW"],
            "FP":    m["FP"],
            "FN":    m["FN"],
            "MT":    m["MT"],
            "PT":    m["PT"],
            "ML":    m["ML"],
            "MT_count": m["MT_count"],
            "PT_count": m["PT_count"],
            "ML_count": m["ML_count"],
            "num_gt_tracks": m["num_gt_tracks"],
            "num_gt_dets":   m["num_gt_dets"],
            "HOTA": m["HOTA"],
            "DetA": m["DetA"],
            "AssA": m["AssA"],
            "HOTA_per_alpha": {str(k): v for k, v in m["HOTA_per_alpha"].items()},
            "DetA_per_alpha": {str(k): v for k, v in m["DetA_per_alpha"].items()},
            "AssA_per_alpha": {str(k): v for k, v in m["AssA_per_alpha"].items()},
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
