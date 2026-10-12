"""Frame-level Pierard ranking score for one ALERT pipeline.

Runs detection + tracking + segmentation on every MOT-val (or MOT-test)
clip, evaluates per-frame satisfaction (S_det, S_seg, S_trk, S_joint)
and three importance profiles (uniform, risk_weighted, vru_present),
and writes:

    outputs/eval/ranking_<det>_<seg>_<trk>_<split>.json   — aggregate R_I
    outputs/eval/ranking_<det>_<seg>_<trk>_<split>.jsonl  — per-frame records

Important:
- Drivable-area ground truth only exists for the still-image keyframe
  per clip.  We load it from the raw alvaromalfaro JSON (rasterising
  poly2d on the fly) and propagate the same GT mask to every labelled
  MOT frame in the clip (option (b) from Step 1).  This assumes the
  drivable area is roughly constant across a 40 s urban clip — see
  docs/ranking_score.md for the discussion.
- The seg keyframe lives in the segmentation TRAINING set (every MOT
  clip is sourced from BDD100K official train).  S_seg numbers are
  therefore optimistic in absolute terms but comparable across the
  pipelines we rank, because every pipeline draws from the same set.

Usage example:
    uv run python scripts/eval_ranking_score.py \
        --detection-config config/detection/yolov8m.yaml \
        --tracking-config  config/tracking/bytetrack.yaml \
        --segmentation-config config/segmentation/segformer_b2.yaml \
        --split val
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--detection-config",    required=True)
    p.add_argument("--tracking-config",     required=True)
    p.add_argument("--segmentation-config", required=True)
    p.add_argument("--base-config", default=str(REPO / "config/base.yaml"))
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--max-videos", type=int, default=None)
    p.add_argument("--iou-thresh",    type=float, default=0.5,
                   help="IoU for det/trk per-frame matching (default 0.5)")
    p.add_argument("--tau-det",       type=float, default=0.5,
                   help="S_det acceptance on per-frame F1 (default 0.5)")
    p.add_argument("--tau-seg",       type=float, default=0.7,
                   help="S_seg acceptance on per-frame direct-drivable IoU "
                        "(default 0.7)")
    p.add_argument("--conf-threshold", type=float, default=0.5,
                   help="Detector confidence (matches eval_tracking default)")
    p.add_argument("--device", default="0",
                   help="Physical GPU index for detector + segmentor.")
    p.add_argument("--output-tag", default=None,
                   help="Optional override for output filename stem; "
                        "defaults to '<det>_<seg>_<trk>'")
    p.add_argument("--no-jsonl", action="store_true",
                   help="Skip per-frame JSONL dump (smaller disk footprint)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# GT loading
# ---------------------------------------------------------------------------


def _load_mot_gt_with_class(split: str) -> dict:
    """Same as eval_tracking._load_gt but keeps the class label.

    Output: {seq: {labeled_idx: [(x1, y1, x2, y2, gt_id, class), ...]}}
    """
    import pandas as pd
    from alert.evaluation.ranking_score import MOT_EVAL_CLASSES
    from alert.utils.paths import data_processed_root

    parquet = data_processed_root() / "bdd100k_videos" / "labels" / f"{split}.parquet"
    if not parquet.exists():
        raise FileNotFoundError(f"GT parquet not found: {parquet}")

    df = pd.read_parquet(parquet)
    df = df[df["category"].isin(MOT_EVAL_CLASSES)].copy()
    df = df[df["haveVideo"] == True]  # noqa: E712

    gt: dict = {}
    for _, row in df.iterrows():
        vid   = row["videoName"]
        fidx  = int(row["frameIndex"])
        gt_id = str(row["id"])
        cat   = row["category"]
        x1    = float(row["box2d.x1"])
        y1    = float(row["box2d.y1"])
        x2    = float(row["box2d.x2"])
        y2    = float(row["box2d.y2"])
        gt.setdefault(vid, {}).setdefault(fidx, []).append(
            (x1, y1, x2, y2, gt_id, cat)
        )
    return gt


def _load_drivable_gt_for_video(stem: str) -> "np.ndarray | None":
    """Rasterise a 720x1280 GT direct-drivable mask from the raw
    alvaromalfaro JSON.  All MOT clips come from the BDD100K train
    split, so the keyframe JSON sits under .../labels/train/.
    """
    import numpy as np  # noqa: F401  (used downstream)

    from alert.data_processing.drivable import (
        IMG_H,
        IMG_W,
        extract_drivable_polygons,
        rasterize_mask,
    )
    from alert.utils.paths import data_raw_root

    json_path = (
        data_raw_root()
        / "bdd100k_alvaromalfaro"
        / "bdd100k"
        / "labels"
        / "train"
        / f"{stem}.json"
    )
    if not json_path.exists():
        return None
    data = json.loads(json_path.read_text())
    polygons = extract_drivable_polygons(data)
    if not polygons:
        # No drivable polygons annotated for this stem — return all-bg
        # so seg satisfaction degenerates to trivially-met when pred is
        # also empty.
        import numpy as np
        return np.full((IMG_H, IMG_W), 2, dtype=np.uint8)
    return rasterize_mask(polygons)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    # Set CUDA_VISIBLE_DEVICES BEFORE torch import (segmentor uses
    # cuda:0 internally — must map to the requested physical GPU).
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    import cv2
    import numpy as np

    from alert.detection.factory import build_detector
    from alert.evaluation.ranking_score import (
        PerFrameRecord,
        VRU_CLASSES,
        aggregate_R_I,
        frame_detection_satisfaction,
        frame_importance_risk_weighted,
        frame_importance_uniform,
        frame_importance_vru_present,
        frame_segmentation_satisfaction,
        frame_tracking_satisfaction,
    )
    from alert.segmentation.factory import build_segmentor
    from alert.tracking.factory import build_tracker
    from alert.utils.config import load_config
    from alert.utils.paths import data_processed_root, output_root

    cfg = load_config(
        args.base_config,
        args.detection_config,
        args.tracking_config,
        args.segmentation_config,
    )
    cfg["device"] = args.device
    cfg.setdefault("detection", {})["confidence_threshold"] = args.conf_threshold

    detector  = build_detector(cfg)
    tracker   = build_tracker(cfg)
    segmentor = build_segmentor(cfg)

    video_dir = data_processed_root() / "bdd100k_videos" / "videos" / args.split
    if not video_dir.exists():
        raise FileNotFoundError(f"MOT data not found: {video_dir}")

    video_paths = sorted(video_dir.glob("*.mov"))
    if args.max_videos:
        video_paths = video_paths[: args.max_videos]
    if not video_paths:
        raise RuntimeError(f"No .mov files in {video_dir}")

    print(f"Loading GT ({args.split} split)...")
    sequences_gt = _load_mot_gt_with_class(args.split)
    available_names = {vp.stem for vp in video_paths}
    sequences_gt = {k: v for k, v in sequences_gt.items() if k in available_names}

    records: list[PerFrameRecord] = []
    n_videos_with_seg_gt = 0

    for v_i, video_path in enumerate(video_paths, 1):
        seq_name = video_path.stem
        gt_seq = sequences_gt.get(seq_name)
        if not gt_seq:
            print(f"  [{v_i}/{len(video_paths)}] {seq_name}: no GT, skip")
            continue

        gt_direct_mask = _load_drivable_gt_for_video(seq_name)
        if gt_direct_mask is not None:
            n_videos_with_seg_gt += 1

        tracker.reset()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  [{v_i}/{len(video_paths)}] {seq_name}: cannot open, skip")
            continue

        total_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n_labeled = max(gt_seq) + 1 if gt_seq else 0
        if total_raw <= 0 or n_labeled <= 0:
            print(f"  [{v_i}/{len(video_paths)}] {seq_name}: bad stride, skip")
            cap.release()
            continue
        stride = max(1, round(total_raw / n_labeled))

        # Per-sequence S_trk state
        last_match: dict[str, str] = {}

        # Pre-resize GT mask to the segmentor output resolution lazily
        # (we resize the seg pred to GT instead — same as eval_segmentation).
        raw_idx = 0
        n_seq_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            dets = detector.detect(frame)
            from alert.evaluation.ranking_score import MOT_EVAL_CLASSES
            dets = [d for d in dets if d.label in MOT_EVAL_CLASSES]
            tracks = tracker.update(detections=dets, frame=frame)

            if raw_idx % stride == 0:
                labeled_idx = raw_idx // stride
                if labeled_idx < n_labeled:
                    gt_frame = gt_seq.get(labeled_idx, [])

                    # Build pred boxes for det F1 (GT classes are the 7
                    # MOT-eval classes, so the comparison stays in-class).
                    pred_dets_box = [
                        (d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.label)
                        for d in dets
                    ]
                    gt_box = [
                        (t[0], t[1], t[2], t[3], t[5]) for t in gt_frame
                    ]
                    s_det = frame_detection_satisfaction(
                        gt_box, pred_dets_box, iou_thresh=args.iou_thresh,
                        tau=args.tau_det,
                    )

                    # Segmentation: run only on labelled frames.  Predictions
                    # may differ in resolution from the GT mask — resize the
                    # pred to GT shape if needed.
                    if gt_direct_mask is not None:
                        seg = segmentor.segment(frame).mask
                        if seg.shape != gt_direct_mask.shape:
                            seg = cv2.resize(
                                seg,
                                (gt_direct_mask.shape[1], gt_direct_mask.shape[0]),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        s_seg = frame_segmentation_satisfaction(
                            seg, gt_direct_mask, tau=args.tau_seg,
                        )
                    else:
                        # No seg GT for this clip — degrade to NaN-like:
                        # exclude from S_joint by setting to 1 (vacuous)
                        # but flag importance_risk_weighted to base.
                        s_seg = 1

                    # Tracking S_trk (mutates last_match in place)
                    pred_tracks_box = [
                        (t.bbox[0], t.bbox[1], t.bbox[2], t.bbox[3],
                         f"{seq_name}__{t.track_id}", t.label)
                        for t in tracks
                    ]
                    gt_tracks_box = [
                        (t[0], t[1], t[2], t[3], t[4], t[5]) for t in gt_frame
                    ]
                    s_trk = frame_tracking_satisfaction(
                        gt_tracks_box, pred_tracks_box, last_match,
                        iou_thresh=args.iou_thresh,
                    )

                    s_joint = int(s_det and s_seg and s_trk)

                    i_uni = frame_importance_uniform()
                    i_rw  = frame_importance_risk_weighted(
                        gt_tracks_box, gt_direct_mask,
                    )
                    i_vp  = frame_importance_vru_present(gt_tracks_box)

                    records.append(PerFrameRecord(
                        seq_name=seq_name,
                        labeled_idx=labeled_idx,
                        s_det=s_det,
                        s_seg=s_seg,
                        s_trk=s_trk,
                        s_joint=s_joint,
                        importance_uniform=i_uni,
                        importance_risk_weighted=i_rw,
                        importance_vru_present=i_vp,
                    ))
                    n_seq_frames += 1

            raw_idx += 1

        cap.release()
        if v_i % 10 == 0 or v_i == len(video_paths):
            print(f"  [{v_i}/{len(video_paths)}] {seq_name}: "
                  f"raw={raw_idx} labeled={n_seq_frames}")

    print(f"\nTotal labelled frames evaluated: {len(records)}")
    print(f"Videos with seg GT loaded:       {n_videos_with_seg_gt}/"
          f"{len(video_paths)}")

    aggregates = aggregate_R_I(records)
    print(f"\n=== Frame-level R_I ({args.split}) ===")
    for profile, scores in aggregates.items():
        print(f"\n  --- Profile: {profile} ---")
        print(f"    R_I (joint)        : {scores['joint']:.4f}")
        print(f"    R_I (det only)     : {scores['det']:.4f}")
        print(f"    R_I (seg only)     : {scores['seg']:.4f}")
        print(f"    R_I (trk only)     : {scores['trk']:.4f}")
        print(f"    n_frames           : {int(scores['n_frames'])}")
        print(f"    n_frames (I>0)     : {int(scores['n_frames_with_I_gt0'])}")
        print(f"    sum I              : {scores['total_importance']:.2f}")

    det_tag = Path(args.detection_config).stem
    trk_tag = Path(args.tracking_config).stem
    seg_tag = Path(args.segmentation_config).stem
    out_tag = args.output_tag or f"{det_tag}_{seg_tag}_{trk_tag}"

    out_dir = output_root() / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    agg_path = out_dir / f"ranking_{out_tag}_{args.split}.json"
    payload = {
        "meta": {
            "stage": "ranking_score",
            "detection_config":    args.detection_config,
            "tracking_config":     args.tracking_config,
            "segmentation_config": args.segmentation_config,
            "split": args.split,
            "iou_thresh": args.iou_thresh,
            "tau_det":    args.tau_det,
            "tau_seg":    args.tau_seg,
            "conf_threshold": args.conf_threshold,
            "vru_classes": sorted(VRU_CLASSES),
            "n_videos_with_seg_gt": n_videos_with_seg_gt,
            "n_videos_total":       len(video_paths),
            "n_frames":             len(records),
            "device": args.device,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        "metrics": aggregates,
    }
    agg_path.write_text(json.dumps(payload, indent=2))
    print(f"\n  Saved aggregate -> {agg_path}")

    if not args.no_jsonl and records:
        jsonl_path = out_dir / f"ranking_{out_tag}_{args.split}.jsonl"
        with jsonl_path.open("w") as f:
            for r in records:
                f.write(json.dumps(r.as_dict()) + "\n")
        print(f"  Saved per-frame -> {jsonl_path}")


if __name__ == "__main__":
    main()
