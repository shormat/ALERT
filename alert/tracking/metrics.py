"""MOT metrics: MOTA, MOTP, IDF1, MT/ML, HOTA/DetA/AssA."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def _iou(a: tuple, b: tuple) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0.0:
        return 0.0
    # Clamp areas to non-negative to neutralize degenerate boxes (x2<x1 or y2<y1).
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _match_frame(
    gt_boxes: list[tuple],
    pred_boxes: list[tuple],
    iou_thresh: float,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    n_gt, n_pred = len(gt_boxes), len(pred_boxes)
    if n_gt == 0 and n_pred == 0:
        return [], [], []
    if n_gt == 0:
        return [], [], list(range(n_pred))
    if n_pred == 0:
        return [], list(range(n_gt)), []

    iou_mat = np.zeros((n_gt, n_pred), dtype=np.float64)
    for i, gb in enumerate(gt_boxes):
        for j, pb in enumerate(pred_boxes):
            iou_mat[i, j] = _iou(gb, pb)

    row_ind, col_ind = linear_sum_assignment(1.0 - iou_mat)

    matches, matched_gt, matched_pred = [], set(), set()
    for r, c in zip(row_ind, col_ind):
        if iou_mat[r, c] >= iou_thresh:
            matches.append((r, c, float(iou_mat[r, c])))
            matched_gt.add(r)
            matched_pred.add(c)

    unmatched_gt   = [i for i in range(n_gt)   if i not in matched_gt]
    unmatched_pred = [j for j in range(n_pred) if j not in matched_pred]
    return matches, unmatched_gt, unmatched_pred


def compute_tracking_metrics(
    sequences_gt:   dict,
    sequences_pred: dict,
    iou_thresh: float = 0.5,
    hota_alphas: list[float] | None = None,
) -> dict:
    """Compute MOT metrics across sequences in
    ``{seq: {frame: [(id, x1, y1, x2, y2), ...]}}`` format.
    """
    if hota_alphas is None:
        hota_alphas = [round(0.05 * i, 2) for i in range(1, 20)]

    total_gt = total_fp = total_fn = total_idsw = total_matches = 0
    total_matched_iou = 0.0

    gt_track_len:   dict[str, int] = {}
    pred_track_len: dict[str, int] = {}
    gt_pred_overlap: dict[tuple[str, str], int] = {}

    hota_acc: dict[float, dict] = {
        a: {"tp": 0, "fp": 0, "fn": 0, "tpr": {}}
        for a in hota_alphas
    }

    for seq_name in sequences_gt:
        gt_seq   = sequences_gt[seq_name]
        pred_seq = sequences_pred.get(seq_name, {})

        all_frames = sorted(set(gt_seq) | set(pred_seq))
        last_match: dict[str, str] = {}

        for frame_idx in all_frames:
            gt_frame   = gt_seq.get(frame_idx, [])
            pred_frame = pred_seq.get(frame_idx, [])

            gt_ids   = [str(t[0]) for t in gt_frame]
            gt_boxes = [t[1:] for t in gt_frame]
            pred_ids   = [str(t[0]) for t in pred_frame]
            pred_boxes = [t[1:] for t in pred_frame]

            for gid in gt_ids:
                gt_track_len[gid] = gt_track_len.get(gid, 0) + 1
            for pid in pred_ids:
                pred_track_len[pid] = pred_track_len.get(pid, 0) + 1

            total_gt += len(gt_frame)

            matches, unmatched_gt, unmatched_pred = _match_frame(
                gt_boxes, pred_boxes, iou_thresh
            )
            total_fp += len(unmatched_pred)
            total_fn += len(unmatched_gt)

            for gi, pi, iou in matches:
                gid, pid = gt_ids[gi], pred_ids[pi]
                total_matched_iou += iou
                total_matches += 1
                if gid in last_match and last_match[gid] != pid:
                    total_idsw += 1
                last_match[gid] = pid
                key = (gid, pid)
                gt_pred_overlap[key] = gt_pred_overlap.get(key, 0) + 1

            for alpha in hota_alphas:
                h_matches, h_ungt, h_unpred = _match_frame(
                    gt_boxes, pred_boxes, alpha
                )
                acc = hota_acc[alpha]
                acc["tp"] += len(h_matches)
                acc["fp"] += len(h_unpred)
                acc["fn"] += len(h_ungt)
                tpr = acc["tpr"]
                for gi, pi, _ in h_matches:
                    gid, pid = gt_ids[gi], pred_ids[pi]
                    if gid not in tpr:
                        tpr[gid] = {}
                    tpr[gid][pid] = tpr[gid].get(pid, 0) + 1

    mota = 1.0 - (total_fp + total_fn + total_idsw) / max(total_gt, 1)

    motp = total_matched_iou / max(total_matches, 1)

    gt_ids_list   = sorted(gt_track_len)
    pred_ids_list = sorted(pred_track_len)

    idtp = 0
    if gt_ids_list and pred_ids_list:
        overlap_mat = np.zeros((len(gt_ids_list), len(pred_ids_list)), dtype=np.float64)
        for i, g in enumerate(gt_ids_list):
            for j, p in enumerate(pred_ids_list):
                overlap_mat[i, j] = gt_pred_overlap.get((g, p), 0)
        ri, ci = linear_sum_assignment(-overlap_mat)
        idtp = int(overlap_mat[ri, ci].sum())

    total_gt_frames   = sum(gt_track_len.values())
    total_pred_frames = sum(pred_track_len.values())
    idfp = total_pred_frames - idtp
    idfn = total_gt_frames - idtp
    idf1 = 2 * idtp / max(2 * idtp + idfp + idfn, 1)

    mt_count = ml_count = pt_count = 0
    for gid in gt_ids_list:
        best = max(
            (gt_pred_overlap.get((gid, p), 0) for p in pred_ids_list),
            default=0,
        )
        ratio = best / max(gt_track_len[gid], 1)
        if ratio >= 0.8:
            mt_count += 1
        elif ratio <= 0.2:
            ml_count += 1
        else:
            pt_count += 1

    n_gt_tracks = len(gt_ids_list)
    mt_pct = mt_count / max(n_gt_tracks, 1)
    ml_pct = ml_count / max(n_gt_tracks, 1)
    pt_pct = pt_count / max(n_gt_tracks, 1)

    hota_per_alpha: dict[float, float] = {}
    deta_per_alpha: dict[float, float] = {}
    assa_per_alpha: dict[float, float] = {}

    for alpha in hota_alphas:
        tp = hota_acc[alpha]["tp"]
        fp = hota_acc[alpha]["fp"]
        fn = hota_acc[alpha]["fn"]
        tpr = hota_acc[alpha]["tpr"]

        det_a = tp / max(tp + fp + fn, 1)

        if tp == 0:
            ass_a = 0.0
        else:
            tp_by_gt:   dict[str, int] = {}
            tp_by_pred: dict[str, int] = {}
            for gid, pred_dict in tpr.items():
                tp_by_gt[gid] = sum(pred_dict.values())
                for pid, cnt in pred_dict.items():
                    tp_by_pred[pid] = tp_by_pred.get(pid, 0) + cnt

            ass_sum = 0.0
            for gid, pred_dict in tpr.items():
                for pid, cnt in pred_dict.items():
                    denom = tp_by_gt[gid] + tp_by_pred[pid] - cnt
                    ass_sum += cnt * cnt / max(denom, 1)
            ass_a = ass_sum / tp

        hota_per_alpha[alpha] = float(np.sqrt(det_a * ass_a))
        deta_per_alpha[alpha] = float(det_a)
        assa_per_alpha[alpha] = float(ass_a)

    hota  = float(np.mean(list(hota_per_alpha.values())))
    det_a = float(np.mean(list(deta_per_alpha.values())))
    ass_a = float(np.mean(list(assa_per_alpha.values())))

    return {
        "MOTA":           mota,
        "MOTP":           motp,
        "IDSW":           total_idsw,
        "FP":             total_fp,
        "FN":             total_fn,
        "IDF1":           idf1,
        "MT":             mt_pct,
        "ML":             ml_pct,
        "PT":             pt_pct,
        "MT_count":       mt_count,
        "ML_count":       ml_count,
        "PT_count":       pt_count,
        "num_gt_tracks":  n_gt_tracks,
        "num_gt_dets":    total_gt,
        "HOTA":           hota,
        "DetA":           det_a,
        "AssA":           ass_a,
        "HOTA_per_alpha": hota_per_alpha,
        "DetA_per_alpha": deta_per_alpha,
        "AssA_per_alpha": assa_per_alpha,
    }
