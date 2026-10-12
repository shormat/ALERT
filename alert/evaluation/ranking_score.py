"""Frame-level Piérard ranking score (Lecture 6 — ELEN0449).

The ranking score for a single random experiment is

    R_I(P) = (sum_omega S(omega) I(omega) P({omega}))
             / (sum_omega I(omega) P({omega}))

omega is one outcome of the random experiment.  In our setup we sample
one labelled frame uniformly at random from the eval set, so

    P({omega}) = 1 / N    for all N frames.

That common factor cancels in numerator and denominator, leaving the
finite-sample estimator

    R_I = sum_frames S(frame) I(frame) / sum_frames I(frame).

``compute_R_I`` keeps the general (P-aware) form so non-uniform sampling
schemes — e.g. day/night rebalancing or sequence-level pre-sampling —
can be plugged in without touching downstream code.

The per-frame primitives below (S_det, S_seg, S_trk, importance) are
deliberately stateless: callers iterate frames in any order and feed the
arrays into ``compute_R_I``.  The only stateful bit is ID-switch
detection, which needs the last GT->pred match map maintained across
frames within a sequence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

VRU_CLASSES: frozenset[str] = frozenset({"pedestrian", "rider", "motorcycle", "bicycle"})

# 7 movable classes evaluated by the BDD100K MOT eval (matches
# scripts/eval_tracking.py:EVAL_CATEGORIES).
MOT_EVAL_CLASSES: frozenset[str] = frozenset(
    {"pedestrian", "rider", "car", "truck", "bus", "motorcycle", "bicycle"}
)

# Direct-drivable class index in the segmentation taxonomy
# (mirrors alert.segmentation.base.SEG_DIRECT).
SEG_DIRECT_ID: int = 0


# ---------------------------------------------------------------------------
# Core ranking-score formula
# ---------------------------------------------------------------------------


def compute_R_I(
    satisfactions: Sequence[float],
    importances: Sequence[float],
    probabilities: Sequence[float] | None = None,
) -> float:
    """Pierard's importance-weighted satisfaction ratio.

    Parameters
    ----------
    satisfactions : 1-D sequence, S(omega) for each omega.  Usually 0 or 1.
    importances   : 1-D sequence, I(omega) for each omega.  Non-negative.
    probabilities : optional 1-D sequence P({omega}).  If None, uniform
                    sampling is assumed and the term cancels analytically.

    Returns
    -------
    float : R_I in [0, 1] when 0 <= S <= 1 and I >= 0.

    Notes
    -----
    Returns 0.0 when sum(I * P) == 0 — i.e. no frame contributes any
    importance — to keep the score well-defined on edge cases (empty
    eval set, vru_present profile on a clip with no VRUs).
    """
    s = np.asarray(satisfactions, dtype=np.float64)
    i = np.asarray(importances, dtype=np.float64)
    if s.shape != i.shape:
        raise ValueError(f"shape mismatch: S {s.shape} vs I {i.shape}")
    if np.any(i < 0):
        raise ValueError("importances must be non-negative")

    if probabilities is None:
        weight = i
    else:
        p = np.asarray(probabilities, dtype=np.float64)
        if p.shape != s.shape:
            raise ValueError(f"shape mismatch: P {p.shape} vs S {s.shape}")
        if np.any(p < 0):
            raise ValueError("probabilities must be non-negative")
        weight = i * p

    denom = float(weight.sum())
    if denom <= 0.0:
        return 0.0
    return float((s * weight).sum() / denom)


# ---------------------------------------------------------------------------
# Geometry helpers (frame-local)
# ---------------------------------------------------------------------------


def _iou(a: tuple, b: tuple) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _hungarian_match(
    gt_boxes: Sequence[tuple],
    pred_boxes: Sequence[tuple],
    iou_thresh: float,
) -> tuple[list[tuple[int, int]], int, int]:
    """Returns (matched_pairs, n_unmatched_gt, n_unmatched_pred)."""
    n_gt, n_pred = len(gt_boxes), len(pred_boxes)
    if n_gt == 0 or n_pred == 0:
        return [], n_gt, n_pred

    iou_mat = np.zeros((n_gt, n_pred), dtype=np.float64)
    for r, gb in enumerate(gt_boxes):
        for c, pb in enumerate(pred_boxes):
            iou_mat[r, c] = _iou(gb, pb)

    rows, cols = linear_sum_assignment(1.0 - iou_mat)
    pairs: list[tuple[int, int]] = []
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    for r, c in zip(rows, cols):
        if iou_mat[r, c] >= iou_thresh:
            pairs.append((int(r), int(c)))
            matched_gt.add(int(r))
            matched_pred.add(int(c))
    return pairs, n_gt - len(matched_gt), n_pred - len(matched_pred)


# ---------------------------------------------------------------------------
# Per-frame satisfaction primitives
# ---------------------------------------------------------------------------


def frame_detection_f1(
    gt_boxes: Sequence[tuple[float, float, float, float, str]],
    pred_boxes: Sequence[tuple[float, float, float, float, str]],
    classes: Iterable[str] = MOT_EVAL_CLASSES,
    iou_thresh: float = 0.5,
) -> float:
    """Macro F1 over the given class set, on a single frame.

    Each box is (x1, y1, x2, y2, class).  Matching is per-class and
    Hungarian.  Empty-on-both side classes contribute F1 = 1 (perfect
    "do nothing" agreement); empty-GT/non-empty-pred classes contribute
    F1 = 0 (false positives); non-empty-GT/empty-pred contribute F1 = 0.
    Macro = arithmetic mean over the union of classes that appear in
    GT or pred (so unused classes do not dilute the score).
    """
    classes = set(classes)
    by_cls_gt: dict[str, list[tuple]] = {c: [] for c in classes}
    by_cls_pr: dict[str, list[tuple]] = {c: [] for c in classes}
    for *box, cls in gt_boxes:
        if cls in classes:
            by_cls_gt[cls].append(tuple(box))
    for *box, cls in pred_boxes:
        if cls in classes:
            by_cls_pr[cls].append(tuple(box))

    active = [c for c in classes if by_cls_gt[c] or by_cls_pr[c]]
    if not active:
        # No GT and no pred among evaluated classes — vacuously satisfied.
        return 1.0

    f1s: list[float] = []
    for c in active:
        gtb, prb = by_cls_gt[c], by_cls_pr[c]
        _, n_fn, n_fp = _hungarian_match(gtb, prb, iou_thresh)
        n_tp = len(gtb) - n_fn
        denom = 2 * n_tp + n_fp + n_fn
        f1s.append(2 * n_tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s))


def frame_detection_satisfaction(
    gt_boxes: Sequence[tuple[float, float, float, float, str]],
    pred_boxes: Sequence[tuple[float, float, float, float, str]],
    classes: Iterable[str] = MOT_EVAL_CLASSES,
    iou_thresh: float = 0.5,
    tau: float = 0.5,
) -> int:
    """S_det = 1 iff frame-level macro F1 (IoU 0.5, MOT classes) >= tau."""
    return int(frame_detection_f1(gt_boxes, pred_boxes, classes, iou_thresh) >= tau)


def frame_segmentation_iou(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    class_id: int = SEG_DIRECT_ID,
) -> float:
    """IoU on a single class between two integer masks of identical shape."""
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(
            f"shape mismatch: pred {pred_mask.shape} vs gt {gt_mask.shape}"
        )
    pred_bin = pred_mask == class_id
    gt_bin = gt_mask == class_id
    inter = int(np.logical_and(pred_bin, gt_bin).sum())
    union = int(np.logical_or(pred_bin, gt_bin).sum())
    if union == 0:
        # No direct-drivable in either pred or GT — vacuously satisfied.
        return 1.0
    return inter / union


def frame_segmentation_satisfaction(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    class_id: int = SEG_DIRECT_ID,
    tau: float = 0.7,
) -> int:
    return int(frame_segmentation_iou(pred_mask, gt_mask, class_id) >= tau)


def frame_tracking_satisfaction(
    gt_tracks: Sequence[tuple[float, float, float, float, str, str]],
    pred_tracks: Sequence[tuple[float, float, float, float, str, str]],
    last_match: dict[str, str],
    vru_classes: Iterable[str] = VRU_CLASSES,
    iou_thresh: float = 0.5,
) -> int:
    """S_trk = 1 iff no VRU ID switch on this frame.

    Each track is (x1, y1, x2, y2, gt_or_pred_id, class).  ``last_match``
    is the running map {gt_id -> last_matched_pred_id} for the current
    sequence; this function reads it AND mutates it in place to keep it
    in sync for the next frame.

    Definition: an ID switch happens for a VRU GT track if it was
    matched to pred id A on a previous frame and is now matched to pred
    id B != A on this frame.  Frames with no VRU GT contribute S_trk = 1
    (vacuously, no track to switch).
    """
    vru_classes = set(vru_classes)
    vru_gt_idx = [i for i, t in enumerate(gt_tracks) if t[5] in vru_classes]
    if not vru_gt_idx:
        return 1

    gt_boxes = [tuple(t[:4]) for t in gt_tracks]
    pred_boxes = [tuple(t[:4]) for t in pred_tracks]
    pairs, _, _ = _hungarian_match(gt_boxes, pred_boxes, iou_thresh)

    matched_gi_to_pi = {gi: pi for gi, pi in pairs}

    switched = False
    for gi in vru_gt_idx:
        gid = str(gt_tracks[gi][4])
        if gi not in matched_gi_to_pi:
            # GT VRU not matched on this frame — that is an FN, not an ID
            # switch.  Do not mutate last_match (keep history) and do not
            # flag a switch.
            continue
        pid = str(pred_tracks[matched_gi_to_pi[gi]][4])
        if gid in last_match and last_match[gid] != pid:
            switched = True
        last_match[gid] = pid
    return 0 if switched else 1


# ---------------------------------------------------------------------------
# Per-frame importance primitives
# ---------------------------------------------------------------------------


def _bbox_footprint_overlap(
    bbox: tuple[float, float, float, float],
    direct_mask: np.ndarray,
) -> float:
    """Fraction of direct-drivable pixels in a thin band below the box.

    Mirrors alert.risk.risk_reasoner._in_path_likelihood (the p_i
    quantity from Section 4.5 of the report) so the importance profile
    is geometrically consistent with the risk module.
    """
    H, W = direct_mask.shape
    x1, y1, x2, y2 = bbox
    ix1 = max(0, int(x1))
    ix2 = min(W - 1, int(x2))
    iy1 = max(0, int(y1))
    iy2 = min(H - 1, int(y2))
    if ix2 <= ix1:
        return 0.0
    band_h = max(4, (iy2 - iy1) // 6)
    by1 = max(0, iy2 - 2)
    by2 = min(H - 1, iy2 + band_h)
    roi = direct_mask[by1:by2, ix1:ix2]
    if roi.size == 0:
        return 0.0
    return float(roi.sum()) / roi.size


def frame_importance_uniform() -> float:
    return 1.0


def frame_importance_vru_present(
    gt_tracks: Sequence[tuple[float, float, float, float, str, str]],
    vru_classes: Iterable[str] = VRU_CLASSES,
) -> float:
    vru_classes = set(vru_classes)
    return 1.0 if any(t[5] in vru_classes for t in gt_tracks) else 0.0


def frame_importance_risk_weighted(
    gt_tracks: Sequence[tuple[float, float, float, float, str, str]],
    gt_direct_mask: np.ndarray | None,
    vru_classes: Iterable[str] = VRU_CLASSES,
    overlap_threshold: float = 0.30,
    high_weight: float = 5.0,
    base_weight: float = 1.0,
) -> float:
    """I = high_weight if any VRU footprint overlaps direct-drivable
    >= overlap_threshold, else base_weight.

    When ``gt_direct_mask`` is None (e.g. seg GT unavailable for this
    clip) the profile degrades to ``base_weight`` everywhere so the
    score remains computable.
    """
    if gt_direct_mask is None:
        return base_weight
    vru_classes = set(vru_classes)
    direct = (gt_direct_mask == SEG_DIRECT_ID).astype(np.uint8)
    for t in gt_tracks:
        if t[5] not in vru_classes:
            continue
        if _bbox_footprint_overlap(tuple(t[:4]), direct) >= overlap_threshold:
            return high_weight
    return base_weight


# ---------------------------------------------------------------------------
# Aggregation container
# ---------------------------------------------------------------------------


@dataclass
class PerFrameRecord:
    seq_name: str
    labeled_idx: int
    s_det: int
    s_seg: int
    s_trk: int
    s_joint: int
    importance_uniform: float
    importance_risk_weighted: float
    importance_vru_present: float

    def as_dict(self) -> dict:
        return {
            "seq_name": self.seq_name,
            "labeled_idx": self.labeled_idx,
            "s_det": self.s_det,
            "s_seg": self.s_seg,
            "s_trk": self.s_trk,
            "s_joint": self.s_joint,
            "importance_uniform": self.importance_uniform,
            "importance_risk_weighted": self.importance_risk_weighted,
            "importance_vru_present": self.importance_vru_present,
        }


def aggregate_R_I(records: Sequence[PerFrameRecord]) -> dict[str, dict[str, float]]:
    """Compute R_I for the joint S and each sub-task S, under each
    importance profile, in one pass over the per-frame records.

    Returns
    -------
    {profile_name: {"joint": R_I, "det": R_I, "seg": R_I, "trk": R_I}}
    """
    if not records:
        empty = {"joint": 0.0, "det": 0.0, "seg": 0.0, "trk": 0.0}
        return {"uniform": dict(empty), "risk_weighted": dict(empty), "vru_present": dict(empty)}

    s_det = np.array([r.s_det for r in records], dtype=np.float64)
    s_seg = np.array([r.s_seg for r in records], dtype=np.float64)
    s_trk = np.array([r.s_trk for r in records], dtype=np.float64)
    s_jnt = np.array([r.s_joint for r in records], dtype=np.float64)

    profiles = {
        "uniform":       np.array([r.importance_uniform        for r in records], dtype=np.float64),
        "risk_weighted": np.array([r.importance_risk_weighted  for r in records], dtype=np.float64),
        "vru_present":   np.array([r.importance_vru_present    for r in records], dtype=np.float64),
    }

    out: dict[str, dict[str, float]] = {}
    for name, I in profiles.items():
        out[name] = {
            "joint": compute_R_I(s_jnt, I),
            "det":   compute_R_I(s_det, I),
            "seg":   compute_R_I(s_seg, I),
            "trk":   compute_R_I(s_trk, I),
            "n_frames":            float(len(records)),
            "n_frames_with_I_gt0": float(int((I > 0).sum())),
            "total_importance":    float(I.sum()),
        }
    return out
