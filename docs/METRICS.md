# ALERT — Evaluation Metrics

All three building blocks report every standard metric so that the choice of primary metric can be deferred. Recommended primary metrics are marked **bold** — use these for tables and plots in the paper.

---

## 1. Detection

Computed via Ultralytics `.val()` at IoU NMS threshold 0.6, `conf=0.001`.

### Metrics

| Metric | Formula / Definition | Range | Notes |
|---|---|---|---|
| **mAP50-95** | Mean Average Precision averaged over 10 IoU thresholds (0.50, 0.55, …, 0.95), then over classes | [0, 1] | COCO and BDD100K official standard. Penalises poor localisation. |
| **mAP50** | mAP at IoU ≥ 0.5 only | [0, 1] | Easier to compare with older papers (VOC era). |
| mAP75 | mAP at IoU ≥ 0.75 only | [0, 1] | Strict localisation; rarely the primary metric but useful as a sanity check. |
| Precision | TP / (TP + FP) at best-F1 threshold | [0, 1] | Per-class and mean. |
| Recall | TP / (TP + FN) at best-F1 threshold | [0, 1] | Per-class and mean. |
| F1 | 2 × P × R / (P + R) | [0, 1] | Harmonic mean of precision and recall. |
| Per-class AP50-95 | AP50-95 for each of the 10 BDD100K classes | [0, 1] | Important for `pedestrian` (highest risk weight in the pipeline). |

### Recommendation

| Priority | Metric | Reason |
|---|---|---|
| Primary (paper table) | **mAP50-95** | BDD100K official; strict enough to reveal localisation quality. |
| Primary (comparison) | **mAP50** | Keep alongside mAP50-95 for comparability with ByteTrack / OC-SORT papers that only report mAP50. |
| Diagnostic | Per-class AP50-95 for `pedestrian` and `car` | These two classes drive risk scores (`w_pedestrian=1.0`, `w_car=0.7`). A model with low pedestrian AP will underperform on risk even if overall mAP is acceptable. |

---

## 2. Segmentation

Evaluated on BDD100K drivable-area masks.  
Three classes: `0=direct` (main drivable lane), `1=alt` (alternative drivable), `2=background`.

### Metrics

| Metric | Formula | Range | Notes |
|---|---|---|---|
| **mIoU** | Mean of per-class IoU; IoU_c = TP_c / (TP_c + FP_c + FN_c) | [0, 1] | BDD100K standard. |
| Per-class IoU | IoU for each of `direct`, `alt`, `background` | [0, 1] | `direct` IoU is the operationally critical number because it drives `p_i(t)` (in-path likelihood) in the risk formula. |
| Pixel Accuracy | (Σ TP_c) / total pixels | [0, 1] | **Misleadingly high** — background dominates (~80% of pixels). Not recommended as primary metric. |
| Mean Pixel Accuracy | Mean of per-class PA; PA_c = TP_c / GT_c | [0, 1] | Corrects class-frequency bias of pixel accuracy. |
| Frequency-weighted IoU | Σ_c (freq_c × IoU_c) | [0, 1] | Weighted by GT pixel frequency. Still biased toward background; use as a secondary cross-check. |

### Recommendation

| Priority | Metric | Reason |
|---|---|---|
| Primary (paper table) | **mIoU** | BDD100K standard; class-balanced. |
| Primary (diagnostic) | **IoU[direct]** | This class feeds directly into the risk formula. A model could have high mIoU while having poor `direct` IoU — that would silently degrade pipeline risk scores. |
| Avoid as primary | Pixel Accuracy | Dominated by background; a trivial "predict all background" model gets ~80% PA. |

---

## 3. Tracking

Evaluated on BDD100K MOT sequences.  
GT filtered to categories: `pedestrian, rider, car, truck, bus, motorcycle, bicycle`.  
Matching: Hungarian algorithm at IoU threshold (default 0.5 for MOTA/IDF1).

### Metrics

#### CLEAR MOT (Bernardin & Stiefelhagen, 2008)

| Metric | Formula | Range | Notes |
|---|---|---|---|
| **MOTA** | 1 − (FP + FN + IDSW) / Σ GT_t | (−∞, 1] | Combines detection errors and identity switches into a single number. Can be negative for poor trackers. |
| MOTP | (Σ matched IoU) / total matches | [0, 1] | Average localisation quality of matched pairs (higher = better). Independent of MOTA. |
| IDSW | Raw count of identity switches | ≥ 0 | Lower = better identity consistency. |
| FP | False positive detections (total) | ≥ 0 | |
| FN | False negatives / missed detections | ≥ 0 | |

#### Identity Metrics

| Metric | Formula | Range | Notes |
|---|---|---|---|
| **IDF1** | 2 × IDTP / (2×IDTP + IDFP + IDFN) | [0, 1] | ID-aware F1. Rewards trackers that maintain consistent identities over long stretches. IDTP = matched GT/pred frame pairs under optimal global assignment. |

#### Track Coverage

| Metric | Definition | Notes |
|---|---|---|
| MT | % of GT tracks tracked ≥ 80% of their lifespan | Higher = better; complementary to IDF1. |
| PT | % of GT tracks tracked 20–80% | Partially tracked. |
| ML | % of GT tracks tracked ≤ 20% of their lifespan | Lower = better. |

#### HOTA (Luiten et al., 2020) — Current Standard

| Metric | Formula | Range | Notes |
|---|---|---|---|
| **HOTA** | √(DetA × AssA) averaged over α ∈ {0.05, 0.10, …, 0.95} | [0, 1] | Official BDD100K MOT metric since 2020. Balances detection and association quality. |
| DetA | TP(α) / (TP(α) + FP(α) + FN(α)) | [0, 1] | Detection sub-score of HOTA (analogous to Jaccard/IoU at the detection level). |
| AssA | (1/TP) Σ_{c∈TP} A(c) | [0, 1] | Association sub-score. A(c) = Jaccard between the set of TPs in the same GT track as c and the set in the same predicted track as c. |
| HOTA@0.5 | HOTA at α = 0.5 | [0, 1] | Single-threshold variant for easy comparison. |
| HOTA@0.75 | HOTA at α = 0.75 | [0, 1] | Strict localisation variant. |

### Recommendation

| Priority | Metric | Reason |
|---|---|---|
| Primary (paper table) | **HOTA** | BDD100K official standard; the only metric that fairly balances detection quality and identity consistency — MOTA can be gamed by improving detection at the cost of ID, and IDF1 vice versa. |
| Secondary | **IDF1** | Widely cited in ByteTrack, OC-SORT, StrongSORT papers — keep for comparability. |
| Secondary | **MOTA** | Older standard; still expected by many reviewers. Report alongside HOTA. |
| Diagnostic | DetA vs AssA split | If HOTA is lower than expected, DetA vs AssA reveals whether the bottleneck is the detector or the association logic. |
| Avoid as primary | MT / ML alone | These are useful for intuition but do not capture localisation quality or overall throughput. |

---

## Summary Table (recommended paper layout)

### Detection

| Model | mAP50 | mAP50-95 | AP[pedestrian] | AP[car] |
|---|---|---|---|---|
| YOLOv8m | | | | |
| RT-DETR-L | | | | |

### Segmentation

| Model | mIoU | IoU[direct] | IoU[alt] |
|---|---|---|---|
| DeepLabV3+ | | | |
| SegFormer-B2 | | | |

### Tracking

| Det + Tracker | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
|---|---|---|---|---|---|---|
| YOLOv8m + ByteTrack | | | | | | |
| YOLOv8m + OC-SORT | | | | | | |
| RT-DETR-L + ByteTrack | | | | | | |
| RT-DETR-L + OC-SORT | | | | | | |

---

## Implementation

- Detection: `scripts/eval_detection.py` — metrics from `results.box.*` (Ultralytics).
- Segmentation: `scripts/eval_segmentation.py` — pixel-level accumulation over all samples.
- Tracking: `scripts/eval_tracking.py` + `alert/tracking/metrics.py`
  - GT loaded from `data/processed/bdd100k_videos/labels/<split>.parquet`.
  - Hungarian matching via `scipy.optimize.linear_sum_assignment`.
  - HOTA computed from scratch over 19 alpha thresholds.
