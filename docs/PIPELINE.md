# ALERT — Pipeline Documentation

## Table of Contents

1. [Overview](#1-overview)
2. [Pipeline Diagram](#2-pipeline-diagram)
3. [Data Flow Summary](#3-data-flow-summary)
4. [Stage A — Detection](#4-stage-a--detection)
5. [Stage B — Drivable Segmentation](#5-stage-b--drivable-segmentation)
6. [Stage C — Tracking](#6-stage-c--tracking)
7. [Stage 1 — Voice Alerts](#7-stage-1--voice-alerts)
8. [Stage 2 — Risk Reasoning](#8-stage-2--risk-reasoning)
9. [Stage 3 — Visualization](#9-stage-3--visualization)
10. [Configuration & Toggles](#10-configuration--toggles)
11. [Key Data Structures](#11-key-data-structures)

---

## 1. Overview

ALERT is a modular dashcam understanding pipeline. It reads a video
source frame-by-frame, runs the perception stack to obtain detections,
tracks, and a drivable mask, then chains the result through risk
reasoning, alerts, and visualization:

```
Frame -> Detector -> Tracker -> Segmentor -> Risk -> Alerts -> Visualization -> annotated frame
```

The orchestrator is `ALERTPipeline` in `alert/pipeline.py`. It
instantiates the perception stack plus the three downstream stages, and
calls them in order inside `process_frame()`.

The perception stack produces a unified `PerceptionResult`:

```python
@dataclass
class PerceptionResult:
    detections:      list[Detection]        # always present
    tracked_objects: list[TrackedObject]    # always present
    drivable_mask:   np.ndarray | None      # binary (H, W) uint8; None if seg disabled
    lane_mask:       np.ndarray | None      # reserved
```

### Class taxonomy

All stages report classes in the BDD100K 10-class taxonomy: `pedestrian`,
`rider`, `car`, `truck`, `bus`, `train`, `motorcycle`, `bicycle`,
`traffic light`, `traffic sign`. A class-mapping layer in each detector
wrapper converts model-native labels (e.g. COCO 80) into this set.

---

## 2. Pipeline Diagram

```
VIDEO FILE / WEBCAM
        |
        v
+-----------------------------------------------------------------------+
|                         VideoReader  (cv2)                            |
|              decodes frames one-by-one as BGR numpy arrays            |
+-----------------------------------------------------------------------+
                               |  raw BGR frame
                               v
+-----------------------------------------------------------------------+
|                        ALERTPipeline                              |
|                           pipeline.py                                 |
|                                                                       |
|   PERCEPTION                                                          |
|  +------------------------------------------------------------------+ |
|  |  +------------+   +-------------+   +----------------------+    | |
|  |  |  DETECTOR  |-->|   TRACKER   |   |   DRIVABLE SEGMENTOR |    | |
|  |  |            |   |             |   |                      |    | |
|  |  | YOLOv8-m   |   | ByteTrack   |   | DeepLabv3+ R101      |    | |
|  |  | RT-DETR-L  |   | OC-SORT     |   | SegFormer-B2         |    | |
|  |  +------------+   +-------------+   +----------------------+    | |
|  |                                                                  | |
|  |  PerceptionResult:                                                | |
|  |  +----------------------------------------------------------+    | |
|  |  | detections:      list[Detection]      (BDD100K 10-class)|    | |
|  |  | tracked_objects: list[TrackedObject]                    |    | |
|  |  | drivable_mask:   np.ndarray (H, W) uint8 binary         |    | |
|  |  +----------------------------------------------------------+    | |
|  +------------------------------------------------------------------+ |
|                                     |                                 |
|                                     |  PerceptionResult               |
|                          +----------+----------+                      |
|                          |                     |                      |
|   STAGE 1                v     STAGE 2         v                      |
|  +---------------------------+  +----------------------------------+  |
|  |       VOICE ALERTS        |  |        RISK REASONING            |  |
|  |   alerts/voice_alerts.py  |  |    risk/risk_reasoner.py         |  |
|  |                           |  |                                  |  |
|  |  Rule 1: Pedestrian       |  |  ri = wci * pi(t) * qi(t)        |  |
|  |  crossing                 |  |       * exp(-TTC / lambda)       |  |
|  |  Rule 2: Vehicle          |  |                                  |  |
|  |  approaching              |  |  Scene risk:                     |  |
|  |  Rule 3: Cyclist nearby   |  |  R(t) = 1 - prod(1 - ri)         |  |
|  |                           |  |                                  |  |
|  |  Cooldown: 5 s / rule     |  |  Alert levels:                   |  |
|  |  TTS: pyttsx3 (offline)   |  |    INFO     R(t) < 0.3           |  |
|  |  background thread        |  |    WARNING  R(t) < 0.6           |  |
|  |                           |  |    CRITICAL R(t) >= 0.6          |  |
|  |  Output: alert_text       |  |  Output: RiskResult              |  |
|  +-------------+-------------+  +----------------+-----------------+  |
|                |  alert_text                       |  RiskResult       |
|                +------------------+----------------+                  |
|                                   |                                   |
|   STAGE 3                         v                                   |
|  +----------------------------------------------------------------+   |
|  |                       VISUALIZATION                            |   |
|  |               utils/visualization.py                           |   |
|  |  compose_frame() — 5 layers applied in order:                  |   |
|  |  1) Seg overlay  — colorize mask, blend at alpha = 0.5         |   |
|  |  2) Tracks       — per-ID-colored bounding boxes               |   |
|  |  3) Risk overlay — per-object colored borders + scene badge    |   |
|  |  4) Alert text   — voice alert message in red, top-left        |   |
|  |  5) FPS counter  — green text, top-right corner                |   |
|  |  Output: annotated BGR frame (np.ndarray)                      |   |
|  +----------------------------------------------------------------+   |
+-----------------------------------+-----------------------------------+
                                    |  annotated frame
                          +---------+----------+
                          v                    v
                   +------------+       +------------+
                   | VideoWriter|       | cv2.imshow |
                   | output.mp4 |       |  live win  |
                   +------------+       +------------+
```

---

## 3. Data Flow Summary

```
raw frame
   |
   v
Perception (Detector -> Tracker -> Segmentor)
   |
   +--> PerceptionResult
            +- detections       (BDD100K 10-class)
            +- tracked_objects  (with track_id)
            +- drivable_mask    (binary HxW or None)
                       |
            +----------+----------+
            v                     v
        Alerts             RiskReasoner
     (alert_text)          (RiskResult)
            |                     |
            +----------+----------+
                       v
                  Visualizer
                       |
              annotated frame
                       |
            VideoWriter / imshow
```

The perception stack is the only component that touches the raw frame.
Risk reasoning, alerts, and visualization consume only `PerceptionResult`,
so swapping models within a stage does not require changes downstream.

---

## 4. Stage A — Detection

**Files:** `alert/detection/base.py`, `alert/detection/factory.py`, `alert/detection/models/`
**Entry point:** `build_detector(config)` returns a `Detector` instance.

### Purpose

Locate objects of interest in each raw BGR frame and return their
bounding boxes, class labels, and confidence scores in the BDD100K
10-class taxonomy.

### Models

| Model | Key | Notes |
|---|---|---|
| YOLOv8-m | `yolov8m` | CNN baseline (Ultralytics). Requires NMS (IoU 0.45). |
| RT-DETR-L | `rtdetr-l` | Transformer end-to-end detector (Ultralytics). No NMS. |

Both use the official COCO pretrained checkpoint as initialization, then
are fine-tuned on BDD100K detection (70K train images, 10 classes).

### Configuration (`config/detection/yolov8m.yaml` or `rtdetr_l.yaml`)

| Key | Default | Description |
|---|---|---|
| `model` | `rtdetr-l` | Model variant |
| `weights` | `checkpoints/detection/{model}_bdd100k/best.pt` | BDD100K-fine-tuned weights |
| `confidence_threshold` | `0.35` | Discard detections below this score |
| `nms_iou_threshold` | `0.45` | NMS IoU threshold (YOLOv8 only) |
| `img_size` | `640` | Inference resolution |

### Output

```python
@dataclass
class Detection:
    bbox: tuple[float, float, float, float]   # (x1, y1, x2, y2) pixels
    label: str                                 # e.g. "pedestrian", "car"
    confidence: float                          # [0, 1]
```

### Class hierarchy

```
alert/detection/base.py
  +-- Detector  (ABC)

alert/detection/models/
  +-- yolov8.py   -> YOLOv8Detector   (wraps ultralytics.YOLO)
  +-- rtdetr.py   -> RTDETRDetector   (wraps ultralytics.RTDETR)

alert/detection/factory.py
  +-- build_detector(config) -> Detector
```

---

## 5. Stage B — Drivable Segmentation

**Files:** `alert/segmentation/base.py`, `alert/segmentation/factory.py`, `alert/segmentation/models/`
**Entry point:** `build_segmentor(config)` returns a `Segmentor` instance.

### Purpose

Produce a drivable-area mask for the current frame. The mask is consumed
by the Alerts stage (in-path heuristic) and Risk Reasoning (`pi(t)`
in-path likelihood).

The pipeline trains the BDD100K 3-class drivable task (directly drivable
/ alternatively drivable / background) and collapses to a binary
"drivable vs not" mask before passing downstream.

### Models

| Model | Key | Backend |
|---|---|---|
| DeepLabv3+ R101 | `deeplabv3plus` | segmentation_models_pytorch (ASPP + depthwise-separable decoder) |
| SegFormer-B2 | `segformer` | HuggingFace Transformers (transformer alternative) |

Both are initialized from ImageNet encoders and fine-tuned on BDD100K
drivable masks.

### Inference pipeline (DeepLabv3+)

```
BGR frame
  -> cv2.cvtColor (BGR -> RGB)
  -> ImageNet normalization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
  -> unsqueeze batch dim
  -> model forward pass  ->  logits (1, 3, H', W')
  -> F.interpolate bilinear -> logits (1, 3, H, W)
  -> argmax dim=1  ->  mask (H, W) int32 with values in {0,1,2}
  -> binarise: drivable = (mask == 0) | (mask == 1)   ->  uint8 (H, W)
```

### Output

```python
drivable_mask: np.ndarray   # (H, W) uint8, 1 = drivable, 0 = not
```

The mask is placed into `PerceptionResult.drivable_mask`.

### Class hierarchy

```
alert/segmentation/base.py
  +-- Segmentor  (ABC)

alert/segmentation/models/
  +-- deeplabv3plus.py  -> DeepLabV3PlusSegmentor  (smp)
  +-- segformer.py      -> SegFormerSegmentor      (HuggingFace)

alert/segmentation/factory.py
  +-- build_segmentor(config) -> Segmentor
```

---

## 6. Stage C — Tracking

**Files:** `alert/tracking/base.py`, `alert/tracking/factory.py`, `alert/tracking/kalman.py`, `alert/tracking/models/`
**Entry point:** `build_tracker(config)` returns a `Tracker` instance.

### Purpose

Associate detections across frames to give each object a persistent
integer identity (`track_id`). Both supported trackers are post-hoc (no
learned parameters) — they consume the detector's output directly.

### Models

| Model | Key | Source | Notes |
|---|---|---|---|
| ByteTrack | `bytetrack` | `vendor/ByteTrack` | Two-round IoU association: high-conf dets first, then low-conf dets against unmatched tracks. |
| OC-SORT | `ocsort` | `vendor/OC_SORT` | Observation-centric Kalman: virtual-trajectory recovery + Velocity Direction Consistency cost. |

### Kalman Filter State

Each track maintains an 8-dimensional state vector:

```
state = [cx, cy, w, h, vx, vy, vw, vh]
```

- `cx, cy` — bounding box center
- `w, h` — width and height
- `vx, vy, vw, vh` — velocities (predicted, not observed)

Transition model: constant velocity (`cx += vx`, etc.).

### Two-Stage ByteTrack Association

```
detections
  +-- high_dets  (conf >= 0.5)
  +-- low_dets   (0.1 <= conf < 0.5)

Stage 1: high_dets  --IoU-->  active tracks     (match_thresh = 0.8)
Stage 2: low_dets   --IoU-->  remaining tracks  (catches occluded objects)
Stage 3: unmatched high_dets --IoU--> lost tracks  (track recovery)
Stage 4: unmatched high_dets with conf >= 0.6 -> new tracks
```

Lost tracks are kept for up to `track_buffer = 30` frames, then pruned.

### Camera Motion Compensation (`use_cmc: true`)

```
prev_gray frame
  -> goodFeaturesToTrack (up to 200 corners)
  -> calcOpticalFlowPyrLK to current frame
  -> median(dx), median(dy) across well-tracked corners
  -> shift every track's [cx, cy] by (dx, dy)
  -> then run standard association
```

This corrects ego-motion so that a stationary object whose image shifts
because the car moved is not mis-matched.

### Configuration (`config/tracking/bytetrack.yaml` or `ocsort.yaml`)

| Key | Default | Description |
|---|---|---|
| `method` | `bytetrack` | `bytetrack` or `ocsort` |
| `use_cmc` | `true` | Enable Camera Motion Compensation |
| `track_high_thresh` | `0.5` | High-confidence detection threshold |
| `track_low_thresh` | `0.1` | Low-confidence threshold |
| `new_track_thresh` | `0.6` | Minimum confidence to start a new track |
| `match_thresh` | `0.8` | IoU matching threshold |
| `track_buffer` | `30` | Frames to keep lost tracks alive |

### Output

```python
@dataclass
class TrackedObject:
    bbox: tuple[float, float, float, float]
    label: str
    confidence: float
    track_id: int        # persistent integer, never reassigned
    age: int             # total frames this track has existed
    hits: int            # frames with successful detections
    time_since_update: int  # frames since last matched detection
```

### Class hierarchy

```
alert/tracking/base.py
  +-- Tracker  (ABC)

alert/tracking/kalman.py
  +-- _KalmanBoxTrack, _iou_batch, _linear_assignment  (shared utilities)

alert/tracking/models/
  +-- bytetrack.py  -> ByteTrackAdapter  (wraps vendor/ByteTrack)
  +-- ocsort.py     -> OCSortAdapter     (wraps vendor/OC_SORT)

alert/tracking/factory.py
  +-- build_tracker(config) -> Tracker
```

---

## 7. Stage 1 — Voice Alerts

**File:** `alert/alerts/voice_alerts.py`
**Classes:** `VoiceAlertSystem`, `_TTSWorker`

### Purpose

Apply hand-crafted safety rules to the tracked objects and segmentation
mask. When a rule fires, issue a spoken warning via pyttsx3 in a
background thread (non-blocking).

### Alert Rules

#### Rule 1 — Pedestrian Crossing

Triggered when both conditions hold:

1. Pedestrian bounding box overlaps road pixels (class 0 in seg mask) by
   >= 30%. Fallback when no segmentation: bottom of bbox is below the
   50% vertical line.
2. The pedestrian has moved laterally > 20 px over their tracked history.

Message: `"Pedestrian crossing"`

#### Rule 2 — Vehicle Approaching

Triggered when bounding box area has grown by > 15% relative to the
oldest entry in a 10-frame history.

Applies to: `car`, `truck`, `bus`, `motorcycle`, `train`.

Message: `"Vehicle approaching"`

#### Rule 3 — Cyclist Nearby

Triggered when either condition holds:

- Bounding box width > 20% of frame width (object is large / close).
- Bottom of bounding box is in the lower 30% of the frame.

Message: `"Cyclist nearby"`

### Cooldown

Each alert type has an independent 5-second cooldown.

### TTS Architecture

```
VoiceAlertSystem.update()
  |
  +-> _try_fire(alert_type, message)
           |
           +-> _TTSWorker.speak(message)  <- non-blocking, queues text
                       |
                   background thread
                       +- pyttsx3.engine.say() + runAndWait()
```

The `_TTSWorker` runs a daemon thread with a `deque(maxlen=5)` queue so
old alerts are dropped if the engine is busy.

### Output

Returns `alert_text: str | None` — the message string if any rule
fired, otherwise `None`.

---

## 8. Stage 2 — Risk Reasoning

**File:** `alert/risk/risk_reasoner.py`
**Classes:** `RiskReasoner`, `RiskResult`, `ObjectRisk`

### Purpose

Compute a continuous risk score for each tracked object and aggregate
them into a single scene-level risk score with a categorical alert level.

### Per-Object Risk Formula

```
ri(t) = wci * pi(t) * qi(t) * exp(-TTCi(t) / lambda)
```

| Term | Symbol | Range | Description |
|---|---|---|---|
| Class weight | `wci` | (0, 1] | Severity by object type |
| In-path likelihood | `pi(t)` | [0, 1] | Probability the object is in the ego driving path |
| Detection confidence | `qi(t)` | [0, 1] | Raw detector/tracker confidence |
| TTC surrogate | `TTCi(t)` | [0.1, 99] | Estimated time to collision in frames |
| TTC decay constant | `lambda` | — | Config: `ttc_lambda = 3.0` |

Class weights (`wci`):

| Class | Weight |
|---|---|
| pedestrian | 1.00 |
| rider | 0.90 |
| motorcycle | 0.85 |
| bicycle | 0.85 |
| car / truck / bus / train | 0.70 |
| traffic light / sign | 0.00 |

### In-Path Likelihood `pi(t)`

With segmentation mask: compute the fraction of the bottom third of the
bounding box that contains road pixels (class index 0). This targets the
feet/wheels contact zone with the road surface.

```
bottom_region = mask[y2 - bbox_h//3 : y2, x1 : x2]
pi(t) = count(road pixels) / total pixels in region
```

Without segmentation (fallback): returns `0.8` if the bounding box
center-x is within +-15% of frame width from the center AND the bottom
of the box is in the lower 60% of the frame, otherwise `0.2`.

### Monocular TTC Surrogate `TTCi(t)`

When no depth sensor is available, bounding box area growth rate is used
as a collision proxy:

```
TTC = current_area / (d(area)/dt)
```

- `d(area)/dt` is estimated via linear regression over the last 10
  frames of stored bounding box areas.
- If area is shrinking or flat, TTC is set to `99.0` (safe).
- Result clamped to `[0.1, 99.0]`.

A fast-approaching object grows rapidly on-screen. A small TTC makes
`exp(-TTC/lambda)` -> 1, driving `ri` toward its maximum. A large TTC
makes the exponential -> 0, suppressing the risk.

### Scene-Level Risk

```
R(t) = 1 - prod(1 - ri(t))
```

This is the complement of the probability that all objects are safe
(assuming independent risks). Any single object with high `ri` is
sufficient to raise the scene risk.

### Alert Levels

| Level | Condition | Visual |
|---|---|---|
| `INFO` | R(t) < 0.3 | Green text badge |
| `WARNING` | 0.3 <= R(t) < 0.6 | Orange filled badge |
| `CRITICAL` | R(t) >= 0.6 | Red filled badge |

### Configuration (`config/base.yaml` -> `risk:`)

| Key | Default | Description |
|---|---|---|
| `ttc_lambda` | `3.0` | TTC exponential decay constant |
| `warning_threshold` | `0.30` | R(t) >= this -> WARNING |
| `critical_threshold` | `0.60` | R(t) >= this -> CRITICAL |

### Temporal State

`RiskReasoner` keeps a per-track area history deque (maxlen=10). Stale
entries are pruned on every update. Call `.reset()` between video clips.

### Output

```python
@dataclass
class RiskResult:
    object_risks: list[ObjectRisk]   # per-object decomposition
    scene_risk: float                # R(t) in [0, 1]
    alert_level: str                 # "INFO" | "WARNING" | "CRITICAL"
    most_critical_id: int | None     # track_id of argmax ri(t)
    most_critical_label: str | None

@dataclass
class ObjectRisk:
    track_id: int
    label: str
    risk_score: float           # ri(t)
    in_path_likelihood: float   # pi(t)
    ttc_surrogate: float        # TTCi(t)
    confidence: float           # qi(t)
    class_weight: float         # wci
```

---

## 9. Stage 3 — Visualization

**File:** `alert/utils/visualization.py`
**Class:** `Visualizer`
**Method:** `compose_frame(frame, perception, alert_text, fps, risk_result) -> np.ndarray`

### Purpose

Compose a single annotated BGR frame from all pipeline outputs, suitable
for writing to the output video or displaying live.

### Layer Order in `compose_frame()`

| Layer | Description |
|---|---|
| 1. Segmentation overlay | Map each pixel's class index to the Cityscapes color palette (19 RGB colors), then blend with original frame at `alpha = 0.5`. |
| 2. Tracks / Detections | If tracking is active: draw per-ID-colored bounding boxes with `"ID:N label"` text. Otherwise draw raw detections in green. |
| 3. Risk overlay | Per-object: `ri > 0.6` -> red border + `"(!)"` label suffix; `ri > 0.3` -> orange border. Scene badge top-left: filled red (CRITICAL) / orange (WARNING) / plain green text (INFO). |
| 4. Alert text | Voice alert message in red at top-left (visible for 3 s after trigger). |
| 5. FPS counter | Current frame rate in green at top-right. |

### Segmentation Palette (Cityscapes)

| Class | Color |
|---|---|
| road | purple |
| sidewalk | magenta |
| building | dark grey |
| person | crimson |
| car | dark blue |
| sky | steel blue |
| vegetation | olive green |

---

## 10. Configuration & Toggles

The pipeline is composed via per-stage configs:

```bash
uv run python scripts/run_pipeline.py \
    --detection-config    config/detection/rtdetr_l.yaml \
    --tracking-config     config/tracking/bytetrack.yaml \
    --segmentation-config config/segmentation/deeplabv3plus_resnet101.yaml \
    --pipeline-config     config/pipeline/default.yaml
```

Toggles for the downstream stages live in `config/pipeline/default.yaml`:

```yaml
pipeline:
  enable_segmentation: true   # if false, drivable_mask is None
  enable_alerts:       true
  enable_risk:         true
```

Degradation behaviour:

| Toggle | Effect when off |
|---|---|
| `enable_segmentation` | `drivable_mask = None`; risk falls back to ego-lane corridor heuristic. |
| `enable_alerts` | No voice warnings; `alert_text` is always `None`. |
| `enable_risk` | No risk overlay; visualization draws tracks only. |

Other pipeline keys:

| Key | Default | Description |
|---|---|---|
| `input` | `null` | Video file path, `"webcam"`, or camera integer index |
| `output` | `outputs/output.mp4` | Output video path |
| `show_display` | `true` | Show live cv2 window (press `q` to quit) |
| `output_fps` | `null` | Matches input FPS if null |
| `output_codec` | `mp4v` | FourCC codec for output video |
| `log_fps` | `true` | Print FPS + counts every 30 frames |
| `max_frames` | `null` | Cap number of frames processed |

---

## 11. Key Data Structures

```
FrameResult                        <- returned by process_frame()
  +- annotated_frame  np.ndarray   <- ready-to-write output frame
  +- perception       PerceptionResult
  +- alert_text       str | None
  +- risk             RiskResult | None

PerceptionResult
  +- detections        list[Detection]
  +- tracked_objects   list[TrackedObject]
  +- drivable_mask     np.ndarray | None     # (H, W) uint8 binary

Detection
  +- bbox        (x1, y1, x2, y2)
  +- label       str   <- BDD100K 10-class taxonomy
  +- confidence  float

TrackedObject
  +- bbox              (x1, y1, x2, y2)
  +- label             str
  +- confidence        float
  +- track_id          int   <- persistent across frames
  +- age               int
  +- hits              int
  +- time_since_update int

RiskResult
  +- object_risks       ObjectRisk[]
  +- scene_risk         float  [0, 1]
  +- alert_level        "INFO" | "WARNING" | "CRITICAL"
  +- most_critical_id   int | None
  +- most_critical_label str | None

ObjectRisk
  +- track_id           int
  +- label              str
  +- risk_score         float  ri(t)
  +- in_path_likelihood float  pi(t)
  +- ttc_surrogate      float  TTCi(t)
  +- confidence         float  qi(t)
  +- class_weight       float  wci
```
