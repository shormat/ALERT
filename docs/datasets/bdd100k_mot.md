# Dataset: bdd100k_mot — Video Dataset Understanding

**Source**: https://www.kaggle.com/datasets/robikscube/driving-video-with-object-tracking  
**Local path**: `data/raw/bdd100k_mot/`

---

## Overview

The robikscube Kaggle release packages 1,000 BDD100K training videos in `.mov` format together with flat CSV/Parquet label files containing multi-object tracking (MOT) annotations. Labels include per-frame bounding boxes with persistent track IDs, enabling temporal sequence training for object trackers.

All videos come from the BDD100K **training** split. No official val/test split is provided — a custom split is created by `scripts/process_dataset.py`.

---

## Dataset Totals

| Item | Count |
|---|---|
| Video files (.mov) | 1,000 |
| Videos with actual footage (`haveVideo=True`) | 961 |
| Videos with no footage (`haveVideo=False`) | 39 |
| Total annotation rows (all 1,000 videos) | 2,890,846 |
| Annotation rows for footage videos | 1,922,517 |
| Total annotated frames (unique) | 190,469 |
| Average annotations per frame | 10.1 |
| Unique videos used by ALERT | 961 |

---

## Directory Layout

```
data/raw/bdd100k_mot/
├── mot_labels.csv                                <- 2,890,846 rows
├── mot_labels.parquet                            <- same data, columnar format (preferred)
└── bdd100k_videos_train_00/
    └── bdd100k/
        └── videos/
            └── train/
                ├── 0000f77c-6257be58.mov         <- H.264 .mov, ~19 MB each
                ├── 0000f77c-62c2a288.mov
                └── ... (1,000 .mov files)
```

---

## Label Format

Both files carry the same flattened annotation table. Use the Parquet for performance:

```python
import pandas as pd
df = pd.read_parquet("data/raw/bdd100k_mot/mot_labels.parquet")
df = df[df["haveVideo"] == True]   # 961 videos, 1,922,517 rows
```

### Column reference

| Column | Type | Description |
|---|---|---|
| `name` | str | Frame image name: `{videoName}-{frameIndex+1:07d}.jpg` (1-indexed) |
| `videoName` | str | Video stem, e.g., `01c71072-718028b8` |
| `frameIndex` | int | 0-based annotation frame index within the video |
| `id` | float | Persistent track ID (unique within a video) |
| `category` | str | Object class (see class list below) |
| `attributes.crowd` | bool | Crowd region flag |
| `attributes.occluded` | bool | Object is occluded |
| `attributes.truncated` | bool | Object is truncated by frame boundary |
| `box2d.x1` | float | Left edge (pixels, 1280×720 frame) |
| `box2d.x2` | float | Right edge |
| `box2d.y1` | float | Top edge |
| `box2d.y2` | float | Bottom edge |
| `haveVideo` | bool | Whether the corresponding `.mov` file has actual footage |

**Note**: `name` uses a **1-based** 7-digit zero-padded index (`frameIndex=0` → `0000001`).

### Sample rows

```
videoName          frameIndex  id       category    occluded  x1       x2        y1       y2
01c71072-718028b8  0           89537.0  car         True      825.17   1003.09   355.01   418.20
01c71072-718028b8  0           89538.0  car         True      484.30   700.46    346.70   424.85
01c71072-718028b8  0           89539.0  pedestrian  True      645.59   663.88    338.38   358.34
```

### Object classes (haveVideo=True)

| Class | Annotation count |
|---|---|
| car | 1,532,819 |
| pedestrian | 181,484 |
| truck | 76,391 |
| bus | 35,293 |
| bicycle | 25,046 |
| rider | 16,892 |
| other vehicle | 15,764 |
| motorcycle | 10,802 |
| other person | 3,978 |
| trailer | 3,510 |
| train | 538 |

Class names follow BDD100K v2 naming (`pedestrian`, `bicycle`, `motorcycle`) — no normalization needed when consuming raw MOT data.

**However**, 3 MOT classes are absent from the detection 10-class taxonomy and must be remapped for tracking evaluation:

| MOT-only class | Map to (detection class) | Ann count |
|---|---|---|
| `other vehicle` | `car` | 15,764 |
| `other person` | `pedestrian` | 3,978 |
| `trailer` | `truck` | 3,510 |

Additionally, `traffic light` and `traffic sign` appear in the detection taxonomy but have **zero annotations in MOT GT** — filter them from tracker predictions when computing HOTA/MOTA/IDF1.

Tracking eval uses 8 canonical classes: `pedestrian`, `rider`, `car`, `truck`, `bus`, `train`, `motorcycle`, `bicycle`.

---

## Video Statistics (haveVideo=True)

| Metric | Value |
|---|---|
| Videos with footage | 961 |
| Annotated frames per video — min | 102 |
| Annotated frames per video — max | 263 |
| Annotated frames per video — mean | 198.2 |
| Unique tracks per video — min | 10 |
| Unique tracks per video — max | 378 |
| Unique tracks per video — mean | 79.0 |
| Track length — min | 1 |
| Track length — max | 263 |
| Track length — mean | 25.3 |
| Track length — median | 15 |

---

## Critical: Container FPS vs Annotation FPS

**The `.mov` files are H.264 containers stored at ~29.97 fps; annotations cover only ~5 fps.**

This means the annotation `frameIndex` (0, 1, 2, …) does **not** correspond directly to the container frame number. For a 30 fps video with 198 annotated frames covering ~40 seconds, the container holds ~1200 raw frames. The ratio (stride) is:

```
stride = total_container_frames / n_annotated_frames   (~6 for a 30fps video)
```

To map annotation index → container frame number:

```python
container_frame = round(ann_idx * stride)
```

### Why `cap.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)` is WRONG

`CAP_PROP_POS_FRAMES` treats `frameIndex` as a container frame number. With stride ≈ 6, seeking to `frameIndex=10` lands on container frame 10 instead of container frame 60. The resulting frame will not match the annotation.

Additionally, H.264 on Linux snaps seeks to the nearest key-frame regardless of the requested position, making seek-based frame extraction unreliable for non-key frames.

### Correct approach: sequential read with pre-computed targets

```python
import cv2
from pathlib import Path
import pandas as pd

VIDEO_DIR = Path("data/raw/bdd100k_mot/bdd100k_videos_train_00/bdd100k/videos/train")
df = pd.read_parquet("data/raw/bdd100k_mot/mot_labels.parquet")
df = df[df["haveVideo"] == True]

def read_annotated_frames(video_name: str, video_df, n: int = 10) -> dict:
    """Return {ann_idx: frame_rgb} for the first n annotated frames."""
    video_path = VIDEO_DIR / f"{video_name}.mov"
    cap   = cv2.VideoCapture(str(video_path))
    n_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_ann = video_df["frameIndex"].nunique()
    stride = n_vid / n_ann

    ann_indices = sorted(video_df["frameIndex"].unique())[:n]
    keep = {round(idx * stride): idx for idx in ann_indices}   # container_frame -> ann_idx

    collected, frame_num = {}, 0
    max_target = max(keep)
    while frame_num <= max_target:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num in keep:
            collected[keep[frame_num]] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_num += 1
    cap.release()
    return collected
```

---

## Video Access — Container Properties

```python
cap = cv2.VideoCapture(str(VIDEO_DIR / "0000f77c-6257be58.mov"))
fps  = cap.get(cv2.CAP_PROP_FPS)              # ~29.97 (container FPS, NOT annotation FPS)
w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 1280
h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 720
nf   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # total container frames
cap.release()

video_df = df[df["videoName"] == "0000f77c-6257be58"]
n_ann   = video_df["frameIndex"].nunique()
stride  = nf / n_ann   # ~6
ann_fps = n_ann / (nf / fps)   # ~5 fps
```

---

## Split Strategy

ALERT creates a custom 761 / 100 / 100 split (train / val / test) from the 961 footage videos, stratified by `timeofday` joined from the alvaromalfaro detection labels (perfect 1-to-1 video name → label file match).

The split is persisted to `data/processed/splits/mot_split.json` and applied by `scripts/process_dataset.py`.

See `docs/datasets/processed.md` for the processed output layout.

### Splitting rules

- Split at the **video level**, never at the track or frame level.
- Stratify by `timeofday` (daytime / night / dawn-dusk / undefined).
- Exclude `haveVideo=False` videos before splitting.
- Fixed seed 42 for reproducibility.

---

## Usage in ALERT Pipeline

This dataset is the **tracking evaluation source** for ByteTrack and OC-SORT. Neither tracker has learnable parameters — this data is used only for evaluating MOTA/IDF1/HOTA, not for training.

```
Frame sequence → Detection → Tracking
                                 ^
                         bdd100k_mot
                  (temporal bbox sequences
                   with persistent track IDs)
```

```python
df = pd.read_parquet("data/raw/bdd100k_mot/mot_labels.parquet")
df = df[df["haveVideo"]].sort_values(["videoName", "frameIndex"])

for video_name, clip_df in df.groupby("videoName"):
    for frame_idx, frame_df in clip_df.groupby("frameIndex"):
        boxes     = frame_df[["box2d.x1", "box2d.y1", "box2d.x2", "box2d.y2"]].values
        track_ids = frame_df["id"].values
        classes   = frame_df["category"].values
        # feed (boxes, track_ids) as ground-truth MOT supervision
```
