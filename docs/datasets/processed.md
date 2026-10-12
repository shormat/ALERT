# Processed Dataset

**Script**: `scripts/process_dataset.py`  
**Output root**: `data/processed/`

---

## Overview

The processing pipeline converts the two raw datasets into training-ready artifacts:

| Raw source | Processed output | Task |
|---|---|---|
| `bdd100k_alvaromalfaro` | `data/processed/bdd100k_frames/` | Detection + drivable segmentation |
| `bdd100k_mot` | `data/processed/bdd100k_videos/` | Multi-object tracking |

Split assignments are persisted under `data/processed/splits/` for reproducibility.

---

## Running the Pipeline

```bash
# Full processing (70k/10k/20k det, 761/100/100 mot)
uv run python scripts/process_dataset.py

# Smoke test (80/10/10 det, 12/3/3 mot)
uv run python scripts/process_dataset.py --tiny

# Run only specific steps
uv run python scripts/process_dataset.py --steps splits,detection
uv run python scripts/process_dataset.py --steps drivable,mot

# Parallel workers (default: 8)
uv run python scripts/process_dataset.py --workers 16
```

Available steps: `splits`, `detection`, `drivable`, `mot`.

---

## Output Directory Layout

```
data/processed/
├── splits/
│   ├── detection_split.json           {stem: "train"|"val"|"test"}  — 100,000 entries
│   └── mot_split.json                 {videoName: "train"|"val"|"test"}  — 961 entries
│
├── bdd100k_frames/                    frame-level pipeline data
│   ├── bdd100k.yaml                   Ultralytics dataset config
│   ├── images/{train,val,test}/       symlinks to raw .jpg files
│   ├── labels/{train,val,test}/       YOLO .txt files (one per image)
│   ├── instances_{train,val,test}.json   COCO-format annotations
│   └── drivable/
│       ├── polygons/{train,val,test}/ per-image drivable polygon JSON
│       ├── masks/{train,val,test}/    single-channel class-ID PNGs
│       └── colormaps/{train,val,test}/ BGR visualization PNGs
│
└── bdd100k_videos/                    tracking pipeline data
    ├── labels/{train,val,test}.parquet   MOT annotations filtered by split
    └── videos/{train,val,test}/       symlinks to raw .mov files
```

---

## Splits

### Detection split

Original BDD100K split — no re-sampling or stratification applied.

| Split | Source | Count |
|---|---|---|
| train | alvaromalfaro `train/` | 70,000 |
| val | alvaromalfaro `val/` | 10,000 |
| test | alvaromalfaro `test/` | 20,000 |

```python
import json
split = json.loads(open("data/processed/splits/detection_split.json").read())
# {"0000f77c-6257be58": "train", "0000f77c-62c2a288": "val", ...}
```

### MOT split

| Split | Count | Strategy |
|---|---|---|
| train | 761 | stratified by `timeofday`, seed 42 |
| val | 100 | stratified by `timeofday`, seed 42 |
| test | 100 | stratified by `timeofday`, seed 42 |

Only the 961 `haveVideo=True` videos are used. Stratification uses `timeofday` joined from the alvaromalfaro train labels (perfect 1-to-1 video name match).

```python
import json
mot_split = json.loads(open("data/processed/splits/mot_split.json").read())
# {"01c71072-718028b8": "train", ...}
```

---

## bdd100k_frames

### File counts

| Split | images | labels | masks | polygons | colormaps |
|---|---|---|---|---|---|
| train | 70,000 | 70,000 | 70,000 | 70,000 | 70,000 |
| val | 10,000 | 10,000 | 10,000 | 10,000 | 10,000 |
| test | 20,000 | 20,000 | 20,000 | 20,000 | 20,000 |

### images/

Symlinks to the raw `.jpg` files under `data/raw/bdd100k_alvaromalfaro/`. Resolution: 1280×720.

### labels/ — YOLO format

One `.txt` per image, one detection per line:

```
<class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>
```

All coordinates are normalized to [0, 1] by image width/height. Empty file = no detections.

**Class mapping** (IDs match `bdd100k.yaml`):

| ID | Class |
|---|---|
| 0 | pedestrian |
| 1 | rider |
| 2 | car |
| 3 | truck |
| 4 | bus |
| 5 | train |
| 6 | motorcycle |
| 7 | bicycle |
| 8 | traffic light |
| 9 | traffic sign |

```python
# Read one YOLO label
from pathlib import Path
lines = Path("data/processed/bdd100k_frames/labels/train/0000f77c-6257be58.txt").read_text().splitlines()
for line in lines:
    cls, cx, cy, w, h = map(float, line.split())
```

### bdd100k.yaml — Ultralytics config

```yaml
path: /absolute/path/to/data/processed/bdd100k_frames
train: images/train
val:   images/val
test:  images/test
names:
  0: pedestrian
  1: rider
  2: car
  3: truck
  4: bus
  5: train
  6: motorcycle
  7: bicycle
  8: traffic light
  9: traffic sign
```

Pass directly to Ultralytics training:

```python
from ultralytics import YOLO
model = YOLO("yolov8m.pt")
model.train(data="data/processed/bdd100k_frames/bdd100k.yaml", epochs=50)
```

### instances_{split}.json — COCO format

Standard COCO object detection format consumable by `pycocotools`:

```json
{
  "info": {...},
  "licenses": [...],
  "categories": [{"id": 1, "name": "pedestrian"}, ...],
  "images": [{"id": 1, "file_name": "0000f77c-6257be58.jpg", "width": 1280, "height": 720}, ...],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 9, "bbox": [x, y, w, h], "area": ..., "iscrowd": 0, "segmentation": []}, ...]
}
```

Note: `bbox` is in COCO format `[x_min, y_min, width, height]` in absolute pixels (not normalized).

```python
from pycocotools.coco import COCO
coco = COCO("data/processed/bdd100k_frames/instances_train.json")
```

| Split | images | annotations |
|---|---|---|
| train | 70,000 | ~1,288,000 |
| val | 10,000 | ~183,000 |
| test | 20,000 | ~365,000 |

### drivable/polygons/ — source-of-truth polygon JSON

One JSON per image; stores the cleaned polygon vertices extracted from the raw label:

```json
{
  "name": "0000f77c-6257be58",
  "polygons": [
    {
      "category": "area/drivable",
      "vertices": [[0.0, 644.9], [529.8, 563.5], ...]
    },
    {
      "category": "area/alternative",
      "vertices": [...]
    }
  ]
}
```

Vertices are in absolute pixel coordinates (float). Polygons with fewer than 3 vertices are excluded.

### drivable/masks/ — class-ID PNG

Single-channel `uint8` PNG, shape `(720, 1280)`.

| Pixel value | Class |
|---|---|
| 0 | `area/drivable` (direct lane) |
| 1 | `area/alternative` (adjacent drivable lanes) |
| 2 | background |

When both `area/drivable` and `area/alternative` overlap, `area/drivable` (ID 0) takes priority.

```python
import cv2, numpy as np
mask = cv2.imread("data/processed/bdd100k_frames/drivable/masks/train/0000f77c-6257be58.png",
                  cv2.IMREAD_UNCHANGED)
# mask.shape == (720, 1280), mask.dtype == uint8
# unique values: subset of {0, 1, 2}
```

### drivable/colormaps/ — BGR visualization PNG

3-channel BGR `uint8` PNG for visual inspection:

| Class | BGR color |
|---|---|
| `area/drivable` | (0, 0, 255) — red |
| `area/alternative` | (255, 0, 0) — blue |
| background | (0, 0, 0) — black |

```python
cmap = cv2.imread("data/processed/bdd100k_frames/drivable/colormaps/train/0000f77c-6257be58.png")
# overlay on original image:
overlay = cv2.addWeighted(original_bgr, 0.7, cmap, 0.3, 0)
```

---

## bdd100k_videos

### labels/{split}.parquet

The raw `mot_labels.parquet` filtered and split. Schema is identical to the raw file:

| Column | Type | Description |
|---|---|---|
| `videoName` | str | Video stem |
| `frameIndex` | int64 | 0-based annotation frame index |
| `id` | str | Track ID (zero-padded 8-digit string, e.g. `"00089537"`) |
| `category` | str | Object class |
| `attributes.occluded` | bool | |
| `attributes.truncated` | bool | |
| `attributes.crowd` | bool | |
| `box2d.x1/x2/y1/y2` | float64 | Pixel coordinates |
| `haveVideo` | bool | Always `True` in processed output |

Note: `id` is stored as a zero-padded string in the processed Parquet (unlike the raw float).

```python
import pandas as pd
train_df = pd.read_parquet("data/processed/bdd100k_videos/labels/train.parquet")
# 761 videos, ~1,524,668 rows
```

| Split | Videos | Rows |
|---|---|---|
| train | 761 | ~1,524,668 |
| val | 100 | ~198,938 |
| test | 100 | ~198,911 |

### videos/{split}/

Symlinks to the raw `.mov` files. Use together with the split Parquet for training:

```python
from pathlib import Path
import pandas as pd, cv2

VIDEO_DIR = Path("data/processed/bdd100k_videos/videos/train")
labels    = pd.read_parquet("data/processed/bdd100k_videos/labels/train.parquet")

for video_name in labels["videoName"].unique():
    video_path = VIDEO_DIR / f"{video_name}.mov"
    video_df   = labels[labels["videoName"] == video_name].sort_values("frameIndex")
    # sequential read — see bdd100k_mot.md for correct frame alignment
```

---

## Regenerating the Processed Data

Processed outputs are fully deterministic given the raw data and the fixed seed (42). To regenerate from scratch:

```bash
# Remove processed outputs
rm -rf data/processed/bdd100k_frames data/processed/bdd100k_videos data/processed/splits

# Rerun
uv run python scripts/process_dataset.py
```

Approximate wall-clock time with 8 workers: ~15 minutes for the full 100k frame dataset.
