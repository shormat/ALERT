# Dataset: bdd100k_alvaromalfaro — Image Dataset Understanding

**Source**: https://www.kaggle.com/datasets/alvaromalfaro/bdd100k  
**Local path**: `data/raw/bdd100k_alvaromalfaro/`

---

## Overview

The alvaromalfaro Kaggle release of BDD100K provides the full 100k image set (70k train / 10k val / 20k test) together with per-image JSON label files. Each per-image JSON contains **both** detection bounding boxes (`box2d`) and drivable-area / lane polygons (`poly2d`) in a single annotation list.

ALERT uses alvaromalfaro as the **single source dataset** for the frame-level pipeline (detection + drivable area + lane). Videos for tracking come from `bdd100k_mot` (see `bdd100k_mot.md`).

---

## Dataset Totals

| Item | Count |
|---|---|
| Train images | 70,000 |
| Val images | 10,000 |
| Test images | 20,000 |
| **Total images** | **100,000** |
| Train label JSONs | 70,000 |
| Val label JSONs | 10,000 |
| Test label JSONs | 20,000 |
| Image resolution | 1280 × 720 px |
| Image format | JPEG (flat directory, no subfolders) |

---

## Directory Layout

```
data/raw/bdd100k_alvaromalfaro/
└── bdd100k/
    ├── images/
    │   └── 100k/
    │       ├── train/          <- 70,000 images (flat, 1280×720 JPEG)
    │       ├── val/            <- 10,000 images (flat)
    │       └── test/           <- 20,000 images (flat)
    └── labels/
        ├── train/              <- 70,000 JSON files (one per image)
        ├── val/                <- 10,000 JSON files
        └── test/               <- 20,000 JSON files
```

Images are flat — no `trainA/trainB` subdirs (unlike the solesensei release).  
`glob("*.jpg")` returns the entire split. No recursive glob needed.

---

## Label Format

Each label file is named `{stem}.json` (stem = image name without `.jpg`):

```json
{
  "name": "0000f77c-6257be58",
  "frames": [
    {
      "timestamp": 10000,
      "objects": [
        {
          "category": "car",
          "box2d": {"x1": 45.24, "y1": 254.53, "x2": 357.81, "y2": 487.91}
        },
        {
          "category": "area/drivable",
          "poly2d": [[1280.20, 626.37, "L"], [1280.20, 371.83, "L"], ...]
        },
        {
          "category": "lane/road curb",
          "poly2d": [[503.67, 373.14, "L"], [357.80, 374.67, "L"]]
        }
      ]
    }
  ],
  "attributes": {"weather": "clear", "scene": "city street", "timeofday": "daytime"}
}
```

### Schema details

| Field | Value |
|---|---|
| Top-level key | `frames` (single-element list) |
| Annotation list key | `objects` (not `labels`) |
| Image name in JSON | `name` field has **no `.jpg`** extension |
| Polygon vertex format | Flat list `[x, y, "L"\|"C"]` per point |
| Box objects | Have `box2d`, no `poly2d` |
| Polygon objects | Have `poly2d`, no `box2d` |
| Scene attributes | Top-level `attributes` (`weather`, `scene`, `timeofday`) |

### Quick load pattern

```python
from pathlib import Path
import json

LABEL_DIR = Path("data/raw/bdd100k_alvaromalfaro/bdd100k/labels/train")
IMG_DIR   = Path("data/raw/bdd100k_alvaromalfaro/bdd100k/images/100k/train")

data = json.loads((LABEL_DIR / "0000f77c-6257be58.json").read_text())
objects = data["frames"][0]["objects"]
boxes   = [o for o in objects if "box2d"  in o]
polys   = [o for o in objects if "poly2d" in o]
```

---

## Detection Annotations

### Category counts (train split)

| Category | Count |
|---|---|
| car | 714,121 |
| traffic sign | 239,961 |
| traffic light | 186,301 |
| person | 91,435 |
| truck | 30,012 |
| bus | 11,688 |
| bike | 7,227 |
| rider | 4,522 |
| motor | 3,002 |
| train | 136 |
| **Total** | **1,288,405** |

### Bounding box statistics (train)

| Metric | Width (px) | Height (px) |
|---|---|---|
| Mean | 56.6 | 49.9 |
| Median | 28.7 | 29.1 |
| Std | 75.8 | 61.7 |

- Mean aspect ratio: 1.22 (wider than tall on average)
- Mean boxes per image: 18.4 — max: 91

### Class name normalization

alvaromalfaro uses BDD100K v1 naming. ALERT normalizes to canonical names at processing time:

| alvaromalfaro (v1) | ALERT canonical |
|---|---|
| `person` | `pedestrian` |
| `bike` | `bicycle` |
| `motor` | `motorcycle` |
| (all others) | unchanged |

---

## Polygon Annotations

### Drivable area counts (train)

| Category | Count |
|---|---|
| `area/drivable` | 64,050 |
| `area/alternative` | 61,799 |
| `area/unknown` | 2 |

Frames with at least one drivable polygon: **64,218** out of 70,000 (91.7%).

### Lane counts (train)

| Category | Count |
|---|---|
| `lane/single white` | 247,108 |
| `lane/road curb` | 109,868 |
| `lane/crosswalk` | 108,284 |
| `lane/double yellow` | 37,519 |
| `lane/single yellow` | 20,220 |
| `lane/double white` | 5,674 |
| `lane/single other` | 249 |

### Polygon vertex format

Each vertex is a flat 3-element list `[x, y, type]`:
- `"L"` — straight line to this point
- `"C"` — cubic bezier control point

```python
poly2d = [[503.67, 373.14, "L"], [357.80, 374.67, "L"]]
vertices = [[pt[0], pt[1]] for pt in poly2d]  # strip the type character
```

---

## Scene Attributes

Every frame has 100% attribute coverage (`timeofday`, `weather`, `scene`).

### Time of day (train)

| Value | Approx % |
|---|---|
| daytime | ~52% |
| night | ~40% |
| dawn/dusk | ~7% |
| undefined | ~1% |

Used for stratified splitting — see `data/processed/splits/detection_split.json`.

---

## Key Quirks

### 1. Flat images (no CycleGAN subdirs)

Unlike the solesensei release, train images are **not** nested in `trainA/`, `trainB/` subdirs.  
A simple `glob("*.jpg")` returns all 70k images.

### 2. JSON key is `objects`, not `labels`

BDD100K v2 per-image JSON uses `frames[0]["objects"]`, whereas the older solesensei list JSON uses top-level `"labels"`.

### 3. `name` field has no extension

The `name` field is the bare stem (e.g. `"0000f77c-6257be58"`), not `"0000f77c-6257be58.jpg"`.

### 4. Test split has drivable labels

BDD100K's official drivable release only covers train+val. alvaromalfaro is the only public source providing drivable-area polygon labels for the **test** split.

### 5. Class imbalance

`car` accounts for 55% of all detection annotations. `train` has only 136 instances — exclude or weight heavily.

---

## Usage in ALERT Pipeline

This dataset is the **single source** for all frame-level tasks:

| Task | From this dataset |
|---|---|
| Object detection | `box2d` annotations → YOLO `.txt` labels |
| Drivable area segmentation | `area/drivable` + `area/alternative` polygons → PNG masks |
| Lane segmentation (future) | `lane/*` polygons |

Processing is handled by `scripts/process_dataset.py` via `alert.data_processing.detection` and `alert.data_processing.drivable`.

See `docs/datasets/processed.md` for the full processed output layout.
