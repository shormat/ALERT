# ALERT — Dataset Overview

ALERT uses two BDD100K Kaggle releases as raw data sources, processed into a unified `data/processed/` layout before training.

---

## Raw Sources

| Source | Kaggle dataset | Task | Local path |
|---|---|---|---|
| **alvaromalfaro** | [alvaromalfaro/bdd100k](https://www.kaggle.com/datasets/alvaromalfaro/bdd100k) | Detection + drivable segmentation (frame-level) | `data/raw/bdd100k_alvaromalfaro/` |
| **robikscube MOT** | [robikscube/driving-video-with-object-tracking](https://www.kaggle.com/datasets/robikscube/driving-video-with-object-tracking) | Multi-object tracking (video-level) | `data/raw/bdd100k_mot/` |

---

## Summary

### alvaromalfaro — image dataset

100k JPEG frames (1280×720) with per-image JSON labels containing both detection bounding boxes and drivable-area / lane polygons.

- **Train**: 70,000 images
- **Val**: 10,000 images
- **Test**: 20,000 images (drivable polygon labels included — unique among public BDD100K releases)
- **Detection annotations**: 1,288,405 boxes across 10 classes
- **Drivable polygons**: 64,050 `area/drivable` + 61,799 `area/alternative` in train

→ Full reference: `docs/datasets/bdd100k_alvaromalfaro.md`

### robikscube MOT — video dataset

1,000 dashcam `.mov` clips with flat Parquet MOT annotations (bounding boxes + persistent track IDs).

- **Videos with footage**: 961 (39 are listed but have no actual video file)
- **Annotation rows**: 1,922,517 (for the 961 footage videos)
- **Annotated frames**: 190,469 unique frames across 961 videos
- **Container FPS**: ~29.97 fps (H.264); **annotation cadence**: ~5 fps
- **Stride**: `total_container_frames / n_annotated_frames` ≈ 6

→ Full reference: `docs/datasets/bdd100k_mot.md`

---

## Processed Output

The script `scripts/process_dataset.py` converts both raw sources into:

| Output | Path | Used for |
|---|---|---|
| YOLO labels | `data/processed/bdd100k_frames/labels/` | Detection training |
| COCO JSON | `data/processed/bdd100k_frames/instances_*.json` | Detection evaluation |
| Drivable masks | `data/processed/bdd100k_frames/drivable/masks/` | Segmentation training/eval |
| MOT parquet splits | `data/processed/bdd100k_videos/labels/*.parquet` | Tracking training/eval |
| Video symlinks | `data/processed/bdd100k_videos/videos/` | Tracking training |
| Split assignments | `data/processed/splits/` | Reproducibility |

**Detection split** (original BDD100K split, no re-sampling):

| Split | Count | Source |
|---|---|---|
| train | 70,000 | alvaromalfaro `train/` |
| val | 10,000 | alvaromalfaro `val/` |
| test | 20,000 | alvaromalfaro `test/` |

**MOT split** (from 961 footage videos, stratified by `timeofday`, seed 42):

| Split | Count |
|---|---|
| train | 761 |
| val | 100 |
| test | 100 |

→ Full reference: `docs/datasets/processed.md`

---

## Class Taxonomy

All pipeline stages use the **BDD100K 10-class taxonomy**:

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

The alvaromalfaro source uses v1 names (`person`, `bike`, `motor`) — these are normalized to the canonical names above by `alert/data_processing/detection.py`.
