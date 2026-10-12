# ALERT — Runbook

Step-by-step instructions for setting up the environment, downloading raw
data, training models, and running the pipeline.

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Environment Setup](#2-environment-setup)
3. [Credentials Setup (.env)](#3-credentials-setup-env)
4. [Download Raw Data](#4-download-raw-data)
5. [Pretrained Checkpoints](#5-pretrained-checkpoints)
6. [Prepare Datasets](#6-prepare-datasets)
7. [Run the Pipeline](#7-run-the-pipeline)
8. [Training](#8-training)
9. [Evaluation](#9-evaluation)
10. [Live Webcam (Windows)](#10-live-webcam-windows)
11. [Pipeline Ranking](#11-pipeline-ranking)

---

## 1. Prerequisites

| Tool | Version | Install |
|---|---|---|
| Python | >= 3.11 | https://www.python.org/downloads/ |
| git | any | https://git-scm.com |
| uv (recommended) | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| CUDA toolkit | 12.1 | https://developer.nvidia.com/cuda-downloads |

Verify:

```bash
python --version
git --version
uv --version
nvidia-smi
```

---

## 2. Environment Setup

### Option A — uv

```bash
git clone https://github.com/shormat/ALERT.git
cd ALERT
uv sync
```

All packages are installed into `.venv/`. Prefix every script with `uv run`:

```bash
uv run python scripts/run_pipeline.py ...
uv run pytest tests/
```

To add a new dependency:

```bash
uv add <package-name>
```

### Option B — pip

```bash
git clone https://github.com/shormat/ALERT.git
cd ALERT
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

With the venv active, run scripts directly:

```bash
python scripts/run_pipeline.py ...
pytest tests/
```

### Vendor libraries

`vendor/ByteTrack/` and `vendor/OC_SORT/` contain the upstream tracker code.
If they are not present after cloning, fetch them manually:

```bash
mkdir -p vendor
git clone https://github.com/ifzhang/ByteTrack.git vendor/ByteTrack
git clone https://github.com/noahcao/OC_SORT.git   vendor/OC_SORT
```

---

## 3. Credentials Setup (.env)

Copy the example file and fill in the credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```
KAGGLE_USERNAME=<kaggle_username>
KAGGLE_KEY=<kaggle_api_key>
DATA_PROCESSED_ROOT=data/processed
OUTPUT_ROOT=outputs
```

To obtain Kaggle credentials, log in at https://www.kaggle.com and use
**Settings -> API -> Create New Token**. Accept the dataset licenses on
Kaggle in the browser before downloading programmatically:

- https://www.kaggle.com/datasets/alvaromalfaro/bdd100k
- https://www.kaggle.com/datasets/robikscube/driving-video-with-object-tracking

---

## 4. Download Raw Data

Two Kaggle datasets are required. Download into `data/raw/`.

```bash
uv run python scripts/download_kaggle_alvaromalfaro_bdd100k.py
uv run python scripts/download_kaggle_driving_video_mot.py
```

Verify:

```bash
find data/raw/ -mindepth 2 -maxdepth 2 | sort
du -sh data/raw/*/
```

Expected layout:

```
data/raw/bdd100k_alvaromalfaro/bdd100k/images/100k/   (train/ val/ test/)
data/raw/bdd100k_alvaromalfaro/bdd100k/labels/         (train/ val/ test/)
data/raw/bdd100k_mot/mot_labels.parquet
data/raw/bdd100k_mot/bdd100k_videos_train_00/bdd100k/videos/train/  (1,000 .mov)
```

Approximate disk usage: 11 GB (images) + 19 GB (videos).

---

## 5. Pretrained Checkpoints

### 5.1 Detection weights

COCO init weights are downloaded by Ultralytics on first use. Move them
to `pretrained/detection/` after download:

```bash
mkdir -p pretrained/detection
uv run python -c "from ultralytics import YOLO, RTDETR; YOLO('yolov8m.pt'); RTDETR('rtdetr-l.pt')"
mv yolov8m.pt  pretrained/detection/yolov8m.pt
mv rtdetr-l.pt pretrained/detection/rtdetr-l.pt
```

Update both detection configs to point to the moved files:

```yaml
# config/detection/yolov8m.yaml
detection:
  weights: pretrained/detection/yolov8m.pt
```

These are COCO-only weights. Fine-tune on BDD100K for best mAP (see
section 8.2), or use zero-shot (COCO classes map to BDD100K's 10 classes).

### 5.2 Segmentation weights

Segmentation models (DeepLabV3+ R-101 and SegFormer-B2) are initialized
from ImageNet encoders provided by `segmentation-models-pytorch`. The
library fetches encoder weights automatically on first use.

Training is required before inference. Segmentation models are not usable
zero-shot for drivable-area classification. See section 8.1 for training
commands.

### 5.3 ByteTrack and OC-SORT

No checkpoints. Both trackers are parameter-free and run directly from
the vendor libs in `vendor/`.

---

## 6. Prepare Datasets

A single script processes both raw datasets and writes the full
`data/processed/` layout.

```bash
# Full processing — 70k/10k/20k det, 761/100/100 mot
uv run python scripts/process_dataset.py

# Local dev subset — 2000/500/500 det, 50/8/8 mot (fits 8 GB VRAM)
uv run python scripts/process_dataset.py --local

# Smoke test only — 80/10/10 det, 12/3/3 mot
uv run python scripts/process_dataset.py --tiny

# Run only specific steps
uv run python scripts/process_dataset.py --local --steps splits,detection
```

The script builds:

```
data/processed/
+-- splits/
|   +-- detection_split.json       {stem: "train"|"val"|"test"}  100k entries
|   +-- mot_split.json             {videoName: "train"|"val"|"test"}  961 entries
+-- bdd100k_frames/
|   +-- bdd100k.yaml               Ultralytics dataset config
|   +-- images/{train,val,test}/   symlinks to raw JPEGs
|   +-- labels/{train,val,test}/   YOLO .txt (class cx cy w h, normalized)
|   +-- instances_{split}.json     COCO format for pycocotools
|   +-- drivable/
|       +-- polygons/              per-image polygon JSON
|       +-- masks/                 single-channel class-ID PNG
|       +-- colormaps/             BGR visualization PNG
+-- bdd100k_videos/
    +-- labels/{train,val,test}.parquet   MOT annotations by split
    +-- videos/{train,val,test}/          symlinks to .mov files
```

See `docs/datasets/processed.md` for the complete format reference.

---

## 7. Run the Pipeline

Each script reads `.env` automatically (`DATA_PROCESSED_ROOT`,
`OUTPUT_ROOT`) and writes a timestamped log to `$OUTPUT_ROOT/logs/`.

### Smoke test (works without segmentation training)

```bash
source .env
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_pipeline.py \
    --input  "$DATA_PROCESSED_ROOT/bdd100k_videos/videos/val/0001542f-ec815219.mov" \
    --output "$OUTPUT_ROOT/smoke.mp4" \
    --detection-config config/detection/yolov8m.yaml \
    --tracking-config  config/tracking/bytetrack.yaml \
    --no-seg \
    --max-frames 200
```

### Full pipeline (after segmentation training)

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_pipeline.py \
    --input  "$DATA_PROCESSED_ROOT/bdd100k_videos/videos/val/0001542f-ec815219.mov" \
    --output "$OUTPUT_ROOT/result_full.mp4" \
    --detection-config    config/detection/yolov8m.yaml \
    --segmentation-config config/segmentation/deeplabv3plus_resnet101.yaml \
    --tracking-config     config/tracking/bytetrack.yaml
```

---

## 8. Training

All training scripts write a timestamped log to `$OUTPUT_ROOT/logs/`.

### 8.1 Segmentation (required — models cannot run zero-shot)

Multi-GPU training with `torchrun`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 uv run torchrun --standalone --nproc_per_node=3 \
    scripts/train_segmentation.py \
    --segmentation-config config/segmentation/deeplabv3plus_resnet101.yaml \
    --epochs 20 --batch 24

CUDA_VISIBLE_DEVICES=0,1,2 uv run torchrun --standalone --nproc_per_node=3 \
    scripts/train_segmentation.py \
    --segmentation-config config/segmentation/segformer_b2.yaml \
    --epochs 20 --batch 24
```

Single-GPU training:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_segmentation.py \
    --segmentation-config config/segmentation/deeplabv3plus_resnet101.yaml \
    --epochs 20 --batch 4
```

After training, update `config/segmentation/<model>.yaml`:

```yaml
segmentation:
  weights: checkpoints/segmentation/deeplabv3plus_bdd100k/best.pt
```

### 8.2 Detection fine-tuning (optional — COCO weights run zero-shot)

Multi-GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 uv run python scripts/train_detection.py \
    --detection-config config/detection/yolov8m.yaml \
    --epochs 50 --batch 48 --amp

CUDA_VISIBLE_DEVICES=0,1,2 uv run python scripts/train_detection.py \
    --detection-config config/detection/rtdetr_l.yaml \
    --epochs 50 --batch 24 --amp
```

Single-GPU:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_detection.py \
    --detection-config config/detection/yolov8m.yaml \
    --epochs 20 --batch 8
```

After training, update `config/detection/<model>.yaml`:

```yaml
detection:
  weights: checkpoints/detection/yolov8m_bdd100k/weights/best.pt
```

---

## 9. Evaluation

### All evaluations at once

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_all.py \
    --detection-config    config/detection/yolov8m.yaml \
    --segmentation-config config/segmentation/deeplabv3plus_resnet101.yaml \
    --tracking-config     config/tracking/bytetrack.yaml
```

### Individual evaluations

```bash
# Detection — mAP50, mAP50-95, mAP75, per-class AP, P/R/F1
uv run python scripts/eval_detection.py \
    --detection-config config/detection/yolov8m.yaml --split val

# Segmentation — mIoU, per-class IoU, pixel accuracy, FW-IoU
uv run python scripts/eval_segmentation.py \
    --segmentation-config config/segmentation/deeplabv3plus_resnet101.yaml --split val

# Tracking — HOTA/DetA/AssA, MOTA, MOTP, IDF1, MT/ML, IDSW
uv run python scripts/eval_tracking.py \
    --detection-config config/detection/yolov8m.yaml \
    --tracking-config  config/tracking/bytetrack.yaml --split val
```

Results are saved to `$OUTPUT_ROOT/eval/` as JSON:

- `detection_<model>_<split>.json`
- `segmentation_<model>_<split>.json`
- `tracking_<det>_<tracker>_<split>.json`

See `docs/METRICS.md` for full metric definitions.

### Ablation grid (2 det x 2 tracker x 3 seg = 12 combos)

```bash
uv run python scripts/run_ablation.py
```

### Tests

```bash
uv run pytest tests/ -m "not slow"   # fast tests only
uv run pytest tests/                 # full suite
```

---

## 10. Live Webcam (Windows)

The `scripts/live_webcam.py` script captures frames from a connected
webcam and runs the pipeline live with on-screen overlay.

```powershell
.venv\Scripts\activate
python scripts\live_webcam.py ^
    --detection-config config/detection/yolov8m.yaml ^
    --tracking-config  config/tracking/bytetrack.yaml ^
    --camera-index 0
```

Press `q` to quit the live window.

---

## 11. Pipeline Ranking

Rank pipeline configurations by an application-dependent score that
fuses detection, tracking, and segmentation metrics. By convention
`detection_score = mAP50` (or `mAP50:95`), `tracking_score = IDF1` (or
`HOTA`), `segmentation_score = mIoU`.

### Continuous ranking across all 4 predefined profiles

```bash
uv run python scripts/rank_pipelines.py outputs/eval/pipelines_example.json --profiles
```

### Threshold ranking

```bash
uv run python scripts/rank_pipelines.py outputs/eval/pipelines_example.json \
    --profiles \
    --mode threshold \
    --thresholds detection_score=0.40 tracking_score=0.55 segmentation_score=0.70
```

### Custom single-profile ranking

```bash
uv run python scripts/rank_pipelines.py outputs/eval/pipelines_example.json \
    --weights detection_score=0.4 tracking_score=0.3 segmentation_score=0.3
```

### Save ranked tables

```bash
uv run python scripts/rank_pipelines.py outputs/eval/pipelines_example.json \
    --profiles --output outputs/eval/ranked.csv
```

### Predefined profiles

| Profile             | w_det | w_trk | w_seg |
|---------------------|-------|-------|-------|
| Balanced            | 0.33  | 0.33  | 0.34  |
| Detection-first     | 0.60  | 0.20  | 0.20  |
| Tracking-first      | 0.20  | 0.60  | 0.20  |
| Segmentation-first  | 0.20  | 0.20  | 0.60  |
