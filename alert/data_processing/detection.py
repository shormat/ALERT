"""Convert per-image annotation JSONs into YOLO + COCO outputs."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ALV_TO_BDD: dict[str, str] = {
    "person": "pedestrian",
    "bike": "bicycle",
    "motor": "motorcycle",
}

BDD_CLASSES: list[str] = [
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
]
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(BDD_CLASSES)}

IMG_W, IMG_H = 1280, 720


def normalize_category(raw: str) -> str:
    return ALV_TO_BDD.get(raw, raw)


@dataclass(frozen=True)
class BoxAnn:
    cls_idx: int
    x1: float
    y1: float
    x2: float
    y2: float

    def to_yolo_line(self, img_w: int = IMG_W, img_h: int = IMG_H) -> str:
        cx = (self.x1 + self.x2) / 2 / img_w
        cy = (self.y1 + self.y2) / 2 / img_h
        w = (self.x2 - self.x1) / img_w
        h = (self.y2 - self.y1) / img_h
        return f"{self.cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

    def to_coco_bbox(self) -> list[float]:
        return [self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1]

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


def parse_alvaromalfaro_boxes(label_json: dict) -> list[BoxAnn]:
    boxes: list[BoxAnn] = []
    for frame in label_json.get("frames", []):
        for obj in frame.get("objects", []):
            if "box2d" not in obj:
                continue
            cat = normalize_category(obj["category"])
            if cat not in CLASS_TO_IDX:
                continue
            b = obj["box2d"]
            x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
            if x2 <= x1 or y2 <= y1:
                continue
            x1 = max(0.0, min(IMG_W, x1))
            y1 = max(0.0, min(IMG_H, y1))
            x2 = max(0.0, min(IMG_W, x2))
            y2 = max(0.0, min(IMG_H, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append(BoxAnn(CLASS_TO_IDX[cat], x1, y1, x2, y2))
    return boxes


def _process_one(args: tuple[Path, Path, Path | None]) -> tuple[str, list[BoxAnn]]:
    src_json, dst_txt, src_img_for_symlink = args
    data = json.loads(src_json.read_text())
    boxes = parse_alvaromalfaro_boxes(data)

    dst_txt.parent.mkdir(parents=True, exist_ok=True)
    if boxes:
        dst_txt.write_text("\n".join(b.to_yolo_line() for b in boxes) + "\n")
    else:
        dst_txt.write_text("")

    if src_img_for_symlink is not None:
        src_img, dst_img = src_img_for_symlink
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        if not dst_img.exists():
            dst_img.symlink_to(src_img.resolve())

    return src_json.stem, boxes


def _build_coco(
    image_records: list[tuple[str, list[BoxAnn]]],
    split: str,
) -> dict:
    images = []
    annotations = []
    ann_id = 1
    for img_id, (stem, boxes) in enumerate(image_records, start=1):
        images.append(
            {
                "id": img_id,
                "file_name": f"{stem}.jpg",
                "width": IMG_W,
                "height": IMG_H,
            }
        )
        for b in boxes:
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": b.cls_idx + 1,
                    "bbox": b.to_coco_bbox(),
                    "area": b.area,
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )
            ann_id += 1
    return {
        "info": {
            "description": f"BDD100K detection {split} split (ALERT processed)",
            "version": "1.0",
        },
        "licenses": [],
        "categories": [
            {"id": i + 1, "name": name, "supercategory": "object"}
            for i, name in enumerate(BDD_CLASSES)
        ],
        "images": images,
        "annotations": annotations,
    }


def process_detection_split(
    *,
    split: str,
    stems: Iterable[str],
    src_label_dir_for_stem: dict[str, Path],
    src_image_dir_for_stem: dict[str, Path],
    out_root: Path,
    workers: int = 8,
    do_symlink: bool = True,
) -> dict:
    stems = sorted(stems)
    label_out = out_root / "labels" / split
    image_out = out_root / "images" / split
    label_out.mkdir(parents=True, exist_ok=True)
    if do_symlink:
        image_out.mkdir(parents=True, exist_ok=True)

    tasks = []
    for stem in stems:
        src_json = src_label_dir_for_stem[stem] / f"{stem}.json"
        dst_txt = label_out / f"{stem}.txt"
        if do_symlink:
            src_img = src_image_dir_for_stem[stem] / f"{stem}.jpg"
            dst_img = image_out / f"{stem}.jpg"
            tasks.append((src_json, dst_txt, (src_img, dst_img)))
        else:
            tasks.append((src_json, dst_txt, None))

    results: list[tuple[str, list[BoxAnn]]] = []
    if workers <= 1:
        for t in tasks:
            results.append(_process_one(t))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_process_one, t) for t in tasks]
            for f in as_completed(futures):
                results.append(f.result())

    results.sort(key=lambda r: r[0])

    coco = _build_coco(results, split)
    (out_root / f"instances_{split}.json").write_text(json.dumps(coco))

    return {
        "split": split,
        "n_images": len(results),
        "n_boxes": sum(len(b) for _, b in results),
    }


def write_yolo_dataset_yaml(out_root: Path) -> Path:
    yaml_path = out_root / "bdd100k.yaml"
    lines = [
        f"path: {out_root.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    for i, name in enumerate(BDD_CLASSES):
        lines.append(f"  {i}: {name}")
    yaml_path.write_text("\n".join(lines) + "\n")
    return yaml_path
