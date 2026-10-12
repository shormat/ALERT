"""Convert drivable-area polygons into per-image JSON, class-ID mask PNG, and BGR colormap PNG."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

DIRECT_ID = 0
ALTERNATIVE_ID = 1
BACKGROUND_ID = 2

CATEGORY_TO_ID: dict[str, int] = {
    "area/drivable": DIRECT_ID,
    "area/alternative": ALTERNATIVE_ID,
}

PALETTE_BGR: dict[int, tuple[int, int, int]] = {
    DIRECT_ID: (0, 0, 255),
    ALTERNATIVE_ID: (255, 0, 0),
    BACKGROUND_ID: (0, 0, 0),
}

IMG_W, IMG_H = 1280, 720


def _vertices_from_poly2d(poly2d: list) -> list[list[float]]:
    return [[float(p[0]), float(p[1])] for p in poly2d]


def extract_drivable_polygons(label_json: dict) -> list[dict]:
    out: list[dict] = []
    for frame in label_json.get("frames", []):
        for obj in frame.get("objects", []):
            cat = obj.get("category")
            if cat not in CATEGORY_TO_ID:
                continue
            if "poly2d" not in obj:
                continue
            verts = _vertices_from_poly2d(obj["poly2d"])
            if len(verts) < 3:
                continue
            out.append({"category": cat, "vertices": verts})
    return out


def rasterize_mask(
    polygons: list[dict], img_w: int = IMG_W, img_h: int = IMG_H
) -> np.ndarray:
    mask = np.full((img_h, img_w), BACKGROUND_ID, dtype=np.uint8)
    # alternative painted first so direct wins on overlap
    for cat in ("area/alternative", "area/drivable"):
        cat_id = CATEGORY_TO_ID[cat]
        for poly in polygons:
            if poly["category"] != cat:
                continue
            pts = np.array(poly["vertices"], dtype=np.int32)
            cv2.fillPoly(mask, [pts], color=int(cat_id))
    return mask


def mask_to_colormap_bgr(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, bgr in PALETTE_BGR.items():
        out[mask == class_id] = bgr
    return out


def _process_one(args: tuple[Path, Path, Path, Path]) -> tuple[str, int]:
    src_json, dst_poly_json, dst_mask_png, dst_colormap_png = args
    data = json.loads(src_json.read_text())
    polygons = extract_drivable_polygons(data)

    dst_poly_json.parent.mkdir(parents=True, exist_ok=True)
    dst_poly_json.write_text(
        json.dumps({"name": src_json.stem, "polygons": polygons})
    )

    mask = rasterize_mask(polygons)
    dst_mask_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_mask_png), mask)

    colormap = mask_to_colormap_bgr(mask)
    dst_colormap_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_colormap_png), colormap)

    return src_json.stem, len(polygons)


def process_drivable_split(
    *,
    split: str,
    stems: Iterable[str],
    src_label_dir_for_stem: dict[str, Path],
    out_root: Path,
    workers: int = 8,
) -> dict:
    stems = sorted(stems)
    poly_out = out_root / "drivable" / "polygons" / split
    mask_out = out_root / "drivable" / "masks" / split
    cmap_out = out_root / "drivable" / "colormaps" / split
    poly_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)
    cmap_out.mkdir(parents=True, exist_ok=True)

    tasks = []
    for stem in stems:
        src = src_label_dir_for_stem[stem] / f"{stem}.json"
        tasks.append((
            src,
            poly_out / f"{stem}.json",
            mask_out / f"{stem}.png",
            cmap_out / f"{stem}.png",
        ))

    n_polys = 0
    if workers <= 1:
        for t in tasks:
            n_polys += _process_one(t)[1]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_process_one, t) for t in tasks]
            for f in as_completed(futures):
                n_polys += f.result()[1]

    return {
        "split": split,
        "n_images": len(stems),
        "n_polygons": n_polys,
    }
