"""Generate quantitative comparison plots from eval JSON outputs.

Reads
  outputs/eval/detection_*_{split}.json
  outputs/eval/segmentation_*_{split}.json
  outputs/eval/tracking_*_{split}.json

Writes
  outputs/plots/detection_overall_{split}.png
  outputs/plots/detection_per_class_ap_{split}.png
  outputs/plots/segmentation_overall_{split}.png
  outputs/plots/segmentation_per_class_iou_{split}.png
  outputs/plots/tracking_overview_{split}.png
  outputs/plots/tracking_hota_alpha_{split}.png
  outputs/plots/tracking_det_ass_scatter_{split}.png
  outputs/plots/tracking_coverage_{split}.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.4,
    "font.size": 10,
})

# ── display names ──────────────────────────────────────────────────────────────
_NAMES = {
    "yolov8m":                 "YOLOv8m",
    "rtdetr_l":                "RT-DETR-L",
    "deeplabv3plus_resnet101": "DeepLabV3+",
    "segformer_b2":            "SegFormer-B2",
    "yolov8m_bytetrack":       "YOLOv8m\n+ByteTrack",
    "yolov8m_ocsort":          "YOLOv8m\n+OC-SORT",
    "rtdetr_l_bytetrack":      "RT-DETR-L\n+ByteTrack",
    "rtdetr_l_ocsort":         "RT-DETR-L\n+OC-SORT",
}
_label = lambda k: _NAMES.get(k, k)

# ── colour palette (colour-blind safe) ────────────────────────────────────────
_C = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _load(eval_dir: Path, stage: str, split: str) -> dict[str, dict]:
    """Return {model_tag: metrics_dict} for every matching JSON in eval_dir."""
    out = {}
    for p in sorted(eval_dir.glob(f"{stage}_*_{split}.json")):
        try:
            payload = json.loads(p.read_text())
        except Exception:
            continue
        meta = payload.get("meta", {})
        m    = payload.get("metrics", {})
        # derive a short tag from the filename
        stem = p.stem  # e.g. "detection_yolov8m_val"
        tag  = stem[len(stage) + 1: -(len(split) + 1)]  # strip stage_ and _split
        out[tag] = m
    return out


def _grouped_bars(ax, groups: dict[str, dict[str, float]],
                  colors: list[str], ylabel: str, ylim: tuple | None = None):
    """groups: {metric_label: {model_tag: value}}  — draws grouped bars."""
    metrics = list(groups.keys())
    models  = list({m for vals in groups.values() for m in vals})
    n_m, n_mod = len(metrics), len(models)
    x   = np.arange(n_m)
    w   = 0.8 / max(n_mod, 1)
    off = (np.arange(n_mod) - (n_mod - 1) / 2) * w

    for i, model in enumerate(models):
        vals = [groups[met].get(model, 0.0) for met in metrics]
        bars = ax.bar(x + off[i], vals, w * 0.9,
                      label=_label(model), color=colors[i % len(colors)])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=8)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path.name}")


# ── detection ─────────────────────────────────────────────────────────────────

def plot_detection(data: dict[str, dict], out_dir: Path, split: str):
    if not data:
        print("  [detection] no JSON files found — skipping")
        return

    # ---- 1. overall metrics ----
    overall_keys = [("mAP50", "mAP50"), ("mAP50-95", "mAP50_95"),
                    ("mAP75", "mAP75"), ("Precision", "precision"),
                    ("Recall", "recall"), ("F1", "f1")]
    groups = {label: {tag: m.get(key, 0.0) for tag, m in data.items()}
              for label, key in overall_keys}

    fig, ax = plt.subplots(figsize=(9, 4))
    _grouped_bars(ax, groups, _C, ylabel="Score", ylim=(0, 1.05))
    ax.set_title(f"Detection — overall metrics ({split})")
    _save(fig, out_dir / f"detection_overall_{split}.png")

    # ---- 2. per-class AP50-95 ----
    all_classes: set[str] = set()
    for m in data.values():
        all_classes |= set(m.get("per_class_ap50_95", {}).keys())
    if not all_classes:
        print("  [detection] no per-class AP data — skipping per-class plot")
        return

    classes = sorted(all_classes)
    groups2 = {cls: {tag: m.get("per_class_ap50_95", {}).get(cls, 0.0)
                     for tag, m in data.items()}
               for cls in classes}

    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 1.5), 4))
    _grouped_bars(ax, groups2, _C, ylabel="AP50-95", ylim=(0, 1.05))
    ax.set_title(f"Detection — per-class AP50-95 ({split})")
    _save(fig, out_dir / f"detection_per_class_ap_{split}.png")


# ── segmentation ──────────────────────────────────────────────────────────────

def plot_segmentation(data: dict[str, dict], out_dir: Path, split: str):
    if not data:
        print("  [segmentation] no JSON files found — skipping")
        return

    # ---- 1. overall ----
    overall_keys = [("mIoU", "mIoU"), ("Pixel Acc", "pixel_accuracy"),
                    ("Mean Pixel Acc", "mean_pixel_accuracy"), ("FW-IoU", "fw_iou")]
    groups = {label: {tag: m.get(key, 0.0) for tag, m in data.items()}
              for label, key in overall_keys}

    fig, ax = plt.subplots(figsize=(8, 4))
    _grouped_bars(ax, groups, _C[2:], ylabel="Score", ylim=(0, 1.05))
    ax.set_title(f"Segmentation — overall metrics ({split})")
    _save(fig, out_dir / f"segmentation_overall_{split}.png")

    # ---- 2. per-class IoU ----
    all_classes: set[str] = set()
    for m in data.values():
        all_classes |= set(m.get("per_class", {}).keys())
    if not all_classes:
        return

    classes = sorted(all_classes)
    groups2 = {cls: {tag: m.get("per_class", {}).get(cls, {}).get("iou", 0.0)
                     for tag, m in data.items()}
               for cls in classes}

    fig, ax = plt.subplots(figsize=(7, 4))
    _grouped_bars(ax, groups2, _C[2:], ylabel="IoU", ylim=(0, 1.05))
    ax.set_title(f"Segmentation — per-class IoU ({split})")
    _save(fig, out_dir / f"segmentation_per_class_iou_{split}.png")


# ── tracking ──────────────────────────────────────────────────────────────────

def plot_tracking(data: dict[str, dict], out_dir: Path, split: str):
    if not data:
        print("  [tracking] no JSON files found — skipping")
        return

    tags    = list(data.keys())
    labels  = [_label(t) for t in tags]
    colors  = [_C[i % len(_C)] for i in range(len(tags))]

    # ---- 1. overview: MOTA / IDF1 / HOTA ----
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    for ax, (key, title) in zip(axes, [("MOTA", "MOTA"), ("IDF1", "IDF1"), ("HOTA", "HOTA")]):
        vals = [data[t].get(key, 0.0) for t in tags]
        bars = ax.bar(range(len(tags)), vals, color=colors)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(tags)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Score")
        if key == "MOTA":
            lo = min(min(vals) - 0.05, 0)
            ax.set_ylim(lo, 1.0)
        else:
            ax.set_ylim(0, 1.0)

    fig.suptitle(f"Tracking — MOTA / IDF1 / HOTA ({split})", fontsize=11)
    fig.tight_layout()
    _save(fig, out_dir / f"tracking_overview_{split}.png")

    # ---- 2. HOTA per alpha curve ----
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (tag, m) in enumerate(data.items()):
        hpa = m.get("HOTA_per_alpha", {})
        if not hpa:
            continue
        alphas = sorted(float(a) for a in hpa)
        hvals  = [hpa[str(a)] for a in alphas]
        ax.plot(alphas, hvals, marker="o", markersize=4,
                color=colors[i], label=_label(tag))
    ax.set_xlabel("α (IoU threshold)")
    ax.set_ylabel("HOTA")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title(f"Tracking — HOTA per α ({split})")
    _save(fig, out_dir / f"tracking_hota_alpha_{split}.png")

    # ---- 3. DetA vs AssA scatter with HOTA isocurves ----
    fig, ax = plt.subplots(figsize=(6, 6))

    # isocurves HOTA = sqrt(DetA * AssA) = h  →  AssA = h² / DetA
    det_grid = np.linspace(0.01, 1.0, 300)
    for h in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        ass_iso = (h ** 2) / det_grid
        mask    = (ass_iso <= 1.0) & (ass_iso >= 0.0)
        ax.plot(det_grid[mask], ass_iso[mask], "--", color="grey",
                alpha=0.35, linewidth=0.8)
        # label at right edge
        x_end = det_grid[mask][-1]
        y_end = ass_iso[mask][-1]
        ax.text(x_end + 0.005, y_end, f"H={h:.1f}", fontsize=6, color="grey", va="center")

    for i, (tag, m) in enumerate(data.items()):
        det_a = m.get("DetA", 0.0)
        ass_a = m.get("AssA", 0.0)
        ax.scatter(det_a, ass_a, color=colors[i], s=100, zorder=5)
        ax.annotate(_label(tag).replace("\n", " "), (det_a, ass_a),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("DetA  (detection accuracy)")
    ax.set_ylabel("AssA  (association accuracy)")
    ax.set_title(f"Tracking — DetA vs AssA ({split})")
    _save(fig, out_dir / f"tracking_det_ass_scatter_{split}.png")

    # ---- 4. MT / PT / ML stacked bar ----
    mt = [data[t].get("MT", 0.0) for t in tags]
    pt = [data[t].get("PT", 0.0) for t in tags]
    ml = [data[t].get("ML", 0.0) for t in tags]
    x  = np.arange(len(tags))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, mt, label="MT  (≥80%)", color="#228833")
    ax.bar(x, pt, bottom=mt, label="PT  (20–80%)", color="#CCBB44")
    ax.bar(x, ml, bottom=[a + b for a, b in zip(mt, pt)], label="ML  (≤20%)", color="#EE6677")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Fraction of GT tracks")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title(f"Tracking — track coverage MT/PT/ML ({split})")
    _save(fig, out_dir / f"tracking_coverage_{split}.png")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Plot eval results from JSON files.")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--eval-dir",    default=None,
                   help="Directory with eval JSONs (default: outputs/eval/)")
    p.add_argument("--output-dir",  default=None,
                   help="Directory for plots (default: outputs/plots/)")
    return p.parse_args()


def main():
    args = parse_args()

    from alert.utils.paths import output_root
    eval_dir = Path(args.eval_dir)   if args.eval_dir   else output_root() / "eval"
    out_dir  = Path(args.output_dir) if args.output_dir else output_root() / "plots"

    print(f"Reading JSONs from : {eval_dir}")
    print(f"Writing plots to   : {out_dir}")
    print()

    det_data = _load(eval_dir, "detection",    args.split)
    seg_data = _load(eval_dir, "segmentation", args.split)
    trk_data = _load(eval_dir, "tracking",     args.split)

    print(f"detection models   : {list(det_data) or '(none)'}")
    print(f"segmentation models: {list(seg_data) or '(none)'}")
    print(f"tracking combos    : {list(trk_data) or '(none)'}")
    print()

    plot_detection(det_data,    out_dir, args.split)
    plot_segmentation(seg_data, out_dir, args.split)
    plot_tracking(trk_data,     out_dir, args.split)

    print()
    print(f"All plots written to {out_dir}/")


if __name__ == "__main__":
    main()
