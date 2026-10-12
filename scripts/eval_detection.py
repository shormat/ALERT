"""CLI wrapper for detection evaluation — all metrics via Ultralytics val."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate detection on BDD100K val/test split.")
    p.add_argument("--detection-config", required=True)
    p.add_argument("--base-config", default=str(REPO / "config/base.yaml"))
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--device", default="0",
                   help="Physical GPU index for Ultralytics, e.g. '0', '1', '0,1,2'. "
                        "Always pass an explicit index — never 'cuda' (Ultralytics overrides "
                        "CUDA_VISIBLE_DEVICES internally).")
    return p.parse_args()


def main():
    args = parse_args()
    from alert.utils.config import load_config
    from alert.detection.datasets.bdd100k import BDD100KDetDataset

    cfg = load_config(args.base_config, args.detection_config)
    det_cfg = cfg.get("detection", {})
    model_key = det_cfg.get("model", "").lower()

    if "rtdetr" in model_key:
        from ultralytics import RTDETR
        model = RTDETR(det_cfg.get("weights") or "rtdetr-l.pt")
    else:
        from ultralytics import YOLO
        model = YOLO(det_cfg.get("weights") or "yolov8m.pt")

    device = args.device  # always a numeric index; Ultralytics sets CUDA_VISIBLE_DEVICES itself
    results = model.val(
        data=str(BDD100KDetDataset.dataset_yaml_path()),
        split=args.split,
        imgsz=det_cfg.get("img_size", 640),
        conf=det_cfg.get("conf_threshold", 0.001),
        iou=det_cfg.get("iou_threshold", 0.6),
        device=device,
    )

    box = results.box
    class_names = results.names  # {idx: name}

    # per-class AP50-95
    per_class_ap = {}
    if hasattr(box, "maps") and box.maps is not None:
        for idx, ap in enumerate(box.maps):
            name = class_names.get(idx, str(idx))
            per_class_ap[name] = float(ap)

    # per-class precision and recall at best-F1 threshold
    per_class_p: dict[str, float] = {}
    per_class_r: dict[str, float] = {}
    if hasattr(box, "p") and box.p is not None and len(box.p) == len(class_names):
        for idx, (p_val, r_val) in enumerate(zip(box.p, box.r)):
            name = class_names.get(idx, str(idx))
            per_class_p[name] = float(p_val)
            per_class_r[name] = float(r_val)

    print(f"\n=== Detection Evaluation ({args.split}) ===")
    print(f"\n  --- Overall ---")
    print(f"  mAP50       : {box.map50:.4f}   (primary: easy-to-compare, IoU≥0.5)")
    print(f"  mAP50-95    : {box.map:.4f}   (primary: COCO/BDD100K standard)")
    print(f"  mAP75       : {box.map75:.4f}   (secondary: strict localisation)")
    mean_p = float(box.mp) if hasattr(box, "mp") else float(sum(per_class_p.values()) / max(len(per_class_p), 1))
    mean_r = float(box.mr) if hasattr(box, "mr") else float(sum(per_class_r.values()) / max(len(per_class_r), 1))
    f1 = 2 * mean_p * mean_r / max(mean_p + mean_r, 1e-9)
    print(f"  Precision   : {mean_p:.4f}   (at best-F1 threshold)")
    print(f"  Recall      : {mean_r:.4f}   (at best-F1 threshold)")
    print(f"  F1          : {f1:.4f}")

    if per_class_ap:
        print(f"\n  --- Per-class AP50-95 ---")
        for name, ap in sorted(per_class_ap.items(), key=lambda x: -x[1]):
            p_str = f"  P={per_class_p[name]:.3f}  R={per_class_r[name]:.3f}" if name in per_class_p else ""
            print(f"    {name:<16s}: {ap:.4f}{p_str}")

    from alert.utils.paths import output_root
    model_tag = Path(args.detection_config).stem
    out_path = output_root() / "eval" / f"detection_{model_tag}_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "stage": "detection",
            "model": model_tag,
            "split": args.split,
            "detection_config": args.detection_config,
            "device": device,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        "metrics": {
            "mAP50": float(box.map50),
            "mAP50_95": float(box.map),
            "mAP75": float(box.map75),
            "precision": mean_p,
            "recall": mean_r,
            "f1": f1,
            "per_class_ap50_95": per_class_ap,
            "per_class_precision": per_class_p,
            "per_class_recall": per_class_r,
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
