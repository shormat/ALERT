"""Run frame-level R_I for the 5 canonical pipelines and emit Section 6.5 LaTeX.

Two modes:

1. ``--run``   : execute eval_ranking_score.py for each pipeline that
                 doesn't already have a cached aggregate JSON.  This is
                 the slow path (one inference pass per pipeline).

2. (default)   : skip inference; just read cached
                 outputs/eval/ranking_<tag>_<split>.json files and emit
                 the LaTeX table fragment + rank-reversal flags.

The 5 canonical pipelines mirror Table 6 in Section 6.5 of the report
so the new R_I rows align row-for-row with $S_{\\mathrm{cont}}$.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


PIPELINES: list[dict] = [
    {
        "name":   "yolov8m + segformer_b2 + bytetrack",
        "short":  "YOLOv8m + SegFormer-B2 + ByteTrack",
        "tag":    "yolov8m_segformer_b2_bytetrack",
        "det":    "config/detection/yolov8m.yaml",
        "seg":    "config/segmentation/segformer_b2.yaml",
        "trk":    "config/tracking/bytetrack.yaml",
        "scont":  {"Bal.": 0.640, "Det.": 0.579, "Trk.": 0.619, "Seg.": 0.715},
    },
    {
        "name":   "yolov8m + segformer_b2 + ocsort",
        "short":  "YOLOv8m + SegFormer-B2 + OC-SORT",
        "tag":    "yolov8m_segformer_b2_ocsort",
        "det":    "config/detection/yolov8m.yaml",
        "seg":    "config/segmentation/segformer_b2.yaml",
        "trk":    "config/tracking/ocsort.yaml",
        "scont":  {"Bal.": 0.638, "Det.": 0.578, "Trk.": 0.616, "Seg.": 0.714},
    },
    {
        "name":   "yolov8m + deeplabv3plus + bytetrack",
        "short":  "YOLOv8m + DeepLabV3+ + ByteTrack",
        "tag":    "yolov8m_deeplabv3plus_resnet101_bytetrack",
        "det":    "config/detection/yolov8m.yaml",
        "seg":    "config/segmentation/deeplabv3plus_resnet101.yaml",
        "trk":    "config/tracking/bytetrack.yaml",
        "scont":  {"Bal.": 0.609, "Det.": 0.561, "Trk.": 0.601, "Seg.": 0.660},
    },
    {
        "name":   "rtdetr_l + segformer_b2 + bytetrack",
        "short":  "RT-DETR-L + SegFormer-B2 + ByteTrack",
        "tag":    "rtdetr_l_segformer_b2_bytetrack",
        "det":    "config/detection/rtdetr_l.yaml",
        "seg":    "config/segmentation/segformer_b2.yaml",
        "trk":    "config/tracking/bytetrack.yaml",
        "scont":  {"Bal.": 0.605, "Det.": 0.540, "Trk.": 0.574, "Seg.": 0.694},
    },
    {
        "name":   "rtdetr_l + deeplabv3plus + ocsort",
        "short":  "RT-DETR-L + DeepLabV3+ + OC-SORT",
        "tag":    "rtdetr_l_deeplabv3plus_resnet101_ocsort",
        "det":    "config/detection/rtdetr_l.yaml",
        "seg":    "config/segmentation/deeplabv3plus_resnet101.yaml",
        "trk":    "config/tracking/ocsort.yaml",
        "scont":  {"Bal.": 0.557, "Det.": 0.511, "Trk.": 0.525, "Seg.": 0.629},
    },
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--device", default="0")
    p.add_argument("--run", action="store_true",
                   help="Run eval_ranking_score.py for any pipeline whose "
                        "aggregate JSON is missing.  Off by default — "
                        "this command is normally a pure post-processor.")
    p.add_argument("--max-videos", type=int, default=None,
                   help="Forwarded to eval_ranking_score.py (debug only).")
    p.add_argument("--output-tex", type=Path, default=None,
                   help="Where to write the LaTeX fragment (default: stdout)")
    return p.parse_args()


def _agg_path(pipe: dict, split: str, output_root_dir: Path) -> Path:
    return output_root_dir / "eval" / f"ranking_{pipe['tag']}_{split}.json"


def _run_one(pipe: dict, split: str, device: str, max_videos: int | None) -> None:
    cmd = [
        "uv", "run", "python", str(REPO / "scripts/eval_ranking_score.py"),
        "--detection-config",    str(REPO / pipe["det"]),
        "--tracking-config",     str(REPO / pipe["trk"]),
        "--segmentation-config", str(REPO / pipe["seg"]),
        "--split", split,
        "--device", device,
        "--output-tag", pipe["tag"],
    ]
    if max_videos is not None:
        cmd += ["--max-videos", str(max_videos)]
    print(f"\n{'='*60}\n{pipe['short']}\n{'='*60}")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"[ERROR] returncode={res.returncode} for {pipe['short']}")


def _format_latex(rows: list[dict], scont_rank: list[int],
                  ri_rank: dict[str, list[int]]) -> str:
    """Build a 5-row LaTeX fragment with three R_I columns and an
    inline rank-reversal note.

    rows: each dict has {short, ri_uniform, ri_risk, ri_vru, joint_seg, ...}
    """
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{ll ccc}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{$R_I$ (joint $S$, frame-level)} \\",
        r"\cmidrule(lr){3-5}",
        r"\# & Pipeline (det+seg+trk) & uniform & risk-weighted & VRU-present \\",
        r"\midrule",
    ]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"{i} & {r['short']} & "
            f"{r['ri_uniform']:.3f} & {r['ri_risk']:.3f} & {r['ri_vru']:.3f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
    ]

    # Rank-reversal note vs Scont (Table 6 / tab:rank).
    notes: list[str] = []
    cont_order = scont_rank
    for prof, ord_ in ri_rank.items():
        if ord_ != cont_order:
            swaps: list[str] = []
            for new_pos, idx in enumerate(ord_, 1):
                old_pos = cont_order.index(idx) + 1
                if new_pos != old_pos:
                    swaps.append(f"P{idx + 1}: {old_pos}$\\to${new_pos}")
            if swaps:
                notes.append(f"\\emph{{{prof}}} reverses " + ", ".join(swaps))

    if notes:
        caption_note = (
            " Rank reversals vs Table~\\ref{tab:rank}: "
            + "; ".join(notes) + "."
        )
    else:
        caption_note = (
            " The ranking under all three importance profiles agrees with "
            "Table~\\ref{tab:rank} ($S_{\\mathrm{cont}}$, Balanced)."
        )

    lines += [
        r"\caption{Frame-level Pi\'erard ranking score $R_I$ "
        r"(Eq.~\ref{eq:rank-RI}) on the MOT val split.  "
        r"$S(\omega) = S_\mathrm{det}\wedge S_\mathrm{seg}\wedge S_\mathrm{trk}$ "
        r"with $\tau_\mathrm{det}{=}0.5$, $\tau_\mathrm{seg}{=}0.7$.  "
        r"Three importance profiles: uniform ($I{=}1$), risk-weighted "
        r"($I{=}5$ when a VRU footprint overlaps the GT direct-drivable "
        r"mask by $\geq 30\%$, else $1$), VRU-present ($I{=}1$ iff the "
        r"frame has any VRU GT, else $0$)."
        + caption_note + "}",
        r"\label{tab:rank-frame}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()

    # Use the eval/ root (resolves env var)
    sys.path.insert(0, str(REPO))
    from alert.utils.paths import output_root  # noqa: E402

    out_root = output_root()

    if args.run:
        for pipe in PIPELINES:
            agg = _agg_path(pipe, args.split, out_root)
            if agg.exists():
                print(f"[skip] {pipe['short']}: {agg.name} already exists")
                continue
            _run_one(pipe, args.split, args.device, args.max_videos)

    # Aggregate phase
    rows: list[dict] = []
    for pipe in PIPELINES:
        agg = _agg_path(pipe, args.split, out_root)
        if not agg.exists():
            print(f"[warn] missing {agg} — pass --run first or pre-compute it")
            continue
        data = json.loads(agg.read_text())
        m = data["metrics"]
        rows.append({
            "tag":    pipe["tag"],
            "short":  pipe["short"],
            "scont":  pipe["scont"],
            "ri_uniform": m["uniform"]["joint"],
            "ri_risk":    m["risk_weighted"]["joint"],
            "ri_vru":     m["vru_present"]["joint"],
            "ri_det":     m["uniform"]["det"],
            "ri_seg":     m["uniform"]["seg"],
            "ri_trk":     m["uniform"]["trk"],
        })

    if not rows:
        print("\nNo aggregate JSONs found — nothing to emit.  "
              "Re-run with --run after the inference pass.")
        return

    # Print plain-text summary
    print("\n=== Frame-level R_I summary ===")
    print(f"{'#':>2}  {'Pipeline':<42}  {'uniform':>8}  {'risk_w':>8}  {'vru_pre':>8}")
    for i, r in enumerate(rows, 1):
        print(f"{i:>2}  {r['short']:<42}  "
              f"{r['ri_uniform']:>8.3f}  {r['ri_risk']:>8.3f}  {r['ri_vru']:>8.3f}")

    print("\n  Sub-task R_I (uniform profile):")
    print(f"{'':>2}  {'Pipeline':<42}  {'det':>6}  {'seg':>6}  {'trk':>6}")
    for r in rows:
        print(f"  {'':>2}  {r['short']:<42}  "
              f"{r['ri_det']:>6.3f}  {r['ri_seg']:>6.3f}  {r['ri_trk']:>6.3f}")

    # Compute rank orderings for reversal flagging.
    # Use Bal. as the canonical Scont order (it's what Table 6 sorts by).
    n = len(rows)
    cont_order = sorted(range(n), key=lambda i: -rows[i]["scont"]["Bal."])
    ri_orders = {
        "uniform":       sorted(range(n), key=lambda i: -rows[i]["ri_uniform"]),
        "risk_weighted": sorted(range(n), key=lambda i: -rows[i]["ri_risk"]),
        "vru_present":   sorted(range(n), key=lambda i: -rows[i]["ri_vru"]),
    }

    print("\n  Rank order (Scont Bal., as in Table 6):")
    print("    " + " > ".join(f"P{cont_order[i]+1}" for i in range(n)))
    for prof, ord_ in ri_orders.items():
        print(f"  Rank order ({prof}):")
        print("    " + " > ".join(f"P{ord_[i]+1}" for i in range(n))
              + ("   (REVERSAL vs Scont)" if ord_ != cont_order else ""))

    tex = _format_latex(rows, cont_order, ri_orders)

    if args.output_tex is None:
        print("\n=== LaTeX fragment ===")
        print(tex)
    else:
        args.output_tex.parent.mkdir(parents=True, exist_ok=True)
        args.output_tex.write_text(tex)
        print(f"\n  LaTeX fragment -> {args.output_tex}")


if __name__ == "__main__":
    main()
