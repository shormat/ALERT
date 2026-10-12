"""Rank ALERT pipeline configurations by preference-based score.

Reads pipeline metric values from a JSON or CSV file and prints a ranked
table. Three modes of weight configuration are supported:

1. Single custom profile via ``--weights``:
       --weights detection_score=0.4 tracking_score=0.3 segmentation_score=0.3

2. Predefined profile grid via ``--profiles`` (Balanced, Detection-first,
   Tracking-first, Segmentation-first).

3. Default equal weights (no flag): w_det = w_trk = w_seg = 1.

Examples
--------
    # default equal weights, continuous mode
    python scripts/rank_pipelines.py outputs/eval/pipelines_example.json

    # all 4 predefined profiles
    python scripts/rank_pipelines.py outputs/eval/pipelines_example.json --profiles

    # all 4 profiles, threshold mode
    python scripts/rank_pipelines.py outputs/eval/pipelines_example.json \
        --profiles --mode threshold \
        --thresholds detection_score=0.40 tracking_score=0.55 segmentation_score=0.70

    # custom single profile
    python scripts/rank_pipelines.py outputs/eval/pipelines_example.json \
        --weights detection_score=0.4 tracking_score=0.3 segmentation_score=0.3

    # write the ranked table back to disk (JSON or CSV)
    python scripts/rank_pipelines.py outputs/eval/pipelines_example.json \
        --profiles --output outputs/eval/ranked.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from alert.evaluation.preference_profiles import (  # noqa: E402
    get_preference_profiles,
    rank_pipelines_across_profiles,
)
from alert.evaluation.preference_ranking import rank_pipelines  # noqa: E402

_METRIC_KEYS = ("detection_score", "tracking_score", "segmentation_score")


def _parse_kv(items: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for tok in items:
        if "=" not in tok:
            raise argparse.ArgumentTypeError(f"expected key=value, got '{tok}'")
        k, v = tok.split("=", 1)
        out[k.strip()] = float(v)
    return out


def _load(path: Path) -> list[dict]:
    if path.suffix.lower() == ".json":
        with path.open() as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            raise ValueError(f"{path} did not contain a list of records")
        return data
    if path.suffix.lower() == ".csv":
        with path.open() as f:
            rows: list[dict] = []
            for r in csv.DictReader(f):
                row = dict(r)
                for k in _METRIC_KEYS:
                    if k in row and row[k] != "":
                        row[k] = float(row[k])
                rows.append(row)
            return rows
    raise ValueError(f"unsupported file extension: {path.suffix}")


def _print_table(profile_name: str, ranked: list[dict]) -> None:
    print(f"\nProfile: {profile_name}")
    name_w = max((len(r.get("name", "")) for r in ranked), default=4)
    print(f"{'rank':>4}  {'name':<{name_w}}  {'det':>6}  {'trk':>6}  {'seg':>6}  {'final':>7}")
    print(f"{'-'*4}  {'-'*name_w}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*7}")
    for i, r in enumerate(ranked, 1):
        print(f"{i:>4}  {r['name']:<{name_w}}  "
              f"{r.get('detection_score', 0):>6.3f}  "
              f"{r.get('tracking_score', 0):>6.3f}  "
              f"{r.get('segmentation_score', 0):>6.3f}  "
              f"{r['final_score']:>7.3f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path,
                   help="JSON or CSV file with pipeline metric records")
    p.add_argument("--output", type=Path, default=None,
                   help="Optional output for the ranked table (.json or .csv)")
    p.add_argument("--mode", choices=("continuous", "threshold"), default="continuous")
    p.add_argument("--profiles", action="store_true",
                   help="Compute rankings for the 4 predefined profiles "
                        "(Balanced, Detection-first, Tracking-first, "
                        "Segmentation-first) instead of a single weight vector")
    p.add_argument("--weights", nargs="+", default=None, metavar="KEY=VAL",
                   help="Custom single-profile weights (e.g. "
                        "detection_score=0.4 tracking_score=0.3 segmentation_score=0.3)")
    p.add_argument("--thresholds", nargs="+", default=None, metavar="KEY=VAL",
                   help="Thresholds for --mode threshold (e.g. "
                        "detection_score=0.40 tracking_score=0.55 segmentation_score=0.70)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    records = _load(args.input)

    thresholds: dict[str, float] | None = None
    if args.mode == "threshold":
        if not args.thresholds:
            raise SystemExit("--mode threshold requires --thresholds key=value pairs")
        thresholds = _parse_kv(args.thresholds)

    rankings: dict[str, list[dict]]
    if args.profiles:
        rankings = rank_pipelines_across_profiles(
            records, profiles=None, thresholds=thresholds, mode=args.mode,
        )
    else:
        weights = _parse_kv(args.weights) if args.weights else {k: 1.0 for k in _METRIC_KEYS}
        ranked = rank_pipelines(records, weights=weights, thresholds=thresholds, mode=args.mode)
        rankings = {"custom": ranked}

    print(f"\nMode: {args.mode}")
    if args.mode == "threshold":
        print(f"Thresholds: {thresholds}")
    for profile_name, ranked in rankings.items():
        _print_table(profile_name, ranked)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix.lower() == ".csv":
            with args.output.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["profile", "rank", "name", "detection_score",
                            "tracking_score", "segmentation_score", "final_score"])
                for profile_name, ranked in rankings.items():
                    for i, r in enumerate(ranked, 1):
                        w.writerow([profile_name, i, r["name"],
                                    r.get("detection_score"),
                                    r.get("tracking_score"),
                                    r.get("segmentation_score"),
                                    r["final_score"]])
        else:
            with args.output.open("w") as f:
                json.dump(rankings, f, indent=2)
        print(f"\n→ {args.output}")


if __name__ == "__main__":
    main()
