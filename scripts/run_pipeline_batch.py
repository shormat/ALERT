"""Run ALERT pipeline on all .mov files in a directory.

Supports GPU sharding (--gpu-rank / --gpu-total) so the eldir wrapper can
launch one process per GPU, each handling a disjoint subset of videos.

Example (single GPU):
    uv run python scripts/run_pipeline_batch.py \\
        --input-dir  data/processed/bdd100k_videos/videos/test \\
        --output-dir outputs/demo/test \\
        --device 0

Example (4-GPU parallel, launched by eldir wrapper):
    for i in 0 1 2 3; do
        uv run python scripts/run_pipeline_batch.py \\
            --input-dir  data/processed/bdd100k_videos/videos/test \\
            --output-dir outputs/demo/test \\
            --device $i --gpu-rank $i --gpu-total 4 &
    done
    wait
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description="Batch ALERT pipeline over a folder of videos.")
    p.add_argument("--input-dir",  required=True, help="Directory with .mov / .mp4 video files")
    p.add_argument("--output-dir", required=True, help="Directory to write annotated .mp4 files")
    p.add_argument("--base-config",          default=str(REPO / "config/base.yaml"))
    p.add_argument("--detection-config",     default=str(REPO / "config/detection/yolov8m.yaml"))
    p.add_argument("--segmentation-config",  default=str(REPO / "config/segmentation/segformer_b2.yaml"))
    p.add_argument("--tracking-config",      default=str(REPO / "config/tracking/bytetrack.yaml"))
    p.add_argument("--pipeline-config",      default=str(REPO / "config/pipeline/default.yaml"))
    p.add_argument("--device", default="0", help="Physical GPU index passed to Ultralytics")
    p.add_argument("--gpu-rank",  type=int, default=0,
                   help="Index of this GPU worker (0-based) for sharding. Default 0.")
    p.add_argument("--gpu-total", type=int, default=1,
                   help="Total number of GPU workers. Default 1 (no sharding).")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Cap frames per video (useful for smoke-testing)")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip videos whose output .mp4 already exists")
    return p.parse_args()


def main():
    args = parse_args()

    import cv2
    from alert.utils.config import load_config
    from alert.pipeline import ALERTPipeline
    from alert.utils.video_io import VideoReader, VideoWriter
    from alert.utils.visualization import Visualizer

    cfg = load_config(
        args.base_config,
        args.detection_config,
        args.segmentation_config,
        args.tracking_config,
        args.pipeline_config,
    )
    cfg["device"] = args.device

    pipeline = ALERTPipeline(cfg)
    viz = Visualizer(cfg)

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(input_dir.glob("*.mov")) + sorted(input_dir.glob("*.mp4"))
    if not videos:
        print(f"[WARN] No .mov / .mp4 files found in {input_dir}")
        return

    # GPU sharding — this worker handles every gpu_total-th video starting at gpu_rank
    my_videos = [v for i, v in enumerate(videos) if i % args.gpu_total == args.gpu_rank]
    print(f"[GPU {args.device}] {len(my_videos)} / {len(videos)} videos assigned "
          f"(rank {args.gpu_rank}/{args.gpu_total})")

    for video_path in my_videos:
        out_path = output_dir / f"{video_path.stem}_pipeline.mp4"
        if args.skip_existing and out_path.exists():
            print(f"  [skip] {video_path.name} → already exists")
            continue

        print(f"\n  Processing {video_path.name} → {out_path.name}")
        pipeline.reset() if hasattr(pipeline, "reset") else None

        with VideoReader(str(video_path)) as reader:
            with VideoWriter(str(out_path), reader.fps, reader.width, reader.height) as writer:
                for i, frame in enumerate(reader):
                    if args.max_frames and i >= args.max_frames:
                        break
                    result = pipeline.process_frame(frame)
                    vis_frame = viz.compose_frame(result)
                    writer.write(vis_frame)
                    if i % 100 == 0:
                        print(f"    frame {i}  fps={result.fps:.1f}")

        print(f"  Done → {out_path}")

    print(f"\n[GPU {args.device}] All assigned videos complete.")


if __name__ == "__main__":
    main()
