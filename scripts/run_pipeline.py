"""CLI wrapper — run ALERT pipeline on a video file."""
from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description="Run ALERT pipeline on a video.")
    p.add_argument("--input", required=True, help="Input video path")
    p.add_argument("--output", default=None, help="Output video path (default: $OUTPUT_ROOT/output.mp4)")
    p.add_argument("--base-config",     default=str(REPO / "config/base.yaml"))
    p.add_argument("--detection-config",    default=str(REPO / "config/detection/yolov8m.yaml"))
    p.add_argument("--segmentation-config", default=None)
    p.add_argument("--tracking-config",     default=str(REPO / "config/tracking/bytetrack.yaml"))
    p.add_argument("--pipeline-config",     default=str(REPO / "config/pipeline/default.yaml"))
    p.add_argument("--no-seg", action="store_true", help="Disable segmentation")
    p.add_argument("--max-frames", type=int, default=None)
    return p.parse_args()


def main():
    import cv2
    args = parse_args()

    from alert.utils.config import load_config
    from alert.pipeline import ALERTPipeline
    from alert.utils.video_io import VideoReader, VideoWriter
    from alert.utils.visualization import Visualizer

    configs = [args.base_config, args.detection_config, args.tracking_config, args.pipeline_config]
    if args.segmentation_config:
        configs.insert(2, args.segmentation_config)

    cfg = load_config(*configs)
    if args.no_seg:
        cfg.setdefault("pipeline", {})["enable_segmentation"] = False

    from alert.utils.paths import output_root
    output_path = args.output or cfg.get("pipeline", {}).get("output") or str(output_root() / "output.mp4")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    pipeline = ALERTPipeline(cfg)
    viz = Visualizer(cfg)

    with VideoReader(args.input) as reader:
        with VideoWriter(output_path, reader.fps, reader.width, reader.height) as writer:
            for i, frame in enumerate(reader):
                if args.max_frames and i >= args.max_frames:
                    break
                result = pipeline.process_frame(frame)
                vis_frame = viz.compose_frame(result)
                writer.write(vis_frame)
                if i % 30 == 0:
                    print(f"  frame {i}  fps={result.fps:.1f}")

    print(f"Done → {output_path}")


if __name__ == "__main__":
    main()
