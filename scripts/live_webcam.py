"""ALERT live webcam demo — capture from cv2.VideoCapture and run the pipeline.

Streams frames from a USB/internal webcam through ALERTPipeline (detection +
tracking, optionally segmentation/risk/alerts) and renders the standard
Visualizer overlay (boxes, track IDs, drivable mask, FPS, risk badge) in an
OpenCV window.

Usage:
    python scripts/live_webcam.py
    python scripts/live_webcam.py --enable-seg
    python scripts/live_webcam.py --camera-index 1 --save-video out.mp4

Press 'q' (or close the window) to quit cleanly.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("alert.live_webcam")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ALERT live webcam demo (detection + tracking, optional segmentation).",
    )
    p.add_argument("--camera-index", type=int, default=0,
                   help="cv2.VideoCapture device index (default: 0)")
    p.add_argument("--detection-config", default=str(REPO / "config/detection/yolov8m.yaml"))
    p.add_argument("--tracking-config", default=str(REPO / "config/tracking/bytetrack.yaml"))
    p.add_argument("--segmentation-config", default=str(REPO / "config/segmentation/segformer_b2.yaml"))
    p.add_argument("--base-config", default=str(REPO / "config/base.yaml"))
    p.add_argument("--enable-seg", action="store_true",
                   help="Enable drivable-area segmentation (slows frame rate)")
    p.add_argument("--enable-risk", action="store_true",
                   help="Enable risk reasoner overlay")
    p.add_argument("--enable-alerts", action="store_true",
                   help="Enable voice/text alert overlay (no audio is played)")
    p.add_argument("--conf-threshold", type=float, default=None,
                   help="Override detection confidence threshold")
    p.add_argument("--device", default="0",
                   help='Torch device: "cpu", "cuda", "0", "cuda:0", ...')
    p.add_argument("--width", type=int, default=1280, help="Webcam capture width")
    p.add_argument("--height", type=int, default=720, help="Webcam capture height")
    p.add_argument("--save-video", default=None,
                   help="Optional path: save annotated stream as mp4")
    p.add_argument("--window-name", default="ALERT Live")
    return p.parse_args()


def _build_config(args: argparse.Namespace) -> dict:
    """Merge YAML configs and apply CLI overrides."""
    from alert.utils.config import load_config

    cfgs: list[str] = [args.base_config, args.detection_config, args.tracking_config]
    if args.enable_seg:
        cfgs.append(args.segmentation_config)
    cfg = load_config(*cfgs)

    cfg["device"] = args.device

    pipe = cfg.setdefault("pipeline", {})
    # Live demo: only what the user explicitly asks for
    pipe["enable_segmentation"] = bool(args.enable_seg)
    pipe["enable_risk"] = bool(args.enable_risk)
    pipe["enable_alerts"] = bool(args.enable_alerts)
    pipe["show_display"] = False  # we manage cv2.imshow ourselves
    pipe["output"] = None
    pipe["log_fps"] = False

    if args.conf_threshold is not None:
        det = cfg.setdefault("detection", {})
        det["confidence_threshold"] = float(args.conf_threshold)
        det["conf_threshold"] = float(args.conf_threshold)

    return cfg


def _open_webcam(index: int, width: int, height: int):
    """Open the webcam and request a target resolution. Returns the VideoCapture."""
    import cv2

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        # Try CAP_DSHOW on Windows / CAP_V4L2 on Linux as fallback
        for backend in (getattr(cv2, "CAP_DSHOW", 0), getattr(cv2, "CAP_V4L2", 0)):
            if backend == 0:
                continue
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                break

    if not cap.isOpened():
        log.error("Could not open webcam at index %d. "
                  "Check that a camera is connected and not in use by another app.",
                  index)
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
    log.info("Webcam %d opened at %dx%d (requested %dx%d)",
             index, actual_w, actual_h, width, height)
    return cap


def _draw_track_count(frame, n_tracks: int):
    """Number of active tracks in top-right (Visualizer reserves the FPS slot there)."""
    import cv2
    text = f"tracks: {n_tracks}"
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(frame, text, (frame.shape[1] - tw - 8, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)


def main() -> int:
    args = parse_args()

    import cv2

    from alert.pipeline import ALERTPipeline
    from alert.utils.visualization import Visualizer

    cfg = _build_config(args)

    log.info("Building ALERTPipeline (seg=%s, risk=%s, alerts=%s, device=%s)",
             args.enable_seg, args.enable_risk, args.enable_alerts, args.device)
    try:
        pipeline = ALERTPipeline(cfg)
        pipeline.warmup()
    except Exception as exc:
        log.error("Failed to build pipeline: %s", exc)
        return 1
    viz = Visualizer(cfg)

    cap = _open_webcam(args.camera_index, args.width, args.height)
    if cap is None:
        return 1

    writer = None
    save_path = args.save_video
    window_name = args.window_name
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    log.info("Starting live loop — press 'q' or close the window to quit.")
    n_frames = 0
    n_errors = 0
    t_start = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                log.warning("Webcam returned no frame; retrying...")
                time.sleep(0.05)
                continue

            try:
                result = pipeline.process_frame(frame)
                vis_frame = viz.compose_frame(result)
                _draw_track_count(vis_frame, len(result.tracks))
            except Exception as exc:  # one bad frame should not kill the demo
                n_errors += 1
                log.exception("Per-frame pipeline error (#%d): %s", n_errors, exc)
                vis_frame = frame  # show raw frame so the window stays alive

            # Lazy-init writer once we know the actual frame size
            if save_path is not None and writer is None:
                h, w = vis_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(save_path, fourcc, 25.0, (w, h))
                if not writer.isOpened():
                    log.error("Could not open VideoWriter for %s", save_path)
                    writer = None
                    save_path = None

            if writer is not None:
                writer.write(vis_frame)

            cv2.imshow(window_name, vis_frame)
            n_frames += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # q or ESC
                break

            # Window closed by user (X button)
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        elapsed = time.perf_counter() - t_start
        avg_fps = n_frames / elapsed if elapsed > 0 else 0.0
        cap.release()
        if writer is not None:
            writer.release()
            log.info("Saved annotated video → %s", save_path)
        cv2.destroyAllWindows()
        log.info("Stopped. %d frames in %.1fs (avg %.1f FPS, %d per-frame errors)",
                 n_frames, elapsed, avg_fps, n_errors)

    return 0


if __name__ == "__main__":
    sys.exit(main())
