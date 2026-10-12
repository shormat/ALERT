"""Run pipeline + generate TTS audio for risk alerts + mux into output video."""
from __future__ import annotations

import argparse
import io
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--base-config",         default=str(REPO / "config/base.yaml"))
    p.add_argument("--detection-config",    default=str(REPO / "config/detection/yolov8m.yaml"))
    p.add_argument("--segmentation-config", default=str(REPO / "config/segmentation/segformer_b2.yaml"))
    p.add_argument("--tracking-config",     default=str(REPO / "config/tracking/bytetrack.yaml"))
    p.add_argument("--pipeline-config",     default=str(REPO / "config/pipeline/default.yaml"))
    p.add_argument("--lang", default="en")
    return p.parse_args()


def main():
    import cv2
    from gtts import gTTS
    from pydub import AudioSegment
    import imageio_ffmpeg

    from alert.utils.config import load_config
    from alert.pipeline import ALERTPipeline
    from alert.utils.visualization import Visualizer

    args = parse_args()
    cfg = load_config(args.base_config, args.detection_config, args.segmentation_config,
                      args.tracking_config, args.pipeline_config)

    pipe = ALERTPipeline(cfg)
    viz = Visualizer(cfg)

    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp_video = str(Path(args.output).with_suffix(".novideo.mp4"))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (W, H))

    alerts: list[tuple[float, str, str]] = []  # (t_sec, level, msg)
    last_alert_at = -10.0  # cooldown
    cooldown = 3.0

    print(f"[1/3] running pipeline on {n_frames} frames @ {fps:.1f} fps...")
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = pipe.process_frame(frame)
        vis_frame = viz.compose_frame(result)
        writer.write(vis_frame)

        rr = result.risk_result
        if rr is not None:
            level = rr.alert_level
            t_sec = i / fps
            if level in ("WARNING", "CRITICAL") and (t_sec - last_alert_at) >= cooldown:
                msg = result.alert_message
                if msg is None:
                    msg = "Caution — potential hazard ahead" if level == "WARNING" else "Critical — obstacle ahead"
                alerts.append((t_sec, level, msg))
                last_alert_at = t_sec
        i += 1
        if i % 60 == 0:
            print(f"  frame {i}/{n_frames}")

    cap.release()
    writer.release()
    duration_sec = n_frames / fps
    print(f"[1/3] done. {len(alerts)} alerts collected over {duration_sec:.1f}s")

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    AudioSegment.converter = ff

    def mp3_buf_to_wav(mp3_bytes: bytes) -> AudioSegment:
        # decode mp3 -> wav via ffmpeg subprocess (avoids needing ffprobe)
        proc = subprocess.run([ff, "-y", "-i", "pipe:0", "-f", "wav", "pipe:1"],
                              input=mp3_bytes, capture_output=True, check=True)
        return AudioSegment.from_wav(io.BytesIO(proc.stdout))

    print(f"[2/3] generating TTS audio for {len(alerts)} alerts...")
    base = AudioSegment.silent(duration=int(duration_sec * 1000) + 500)
    cache: dict[str, AudioSegment] = {}
    for t_sec, level, msg in alerts:
        if msg not in cache:
            tts = gTTS(msg, lang=args.lang)
            buf = io.BytesIO(); tts.write_to_fp(buf)
            cache[msg] = mp3_buf_to_wav(buf.getvalue())
        seg = cache[msg]
        base = base.overlay(seg, position=int(t_sec * 1000))
        print(f"  {t_sec:6.2f}s  [{level}]  {msg}")

    tmp_audio = str(Path(args.output).with_suffix(".audio.wav"))
    base.export(tmp_audio, format="wav")
    print(f"[2/3] audio track → {tmp_audio}")

    print(f"[3/3] muxing video + audio → {args.output}")
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ff, "-y", "-i", tmp_video, "-i", tmp_audio,
                    "-c:v", "libx264", "-crf", "26", "-preset", "medium",
                    "-c:a", "aac", "-b:a", "96k", "-shortest", args.output],
                   check=True)
    Path(tmp_video).unlink(missing_ok=True)
    Path(tmp_audio).unlink(missing_ok=True)
    print(f"Done → {args.output}")


if __name__ == "__main__":
    main()
