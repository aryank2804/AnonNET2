#!/usr/bin/env python3
"""
Batch baseline anonymization for AnonNET2 comparisons.

Supports three methods:
  blur         — Gaussian blur applied to detected face regions
  pixelation   — Pixelation applied to detected face regions
  deepprivacy2 — DeepPrivacy2 (WACV 2023), run frame-by-frame, reassembled to video

Dependencies:
  pip install opencv-python-headless  # for blur / pixelation (no mediapipe needed)
  pip install git+https://github.com/hukkelas/deep_privacy2  # for deepprivacy2
  ffmpeg must be on PATH for all methods

Usage examples:
  python run_baselines.py --method blur        --input_dir data/videos --output_dir out/blur
  python run_baselines.py --method pixelation  --input_dir data/videos --output_dir out/pixelation
  python run_baselines.py --method deepprivacy2 --input_dir data/videos --output_dir out/dp2 \
      --dp2_dir /path/to/deep_privacy2 --dp2_config configs/anonymizers/face.py

Notes:
  - Input directory is walked recursively; subdirectory structure is mirrored in output.
  - Blur kernel size and pixelation block size are set proportionally to each face bounding box.
  - DeepPrivacy2 has no native video pipeline; this script handles frame extraction
    and reassembly, preserving original audio in all methods.
  - Already-processed files are skipped (re-run safe).
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


# ── Video helpers ─────────────────────────────────────────────────────────────

def get_video_fps(video_path: Path) -> str:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    streams = json.loads(result.stdout)["streams"]
    video_stream = next(s for s in streams if s["codec_type"] == "video")
    return video_stream["r_frame_rate"]


def mux_audio(silent_video: str, original_video: str, output_path: str) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", silent_video,
            "-i", original_video,
            "-c:v", "copy",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-c:a", "copy",
            output_path,
        ],
        check=True,
    )


def extract_frames(video_path: Path, out_dir: str) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(video_path),
            os.path.join(out_dir, "frame_%06d.png"),
        ],
        check=True,
    )


def reassemble_frames(frames_dir: str, fps: str, tmp_output: str) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", fps,
            "-i", os.path.join(frames_dir, "frame_%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            tmp_output,
        ],
        check=True,
    )


# ── Face detection (OpenCV Haar cascade — no extra downloads needed) ──────────

def make_face_detector():
    cascade_path = os.environ.get("VEIL_ROOT", "veilbench_data") + "/haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        sys.exit("Failed to load Haar cascade. Check your OpenCV installation.")
    return detector


def detect_faces(frame: np.ndarray, detector) -> list:
    """Return list of (x, y, w, h) pixel bounding boxes."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if len(boxes) == 0:
        return []
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in boxes]


# ── Blur ──────────────────────────────────────────────────────────────────────

def apply_gaussian_blur(frame: np.ndarray, boxes: list) -> np.ndarray:
    out = frame.copy()
    for (x, y, w, h) in boxes:
        k = max(3, int(h * 0.3))
        k = k if k % 2 == 1 else k + 1
        out[y:y+h, x:x+w] = cv2.GaussianBlur(out[y:y+h, x:x+w], (k, k), 0)
    return out


# ── Pixelation ────────────────────────────────────────────────────────────────

def apply_pixelation(frame: np.ndarray, boxes: list) -> np.ndarray:
    out = frame.copy()
    for (x, y, w, h) in boxes:
        small_h = max(1, int(h * 0.05))
        small_w = max(1, int(w * 0.05))
        roi = out[y:y+h, x:x+w]
        small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        out[y:y+h, x:x+w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return out


# ── Per-method video processors ───────────────────────────────────────────────

def process_blur_or_pixel(video_path: Path, output_path: Path, method: str) -> None:
    cap = cv2.VideoCapture(str(video_path))
    fps_float = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp_video = str(output_path.with_suffix(".tmp.mp4"))
    writer = cv2.VideoWriter(
        tmp_video, cv2.VideoWriter_fourcc(*"mp4v"), fps_float, (width, height)
    )

    detector = make_face_detector()
    frame_idx = 0
    faces_last = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        boxes = detect_faces(frame, detector)
        faces_last = len(boxes)
        if method == "blur":
            frame = apply_gaussian_blur(frame, boxes)
        else:
            frame = apply_pixelation(frame, boxes)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mux_audio(tmp_video, str(video_path), str(output_path))
    os.remove(tmp_video)
    print(f"    -> {frame_idx} frames, {faces_last} faces in last frame")


def process_deepprivacy2(
    video_path: Path, output_path: Path, dp2_dir: str, dp2_config: str
) -> None:
    fps = get_video_fps(video_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        frames_in = os.path.join(tmpdir, "frames_in")
        frames_out = os.path.join(tmpdir, "frames_out")
        os.makedirs(frames_in)
        os.makedirs(frames_out)

        extract_frames(video_path, frames_in)
        n_frames = len(list(Path(frames_in).glob("*.png")))
        print(f"    Extracted {n_frames} frames")

        subprocess.run(
            [
                "bash", "run_dp2.sh", dp2_config,
                "-i", frames_in,
                "--output_path", frames_out,
            ],
            cwd=dp2_dir, env={**os.environ, "PYTHONPATH": dp2_dir},
            check=True,
        )

        out_frames = sorted(Path(frames_out).glob("*.png"))
        if len(out_frames) == 0:
            raise RuntimeError(
                f"DeepPrivacy2 produced no output frames in {frames_out}. "
                "Check config path and model weights."
            )

        if out_frames[0].name != "frame_000001.png":
            for i, f in enumerate(sorted(out_frames), start=1):
                f.rename(Path(frames_out) / f"frame_{i:06d}.png")

        tmp_video = str(output_path.with_suffix(".tmp.mp4"))
        reassemble_frames(frames_out, fps, tmp_video)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        mux_audio(tmp_video, str(video_path), str(output_path))
        os.remove(tmp_video)
        print(f"    -> reassembled {len(out_frames)} frames")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Batch baseline anonymization for AnonNET2")
    p.add_argument("--method", required=True, choices=["blur", "pixelation", "deepprivacy2"])
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--ext", default=".mp4")
    p.add_argument("--dp2_dir", default=None)
    p.add_argument("--dp2_config", default="configs/anonymizers/face.py")
    p.add_argument("--skip_existing", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()

    if args.method == "deepprivacy2" and not args.dp2_dir:
        sys.exit("--dp2_dir is required when --method deepprivacy2")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(input_dir.rglob(f"*{args.ext}"))
    if not videos:
        sys.exit(f"No {args.ext} files found under {input_dir}")

    print(f"Method : {args.method}")
    print(f"Input  : {input_dir}  ({len(videos)} videos)")
    print(f"Output : {output_dir}")
    print()

    ok, skipped, failed = 0, 0, 0

    for i, video_path in enumerate(videos, start=1):
        rel = video_path.relative_to(input_dir)
        out_path = output_dir / rel
        print(f"[{i}/{len(videos)}] {rel}")

        if args.skip_existing and out_path.exists():
            print("    Skipped (already exists)")
            skipped += 1
            continue

        try:
            if args.method in ("blur", "pixelation"):
                process_blur_or_pixel(video_path, out_path, args.method)
            else:
                process_deepprivacy2(video_path, out_path, args.dp2_dir, args.dp2_config)
            ok += 1
        except Exception as e:
            print(f"    ERROR: {e}")
            failed += 1

    print()
    print(f"Done -- {ok} processed, {skipped} skipped, {failed} failed.")


if __name__ == "__main__":
    main()
