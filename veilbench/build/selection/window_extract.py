"""
window_extract.py — VEIL-Bench v2.0
Extract exactly 81 frames from each source clip.

Policies:
  max_yaw  : sparse SixDRepNet pass over full clip, pick window with
             highest mean |yaw|. Used for veil_id (VoxCeleb2).
  truncate : take frames 0-80. Used for veil_av (MEAD/CREMA-D).
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from sixdrepnet import SixDRepNet
from sixdrepnet.utils import compute_euler_angles_from_rotation_matrices

WINDOW = 81
POSE_SAMPLE_FPS = 5

WORKDIR = Path(os.environ.get("VEIL_ROOT", "veilbench_data"))
SHARE   = Path(os.environ.get("SHARE_ROOT", "raw_corpora"))

TRANSFORM = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Path resolver
# ---------------------------------------------------------------------------

def resolve_src(clip_id: str, corpus: str) -> tuple:
    """Returns (path, src_type) where src_type is 'video' or 'jpeg_dir'."""
    if corpus == "voxceleb2_test":
        parts = clip_id.split("__")
        path = SHARE / "vox-celeb-dataset/vox-celeb-raw/mp4" / parts[0] / parts[1] / (parts[2] + ".mp4")
        return path, "video"

    elif corpus == "cremad":
        parts = clip_id.split("____")
        path = WORKDIR / "raw/cremad/crema-d-mirror/VideoFlash" / (parts[1] + ".flv")
        return path, "video"

    elif corpus == "mead":
        parts = clip_id.split("____")
        path = SHARE / "MEAD/videos" / parts[0] / "video" / parts[1] / parts[2] / parts[3] / (parts[4] + ".mp4")
        return path, "video"

    elif corpus == "celebv_hq":
        parts = clip_id.split("____")
        filename = parts[1] + ".mp4"
        video_root = SHARE / "CelebV-HQ/path_to_videos_256_for_inf"
        for split in ["train", "test", "val"]:
            path = video_root / split / filename
            if path.exists():
                return path, "video"
        raise FileNotFoundError(f"CelebV-HQ clip not found: {filename}")

    else:
        raise ValueError(f"Unknown corpus: {corpus}")


# ---------------------------------------------------------------------------
# Frame reading
# ---------------------------------------------------------------------------

def get_video_info(src: Path, src_type: str) -> tuple:
    """Returns (total_frames, fps)."""
    if src_type == "video":
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {src}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total, fps
    elif src_type == "jpeg_dir":
        jpegs = sorted(src.glob("*.jpg")) or sorted(src.glob("*.jpeg"))
        if not jpegs:
            raise RuntimeError(f"No JPEGs in {src}")
        return len(jpegs), 30.0


def iter_frames_sparse(src: Path, src_type: str, fps: float) -> list:
    """Returns list of (frame_idx, PIL_RGB_image) at ~POSE_SAMPLE_FPS rate."""
    step = max(1, round(fps / POSE_SAMPLE_FPS))
    frames = []
    if src_type == "video":
        cap = cv2.VideoCapture(str(src))
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append((idx, Image.fromarray(rgb)))
            idx += 1
        cap.release()
    elif src_type == "jpeg_dir":
        jpegs = sorted(src.glob("*.jpg")) or sorted(src.glob("*.jpeg"))
        for idx, p in enumerate(jpegs):
            if idx % step == 0:
                frame = cv2.imread(str(p))
                if frame is not None:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append((idx, Image.fromarray(rgb)))
    return frames


# ---------------------------------------------------------------------------
# Pose model — copied verbatim from pose_detect.py
# ---------------------------------------------------------------------------

def load_pose_model(device: str):
    gpu_id = 0 if device == "cuda" else -1
    weights_path = os.environ.get("SIXDREPNET_CKPT", str(WORKDIR / "6DRepNet_300W_LP_AFLW2000.pth"))
    wrapper = SixDRepNet(gpu_id=gpu_id, dict_path=weights_path)
    model = wrapper.model
    model.to(device)
    model.eval()
    return model


def estimate_pose_pil(model, face_pil: Image.Image, device: str) -> float:
    """Returns yaw in degrees. Copied from pose_detect.py::estimate_pose."""
    tensor = TRANSFORM(face_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        R_pred = model(tensor)
    euler = compute_euler_angles_from_rotation_matrices(R_pred) * (180.0 / np.pi)
    yaw = euler[0, 1].item()
    return float(yaw)


# ---------------------------------------------------------------------------
# Window selection
# ---------------------------------------------------------------------------

def build_yaw_map(src: Path, src_type: str, fps: float,
                  pose_model, device: str) -> dict:
    """Returns {frame_idx: yaw_degrees} for sparsely sampled frames."""
    sparse = iter_frames_sparse(src, src_type, fps)
    yaw_map = {}
    for frame_idx, pil_img in sparse:
        try:
            yaw = estimate_pose_pil(pose_model, pil_img, device)
            yaw_map[frame_idx] = yaw
        except Exception:
            pass  # skip frames where pose fails
    return yaw_map


def select_max_yaw_window(yaw_map: dict, total_frames: int) -> tuple:
    """Returns (best_start_frame, mean_abs_yaw_in_window)."""
    if total_frames <= WINDOW:
        vals = list(yaw_map.values())
        return 0, float(np.mean([abs(y) for y in vals]) if vals else 0.0)

    max_start = total_frames - WINDOW
    sampled = sorted(yaw_map.keys())
    step = max(1, sampled[1] - sampled[0]) if len(sampled) > 1 else 1

    best_start, best_score = 0, -1.0
    for start in range(0, max_start + 1, step):
        end = start + WINDOW
        in_window = [abs(yaw_map[f]) for f in sampled if start <= f < end]
        if not in_window:
            continue
        score = float(np.mean(in_window))
        if score > best_score:
            best_score, best_start = score, start

    return min(best_start, max_start), best_score


# ---------------------------------------------------------------------------
# FFmpeg extraction
# ---------------------------------------------------------------------------

def extract_from_video(src: Path, dst: Path, start_frame: int, fps: float,
                       n_frames: int = WINDOW):
    dst.parent.mkdir(parents=True, exist_ok=True)
    start_sec = start_frame / fps
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.6f}",
        "-i", str(src),
        "-frames:v", str(n_frames),
        "-c:v", "libx264", "-crf", "18",
        "-an", str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{r.stderr}")
    _verify_frame_count(dst, expected=n_frames)


def extract_from_jpeg_dir(src: Path, dst: Path, start_frame: int, fps: float):
    dst.parent.mkdir(parents=True, exist_ok=True)
    jpegs = sorted(src.glob("*.jpg")) or sorted(src.glob("*.jpeg"))
    selected = jpegs[start_frame: start_frame + WINDOW]
    if len(selected) < WINDOW:
        raise RuntimeError(f"Not enough frames in {src}")

    duration = 1.0 / fps
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        concat_path = f.name
        for jpeg in selected:
            f.write(f"file '{jpeg.resolve()}'\n")
            f.write(f"duration {duration:.6f}\n")
        f.write(f"file '{selected[-1].resolve()}'\n")

    try:
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_path,
            "-c:v", "libx264", "-crf", "18",
            "-an", str(dst),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{r.stderr}")
    finally:
        os.unlink(concat_path)
    _verify_frame_count(dst)


def _verify_frame_count(dst: Path, expected: int = WINDOW):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "json", str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(r.stdout)
    n = int(data["streams"][0].get("nb_read_frames", 0))
    if n != expected:
        raise RuntimeError(f"Frame count mismatch: expected {expected}, got {n}")


# ---------------------------------------------------------------------------
# Per-clip processing
# ---------------------------------------------------------------------------

def process_clip(clip_id, corpus, policy, device, pose_model=None):
    dst = WORKDIR / "clips_81f" / f"{clip_id}.mp4"
    if dst.exists():
        return {"clip_id": clip_id, "status": "skipped"}

    try:
        src, src_type = resolve_src(clip_id, corpus)
        if not src.exists():
            return {"clip_id": clip_id, "status": "error",
                    "error": f"Source not found: {src}"}

        total_frames, fps = get_video_info(src, src_type)

        if total_frames <= WINDOW:
            # Clip already short enough — encode directly from frame 0
            if src_type == "jpeg_dir":
                extract_from_jpeg_dir(src, dst, 0, fps)
            else:
                extract_from_video(src, dst, 0, fps, n_frames=total_frames)
            return {"clip_id": clip_id, "status": "ok",
                    "start_frame": 0, "end_frame": total_frames - 1,
                    "mean_abs_yaw": None, "policy": "passthrough",
                    "fps": fps, "total_src_frames": total_frames}

        if policy == "truncate":
            start_frame, mean_abs_yaw = 0, None

        elif policy == "max_yaw":
            yaw_map = build_yaw_map(src, src_type, fps, pose_model, device)
            if not yaw_map:
                # Pose failed on all frames — fall back to truncate
                start_frame, mean_abs_yaw = 0, None
            else:
                start_frame, mean_abs_yaw = select_max_yaw_window(yaw_map, total_frames)

        if src_type == "jpeg_dir":
            extract_from_jpeg_dir(src, dst, start_frame, fps)
        else:
            extract_from_video(src, dst, start_frame, fps)

        return {"clip_id": clip_id, "status": "ok",
                "start_frame": start_frame,
                "end_frame": start_frame + WINDOW - 1,
                "mean_abs_yaw": mean_abs_yaw,
                "policy": policy, "fps": fps,
                "total_src_frames": total_frames}

    except Exception as e:
        return {"clip_id": clip_id, "status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",         required=True)
    ap.add_argument("--subset",           default=None)
    ap.add_argument("--policy",           choices=["max_yaw", "truncate"], default="max_yaw")
    ap.add_argument("--device",           default="cuda")
    ap.add_argument("--workers",          type=int, default=1)
    ap.add_argument("--window_manifest",  default="window_manifest.json")
    args = ap.parse_args()

    with open(args.manifest) as f:
        raw = json.load(f)
    clips = raw.get("clips", raw)

    if args.subset:
        clips = {k: v for k, v in clips.items()
                 if v.get("subset") == args.subset}

    print(f"Processing {len(clips)} clips  policy={args.policy}", flush=True)

    # Load pose model once if needed
    pose_model = None
    if args.policy == "max_yaw":
        print("Loading SixDRepNet ...", flush=True)
        pose_model = load_pose_model(args.device)
        print("SixDRepNet ready.", flush=True)

    (WORKDIR / "clips_81f").mkdir(parents=True, exist_ok=True)

    results = {}
    errors  = []

    for i, (clip_id, meta) in enumerate(clips.items()):
        corpus = meta.get("corpus", "")
        r = process_clip(clip_id, corpus, args.policy, args.device, pose_model)
        results[clip_id] = r

        status = r["status"]
        yaw_s  = f"  mean|yaw|={r['mean_abs_yaw']:.1f}°" if r.get("mean_abs_yaw") else ""
        start_s = f"  start={r.get('start_frame')}" if status == "ok" else ""
        err_s   = f"  {r.get('error','')}" if status == "error" else ""
        print(f"[{i+1}/{len(clips)}] {status:<12} {clip_id}{start_s}{yaw_s}{err_s}",
              flush=True)

        if status == "error":
            errors.append(r)

    with open(args.window_manifest, "w") as f:
        json.dump(results, f, indent=2)

    ok   = sum(1 for r in results.values() if r["status"] == "ok")
    skip = sum(1 for r in results.values() if r["status"] == "skipped")
    err  = len(errors)
    print(f"\nDone: ok={ok}  skipped={skip}  errors={err}")
    print(f"Window manifest → {args.window_manifest}")

    if errors:
        print("\nFailed clips:")
        for e in errors:
            print(f"  {e['clip_id']}: {e.get('error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
