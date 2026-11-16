#!/usr/bin/env python3
"""
pose_detect.py  —  Stage 2b: head pose annotation via SixDRepNet.

Reads annotations/{corpus}/face_detect.json, extracts best_frame per clip,
re-runs SCRFD on it to get the face bbox, crops and runs SixDRepNet for
yaw / pitch / roll.

MEAD is excluded — yaw comes from angle_label ground truth, not the model.

Idempotent. Safe to kill and re-run.

Usage (GPU node, bench env):
    export LD_LIBRARY_PATH=$(python -c 'import os,glob,sysconfig; print(":".join(glob.glob(os.path.join(sysconfig.get_paths()["purelib"],"nvidia","*","lib"))))'):$LD_LIBRARY_PATH
    python pose_detect.py --corpus voxceleb2_test
    python pose_detect.py --corpus cremad
    python pose_detect.py --corpus hdtf
"""

import os
# Suppress ONNX Runtime pthread_setaffinity_np warnings
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from insightface.app import FaceAnalysis
from sixdrepnet import SixDRepNet
from sixdrepnet.utils import compute_euler_angles_from_rotation_matrices

# ── Paths ──────────────────────────────────────────────────────────────────────
WORKDIR = Path(os.environ.get("VEIL_ROOT", "veilbench_data"))
SHARE   = Path(os.environ.get("SHARE_ROOT", "raw_corpora"))

CORPUS_CFG = {
    "voxceleb2_test": {
        "base":       SHARE / "vox-celeb-dataset/vox-celeb-raw/mp4",
        "mode":       "mp4",
        "native_fps": 25.0,
        "pad":        100,   # boundary suppression fix — same as face_detect.py
    },
    "cremad": {
        "base":       WORKDIR / "raw/cremad/crema-d-mirror/VideoFlash",
        "mode":       "mp4",
        "native_fps": 29.97,
        "pad":        0,
    },
    "hdtf": {
        "base":       SHARE / "HDTF-dataset/HDTF-RGB",
        "mode":       "mp4",
        "native_fps": 25.0,
        "pad":        0,
    },
}

# SixDRepNet preprocessing — ImageNet normalisation, 224px square crop
TRANSFORM = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

CROP_MARGIN = 0.25   # expand bbox by this fraction on each side before crop


# ── Yaw binning ────────────────────────────────────────────────────────────────
def yaw_bin(yaw_deg: float) -> str:
    a = abs(yaw_deg)
    if a <= 15.0:
        return "le15"
    elif a <= 40.0:
        return "r15_40"
    else:
        return "gt40"


# ── Clip-id → file path ────────────────────────────────────────────────────────
def clip_to_path(corpus: str, clip_id: str, cfg: dict) -> Path:
    base = cfg["base"]
    if corpus == "voxceleb2_test":
        parts = clip_id.split("__")
        return base / parts[0] / parts[1] / (parts[2] + ".mp4")
    elif corpus == "cremad":
        stem = clip_id.split("____", 1)[1]
        return base / (stem + ".flv")
    elif corpus == "hdtf":
        stem = clip_id.split("__", 1)[1]
        return base / (stem + ".mp4")
    return base / clip_id


# ── Frame extraction ───────────────────────────────────────────────────────────
def extract_frame(path: Path, frame_idx: int):
    """Return BGR frame at frame_idx, or None on failure."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


# ── Face crop ──────────────────────────────────────────────────────────────────
def get_face_crop(detector, frame_bgr: np.ndarray, pad: int):
    """
    Run SCRFD on (optionally padded) frame; return face crop from the
    ORIGINAL (unpadded) frame as a PIL RGB image, or None if no detection.

    pad > 0: add black border before detection, subtract offset from bbox
             so the returned crop is taken from real image pixels.
    """
    if pad > 0:
        padded = cv2.copyMakeBorder(
            frame_bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
    else:
        padded = frame_bgr

    bboxes, _ = detector.detect(padded, input_size=(640, 640))
    if bboxes is None or len(bboxes) == 0:
        return None

    # Best detection by confidence
    top  = int(np.argmax(bboxes[:, 4]))
    bbox = bboxes[top]
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])

    # Undo padding offset so coordinates are in original-frame space
    x1 -= pad; y1 -= pad; x2 -= pad; y2 -= pad

    h_orig, w_orig = frame_bgr.shape[:2]

    # Expand bbox by margin
    bw = x2 - x1
    bh = y2 - y1
    x1 = max(0, x1 - CROP_MARGIN * bw)
    y1 = max(0, y1 - CROP_MARGIN * bh)
    x2 = min(w_orig, x2 + CROP_MARGIN * bw)
    y2 = min(h_orig, y2 + CROP_MARGIN * bh)

    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    if x2 <= x1 or y2 <= y1:
        return None

    crop_bgr = frame_bgr[y1:y2, x1:x2]
    if crop_bgr.size == 0:
        return None

    return Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))


# ── Pose inference ─────────────────────────────────────────────────────────────
def estimate_pose(pose_model, face_pil: Image.Image, device: torch.device):
    """Return (yaw, pitch, roll) in degrees."""
    tensor = TRANSFORM(face_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        # The raw model outputs a 3x3 rotation matrix [1, 3, 3]
        R_pred = pose_model(tensor)
        
        # Convert the rotation matrix to Euler Angles (radians), then to degrees
        euler = compute_euler_angles_from_rotation_matrices(R_pred) * (180.0 / np.pi)
        
        # Extract Pitch, Yaw, Roll for the single item in our batch [0]
        p = euler[0, 0].item()
        y = euler[0, 1].item()
        r = euler[0, 2].item()
        
    return float(y), float(p), float(r)

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Stage 2b: SixDRepNet head pose")
    parser.add_argument("--corpus",     required=True, choices=list(CORPUS_CFG))
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--batch-save", type=int, default=50)
    args = parser.parse_args()

    corpus = args.corpus
    cfg    = CORPUS_CFG[corpus]
    pad    = cfg["pad"]

    face_path = WORKDIR / "annotations" / corpus / "face_detect.json"
    out_path  = WORKDIR / "annotations" / corpus / "pose.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not face_path.exists():
        sys.exit(f"ERROR: face_detect.json not found at {face_path}")

    with open(face_path) as f:
        face_data = json.load(f)

    # Only process clips where SCRFD found a face and best_frame_idx is set
    eligible = {
        cid: v for cid, v in face_data.items()
        if v.get("detected") is True and v.get("best_frame_idx") is not None
    }
    print(f"[{corpus}] eligible clips (detected=True): {len(eligible)}", flush=True)

    results = {}
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
        print(f"[{corpus}] resuming — {len(results)} clips already done", flush=True)

    pending = [cid for cid in eligible if cid not in results]
    print(f"[{corpus}] to process: {len(pending)}", flush=True)

    if not pending:
        print("Nothing to do.", flush=True)
        return

    # ── CUDA check ─────────────────────────────────────────────────────────────
    import onnxruntime as ort
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        print("WARNING: CUDAExecutionProvider not available — set LD_LIBRARY_PATH first")
        print("Continuing on CPU (will be slow).\n", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch device: {device}", flush=True)

    # ── Load SCRFD ─────────────────────────────────────────────────────────────
    print("Loading SCRFD-10G-KPS (buffalo_l) ...", flush=True)
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    detector = app.det_model
    print("SCRFD ready.", flush=True)

    # ── Load SixDRepNet ────────────────────────────────────────────────────────
    # ── Load SixDRepNet ────────────────────────────────────────────────────────
    print("Loading SixDRepNet ...", flush=True)
    
    # The pip package uses a wrapper class. We pass gpu_id here so it handles weights properly.
    gpu_id = 0 if device.type == "cuda" else -1
    
    # Provide the path to the manually downloaded weights to bypass the broken 404 URL
    weights_path = str(WORKDIR / "6DRepNet_300W_LP_AFLW2000.pth")
    detector_wrapper = SixDRepNet(gpu_id=gpu_id, dict_path=weights_path)
    
    # We extract the underlying PyTorch nn.Module from the wrapper.
    # This ensures your custom `estimate_pose()` tensor logic works as intended.
    pose_model = detector_wrapper.model
    pose_model.to(device)
    pose_model.eval()
    
    print(f"SixDRepNet ready. pad={pad}px\n", flush=True)

    n_err = 0

    for i, clip_id in enumerate(pending):
        frame_idx = eligible[clip_id]["best_frame_idx"]
        path      = clip_to_path(corpus, clip_id, cfg)

        try:
            frame = extract_frame(path, frame_idx)
            if frame is None:
                results[clip_id] = {"error": "frame_read_failed"}
                n_err += 1
                continue

            face_pil = get_face_crop(detector, frame, pad)
            if face_pil is None:
                results[clip_id] = {"error": "no_face_on_best_frame"}
                n_err += 1
                continue

            yaw, pitch, roll = estimate_pose(pose_model, face_pil, device)
            results[clip_id] = {
                "yaw":        round(yaw,   3),
                "pitch":      round(pitch, 3),
                "roll":       round(roll,  3),
                "yaw_bin":    yaw_bin(yaw),
                "yaw_source": "6drepnet",
            }

        except Exception as e:
            # Let's print the exact error and force the script to stop so we can read it
            print(f"\nCRASH EXCEPTION on clip {clip_id}:")
            print(repr(e))
            import traceback
            traceback.print_exc()
            import sys
            sys.exit(1)

        if (i + 1) % args.batch_save == 0:
            with open(out_path, "w") as f:
                json.dump(results, f)
            n_ok = sum(1 for v in results.values() if "yaw" in v)
            print(f"  [{i+1:>5}/{len(pending)}]  ok={n_ok}  err={n_err}", flush=True)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    n_ok = sum(1 for v in results.values() if "yaw" in v)
    print(f"\n[{corpus}] DONE")
    print(f"  total   : {len(results)}")
    print(f"  ok      : {n_ok}")
    print(f"  errors  : {n_err}")
    print(f"  output  : {out_path}")


if __name__ == "__main__":
    main()