#!/usr/bin/env python3
"""
pose_detect_v2.py — Stage 2b: V2.0 head pose annotation via SixDRepNet.
Reads the exact 81-frame clips and calculates pose using the new yaw bins.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import json
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
CLIPS_DIR = WORKDIR / "clips_81f"
FACE_JSON = WORKDIR / "annotations/v2_face_detect.json"
OUT_PATH = WORKDIR / "annotations/v2_pose.json"
WEIGHTS_PATH = Path(os.environ.get("SIXDREPNET_CKPT", WORKDIR / "6DRepNet_300W_LP_AFLW2000.pth"))

# ── Config ─────────────────────────────────────────────────────────────────────
CROP_MARGIN = 0.25
TRANSFORM = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def yaw_bin(yaw_deg: float) -> str:
    """V2.0 Yaw Bins: frontal (0-30), profile (30-60), extreme (>60)"""
    a = abs(yaw_deg)
    if a <= 30.0: return "frontal"
    elif a <= 60.0: return "profile"
    else: return "extreme"

def extract_frame(path: Path, frame_idx: int):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened(): return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def get_face_crop(detector, frame_bgr: np.ndarray, pad: int):
    padded = cv2.copyMakeBorder(frame_bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0]) if pad > 0 else frame_bgr
    bboxes, _ = detector.detect(padded, input_size=(640, 640))
    if bboxes is None or len(bboxes) == 0: return None

    top = int(np.argmax(bboxes[:, 4]))
    bbox = bboxes[top]
    x1, y1, x2, y2 = float(bbox[0]) - pad, float(bbox[1]) - pad, float(bbox[2]) - pad, float(bbox[3]) - pad
    h_orig, w_orig = frame_bgr.shape[:2]

    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - CROP_MARGIN * bw))
    y1 = max(0, int(y1 - CROP_MARGIN * bh))
    x2 = min(w_orig, int(x2 + CROP_MARGIN * bw))
    y2 = min(h_orig, int(y2 + CROP_MARGIN * bh))

    if x2 <= x1 or y2 <= y1: return None
    crop_bgr = frame_bgr[y1:y2, x1:x2]
    if crop_bgr.size == 0: return None

    return Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))

def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not FACE_JSON.exists():
        print(f"ERROR: {FACE_JSON} not found. Run face_detect_v2.py first.")
        return

    with open(FACE_JSON) as f:
        face_data = json.load(f)

    # Filter eligible clips: Must have a detected face, must NOT be MEAD
    eligible = {
        cid: v for cid, v in face_data.items()
        if v.get("detected") is True 
        and v.get("best_frame_idx") is not None
        and not ((cid.startswith("M") or cid.startswith("W")) and "____" in cid) # Skip MEAD
    }

    results = {}
    if OUT_PATH.exists():
        with open(OUT_PATH) as f: results = json.load(f)

    pending = [cid for cid in eligible if cid not in results]
    if not pending:
        print("All clips processed.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch device: {device}")

    print("Loading SCRFD-10G-KPS...")
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    detector = app.det_model

    print("Loading SixDRepNet...")
    gpu_id = 0 if device.type == "cuda" else -1
    detector_wrapper = SixDRepNet(gpu_id=gpu_id, dict_path=str(WEIGHTS_PATH))
    pose_model = detector_wrapper.model
    pose_model.to(device)
    pose_model.eval()

    n_err = 0
    for i, cid in enumerate(pending):
        frame_idx = eligible[cid]["best_frame_idx"]
        vid_path = CLIPS_DIR / f"{cid}.mp4"
        pad = 100 if cid.startswith("id") else 0 # 100px pad for VoxCeleb

        try:
            frame = extract_frame(vid_path, frame_idx)
            if frame is None:
                results[cid] = {"error": "frame_read_failed"}; n_err += 1; continue

            face_pil = get_face_crop(detector, frame, pad)
            if face_pil is None:
                results[cid] = {"error": "no_face_on_best_frame"}; n_err += 1; continue

            tensor = TRANSFORM(face_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                R_pred = pose_model(tensor)
                euler = compute_euler_angles_from_rotation_matrices(R_pred) * (180.0 / np.pi)
                y, p, r = euler[0, 1].item(), euler[0, 0].item(), euler[0, 2].item()

            results[cid] = {
                "yaw": round(float(y), 3),
                "pitch": round(float(p), 3),
                "roll": round(float(r), 3),
                "yaw_bin": yaw_bin(float(y)),
                "yaw_source": "6drepnet",
            }
        except Exception as e:
            results[cid] = {"error": str(e)}; n_err += 1

        if (i + 1) % 50 == 0:
            with open(OUT_PATH, "w") as f: json.dump(results, f)
            n_ok = sum(1 for v in results.values() if "yaw" in v)
            print(f"[{i+1}/{len(pending)}] ok={n_ok} err={n_err}")

    with open(OUT_PATH, "w") as f: json.dump(results, f, indent=4)
    print("Stage 2b Complete.")

if __name__ == "__main__":
    main()
