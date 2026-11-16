#!/usr/bin/env python3
"""
face_detect_v2.py - V2.0 Face Detection on 81-frame clips
"""

import os
import json
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis

WORKDIR = Path(os.environ.get("VEIL_ROOT", "veilbench_data"))
CLIPS_DIR = WORKDIR / "clips_81f"
ACTIVE_LIST = Path(os.environ.get("ACTIVE_LIST", WORKDIR / "manifests/clips_756.txt"))
OUT_PATH = WORKDIR / "annotations/v2_face_detect.json"

def face_size_bin(height_px: float) -> str:
    if height_px >= 160: return "ge160"
    elif height_px >= 80: return "r80_160"
    else: return "lt80"

def sample_mp4(path: Path, target_fps: float = 1.0):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened(): return
    native = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(native / target_fps))
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % step == 0: yield idx, frame
        idx += 1
    cap.release()

def frontality(bbox, kps):
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    face_w, face_h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    re, le = kps[0].astype(float), kps[1].astype(float)
    
    level = max(0.0, 1.0 - abs(re[1] - le[1]) / (0.3 * face_h))
    symmetry = max(0.0, 1.0 - abs(((re[0] + le[0]) / 2.0) - ((x1 + x2) / 2.0)) / (0.5 * face_w))
    span = min(abs(re[0] - le[0]) / (0.45 * face_w), 1.0)
    
    return float(level * symmetry * span)

def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with open(ACTIVE_LIST) as f:
        active_clips = [line.strip() for line in f if line.strip()]

    results = {}
    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            results = json.load(f)
            
    pending = [c for c in active_clips if c not in results]
    if not pending:
        print("All active clips processed.")
        return

    print("Loading SCRFD-10G-KPS...")
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    detector = app.det_model

    for i, cid in enumerate(pending):
        # Apply 100px pad ONLY for VoxCeleb face tracks
        pad = 100 if cid.startswith("id") else 0
        vid_path = CLIPS_DIR / f"{cid}.mp4"
        
        frames = list(sample_mp4(vid_path, target_fps=1.0))
        if not frames:
            results[cid] = {"error": "no_frames"}
            continue

        heights, front_scores, best_conf, best_frame = [], [], -1.0, -1
        n_detected = 0

        for frame_idx, bgr in frames:
            if pad > 0:
                bgr = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            
            bboxes, kpss = detector.detect(bgr, input_size=(640, 640))
            if bboxes is None or len(bboxes) == 0: continue

            n_detected += 1
            top = int(np.argmax(bboxes[:, 4]))
            bbox, conf = bboxes[top], float(bboxes[top][4])
            heights.append(float(bbox[3] - bbox[1]))
            front_scores.append(frontality(bbox, kpss[top]) if kpss is not None else 0.0)

            if conf > best_conf:
                best_conf, best_frame = conf, frame_idx

        if not heights:
            results[cid] = {"detected": False, "detect_rate": 0.0}
        else:
            results[cid] = {
                "detected": True,
                "detect_rate": round(n_detected / len(frames), 4),
                "face_size_median": round(float(np.median(heights)), 2),
                "face_size_bin": face_size_bin(float(np.median(heights))),
                "best_frame_idx": int(best_frame),
            }

        if (i + 1) % 50 == 0:
            with open(OUT_PATH, "w") as f: json.dump(results, f)
            print(f"[{i+1}/{len(pending)}] Processed.")

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=4)
    print("Stage 2a Complete.")

if __name__ == "__main__":
    main()
