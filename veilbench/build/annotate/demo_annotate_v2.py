#!/usr/bin/env python3
"""
demo_annotate_v2.py - V2.0 Demographics Annotation via FaceXFormer.
Reads directly from clips_81f/ and v2_face_detect.json.
Skips CREMA-D (uses ground-truth CSV).
"""

import os
import sys
import json
import argparse
from pathlib import Path
from PIL import Image

import cv2
import torch
import torchvision
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from insightface.app import FaceAnalysis

# Add facexformer to path
sys.path.append(str(Path(os.environ.get("FACEXFORMER_REPO", "./facexformer")).resolve()))
from network import FaceXFormer

# ── Paths ──────────────────────────────────────────────────────────────────────
WORKDIR = Path(os.environ.get("VEIL_ROOT", "veilbench_data"))
CLIPS_DIR = WORKDIR / "clips_81f"
ACTIVE_LIST = Path(os.environ.get("ACTIVE_LIST", WORKDIR / "manifests/clips_756.txt"))
FACE_JSON = WORKDIR / "annotations/v2_face_detect.json"
OUT_PATH = WORKDIR / "annotations/v2_demographics.json"
MODEL_PATH = Path(os.environ.get("FACEXFORMER_CKPT", WORKDIR / "facexformer/ckpts/model.pt"))

# Standard FairFace mappings used by FaceXFormer
AGE_MAP = {0: "0-2", 1: "3-9", 2: "10-19", 3: "20-29", 4: "30-39", 5: "40-49", 6: "50-59", 7: "60-69", 8: "70+"}
GENDER_MAP = {0: "Male", 1: "Female"}
RACE_MAP = {0: "White", 1: "Black", 2: "Latino_Hispanic", 3: "East Asian", 4: "Southeast Asian", 5: "Indian", 6: "Middle Eastern"}

def main(args):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── Load InsightFace ───────────────────────────────────────────────────────
    print("Loading InsightFace (SCRFD)...")
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=args.gpu_num, det_size=(640, 640))

    # ── Load FaceXFormer ───────────────────────────────────────────────────────
    print("Loading FaceXFormer...")
    model = FaceXFormer().to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict_backbone'])
    model.eval()

    transforms_image = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(224, 224), interpolation=InterpolationMode.BICUBIC),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ── Filter Eligible Clips ──────────────────────────────────────────────────
    with open(ACTIVE_LIST) as f:
        active_clips = [line.strip() for line in f if line.strip()]

    if not FACE_JSON.exists():
        print(f"ERROR: {FACE_JSON} missing. Run face_detect_v2.py first.")
        return

    with open(FACE_JSON, "r") as f:
        face_data = json.load(f)

    demo_results = {}
    if OUT_PATH.exists():
        with open(OUT_PATH, "r") as f:
            demo_results = json.load(f)
        print(f"Resuming: Loaded {len(demo_results)} existing annotations.")

    # Eligible: In active list, NOT cremad, has best_frame_idx, not already processed
    clips_to_process = [
        cid for cid in active_clips
        if not cid.startswith("cremad")
        and cid in face_data
        and face_data[cid].get("detected", False)
        and face_data[cid].get("best_frame_idx") is not None
        and cid not in demo_results
    ]
    
    print(f"Clips remaining to process: {len(clips_to_process)}")
    if not clips_to_process:
        print("All active clips processed.")
        return

    # ── FaceXFormer Setup ──────────────────────────────────────────────────────
    dummy_labels = {
        "segmentation": torch.zeros([1, 224, 224]).to(device),
        "lnm_seg": torch.zeros([1, 5, 2]).to(device),
        "landmark": torch.zeros([1, 68, 2]).to(device),
        "headpose": torch.zeros([1, 3]).to(device),
        "attribute": torch.zeros([1, 40]).to(device),
        "a_g_e": torch.zeros([1, 3]).to(device),
        "visibility": torch.zeros([1, 29]).to(device)
    }
    task_tensor = torch.tensor([4]).to(device)

    # ── Inference Loop ─────────────────────────────────────────────────────────
    for i, clip_id in enumerate(tqdm(clips_to_process, desc="Annotating demographics")):
        vid_path = CLIPS_DIR / f"{clip_id}.mp4"
        if not vid_path.exists():
            continue

        best_frame_idx = face_data[clip_id]["best_frame_idx"]

        cap = cv2.VideoCapture(str(vid_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            continue

        # VoxCeleb requires 100px padding to avoid boundary suppression
        pad = 100 if clip_id.startswith("id") else 0
        if pad > 0:
            padded_frame = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0,0,0])
            faces = app.get(padded_frame)
        else:
            faces = app.get(frame)

        if not faces:
            continue

        # Get largest face and adjust bbox for padding
        largest_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        x1, y1, x2, y2 = [int(v) for v in largest_face.bbox]
        x1 -= pad; y1 -= pad; x2 -= pad; y2 -= pad

        # Clamp bounds
        h_img, w_img = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = image.crop((x1, y1, x2, y2))
        img_tensor = transforms_image(image).unsqueeze(0).to(device)

        with torch.no_grad():
            _, _, _, _, age_out, gender_out, race_out, _ = model(img_tensor, dummy_labels, task_tensor)
            age_pred = torch.argmax(age_out, dim=1)[0].item()
            gender_pred = torch.argmax(gender_out, dim=1)[0].item()
            race_pred = torch.argmax(race_out, dim=1)[0].item()

        demo_results[clip_id] = {
            "age_group": AGE_MAP.get(age_pred, str(age_pred)),
            "gender": GENDER_MAP.get(gender_pred, str(gender_pred)),
            "ethnicity": RACE_MAP.get(race_pred, str(race_pred)),
            "demo_source": "facexformer"
        }

        if (i + 1) % args.batch_save == 0:
            with open(OUT_PATH, "w") as f:
                json.dump(demo_results, f, indent=2)

    with open(OUT_PATH, "w") as f:
        json.dump(demo_results, f, indent=2)
    print(f"\nFinished. Total demographics saved: {len(demo_results)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_num", type=int, default=0)
    parser.add_argument("--batch_save", type=int, default=50)
    args = parser.parse_args()
    main(args)
