#!/usr/bin/env python3
"""
metrics_pose_expr_v2.py — Pose and Expression preservation for VEIL-Bench v2.0

Env: bench  (pip install sixdrepnet mediapipe)

Computes two attribute preservation metrics from DreamID-V Table 1:

  Pose       Mean absolute yaw error vs manifest ground truth (SixDRepNet).
             Lower = anonymized face preserves original head pose better.
             Ground truth pose is already in the manifest from pose_detect.py.

  Expression Mean L2 distance of normalised MediaPipe FaceMesh landmarks
             between original and anonymized frames at best_frame_idx.
             Lower = expression better preserved after anonymization.

Before running:
    export LD_LIBRARY_PATH=$(python -c 'import os,glob,sysconfig; \
      print(":".join(glob.glob(os.path.join(sysconfig.get_paths()["purelib"],\
      "nvidia","*","lib"))))'):$LD_LIBRARY_PATH
    module load cuda/12.2.1_gcc-10.4.0

Usage:
    python metrics_pose_expr_v2.py \\
        --manifest    $VEIL_ROOT/manifests/veilbench_manifest_v2.json \\
        --clips-dir   $VEIL_ROOT/clips_756/ \\
        --anon-dv     $VEIL_ROOT/anonymized/dreamidv_pad100/anonymized/ \\
        --anon-an     $VEIL_ROOT/anonymized/anonnet/ \\
        --sixd-weights $VEIL_ROOT/6DRepNet_300W_LP_AFLW2000.pth \\
        --out         $VEIL_ROOT/results/pose_expr_results.json

Output schema:
    {
      "dreamidv": {
        "overall": {"pose_mae": 3.2, "expression_err": 0.041, "n_clips": 600},
        "by_yaw_bin": {"frontal": {...}, "profile": {...}, "extreme": {...}}
      },
      "anonnet": { ... }
    }
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

logging.basicConfig(format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)
log = logging.getLogger(__name__)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov")
YAW_BINS   = ("frontal", "profile", "extreme")

BORDER_PX    = 100   # VoxCeleb2 boundary suppression fix (same as pose_detect.py)
CROP_MARGIN  = 0.25  # expand bbox by this fraction before SixDRepNet crop

# SixDRepNet preprocessing — identical to pose_detect.py
SIXD_TRANSFORM = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


# ── model loading ─────────────────────────────────────────────────────────────

def load_detector():
    """SCRFD face detector from InsightFace buffalo_l."""
    from insightface.app import FaceAnalysis  # type: ignore
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app.det_model


def load_sixdrepnet(weights_path: str):
    """Load SixDRepNet — identical pattern to pose_detect.py."""
    from sixdrepnet import SixDRepNet  # type: ignore
    gpu_id = 0 if DEVICE.type == "cuda" else -1
    wrapper = SixDRepNet(gpu_id=gpu_id, dict_path=weights_path)
    model = wrapper.model
    model.to(DEVICE).eval()
    return model


def load_mediapipe():
    """MediaPipe FaceMesh for expression landmarks."""
    import mediapipe as mp  # type: ignore
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )


# ── video I/O ─────────────────────────────────────────────────────────────────

def find_video(directory: Path, clip_id: str) -> Optional[Path]:
    for ext in VIDEO_EXTS:
        p = directory / f"{clip_id}{ext}"
        if p.exists():
            return p
    return None


def read_frame(path: Path, frame_idx: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = min(frame_idx, total - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


# ── face crop (identical to pose_detect.py) ───────────────────────────────────

def get_face_crop(detector, frame_bgr: np.ndarray, pad: int) -> Optional[Image.Image]:
    if pad > 0:
        padded = cv2.copyMakeBorder(
            frame_bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0
        )
    else:
        padded = frame_bgr

    bboxes, _ = detector.detect(padded, input_size=(640, 640))
    if bboxes is None or len(bboxes) == 0:
        return None

    top  = int(np.argmax(bboxes[:, 4]))
    bbox = bboxes[top]
    x1, y1, x2, y2 = float(bbox[0])-pad, float(bbox[1])-pad, \
                      float(bbox[2])-pad, float(bbox[3])-pad

    h, w = frame_bgr.shape[:2]
    bw, bh = x2-x1, y2-y1
    x1 = max(0, x1 - CROP_MARGIN*bw);  y1 = max(0, y1 - CROP_MARGIN*bh)
    x2 = min(w, x2 + CROP_MARGIN*bw);  y2 = min(h, y2 + CROP_MARGIN*bh)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame_bgr[y1:y2, x1:x2]
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)) if crop.size else None


# ── pose estimation (identical to pose_detect.py) ────────────────────────────

def estimate_pose(model, face_pil: Image.Image) -> tuple[float, float, float]:
    """Returns (yaw, pitch, roll) in degrees."""
    from sixdrepnet.utils import compute_euler_angles_from_rotation_matrices  # type: ignore
    tensor = SIXD_TRANSFORM(face_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        R = model(tensor)
        euler = compute_euler_angles_from_rotation_matrices(R) * (180.0 / np.pi)
        y, p, r = euler[0,1].item(), euler[0,0].item(), euler[0,2].item()
    return float(y), float(p), float(r)


# ── expression estimation (MediaPipe FaceMesh) ────────────────────────────────

def get_expression_landmarks(mp_face_mesh, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Return normalised expression-sensitive landmarks as a flat vector.
    Uses MediaPipe FaceMesh 478-landmark model.
    Normalises by face bounding box size so expression is pose-invariant.
    Returns None if no face detected.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = mp_face_mesh.process(rgb)
    if not result.multi_face_landmarks:
        return None

    lm = result.multi_face_landmarks[0].landmark
    pts = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32)  # (478, 3)

    # Normalise by bounding box
    bbox_range = pts.max(0) - pts.min(0)
    scale = np.linalg.norm(bbox_range[:2]) + 1e-8
    pts_norm = (pts - pts.mean(0)) / scale

    # Expression-sensitive subset: mouth, brows, eyes (indices from MediaPipe docs)
    expr_indices = list(range(0, 17)) + list(range(17, 68))  # mouth + brow region
    return pts_norm[expr_indices].flatten()


def expression_error(mp_face_mesh, orig_bgr: np.ndarray,
                     anon_bgr: np.ndarray) -> Optional[float]:
    """Mean L2 distance of normalised expression landmarks."""
    orig_lm = get_expression_landmarks(mp_face_mesh, orig_bgr)
    anon_lm = get_expression_landmarks(mp_face_mesh, anon_bgr)
    if orig_lm is None or anon_lm is None:
        return None
    return float(np.linalg.norm(orig_lm - anon_lm))


# ── per-clip processing ───────────────────────────────────────────────────────

def process_clip(
    clip_id:      str,
    meta:         dict,
    orig_dir:     Path,
    anon_dir:     Path,
    detector,
    sixd_model,
    mp_face_mesh,
) -> Optional[dict]:
    """
    Returns {"pose_mae": float, "expression_err": float} or None if processing fails.
    """
    frame_idx = meta.get("best_frame_idx")
    if frame_idx is None:
        return None

    gt_yaw   = meta.get("yaw")
    gt_pitch = meta.get("pitch")
    gt_roll  = meta.get("roll")
    if gt_yaw is None:
        return None

    # Find anonymized video
    anon_path = find_video(anon_dir, clip_id)
    if anon_path is None:
        return None

    # Find original video (for expression comparison)
    orig_path = find_video(orig_dir, clip_id)

    # Read frames
    anon_frame = read_frame(anon_path, frame_idx)
    if anon_frame is None:
        return None

    # ── Pose ──────────────────────────────────────────────────────────────────
    face_pil = get_face_crop(detector, anon_frame, BORDER_PX)
    pose_mae = None
    if face_pil is not None:
        try:
            yaw, pitch, roll = estimate_pose(sixd_model, face_pil)
            pose_mae = float(abs(yaw - gt_yaw))   # yaw MAE
        except Exception:
            pass

    # ── Expression ────────────────────────────────────────────────────────────
    expr_err = None
    if orig_path is not None and mp_face_mesh is not None:
        orig_frame = read_frame(orig_path, frame_idx)
        if orig_frame is not None:
            try:
                expr_err = expression_error(mp_face_mesh, orig_frame, anon_frame)
            except Exception:
                pass

    if pose_mae is None and expr_err is None:
        return None

    return {"pose_mae": pose_mae, "expression_err": expr_err}


# ── aggregation ───────────────────────────────────────────────────────────────

def aggregate(per_clip: dict, ids: list[str]) -> dict:
    pose_vals = [per_clip[c]["pose_mae"]      for c in ids
                 if per_clip.get(c) and per_clip[c].get("pose_mae") is not None]
    expr_vals = [per_clip[c]["expression_err"] for c in ids
                 if per_clip.get(c) and per_clip[c].get("expression_err") is not None]
    return {
        "pose_mae":      float(np.mean(pose_vals)) if pose_vals else None,
        "expression_err": float(np.mean(expr_vals)) if expr_vals else None,
        "n_clips":       len(pose_vals),
    }


def build_results(clips_meta: dict, all_ids: list[str], per_clip: dict) -> dict:
    by_yaw: dict = {y: [] for y in YAW_BINS}
    for cid, m in clips_meta.items():
        yb = m.get("yaw_bin")
        if yb in by_yaw:
            by_yaw[yb].append(cid)

    return {
        "overall":    aggregate(per_clip, all_ids),
        "by_yaw_bin": {yb: aggregate(per_clip, ids) for yb, ids in by_yaw.items() if ids},
    }


# ── main ──────────────────────────────────────────────────────────────────────

def process_split(tag: str, clip_ids: list[str], clips_meta: dict,
                  orig_dir: Path, anon_dir: Path,
                  detector, sixd_model, mp_face_mesh) -> dict:
    per_clip: dict = {}
    n = len(clip_ids)
    for i, cid in enumerate(clip_ids):
        if i % 50 == 0 or i == n - 1:
            log.info(f"[{tag}] {i+1}/{n}")
        per_clip[cid] = process_clip(
            cid, clips_meta[cid], orig_dir, anon_dir,
            detector, sixd_model, mp_face_mesh
        )
    return per_clip


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest",      required=True)
    p.add_argument("--clips-dir",     required=True,
                   help="Original clips dir (for expression comparison)")
    p.add_argument("--anon-dv",       required=True)
    p.add_argument("--anon-an",       default=None)
    p.add_argument("--sixd-weights",  required=True,
                   help="Path to 6DRepNet_300W_LP_AFLW2000.pth")
    p.add_argument("--out",           default="results/pose_expr_results.json")
    p.add_argument("--no-expression", action="store_true",
                   help="Skip expression metric (skip mediapipe)")
    p.add_argument("--max-clips",     type=int, default=None)
    args = p.parse_args()

    with open(args.manifest) as fh:
        manifest = json.load(fh)

    clips_meta = manifest["clips"]
    # Pose only meaningful for VEIL-ID (has gt pose in manifest)
    veil_id = [c for c, m in clips_meta.items()
               if m.get("subset") == "veil_id" and m.get("yaw") is not None]
    if args.max_clips:
        veil_id = veil_id[:args.max_clips]
    log.info(f"VEIL-ID clips with pose gt: {len(veil_id)}")

    log.info("Loading models ...")
    detector    = load_detector()
    sixd_model  = load_sixdrepnet(args.sixd_weights)
    mp_face_mesh = None if args.no_expression else load_mediapipe()
    if mp_face_mesh:
        log.info("MediaPipe FaceMesh loaded.")

    results: dict = {}

    for tag, anon_dir in [("dreamidv", Path(args.anon_dv)),
                           ("anonnet",  Path(args.anon_an) if args.anon_an else None)]:
        if anon_dir is None:
            continue
        log.info(f"\n=== {tag} ===")
        per_clip = process_split(
            tag, veil_id, clips_meta,
            Path(args.clips_dir), anon_dir,
            detector, sixd_model, mp_face_mesh
        )
        results[tag] = build_results(
            {c: clips_meta[c] for c in veil_id}, veil_id, per_clip
        )
        ov = results[tag]["overall"]
        log.info(f"{tag} — pose_mae={ov['pose_mae']:.3f}°  "
                 f"expression_err={ov['expression_err']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    log.info(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
