#!/usr/bin/env python3
"""
face_detect.py - Stage 2a: face detection annotation via SCRFD-10G-KPS.

Reads annotations/{corpus}/pool.json, samples frames at 1 fps, runs
SCRFD-10G-KPS (InsightFace / buffalo_l), and writes per-clip face stats to
annotations/{corpus}/face_detect.json.

Idempotent: resumes from existing output file. Safe to kill and re-run.

Usage (GPU node, bench env):
    export LD_LIBRARY_PATH=$(python -c 'import os,glob,sysconfig; print(":".join(glob.glob(os.path.join(sysconfig.get_paths()["purelib"],"nvidia","*","lib"))))'):$LD_LIBRARY_PATH
    python face_detect.py --corpus voxceleb2_test --pad 100
    python face_detect.py --corpus cremad
    python face_detect.py --corpus hdtf
    python face_detect.py --corpus mead   # run after snr_mead + pool_filter
"""

import os
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis

# ── Paths ──────────────────────────────────────────────────────────────────────
WORKDIR = Path(os.environ.get("VEIL_ROOT", "veilbench_data"))
SHARE   = Path(os.environ.get("SHARE_ROOT", "raw_corpora"))

CORPUS_CFG = {
    "voxceleb2_test": {
        "base":       SHARE / "vox-celeb-dataset/vox-celeb-raw/mp4",
        "mode":       "mp4",
        "native_fps": 25.0,
    },
    "cremad": {
        "base":       WORKDIR / "raw/cremad/crema-d-mirror/VideoFlash",
        "mode":       "mp4",        # cv2 handles .flv
        "native_fps": 29.97,
    },
    "hdtf": {
        "base":       SHARE / "HDTF-dataset/HDTF-RGB",
        "mode":       "mp4",   # was jpeg_noext — actual files are mp4
        "native_fps": 25.0,
        "pad":        0,
    },
    "mead": {
        "base":       SHARE / "MEAD/videos",
        "mode":       "jpeg",       # JPEG frames with .jpg extension
        "native_fps": 30.0,
    },
}

# MEAD angles excluded — steep pitch makes face bbox unreliable
MEAD_SKIP_ANGLES = {"top", "down"}


# ── Face-size binning ──────────────────────────────────────────────────────────
def face_size_bin(height_px: float) -> str:
    if height_px >= 160:
        return "ge160"
    elif height_px >= 80:
        return "r80_160"
    else:
        return "lt80"


# ── Frame sampling ─────────────────────────────────────────────────────────────
def sample_mp4(path: Path, target_fps: float = 1.0):
    """Yield (frame_idx, bgr) at ~target_fps from an mp4/flv."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return
    native = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step   = max(1, round(native / target_fps))
    idx    = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            yield idx, frame
        idx += 1
    cap.release()


def sample_jpeg_dir(path: Path, native_fps: float,
                    no_ext: bool = False, target_fps: float = 1.0):
    """Yield (frame_idx, bgr) from a directory of JPEG frames."""
    if not path.is_dir():
        return
    if no_ext:
        files = sorted(
            [f for f in path.iterdir() if f.is_file() and f.suffix == ""],
            key=lambda f: f.name,
        )
    else:
        files = sorted(
            [f for f in path.iterdir()
             if f.suffix.lower() in {".jpg", ".jpeg", ".png"}],
            key=lambda f: f.name,
        )
    step = max(1, round(native_fps / target_fps))
    for i, f in enumerate(files):
        if i % step == 0:
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if img is not None:
                yield i, img


def get_frames(corpus: str, clip_id: str, cfg: dict, target_fps: float):
    """Resolve clip_id → path, return list of (frame_idx, bgr)."""
    base = cfg["base"]
    mode = cfg["mode"]

    if corpus == "voxceleb2_test":
        # id01460__cH23Eq0_fD4__00191 → id01460/cH23Eq0_fD4/00191.mp4
        parts = clip_id.split("__")
        path  = base / parts[0] / parts[1] / (parts[2] + ".mp4")

    elif corpus == "cremad":
        # cremad____1065_IEO_DIS_MD → 1065_IEO_DIS_MD.flv
        stem = clip_id.split("____", 1)[1]
        path = base / (stem + ".flv")

    elif corpus == "hdtf":
        stem = clip_id.split("__", 1)[1]
        path = base / (stem + ".mp4")

    elif corpus == "mead":
        parts = clip_id.split("____")  # ['M019', 'front', 'happy', 'level_2', '011']
        base_path = base / parts[0] / "video" / parts[1] / parts[2] / parts[3] / parts[4]
        # try as JPEG dir first, fall back to .mp4
        if base_path.is_dir():
            path = base_path
        elif (base_path.parent / (parts[4] + ".mp4")).exists():
            return list(sample_mp4(base_path.parent / (parts[4] + ".mp4"), target_fps))
        else:
            path = base_path  # will fail gracefully
        return list(sample_jpeg_dir(path, cfg["native_fps"],
                                    no_ext=False, target_fps=target_fps))
    else:
        path = base / clip_id

    if mode == "mp4":
        return list(sample_mp4(path, target_fps))
    elif mode == "jpeg_noext":
        return list(sample_jpeg_dir(path, cfg["native_fps"],
                                    no_ext=True, target_fps=target_fps))
    elif mode == "jpeg":
        return list(sample_jpeg_dir(path, cfg["native_fps"],
                                    no_ext=False, target_fps=target_fps))
    return []


# ── Frontality score ───────────────────────────────────────────────────────────
def frontality(bbox: np.ndarray, kps: np.ndarray) -> float:
    """
    Score in [0, 1] — high = frontal.
    bbox: [x1, y1, x2, y2, conf]  (in padded-frame coordinates; relative math
          cancels the offset so result is identical to unpadded)
    kps:  (5, 2) — right_eye, left_eye, nose, right_mouth, left_mouth

    Three sub-scores multiplied:
      1. level    — eyes at similar height         (penalises roll)
      2. symmetry — eye midpoint near face centre  (penalises yaw)
      3. span     — inter-eye distance sane fraction of face width
                    (collapses for strong profiles)
    """
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    face_w = max(x2 - x1, 1.0)
    face_h = max(y2 - y1, 1.0)

    re = kps[0].astype(float)   # right eye
    le = kps[1].astype(float)   # left eye

    eye_dy   = abs(re[1] - le[1])
    level    = max(0.0, 1.0 - eye_dy / (0.3 * face_h))

    mid_x      = (re[0] + le[0]) / 2.0
    face_mid_x = (x1 + x2) / 2.0
    offset     = abs(mid_x - face_mid_x) / (0.5 * face_w)
    symmetry   = max(0.0, 1.0 - offset)

    eye_dx = abs(re[0] - le[0])
    span   = min(eye_dx / (0.45 * face_w), 1.0)

    return float(level * symmetry * span)


# ── Per-clip detection ─────────────────────────────────────────────────────────
def detect_clip(detector, frames: list, pad: int = 0) -> dict:
    """
    Run SCRFD on sampled frames; return per-clip stats dict.
    pad > 0: add black border before detection (use for VoxCeleb2 face tracks
             that touch frame borders and trigger boundary suppression).
    Face bbox height is unaffected by padding (absolute pixel difference).
    Frontality math uses relative coordinates so padding cancels out.
    """
    if not frames:
        return {"error": "no_frames"}

    heights      = []
    front_scores = []
    n_detected   = 0
    best_conf    = -1.0
    best_frame   = -1

    for frame_idx, bgr in frames:
        if pad > 0:
            bgr = cv2.copyMakeBorder(
                bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )

        bboxes, kpss = detector.detect(bgr, input_size=(640, 640))

        if bboxes is None or len(bboxes) == 0:
            continue

        n_detected += 1
        top  = int(np.argmax(bboxes[:, 4]))
        bbox = bboxes[top]
        conf = float(bbox[4])
        h    = float(bbox[3] - bbox[1])
        heights.append(h)

        fs = frontality(bbox, kpss[top]) if (kpss is not None and len(kpss) > 0) else 0.0
        front_scores.append(fs)

        if conf > best_conf:
            best_conf  = conf
            best_frame = frame_idx

    n_sampled = len(frames)

    if not heights:
        return {
            "detected":         False,
            "detect_rate":      0.0,
            "face_size_median": None,
            "face_size_max":    None,
            "face_size_min":    None,
            "face_size_bin":    None,
            "anchor_score":     None,
            "best_frame_idx":   None,
            "n_frames_sampled": n_sampled,
        }

    med_h = float(np.median(heights))
    return {
        "detected":         True,
        "detect_rate":      round(n_detected / n_sampled, 4),
        "face_size_median": round(med_h, 2),
        "face_size_max":    round(float(np.max(heights)), 2),
        "face_size_min":    round(float(np.min(heights)), 2),
        "face_size_bin":    face_size_bin(med_h),
        "anchor_score":     round(float(np.mean(front_scores)), 4),
        "best_frame_idx":   int(best_frame),
        "n_frames_sampled": n_sampled,
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Stage 2a: SCRFD face detection")
    parser.add_argument("--corpus",     required=True, choices=list(CORPUS_CFG))
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--target-fps", type=float, default=1.0)
    parser.add_argument("--batch-save", type=int, default=50)
    parser.add_argument("--pad",        type=int, default=0,
                        help="Black border px to add before detection (default 0). "
                             "Use --pad 100 for VoxCeleb2-style tightly cropped face tracks.")
    args = parser.parse_args()

    corpus = args.corpus
    cfg    = CORPUS_CFG[corpus]
    pad    = args.pad

    # ── CUDA check ─────────────────────────────────────────────────────────────
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" not in available:
        print(f"WARNING: CUDAExecutionProvider not available. Got: {available}")
        print("Set LD_LIBRARY_PATH before launching — see file header.")
        print("Continuing on CPU (will be slow).\n", flush=True)

    pool_path = WORKDIR / "annotations" / corpus / "pool.json"
    out_path  = WORKDIR / "annotations" / corpus / "face_detect.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not pool_path.exists():
        sys.exit(f"ERROR: pool not found at {pool_path}")
    with open(pool_path) as f:
        pool = json.load(f)
    clip_ids = pool if isinstance(pool, list) else pool.get("clip_ids", [])
    print(f"[{corpus}] pool size: {len(clip_ids)}", flush=True)

    results = {}
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
        print(f"[{corpus}] resuming — {len(results)} clips already done", flush=True)

    pending = [c for c in clip_ids if c not in results]
    print(f"[{corpus}] to process: {len(pending)}", flush=True)

    if not pending:
        print("Nothing to do.", flush=True)
        return

    print("Loading SCRFD-10G-KPS (buffalo_l) ...", flush=True)
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    detector = app.det_model
    print(f"Model ready. pad={pad}px\n", flush=True)

    n_skip = n_err = 0

    for i, clip_id in enumerate(pending):

        if corpus == "mead":
            parts = clip_id.split("____")
            angle = parts[1] if len(parts) >= 2 else ""
            if angle in MEAD_SKIP_ANGLES:
                results[clip_id] = {"skipped": True, "reason": "top_down_angle"}
                n_skip += 1
                continue

        try:
            frames = get_frames(corpus, clip_id, cfg, args.target_fps)
            stats  = detect_clip(detector, frames, pad=pad)
            results[clip_id] = stats
            if "error" in stats:
                n_err += 1
        except Exception as e:
            results[clip_id] = {"error": str(e)[:200]}
            n_err += 1

        if (i + 1) % args.batch_save == 0:
            with open(out_path, "w") as f:
                json.dump(results, f)
            n_det     = sum(1 for v in results.values() if v.get("detected") is True)
            n_no_face = sum(1 for v in results.values() if v.get("detected") is False)
            print(f"  [{i+1:>5}/{len(pending)}]  "
                  f"detected={n_det}  no_face={n_no_face}  "
                  f"skip={n_skip}  err={n_err}",
                  flush=True)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    n_det     = sum(1 for v in results.values() if v.get("detected") is True)
    n_no_face = sum(1 for v in results.values() if v.get("detected") is False)
    print(f"\n[{corpus}] DONE")
    print(f"  total    : {len(results)}")
    print(f"  detected : {n_det}")
    print(f"  no face  : {n_no_face}")
    print(f"  skipped  : {n_skip}  (MEAD top/down)")
    print(f"  errors   : {n_err}")
    print(f"  output   : {out_path}")


if __name__ == "__main__":
    main()