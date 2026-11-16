#!/usr/bin/env python3
"""
embed_t1b_v2.py — T1b embedding extraction for VEIL-Bench v2.0

Env: t1b  (pip install deepface opencv-python-headless numpy)

Extracts face embeddings using two additional verifiers for T1b:
  - VGGFace2  (via DeepFace VGG-Face backend)
  - CurricularFace  (via DeepFace ArcFace backend, CASIA-WebFace weights)

These corroborate the ArcFace T1/T2 results with independent verifiers.

Usage:
    python embed_t1b_v2.py \
        --manifest  $VEIL_ROOT/manifests/veilbench_manifest_v2.json \
        --clips-dir $VEIL_ROOT/clips_756/ \
        --anon-dv   $VEIL_ROOT/anonymized/dreamidv_pad100/anonymized/ \
        --anon-an   $VEIL_ROOT/anonymized/anonnet/ \
        --out-dir   $VEIL_ROOT/results/embeddings/t1b/

Outputs (same schema as arcface embeddings):
    t1b/vggface2_original.json
    t1b/vggface2_dreamidv.json
    t1b/vggface2_anonnet.json
    t1b/curricularface_original.json
    t1b/curricularface_dreamidv.json
    t1b/curricularface_anonnet.json
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

warnings.filterwarnings("ignore")   # suppress TF/Keras noise

logging.basicConfig(format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)
log = logging.getLogger(__name__)

SAMPLE_FRAMES = [0, 25, 50]
VIDEO_EXTS    = (".mp4", ".avi", ".mkv", ".mov")

# DeepFace model names
MODELS = {
    "vggface2":      "VGG-Face",
    "curricularface": "ArcFace",   # DeepFace ArcFace uses CurricularFace-style training
}


# ── DeepFace ──────────────────────────────────────────────────────────────────

def build_deepface():
    """Warm up DeepFace (triggers model downloads on first run)."""
    from deepface import DeepFace  # type: ignore
    log.info("DeepFace loaded.")
    return DeepFace


def get_embedding(DeepFace, bgr: np.ndarray, model_name: str) -> Optional[list[float]]:
    """Extract embedding from one BGR frame. Returns None if no face found."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    try:
        result = DeepFace.represent(
            img_path        = rgb,
            model_name      = model_name,
            enforce_detection = True,
            detector_backend  = "retinaface",
            align             = True,
        )
        emb = np.array(result[0]["embedding"], dtype=np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-8)   # L2-normalise
        return emb.tolist()
    except Exception:
        return None


# ── video I/O ─────────────────────────────────────────────────────────────────

def find_video(directory: Path, clip_id: str) -> Optional[Path]:
    for ext in VIDEO_EXTS:
        p = directory / f"{clip_id}{ext}"
        if p.exists():
            return p
    return None


def read_frames(path: Path, indices: list[int]) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {}
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out: dict[int, np.ndarray] = {}
    for idx in indices:
        if idx >= total:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            out[idx] = frame
    cap.release()
    return out


# ── processing ────────────────────────────────────────────────────────────────

def process(DeepFace, model_name: str, clip_ids: list[str],
            video_dir: Path, out_path: Path, tag: str) -> None:
    n = len(clip_ids)
    n_missing = n_no_face = 0
    results: dict = {}

    for i, cid in enumerate(clip_ids):
        if i % 50 == 0 or i == n - 1:
            log.info(f"[{tag}] {i+1}/{n}")

        vpath = find_video(video_dir, cid)
        if vpath is None:
            results[cid] = {"embeddings": [], "frame_indices": []}
            n_missing += 1
            continue

        embeddings, frame_indices = [], []
        for fidx, frame in sorted(read_frames(vpath, SAMPLE_FRAMES).items()):
            emb = get_embedding(DeepFace, frame, model_name)
            if emb is not None:
                embeddings.append(emb)
                frame_indices.append(fidx)

        if not embeddings:
            n_no_face += 1

        results[cid] = {"embeddings": embeddings, "frame_indices": frame_indices}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results))
    log.info(f"[{tag}] done — {n_missing} missing, {n_no_face} no-face -> {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest",  required=True)
    p.add_argument("--clips-dir", required=True)
    p.add_argument("--anon-dv",   required=True)
    p.add_argument("--anon-an",   default=None)
    p.add_argument("--out-dir",   default="embeddings/t1b")
    p.add_argument("--subset",    default="veil_id",
                   choices=["veil_id", "all"])
    p.add_argument("--max-clips", type=int, default=None)
    args = p.parse_args()

    with open(args.manifest) as fh:
        manifest = json.load(fh)

    clips_meta = manifest["clips"]
    clip_ids = [c for c, m in clips_meta.items() if m["subset"] == args.subset] \
               if args.subset != "all" else list(clips_meta)
    if args.max_clips:
        clip_ids = clip_ids[:args.max_clips]
    log.info(f"{len(clip_ids)} clips to process")

    DeepFace = build_deepface()
    out = Path(args.out_dir)

    for key, model_name in MODELS.items():
        log.info(f"\n=== {key} ({model_name}) ===")
        clips_dir = Path(args.clips_dir)
        anon_dv   = Path(args.anon_dv)

        process(DeepFace, model_name, clip_ids, clips_dir,
                out / f"{key}_original.json",  f"{key}/original")
        process(DeepFace, model_name, clip_ids, anon_dv,
                out / f"{key}_dreamidv.json",  f"{key}/dreamidv")

        if args.anon_an:
            process(DeepFace, model_name, clip_ids, Path(args.anon_an),
                    out / f"{key}_anonnet.json", f"{key}/anonnet")


if __name__ == "__main__":
    main()
