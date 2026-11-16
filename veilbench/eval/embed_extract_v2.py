#!/usr/bin/env python3
"""
embed_extract_v2.py — ArcFace embedding extraction for VEIL-Bench v2.0

Env: embed   (see env_embed.yml)

Before running, export the InsightFace CUDA lib path:
    export LD_LIBRARY_PATH=$(python -c 'import os,glob,sysconfig; \
      print(":".join(glob.glob(os.path.join(sysconfig.get_paths()["purelib"],\
      "nvidia","*","lib"))))'):$LD_LIBRARY_PATH

Usage:
    python embed_extract_v2.py \
        --manifest  $VEIL_ROOT/manifests/veilbench_manifest_v2.json \
        --clips-dir $VEIL_ROOT/clips_756/ \
        --anon-dv   $VEIL_ROOT/anonymized/dreamidv_pad100/anonymized/ \
        --out-dir   $VEIL_ROOT/embeddings/

Outputs:
    embeddings/arcface_original.json
    embeddings/arcface_dreamidv.json

    Schema: { clip_id: { "embeddings": [[512-d], ...], "frame_indices": [0,25,50] } }
"""

from __future__ import annotations
import argparse, json, logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logging.basicConfig(format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)
log = logging.getLogger(__name__)

# 1 fps from 72-frame 25 fps clips → frames 0, 25, 50
SAMPLE_FRAMES = [0, 25, 50]

# Black border prevents InsightFace from suppressing detections whose bbox
# touches the image edge on the pre-cropped 224×224 VoxCeleb2 face tracks.
BORDER_PX = 100

VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov")


# ── InsightFace ───────────────────────────────────────────────────────────────

def build_app():
    from insightface.app import FaceAnalysis  # type: ignore
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def get_embedding(app, bgr: np.ndarray) -> Optional[list[float]]:
    """Pad, detect, return the 512-d normed ArcFace embedding or None."""
    padded = cv2.copyMakeBorder(bgr, BORDER_PX, BORDER_PX, BORDER_PX, BORDER_PX,
                                cv2.BORDER_CONSTANT, value=0)
    faces = app.get(padded)
    if not faces:
        return None
    # Largest bbox wins
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    return face.normed_embedding.tolist()   # already L2-normalised


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

def process(app, clip_ids: list[str], video_dir: Path, out_path: Path, tag: str) -> None:
    n = len(clip_ids)
    n_missing = n_no_face = 0
    results: dict = {}

    for i, cid in enumerate(clip_ids):
        if i % 100 == 0 or i == n - 1:
            log.info(f"[{tag}] {i+1}/{n}")

        vpath = find_video(video_dir, cid)
        if vpath is None:
            log.warning(f"[{tag}] no video: {cid}")
            results[cid] = {"embeddings": [], "frame_indices": []}
            n_missing += 1
            continue

        embeddings, frame_indices = [], []
        for fidx, frame in sorted(read_frames(vpath, SAMPLE_FRAMES).items()):
            emb = get_embedding(app, frame)
            if emb is not None:
                embeddings.append(emb)
                frame_indices.append(fidx)

        if not embeddings:
            n_no_face += 1

        results[cid] = {"embeddings": embeddings, "frame_indices": frame_indices}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results))
    log.info(f"[{tag}] done — {n_missing} missing, {n_no_face} no-face → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest",  required=True)
    p.add_argument("--clips-dir", required=True)
    p.add_argument("--anon-dv",   required=True)
    p.add_argument("--out-dir",   default="embeddings")
    p.add_argument("--max-clips", type=int, default=None, help="Truncate to N clips (smoke-test)")
    p.add_argument("--subset",    default="veil_id", choices=["veil_id", "veil_hq", "all"],
                   help="Subset to process (attackers only need veil_id)")
    args = p.parse_args()

    with open(args.manifest) as fh:
        manifest = json.load(fh)

    clips_meta = manifest["clips"]
    clip_ids = (list(clips_meta)
                if args.subset == "all"
                else [c for c, m in clips_meta.items() if m["subset"] == args.subset])
    if args.max_clips:
        clip_ids = clip_ids[:args.max_clips]
    log.info(f"Subset '{args.subset}': {len(clip_ids)} clips")

    app = build_app()
    out = Path(args.out_dir)
    process(app, clip_ids, Path(args.clips_dir), out / "arcface_original.json", "original")
    process(app, clip_ids, Path(args.anon_dv),   out / "arcface_dreamidv.json", "dreamidv")


if __name__ == "__main__":
    main()
