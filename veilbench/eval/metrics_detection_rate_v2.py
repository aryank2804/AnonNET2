#!/usr/bin/env python3
"""
metrics_detection_rate_v2.py — Face detection rate for VEIL-Bench v2.0

Env: embed  (reuses InsightFace already installed)

Measures the percentage of frames in each clip where a face is detectable.
A low detection rate means the anonymizer produced unrecognisable/broken faces —
this reframes "good privacy at extreme angles" as catastrophic quality failure.

Expected result:
  - DreamID-V: near 100% detection at all pose bins (DiT synthesises a clean face)
  - AnonNET: drops at extreme angles (LivePortrait warping produces face-less frames)

Before running:
    export LD_LIBRARY_PATH=$(python -c 'import os,glob,sysconfig; \
      print(":".join(glob.glob(os.path.join(sysconfig.get_paths()["purelib"],\
      "nvidia","*","lib"))))'):$LD_LIBRARY_PATH

Usage:
    python metrics_detection_rate_v2.py \
        --manifest  $VEIL_ROOT/manifests/veilbench_manifest_v2.json \
        --anon-dv   $VEIL_ROOT/anonymized/dreamidv_pad100/anonymized/ \
        --anon-an   $VEIL_ROOT/anonymized/anonnet/ \
        --out       $VEIL_ROOT/results/detection_rate.json

Output schema:
    {
      "dreamidv": {
        "overall": {"detection_rate": 0.98, "n_clips": 756},
        "by_yaw_bin": {
          "frontal": {"detection_rate": 0.99, "n_clips": 555},
          "profile":  {"detection_rate": 0.97, "n_clips": 173},
          "extreme":  {"detection_rate": 0.84, "n_clips": 28}
        }
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

logging.basicConfig(format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)
log = logging.getLogger(__name__)

SAMPLE_FRAMES = [0, 25, 50]
BORDER_PX     = 100
VIDEO_EXTS    = (".mp4", ".avi", ".mkv", ".mov")
YAW_BINS      = ("frontal", "profile", "extreme")


# ── InsightFace (detector only) ───────────────────────────────────────────────

def build_detector():
    from insightface.app import FaceAnalysis  # type: ignore
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def has_face(app, bgr: np.ndarray) -> bool:
    padded = cv2.copyMakeBorder(bgr, BORDER_PX, BORDER_PX, BORDER_PX, BORDER_PX,
                                cv2.BORDER_CONSTANT, value=0)
    return len(app.get(padded)) > 0


# ── video I/O ─────────────────────────────────────────────────────────────────

def find_video(directory: Path, clip_id: str) -> Optional[Path]:
    for ext in VIDEO_EXTS:
        p = directory / f"{clip_id}{ext}"
        if p.exists():
            return p
    return None


def clip_detection_rate(app, video_path: Path) -> Optional[float]:
    """Fraction of sampled frames with at least one detected face."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detected = attempts = 0
    for idx in SAMPLE_FRAMES:
        if idx >= total:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        attempts += 1
        if has_face(app, frame):
            detected += 1
    cap.release()

    return detected / attempts if attempts > 0 else None


# ── processing ────────────────────────────────────────────────────────────────

def collect_rates(app, clip_ids: list[str], video_dir: Path,
                  tag: str) -> dict[str, Optional[float]]:
    rates: dict[str, Optional[float]] = {}
    n = len(clip_ids)
    for i, cid in enumerate(clip_ids):
        if i % 100 == 0 or i == n - 1:
            log.info(f"[{tag}] {i+1}/{n}")
        vp = find_video(video_dir, cid)
        rates[cid] = clip_detection_rate(app, vp) if vp else None
    return rates


def aggregate(rates: dict[str, Optional[float]], ids: list[str], label: str) -> dict:
    vals = [rates[c] for c in ids if rates.get(c) is not None]
    dr   = float(np.mean(vals)) if vals else None
    log.info(f"  {label}: {dr:.4f}" if dr is not None else f"  {label}: NA")
    return {"detection_rate": dr, "n_clips": len(vals)}


def build_results(clips_meta: dict, all_ids: list[str],
                  rates: dict[str, Optional[float]]) -> dict:
    by_yaw: dict[str, list[str]] = {y: [] for y in YAW_BINS}
    for cid, m in clips_meta.items():
        yb = m.get("yaw_bin")
        if yb in by_yaw:
            by_yaw[yb].append(cid)

    result = {
        "overall":    aggregate(rates, all_ids, "overall"),
        "by_yaw_bin": {
            yb: aggregate(rates, ids, yb)
            for yb, ids in by_yaw.items() if ids
        },
    }
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest",  required=True)
    p.add_argument("--anon-dv",   required=True)
    p.add_argument("--anon-an",   default=None)
    p.add_argument("--out",       default="results/detection_rate.json")
    p.add_argument("--max-clips", type=int, default=None)
    args = p.parse_args()

    with open(args.manifest) as fh:
        manifest = json.load(fh)

    clips_meta = manifest["clips"]
    all_ids    = sorted(clips_meta)
    if args.max_clips:
        all_ids = all_ids[:args.max_clips]
        clips_meta = {c: clips_meta[c] for c in all_ids}

    app = build_detector()
    results: dict = {}

    log.info("DreamID-V detection rate ...")
    dv_rates = collect_rates(app, all_ids, Path(args.anon_dv), "dreamidv")
    results["dreamidv"] = build_results(clips_meta, all_ids, dv_rates)
    log.info(f"DreamID-V overall: {results['dreamidv']['overall']['detection_rate']:.4f}")

    if args.anon_an:
        log.info("AnonNET detection rate ...")
        an_rates = collect_rates(app, all_ids, Path(args.anon_an), "anonnet")
        results["anonnet"] = build_results(clips_meta, all_ids, an_rates)
        log.info(f"AnonNET overall: {results['anonnet']['overall']['detection_rate']:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    log.info(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
