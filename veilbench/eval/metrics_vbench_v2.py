#!/usr/bin/env python3
"""
metrics_vbench_v2.py — VBench-style metrics for VEIL-Bench v2.0

Env: vbench  (pip install vbench torch torchvision)

Computes three metrics from VBench [VBench, CVPR 2024]:
  background_consistency  CLIP similarity of background regions across frames
  subject_consistency     DINO similarity of foreground subject across frames
  motion_smoothness       RAFT optical flow smoothness score

These are the same metrics cited in the DreamID-V paper (Table 1).
DreamID-V is expected to win all three since it only modifies the face region
and has temporal modelling; AnonNET's warping bleeds into background and
introduces flicker at non-frontal angles.

Usage:
    python metrics_vbench_v2.py \
        --manifest  $VEIL_ROOT/manifests/veilbench_manifest_v2.json \
        --anon-dv   $VEIL_ROOT/anonymized/dreamidv_pad100/anonymized/ \
        --anon-an   $VEIL_ROOT/anonymized/anonnet/ \
        --out       $VEIL_ROOT/results/vbench_results.json

Output schema:
    {
      "dreamidv": {
        "overall": {
          "background_consistency": 0.97,
          "subject_consistency":    0.94,
          "motion_smoothness":      0.99
        },
        "by_yaw_bin": { "frontal": {...}, "profile": {...}, "extreme": {...} }
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

logging.basicConfig(format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)
log = logging.getLogger(__name__)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov")
YAW_BINS   = ("frontal", "profile", "extreme")

# Sample every N frames for efficiency (VBench default is all frames;
# we sample at 5fps from 25fps clips = every 5th frame)
FRAME_STEP = 5


# ── model loading ─────────────────────────────────────────────────────────────

def load_clip():
    """CLIP ViT-B/32 for background consistency."""
    import clip  # type: ignore  (pip install clip)
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    model.eval()
    return model, preprocess


def load_dino():
    """DINO ViT-B/16 for subject consistency."""
    model = torch.hub.load("facebookresearch/dino:main", "dino_vitb16",
                           pretrained=True, verbose=False)
    model.eval().to(DEVICE)
    return model


def load_raft():
    """RAFT optical flow for motion smoothness."""
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights  # type: ignore
    weights = Raft_Small_Weights.DEFAULT
    model   = raft_small(weights=weights).eval().to(DEVICE)
    return model, weights.transforms()


# ── video I/O ─────────────────────────────────────────────────────────────────

def find_video(directory: Path, clip_id: str) -> Optional[Path]:
    for ext in VIDEO_EXTS:
        p = directory / f"{clip_id}{ext}"
        if p.exists():
            return p
    return None


def read_frames_rgb(path: Path, step: int = FRAME_STEP) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    frames, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    return frames


# ── metrics ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def background_consistency(frames: list[np.ndarray], clip_model, preprocess) -> Optional[float]:
    """
    Mean CLIP cosine similarity between consecutive frame pairs.
    Uses full frame (not masked) as a proxy for background consistency —
    faces are a small fraction of VoxCeleb2 frames.
    """
    if len(frames) < 2:
        return None
    from PIL import Image  # type: ignore
    import torch.nn.functional as F

    embs = []
    for f in frames:
        img    = preprocess(Image.fromarray(f)).unsqueeze(0).to(DEVICE)
        emb    = clip_model.encode_image(img).float()
        emb    = F.normalize(emb, dim=-1)
        embs.append(emb)

    sims = [
        (embs[i] * embs[i+1]).sum().item()
        for i in range(len(embs)-1)
    ]
    return float(np.mean(sims))


@torch.no_grad()
def subject_consistency(frames: list[np.ndarray], dino_model) -> Optional[float]:
    """
    Mean DINO cosine similarity between consecutive frame [CLS] tokens.
    DINO focuses on foreground subject — measures face/person stability.
    """
    if len(frames) < 2:
        return None
    from torchvision import transforms
    import torch.nn.functional as F

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    embs = []
    for f in frames:
        from PIL import Image  # type: ignore
        t   = tfm(Image.fromarray(f)).unsqueeze(0).to(DEVICE)
        emb = dino_model(t).float()
        emb = F.normalize(emb, dim=-1)
        embs.append(emb)

    sims = [
        (embs[i] * embs[i+1]).sum().item()
        for i in range(len(embs)-1)
    ]
    return float(np.mean(sims))


@torch.no_grad()
def motion_smoothness(frames: list[np.ndarray], raft_model, raft_transforms) -> Optional[float]:
    """
    Mean normalised optical flow magnitude across consecutive frame pairs.
    Lower temporal variation = smoother motion.
    Score = 1 - mean(flow_magnitude / diagonal) so higher = smoother.
    """
    if len(frames) < 2:
        return None

    scores = []
    for i in range(len(frames)-1):
        img1 = torch.from_numpy(frames[i]).permute(2,0,1).float() / 255.0
        img2 = torch.from_numpy(frames[i+1]).permute(2,0,1).float() / 255.0
        img1, img2 = raft_transforms(img1.unsqueeze(0), img2.unsqueeze(0))
        # RAFT requires H and W divisible by 8
        h, w = img1.shape[-2], img1.shape[-1]
        new_h = (h // 8) * 8
        new_w = (w // 8) * 8
        if new_h != h or new_w != w:
            import torch.nn.functional as F
            img1 = F.interpolate(img1, size=(new_h, new_w), mode='bilinear', align_corners=False)
            img2 = F.interpolate(img2, size=(new_h, new_w), mode='bilinear', align_corners=False)
        img1, img2 = img1.to(DEVICE), img2.to(DEVICE)

        flow_preds = raft_model(img1, img2)
        flow       = flow_preds[-1][0]   # (2, H, W)

        H, W        = flow.shape[1], flow.shape[2]
        diagonal    = (H**2 + W**2) ** 0.5
        magnitude   = flow.norm(dim=0).mean().item()
        scores.append(magnitude / diagonal)

    # Invert: high magnitude = low smoothness
    return float(1.0 - np.mean(scores))


# ── per-clip ──────────────────────────────────────────────────────────────────

def compute_clip_metrics(
    clip_id: str,
    video_path: Optional[Path],
    clip_model, clip_preprocess,
    dino_model,
    raft_model, raft_transforms,
) -> Optional[dict]:
    if video_path is None or not video_path.exists():
        return None
    frames = read_frames_rgb(video_path)
    if len(frames) < 2:
        return None
    return {
        "background_consistency": background_consistency(frames, clip_model, clip_preprocess),
        "subject_consistency":    subject_consistency(frames, dino_model),
        "motion_smoothness":      motion_smoothness(frames, raft_model, raft_transforms),
    }


def collect(clip_ids: list[str], video_dir: Path, tag: str,
            clip_model, clip_preprocess, dino_model, raft_model, raft_transforms) -> dict:
    results: dict = {}
    n = len(clip_ids)
    for i, cid in enumerate(clip_ids):
        if i % 50 == 0 or i == n - 1:
            log.info(f"[{tag}] {i+1}/{n}")
        vp = find_video(video_dir, cid)
        results[cid] = compute_clip_metrics(
            cid, vp, clip_model, clip_preprocess, dino_model, raft_model, raft_transforms
        )
    return results


# ── aggregation ───────────────────────────────────────────────────────────────

def aggregate(per_clip: dict, ids: list[str]) -> dict:
    out = {}
    for metric in ("background_consistency", "subject_consistency", "motion_smoothness"):
        vals = [per_clip[c][metric]
                for c in ids
                if per_clip.get(c) and per_clip[c].get(metric) is not None]
        out[metric] = float(np.mean(vals)) if vals else None
    out["n_clips"] = sum(1 for c in ids if per_clip.get(c) is not None)
    return out


def build_results(clips_meta: dict, all_ids: list[str], per_clip: dict) -> dict:
    by_yaw: dict[str, list[str]] = {y: [] for y in YAW_BINS}
    for cid, m in clips_meta.items():
        yb = m.get("yaw_bin")
        if yb in by_yaw:
            by_yaw[yb].append(cid)

    result = {
        "overall":    aggregate(per_clip, all_ids),
        "by_yaw_bin": {yb: aggregate(per_clip, ids) for yb, ids in by_yaw.items() if ids},
    }
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest",  required=True)
    p.add_argument("--anon-dv",   required=True)
    p.add_argument("--anon-an",   default=None)
    p.add_argument("--out",       default="results/vbench_results.json")
    p.add_argument("--max-clips", type=int, default=None)
    args = p.parse_args()

    with open(args.manifest) as fh:
        manifest = json.load(fh)

    clips_meta = manifest["clips"]
    all_ids    = sorted(clips_meta)
    if args.max_clips:
        all_ids = all_ids[:args.max_clips]
        clips_meta = {c: clips_meta[c] for c in all_ids}

    log.info("Loading models ...")
    clip_model, clip_preprocess = load_clip()
    dino_model                  = load_dino()
    raft_model, raft_transforms = load_raft()

    results: dict = {}

    log.info("DreamID-V ...")
    dv_per_clip = collect(all_ids, Path(args.anon_dv), "dreamidv",
                          clip_model, clip_preprocess, dino_model, raft_model, raft_transforms)
    results["dreamidv"] = build_results(clips_meta, all_ids, dv_per_clip)
    ov = results["dreamidv"]["overall"]
    log.info(f"DreamID-V: bg={ov['background_consistency']:.4f}  "
             f"subj={ov['subject_consistency']:.4f}  "
             f"smooth={ov['motion_smoothness']:.4f}")

    if args.anon_an:
        log.info("AnonNET ...")
        an_per_clip = collect(all_ids, Path(args.anon_an), "anonnet",
                              clip_model, clip_preprocess, dino_model, raft_model, raft_transforms)
        results["anonnet"] = build_results(clips_meta, all_ids, an_per_clip)
        ov = results["anonnet"]["overall"]
        log.info(f"AnonNET:   bg={ov['background_consistency']:.4f}  "
                 f"subj={ov['subject_consistency']:.4f}  "
                 f"smooth={ov['motion_smoothness']:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    log.info(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
