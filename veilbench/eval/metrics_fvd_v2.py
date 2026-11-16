#!/usr/bin/env python3
"""
metrics_fvd_v2.py — Fréchet Video Distance (FVD) for VEIL-Bench v2.0

Env: fvd   (see env_fvd.yml)

Uses the canonical I3D TorchScript model from HuggingFace:
    flateon/FVD-I3D-torchscript  (downloaded automatically on first run)

Usage:
    python metrics_fvd_v2.py \
        --manifest      $VEIL_ROOT/manifests/veilbench_manifest_v2.json \
        --clips-dir     $VEIL_ROOT/clips_756/ \
        --anon-dv       $VEIL_ROOT/anonymized/dreamidv_pad100/anonymized/ \
        --anon-an       $VEIL_ROOT/anonymized/anonnet/ \
        --out           $VEIL_ROOT/results/fvd_results.json \
        --save-features $VEIL_ROOT/results/features/

    # Re-stratify without re-extracting (pose data was missing last time):
    python metrics_fvd_v2.py \
        --manifest      $VEIL_ROOT/manifests/veilbench_manifest_v2.json \
        --clips-dir     $VEIL_ROOT/clips_756/ \
        --anon-dv       $VEIL_ROOT/anonymized/dreamidv_pad100/anonymized/ \
        --anon-an       $VEIL_ROOT/anonymized/anonnet/ \
        --out           $VEIL_ROOT/results/fvd_results.json \
        --save-features $VEIL_ROOT/results/features/
        (features already cached — extraction is skipped automatically)

Output schema:
    {
      "dreamidv": {
        "overall":    {"fvd": 123.4, "n_clips": 756},
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
from huggingface_hub import hf_hub_download  # type: ignore
from scipy import linalg  # type: ignore

logging.basicConfig(format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)
log = logging.getLogger(__name__)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov")
CLIP_LEN   = 16
STRIDE     = 8
YAW_BINS   = ("frontal", "profile", "extreme")


# ── model ─────────────────────────────────────────────────────────────────────

def load_model() -> torch.jit.ScriptModule:
    log.info("Downloading I3D TorchScript model from HuggingFace ...")
    path = hf_hub_download(repo_id="flateon/FVD-I3D-torchscript",
                           filename="i3d_torchscript.pt")
    model = torch.jit.load(path, map_location=DEVICE)
    model.eval()
    log.info(f"I3D loaded on {DEVICE}")
    return model


# ── video I/O ─────────────────────────────────────────────────────────────────

def find_video(directory: Path, clip_id: str) -> Optional[Path]:
    for ext in VIDEO_EXTS:
        p = directory / f"{clip_id}{ext}"
        if p.exists():
            return p
    return None


def read_video_tensor(path: Path) -> Optional[torch.Tensor]:
    """All frames -> float32 (T, C, H, W) in [0, 1]."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        return None
    arr = np.stack(frames).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(0, 3, 1, 2)   # (T, C, H, W)


def make_clips(video: torch.Tensor) -> torch.Tensor:
    """(T, C, H, W) -> (N, C, T_clip, H, W)"""
    T = video.shape[0]
    clips = []
    for s in range(0, T - CLIP_LEN + 1, STRIDE):
        clips.append(video[s:s+CLIP_LEN].permute(1, 0, 2, 3))
    if not clips:
        pad = CLIP_LEN - T
        video = torch.cat([video, video[-1:].expand(pad, -1, -1, -1)], dim=0)
        clips.append(video.permute(1, 0, 2, 3))
    return torch.stack(clips)


# ── feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def extract(model, path: Path) -> Optional[np.ndarray]:
    """One 400-d feature vector per video (max-pooled across temporal clips)."""
    video = read_video_tensor(path)
    if video is None:
        return None
    clips = make_clips(video).to(DEVICE)
    feats = model(clips, rescale=False, resize=True, return_features=True)
    return feats.cpu().float().numpy().max(axis=0)   # (400,)


def collect(clip_ids: list[str], video_dir: Path, model, tag: str) -> dict:
    out: dict = {}
    n = len(clip_ids)
    for i, cid in enumerate(clip_ids):
        if i % 50 == 0 or i == n - 1:
            log.info(f"[{tag}] {i+1}/{n}")
        vp = find_video(video_dir, cid)
        out[cid] = extract(model, vp) if vp else None
    return out


# ── feature caching ───────────────────────────────────────────────────────────

def save_features(feats: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), **{cid: v for cid, v in feats.items() if v is not None})
    log.info(f"Features saved -> {path}")


def load_features(path: Path) -> dict:
    npz = np.load(str(path))
    return {cid: npz[cid] for cid in npz.files}


def maybe_collect(all_ids: list[str], video_dir: Optional[Path], model,
                  tag: str, features_dir: Optional[Path]) -> dict:
    """Load from cache if available, otherwise extract (and save if --save-features)."""
    if features_dir is not None:
        feat_path = features_dir / f"i3d_{tag}.npz"
        if feat_path.exists():
            log.info(f"[{tag}] loading cached features from {feat_path}")
            return load_features(feat_path)

    log.info(f"Extracting features: {tag} ...")
    feats = collect(all_ids, video_dir, model, tag)

    if features_dir is not None:
        save_features(feats, features_dir / f"i3d_{tag}.npz")

    return feats


# ── FVD ───────────────────────────────────────────────────────────────────────

def frechet(mu1, s1, mu2, s2) -> float:
    diff = mu1 - mu2
    cov, _ = linalg.sqrtm(s1 @ s2, disp=False)
    if np.iscomplexobj(cov):
        cov = cov.real
    return float(diff @ diff + np.trace(s1 + s2 - 2.0 * cov))


def fvd(real: dict, fake: dict, ids: list[str], label: str) -> Optional[float]:
    R = np.vstack([real[c] for c in ids if real.get(c) is not None])
    F = np.vstack([fake[c] for c in ids if fake.get(c) is not None])
    if len(R) < 2 or len(F) < 2:
        log.warning(f"[{label}] too few clips: real={len(R)}, fake={len(F)}")
        return None
    if len(R) < 50:
        log.warning(f"[{label}] only {len(R)} clips -- FVD may be unstable")
    d   = R.shape[1]
    eps = np.eye(d) * 1e-6
    return frechet(R.mean(0), np.cov(R.T) + eps,
                   F.mean(0), np.cov(F.T) + eps)


# ── main ──────────────────────────────────────────────────────────────────────

def build_results(clips_meta: dict, all_ids: list[str],
                  orig: dict, fake: dict) -> dict:
    by_yaw: dict = {y: [] for y in YAW_BINS}
    for cid, m in clips_meta.items():
        yb = m.get("yaw_bin")
        if yb in by_yaw:
            by_yaw[yb].append(cid)

    n_ok = sum(1 for c in all_ids if fake.get(c) is not None)
    result = {
        "overall": {"fvd": fvd(orig, fake, all_ids, "overall"), "n_clips": n_ok},
        "by_yaw_bin": {},
    }
    for yaw_bin, ids in by_yaw.items():
        if not ids:
            continue
        n = sum(1 for c in ids if fake.get(c) is not None)
        result["by_yaw_bin"][yaw_bin] = {
            "fvd":    fvd(orig, fake, ids, yaw_bin),
            "n_clips": n,
        }
        log.info(f"  {yaw_bin}: {result['by_yaw_bin'][yaw_bin]['fvd']:.2f}  (n={n})")
    return result


def run_all(manifest, clips_dir, anon_dv_dir, anon_an_dir, model,
            out_path, features_dir):
    clips_meta = manifest["clips"]
    all_ids    = sorted(clips_meta)

    orig = maybe_collect(all_ids, clips_dir,   model, "original", features_dir)
    dv   = maybe_collect(all_ids, anon_dv_dir, model, "dreamidv", features_dir)
    an: dict = {}
    if anon_an_dir:
        an = maybe_collect(all_ids, anon_an_dir, model, "anonnet", features_dir)

    results: dict = {}

    log.info("Computing FVD -- DreamID-V ...")
    results["dreamidv"] = build_results(clips_meta, all_ids, orig, dv)
    log.info(f"DreamID-V overall FVD: {results['dreamidv']['overall']['fvd']:.2f}")

    if an:
        log.info("Computing FVD -- AnonNET ...")
        results["anonnet"] = build_results(clips_meta, all_ids, orig, an)
        log.info(f"AnonNET overall FVD:   {results['anonnet']['overall']['fvd']:.2f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    log.info(f"Saved -> {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest",       required=True)
    p.add_argument("--clips-dir",      required=True)
    p.add_argument("--anon-dv",        required=True)
    p.add_argument("--anon-an",        default=None)
    p.add_argument("--out",            default="results/fvd_results.json")
    p.add_argument("--save-features",  default=None, metavar="DIR",
                   help="Save/load per-clip I3D features to this directory. "
                        "If .npz files already exist there, extraction is skipped.")
    p.add_argument("--max-clips",      type=int, default=None,
                   help="Truncate to N clips (smoke-test)")
    args = p.parse_args()

    with open(args.manifest) as fh:
        manifest = json.load(fh)
    if args.max_clips:
        manifest["clips"] = dict(list(manifest["clips"].items())[:args.max_clips])

    features_dir = Path(args.save_features) if args.save_features else None

    # Only load the model if we actually need to extract something
    need_extraction = features_dir is None or not all(
        (features_dir / f"i3d_{tag}.npz").exists()
        for tag in (["original", "dreamidv"] + (["anonnet"] if args.anon_an else []))
    )
    model = load_model() if need_extraction else None

    run_all(manifest,
            Path(args.clips_dir),
            Path(args.anon_dv),
            Path(args.anon_an) if args.anon_an else None,
            model,
            Path(args.out),
            features_dir)


if __name__ == "__main__":
    main()
