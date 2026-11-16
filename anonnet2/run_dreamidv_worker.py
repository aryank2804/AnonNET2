#!/usr/bin/env python3
"""
run_dreamidv_worker.py  —  DreamID-V batch worker  [FIXED]
Env: a2

Loads DreamID-V pipeline ONCE, then processes all jobs from a JSON file.
Previously the DreamIDVPipeline import always failed (class doesn't exist),
causing the model to reload per clip via subprocess — catastrophically slow
for 756 clips.

This version:
  - Imports dreamidv_wan_faster.DreamIDV directly (the actual class)
  - Loads the pipeline once
  - Runs DWPose in-process per job (fast, no subprocess)
  - Writes output tensor directly to out_path (no glob fragility)

Confirmed API (from ComfyUI nodes.py / generate_dreamidv_faster.py):
  pipeline.generate(text_prompt, ref_paths, size, frame_num, shift,
                    sample_solver, sampling_steps, guide_scale_img, seed,
                    offload_model)
  ref_paths for wan_faster = [ref_video, ref_mask, ref_image]  ← 3 paths, NO pose

jobs.json format:
    [
        {"ref_video": "/path/orig_clip.mp4",
         "ref_image": "/path/face_crop_512.png",
         "out_path":  "/path/anon_clip.mp4"},
        ...
    ]
"""

import argparse
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import logging
import os
import sys
import types

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [dreamidv-worker] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Args ──────────────────────────────────────────────────────────────────────

ap = argparse.ArgumentParser()
ap.add_argument("--jobs",     required=True,  help="Path to jobs JSON file")
ap.add_argument("--repo",     required=True,  help="Path to dreamidv-inference repo")
ap.add_argument("--wan-dir",  required=True,  help="Path to wan2.1-1.3B model dir")
ap.add_argument("--ckpt",     required=True,  help="Path to dreamidv_faster.pth")
ap.add_argument("--size",     default="832*480")
ap.add_argument("--steps",    type=int, default=16)
ap.add_argument("--seed",     type=int, default=42)
ap.add_argument("--frame-num", type=int, default=81,
                help="Frames per clip (VoxCeleb2 = 81)")
ap.add_argument("--t5-cpu",   action="store_true", default=True,
                help="Run T5 encoder on CPU to save GPU VRAM (default: True)")
ap.add_argument("--guide-scale", type=float, default=4.0)
args = ap.parse_args()

# ── Parse size ────────────────────────────────────────────────────────────────

try:
    W, H = (int(x) for x in args.size.split("*"))
    size_tuple = (W, H)
except ValueError:
    log.error(f"Invalid --size format '{args.size}', expected WxH e.g. 832*480")
    sys.exit(1)

# ── Load jobs ─────────────────────────────────────────────────────────────────

with open(args.jobs) as f:
    jobs = json.load(f)
log.info(f"{len(jobs)} jobs to process")

if not jobs:
    log.info("No jobs — exiting")
    sys.exit(0)

# ── Add repo to Python path ───────────────────────────────────────────────────

sys.path.insert(0, args.repo)
os.chdir(args.repo)
sys.path.insert(0, os.path.join(args.repo, "pose"))

# ── Check FlashAttention 2 (required for wan_faster) ─────────────────────────

try:
    import flash_attn
    log.info(f"FlashAttention 2 OK (version {flash_attn.__version__})")
except ImportError:
    log.error("=" * 60)
    log.error("FlashAttention 2 is REQUIRED for dreamidv_faster but not installed.")
    log.error("Install it with:")
    log.error("  pip install flash-attn==2.8.2 --no-build-isolation")
    log.error("L40S (SM 8.9) is supported. Ensure CUDA 12.x and PyTorch match.")
    log.error("=" * 60)
    sys.exit(1)

# ── Imports (after sys.path) ──────────────────────────────────────────────────

import cv2
import numpy as np
import torch

try:
    import dreamidv_wan_faster
    from dreamidv_wan_faster.configs import WAN_CONFIGS
except ImportError as e:
    log.error(f"Could not import dreamidv_wan_faster: {e}")
    log.error(f"Is --repo correct? {args.repo}")
    sys.exit(1)

try:
    from pose.extract import process_dwpose
except ImportError as e:
    log.error(f"Could not import pose.extract.process_dwpose: {e}")
    log.error("Ensure pose/extract.py exists in the dreamidv-inference repo.")
    sys.exit(1)

# ── DWPose model paths ────────────────────────────────────────────────────────

DWPOSE_DIR = os.path.join(args.repo, "pose", "models")
DET_MODEL  = os.path.join(DWPOSE_DIR, "yolox_l.onnx")
POSE_MODEL = os.path.join(DWPOSE_DIR, "dw-ll_ucoco_384.onnx")

for p in [DET_MODEL, POSE_MODEL]:
    if not os.path.exists(p):
        log.error(f"DWPose model missing: {p}")
        log.error("Place yolox_l.onnx and dw-ll_ucoco_384.onnx in:")
        log.error(f"  {DWPOSE_DIR}")
        sys.exit(1)

log.info("DWPose models OK")

# ── Build config ──────────────────────────────────────────────────────────────
#
# Mirrors the ComfyUI loader (nodes.py RunningHub_DreamID_V_Loader.load):
#   cfg = types.SimpleNamespace(**dict(WAN_CONFIGS_FASTER["swapface"]))
#   cfg.t5_device = "cpu"   # or "cuda"

cfg_dict = dict(WAN_CONFIGS["swapface"])
cfg_dict["t5_device"] = "cpu" if args.t5_cpu else "cuda"
cfg = types.SimpleNamespace(**cfg_dict)

# ── Load pipeline ONCE ────────────────────────────────────────────────────────

log.info("Loading DreamID-V pipeline (wan_faster) — this takes ~2-3 min on first load...")
log.info(f"  ckpt_dir:      {args.wan_dir}")
log.info(f"  dreamidv_ckpt: {args.ckpt}")

try:
    pipeline = dreamidv_wan_faster.DreamIDV(
        config         = cfg,
        checkpoint_dir = args.wan_dir,
        dreamidv_ckpt  = args.ckpt,
        device_id      = 0,
        rank           = 0,
        t5_fsdp        = False,
        dit_fsdp       = False,
        use_usp        = False,
        t5_cpu         = args.t5_cpu,
    )
except Exception as e:
    log.error(f"Pipeline loading failed: {e}")
    raise

log.info("Pipeline loaded — starting batch inference")

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_video_fps(path: str, fallback: float = 25.0) -> float:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else fallback


def tensor_to_video(tensor: torch.Tensor, out_path: str, fps: float = 25.0) -> None:
    """
    Convert pipeline output tensor → mp4.

    DreamID-V generate() returns shape (C, T, H, W), values in [-1, 1].
    We convert to (T, H, W, C) in [0, 1], then write with cv2.
    """
    # (C, T, H, W) → (T, H, W, C), [-1,1] → [0,1]
    frames_np = (tensor.clamp(-1, 1).cpu().permute(1, 2, 3, 0).numpy() + 1.0) / 2.0
    T, H, W, C = frames_np.shape

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter could not open {out_path}")

    for i in range(T):
        frame_bgr = cv2.cvtColor(
            (frames_np[i] * 255).clip(0, 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        )
        writer.write(frame_bgr)

    writer.release()


# ── Process jobs ──────────────────────────────────────────────────────────────

n_ok = n_skip = n_fail = 0

for i, job in enumerate(jobs, 1):
    ref_video = job["ref_video"]
    ref_image = job["ref_image"]
    out_path  = job["out_path"]

    log.info(f"── Job {i}/{len(jobs)}: {os.path.basename(ref_video)}")

    if os.path.exists(out_path):
        log.info("  Already exists — skipping")
        n_skip += 1
        continue

    # ── Step 1: Generate pose + mask with DWPose ──────────────────────────
    video_stem = os.path.splitext(os.path.basename(ref_video))[0]
    temp_dir   = os.path.join(os.path.dirname(ref_video), "temp_generated")
    os.makedirs(temp_dir, exist_ok=True)

    mask_path = os.path.join(temp_dir, f"{video_stem}_mask.mp4")
    pose_path = os.path.join(temp_dir, f"{video_stem}_pose.mp4")

    if os.path.exists(mask_path) and os.path.exists(pose_path):
        log.info("  Pose/mask already cached")
    else:
        log.info("  Running DWPose...")
        try:
            process_dwpose(ref_video, pose_path, mask_path)
        except Exception as e:
            log.error(f"  DWPose failed: {e}")
            log.error("  Tip: check if the video has a detectable face. "
                      "For tight face-crops (VoxCeleb2), pad the video first.")
            n_fail += 1
            continue

        if not os.path.exists(mask_path):
            log.error("  DWPose ran but produced no mask — face likely not detected.")
            log.error("  This is the VoxCeleb2 tight-crop problem — pad the video.")
            n_fail += 1
            continue

    # ── Step 2: Run DreamID-V inference ───────────────────────────────────
    #
    # wan_faster ref_paths = [ref_video, ref_mask, ref_image]   ← 3 paths
    # (confirmed from ComfyUI nodes.py; pose video is NOT used by wan_faster)
    # sample_solver MUST be "unipc" for wan_faster.

    ref_paths = [ref_video, mask_path, ref_image]
    log.info(f"  Inferring at {args.size}, {args.steps} steps, seed {args.seed}...")

    # Use actual clip frame count rather than hardcoded default
    _cap = cv2.VideoCapture(ref_video)
    actual_frame_num = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    _cap.release()
    frame_num = actual_frame_num if actual_frame_num > 0 else args.frame_num
    log.info(f"  frame_num={frame_num} (from clip)")

    try:
        generated = pipeline.generate(
            "change face",
            ref_paths,
            size           = size_tuple,
            frame_num      = frame_num,
            shift          = 5.0,
            sample_solver  = "unipc",      # REQUIRED for wan_faster
            sampling_steps = args.steps,
            guide_scale_img = args.guide_scale,
            seed           = args.seed,
            offload_model  = True,
        )
    except Exception as e:
        log.error(f"  pipeline.generate() failed: {e}")
        n_fail += 1
        continue

    # ── Step 3: Save output ───────────────────────────────────────────────

    src_fps = get_video_fps(ref_video)
    try:
        tensor_to_video(generated, out_path, fps=src_fps)
        log.info(f"  Saved → {out_path}")
        n_ok += 1
    except Exception as e:
        log.error(f"  Failed to save video: {e}")
        n_fail += 1
        continue

    # Free tensor memory between clips to avoid VRAM accumulation
    del generated
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ── Summary ───────────────────────────────────────────────────────────────────

log.info("=" * 50)
log.info(f"Done. OK={n_ok}  skipped={n_skip}  failed={n_fail}")
log.info("=" * 50)

if n_fail > 0:
    sys.exit(1)   # signal to run_batch.py that some jobs failed
