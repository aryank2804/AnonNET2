#!/usr/bin/env python3
"""
run_phase3_pad100.py — Full Phase 3 batch with pad100 (pad_frac=1.0)

Scans all workdirs with completed Phase 1 & 2, pads scene clips at pad_frac=1.0,
runs DreamID-V, unpads output, saves to anonymized/dreamidv_pad100/anonymized/
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PAD_FRAC = 1.0  # pad100

def get_video_props(path):
    cap = cv2.VideoCapture(str(path))
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return w, h, fps

def pad_video(src, dst, pad_frac):
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pad_px = int(min(w, h) * pad_frac)
    new_w, new_h = w + 2*pad_px, h + 2*pad_px
    out = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (new_w, new_h))
    while True:
        ret, frame = cap.read()
        if not ret: break
        out.write(cv2.copyMakeBorder(frame, pad_px, pad_px, pad_px, pad_px,
                                     cv2.BORDER_CONSTANT, value=[0,0,0]))
    cap.release(); out.release()
    return pad_px, pad_px, w, h

def unpad_video(src, dst, pad_left, pad_top, orig_w, orig_h, target_fps=25.0):
    cap = cv2.VideoCapture(str(src))
    out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    padded_w = orig_w + 2*pad_left
    padded_h = orig_h + 2*pad_top
    sx, sy = out_w/padded_w, out_h/padded_h
    cx, cy = max(0,int(pad_left*sx)), max(0,int(pad_top*sy))
    cw, ch = min(int(orig_w*sx), out_w-cx), min(int(orig_h*sy), out_h-cy)
    out = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (orig_w, orig_h))
    while True:
        ret, frame = cap.read()
        if not ret: break
        cropped = frame[cy:cy+ch, cx:cx+cw]
        resized = cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
        out.write(resized)
    cap.release(); out.release()

def run_worker(jobs, args):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, dir="/tmp") as f:
        json.dump(jobs, f, indent=2)
        jobs_file = f.name
    worker = str(Path(args.main1_dir) / "run_dreamidv_worker.py")
    bash_cmd = (
        f"(module load conda 2>/dev/null || true); "
        f"conda run --no-capture-output -n {args.dreamidv_env} "
        f"python {worker} "
        f"--jobs {jobs_file} "
        f"--repo {args.dreamidv_repo} "
        f"--wan-dir {args.wan_dir} "
        f"--ckpt {args.dreamidv_ckpt} "
        f"--size 624*480 --steps 16 --seed 42"
    )
    result = subprocess.run(["bash", "-c", bash_cmd])
    try: os.unlink(jobs_file)
    except: pass
    return result.returncode == 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dreamidv-dir",  required=True, help="anonymized/dreamidv/ directory")
    ap.add_argument("--main1-dir",     default=os.path.dirname(os.path.abspath(__file__)),
                    help="directory containing run_dreamidv_worker.py (default: this script's directory)")
    ap.add_argument("--dreamidv-repo", required=True)
    ap.add_argument("--wan-dir",       required=True)
    ap.add_argument("--dreamidv-ckpt", required=True)
    ap.add_argument("--dreamidv-env",  default="dreamidv")
    ap.add_argument("--out-name",     default="dreamidv_pad100")
    ap.add_argument("--batch-size",    type=int, default=50,
                    help="Jobs per worker call (model stays loaded within batch)")
    args = ap.parse_args()

    dreamidv_dir = Path(args.dreamidv_dir)
    workdirs     = dreamidv_dir / "workdirs"
    out_dir      = dreamidv_dir.parent / args.out_name / "anonymized"
    pad_dir      = dreamidv_dir.parent / args.out_name / "padded_scenes"
    out_dir.mkdir(parents=True, exist_ok=True)
    pad_dir.mkdir(parents=True, exist_ok=True)

    # Find all clips with completed Phase 1 & 2
    all_jobs = []
    pad_meta = {}

    for clip_dir in sorted(workdirs.iterdir()):
        face_png = clip_dir / "identity_0" / "generated_face.png"
        scenes   = clip_dir / "identity_0" / "scenes"
        if not face_png.exists() or not scenes.is_dir():
            continue

        for orig_scene in sorted(scenes.glob("*_orig.mp4")):
            scene_id  = orig_scene.stem.replace("_orig", "")
            key       = f"{clip_dir.name}__{scene_id}"
            padded = pad_dir / f"{key}_padded.mp4"
            raw    = pad_dir / f"{key}_raw.mp4"
            final_out = out_dir / f"{key}.mp4"

            if final_out.exists():
                log.info("Skipping %s (done)", key)
                continue

            # If raw already exists, just unpad and skip DreamID-V
            if raw.exists():
                log.info("Raw exists for %s, unpadding only", key)
                pl, pt, ow, oh = pad_meta.get(key, (0, 0, 0, 0))
                if ow > 0:
                    unpad_video(raw, final_out, pl, pt, ow, oh)
                continue


            if not padded.exists():
                pl, pt, ow, oh = pad_video(orig_scene, padded, PAD_FRAC)
                pad_meta[key] = (pl, pt, ow, oh)
            else:
                w, h, _ = get_video_props(orig_scene)
                pad_px  = int(min(w, h) * PAD_FRAC)
                pad_meta[key] = (pad_px, pad_px, w, h)

            all_jobs.append({
                "ref_video": str(padded),
                "ref_image": str(face_png),
                "out_path":  str(raw),
                "_key":      key,
                "_final":    str(final_out),
                "_meta":     pad_meta[key],
            })

    log.info("Total jobs to run: %d", len(all_jobs))

    # Process in batches
    for i in range(0, len(all_jobs), args.batch_size):
        batch = all_jobs[i:i+args.batch_size]
        log.info("Batch %d/%d (%d jobs)...",
                 i//args.batch_size+1, (len(all_jobs)-1)//args.batch_size+1, len(batch))

        worker_jobs = [{"ref_video": j["ref_video"],
                        "ref_image": j["ref_image"],
                        "out_path":  j["out_path"]} for j in batch]
        run_worker(worker_jobs, args)

        # Unpad whatever was produced
        for j in batch:
            raw   = Path(j["out_path"])
            final = Path(j["_final"])
            pl, pt, ow, oh = j["_meta"]
            if raw.exists() and not final.exists():
                unpad_video(raw, final, pl, pt, ow, oh)
                log.info("  → %s", final.name)
            elif not raw.exists():
                log.warning("  raw missing: %s", j["_key"])

    log.info("Done.")

if __name__ == "__main__":
    main()
