"""
Staged batch runner for AnonNET.

Stage 1 (CPU):  clip_creation for ALL videos  — scene detection, frontal frame selection
Stage 2 (GPU):  Predictor + Segment loaded ONCE, inpainting runs across all videos
Stage 3 (GPU):  motion_lp / motion_lia — called per video for now (see note below)

Progress is checkpointed to <save_folder>/batch_progress.json after every video.
A crashed or evicted OAR job resumes automatically from the last completed stage.

NOTE on stage 3 model loading
-------------------------------
motion_lp / motion_lia are currently called as main() functions, which means
LivePortrait / LIA is re-initialised for every video. To fully fix this, share
gen_lp.py and gen_lia.py and we can pull the model instantiation out of main()
into a class that lives here for the whole loop. The big win (SD + ControlNet)
is already solved in stage 2.
"""

import os
import sys
import gc
import json
import pickle
import traceback
import argparse
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CHECKPOINT_FILE = "batch_progress.json"
STAGE1_CACHE_DIR = "stage1_cache"


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint(save_folder: str) -> dict:
    path = os.path.join(save_folder, CHECKPOINT_FILE)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_checkpoint(ckpt: dict, save_folder: str):
    """Atomic write so a crash mid-save doesn't corrupt the checkpoint."""
    path = os.path.join(save_folder, CHECKPOINT_FILE)
    tmp  = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ckpt, f, indent=2)
    os.replace(tmp, path)


def print_summary(ckpt: dict):
    total = len(ckpt)
    s1    = sum(1 for v in ckpt.values() if v.get("stage1_done"))
    s2    = sum(1 for v in ckpt.values() if v.get("stage2_done"))
    done  = sum(1 for v in ckpt.values() if v.get("stage3_done"))
    errs  = sum(1 for v in ckpt.values() if v.get("error"))
    print(f"\n{'='*60}")
    print(f"  {total} videos total | S1: {s1} | S2: {s2} | done: {done} | errors: {errs}")
    print(f"{'='*60}\n")


# ── Video discovery ───────────────────────────────────────────────────────────

def find_videos(input_dir: str):
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    return sorted(
        str(p) for p in Path(input_dir).iterdir()
        if p.suffix.lower() in exts
    )


# ── Stage 1: scene detection + frontal frame selection ───────────────────────

def run_stage1(videos, args, ckpt, save_folder):
    from scene_clips import clip_creation

    cache_dir = os.path.join(save_folder, STAGE1_CACHE_DIR)
    os.makedirs(cache_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  STAGE 1  —  scene detection + frontal frame selection")
    print(f"  {sum(1 for v in videos if not ckpt.get(Path(v).stem, {}).get('stage1_done'))} videos to process")
    print(f"{'='*60}")

    for video_path in videos:
        vid = Path(video_path).stem

        if ckpt.get(vid, {}).get("stage1_done"):
            print(f"  [skip] {vid}")
            continue

        ckpt.setdefault(vid, {})
        print(f"\n  [S1] {vid}")

        try:
            args.driving_path = video_path
            scene_paths, attributes = clip_creation(args)

            cache_path = os.path.join(cache_dir, f"{vid}.pkl")
            with open(cache_path, "wb") as f:
                pickle.dump({"scene_paths": scene_paths, "attributes": attributes}, f)

            ckpt[vid].update(
                stage1_done=True,
                cache_path=cache_path,
                n_scenes=len(scene_paths),
                error=None,
            )
            print(f"       → {len(scene_paths)} scenes found")

        except Exception as e:
            print(f"  [ERROR] {vid}: {e}")
            traceback.print_exc()
            ckpt[vid].update(stage1_done=False, error=str(e))

        save_checkpoint(ckpt, save_folder)

    print("\n  STAGE 1 complete.")


# ── Stage 2: diffusion inpainting ────────────────────────────────────────────

def run_stage2(videos, args, ckpt, save_folder):
    from PIL import Image
    from AnonHead.predict_multiple import Predictor
    from AnonHead.segment_multiple import Segment

    pending = [
        v for v in videos
        if ckpt.get(Path(v).stem, {}).get("stage1_done")
        and not ckpt.get(Path(v).stem, {}).get("stage2_done")
    ]

    print(f"\n{'='*60}")
    print(f"  STAGE 2  —  diffusion inpainting  ({len(pending)} videos pending)")
    print(f"{'='*60}")

    if not pending:
        print("  Nothing to do.")
        return

    # ── Load once ──────────────────────────────────────────────────────────────
    print("  Loading Predictor + Segment …")
    predictor = Predictor()
    segment   = Segment(load_det=False, load_seg=(args.segment == "head"))
    print("  Models ready.\n")

    for video_path in pending:
        vid = Path(video_path).stem
        print(f"  [S2] {vid}")

        try:
            with open(ckpt[vid]["cache_path"], "rb") as f:
                data = pickle.load(f)
            scene_paths = data["scene_paths"]
            attributes  = data["attributes"]

            src_dir = os.path.join(args.save_folder, "source_images")
            os.makedirs(src_dir, exist_ok=True)

            scenes_info: list = []
            prompts:     dict = {}

            for i, (scene_number, sid, vid_path, im_path, bboxes) in enumerate(scene_paths):
                basename = os.path.splitext(os.path.basename(im_path))[0]

                if "noface" in basename:
                    Image.open(im_path).save(
                        os.path.join(src_dir, f"{vid}_source_{i}_noface.png"))
                    continue

                outputs = segment.retinaface_detect_and_annotate(
                    img=im_path, margin=0.8, method=args.segment, bboxes=bboxes)

                if len(outputs) == 0:
                    print(f"       scene {i}: no face — skipping")
                    Image.open(im_path).save(
                        os.path.join(src_dir, f"{vid}_source_{i}_noface.png"))
                    continue
                if len(outputs) > 1:
                    print(f"       scene {i}: multiple faces — skipping")
                    continue

                try:
                    seed_val = int(basename.split("_")[-1])
                except Exception:
                    seed_val = 6000

                if seed_val not in prompts:
                    prompts[seed_val], _ = predictor.create_prompt(
                        im_path, deepface_result=attributes[seed_val][1])

                crop, _, _ = predictor.anonymize(
                    image=outputs[0][1],
                    prompt=prompts[seed_val],
                    mask=outputs[0][0],
                    negative_prompt=args.negative_prompt,
                    strength=[0.9, 0.4, 0.3],
                    max_height=args.max_height,
                    max_width=args.max_width,
                    steps=35,
                    seed=seed_val,
                    guidance_scale=8.0,
                    im_path=im_path,
                )
                crop      = crop.resize(outputs[0][1].size, Image.LANCZOS)
                out_image = segment.merge_crops(
                    Image.open(im_path), [(outputs[0][0], crop, outputs[0][2])])

                out_path = os.path.join(src_dir, f"{vid}_source_{i}.png")
                out_image.save(out_path)
                scenes_info.append((scene_number, sid, vid_path, out_path))

            # Persist scenes_info so stage 3 can pick it up
            with open(ckpt[vid]["cache_path"], "wb") as f:
                pickle.dump({**data, "scenes_info": scenes_info}, f)

            ckpt[vid].update(stage2_done=True, error=None)
            print(f"       → {len(scenes_info)} scenes inpainted")

        except Exception as e:
            print(f"  [ERROR] {vid}: {e}")
            traceback.print_exc()
            ckpt[vid].update(stage2_done=False, error=str(e))

        save_checkpoint(ckpt, save_folder)

    # ── Unload before stage 3 ─────────────────────────────────────────────────
    del predictor, segment
    torch.cuda.empty_cache()
    gc.collect()
    print("\n  STAGE 2 complete — inpainting models unloaded.")


# ── Stage 3: motion transfer ─────────────────────────────────────────────────

def run_stage3(videos, args, ckpt, save_folder):
    """
    NOTE: to load LivePortrait / LIA truly once across all videos, share
    gen_lp.py / gen_lia.py so we can pull the model class out of main().
    For now the SD memory is freed, so at least that's not competing on VRAM.
    """
    pending = [
        v for v in videos
        if ckpt.get(Path(v).stem, {}).get("stage2_done")
        and not ckpt.get(Path(v).stem, {}).get("stage3_done")
    ]

    print(f"\n{'='*60}")
    print(f"  STAGE 3  —  motion transfer [{args.motion}]  ({len(pending)} videos pending)")
    print(f"{'='*60}")

    if not pending:
        print("  Nothing to do.")
        return

    if args.motion == "lp":
        from Generation.gen_lp import main as motion_fn
    else:
        from Generation.gen_lia import main as motion_fn

    for video_path in pending:
        vid = Path(video_path).stem
        print(f"\n  [S3] {vid}")

        try:
            with open(ckpt[vid]["cache_path"], "rb") as f:
                data = pickle.load(f)
            scenes_info = data.get("scenes_info", [])

            if not scenes_info:
                print(f"       no scenes_info (all scenes were skipped) — marking done")
                ckpt[vid].update(stage3_done=True)
                save_checkpoint(ckpt, save_folder)
                continue

            if args.motion == "lp":
                motion_fn(
                    scenes=scenes_info,
                    file_name=vid,
                    stitch=args.no_stitch,
                    save_folder=args.save_folder,
                )
            else:
                motion_fn(
                    scenes=scenes_info,
                    file_name=vid,
                    save_folder=args.save_folder,
                    original=video_path,
                )

            ckpt[vid].update(stage3_done=True, error=None)

        except Exception as e:
            print(f"  [ERROR] {vid}: {e}")
            traceback.print_exc()
            ckpt[vid].update(stage3_done=False, error=str(e))

        save_checkpoint(ckpt, save_folder)

    torch.cuda.empty_cache()
    gc.collect()
    print("\n  STAGE 3 complete.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.save_folder, exist_ok=True)

    videos = find_videos(args.input_dir)
    if not videos:
        print(f"No videos found in {args.input_dir}")
        return

    print(f"Found {len(videos)} videos in {args.input_dir}")

    ckpt = load_checkpoint(args.save_folder)
    print_summary(ckpt)

    if not args.skip_stage1:
        run_stage1(videos, args, ckpt, args.save_folder)
        print_summary(ckpt)

    if not args.skip_stage2:
        run_stage2(videos, args, ckpt, args.save_folder)
        print_summary(ckpt)

    if not args.skip_stage3:
        run_stage3(videos, args, ckpt, args.save_folder)

    print_summary(ckpt)
    print("Batch complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Staged batch runner for AnonNET")

    # I/O
    p.add_argument("--input_dir",    required=True, help="Folder of input videos")
    p.add_argument("--save_folder",  default="./results")

    # Pipeline knobs (same as anonymize.py)
    p.add_argument("--motion",    choices=["lp", "lia"], default="lp")
    p.add_argument("--segment",   choices=["head", "face"], default="head")
    p.add_argument("--negative_prompt", type=str, default="")
    p.add_argument("--max_height", type=int, default=612)
    p.add_argument("--max_width",  type=int, default=612)
    p.add_argument("--max_len",    type=int, default=None)
    p.add_argument("--no_stitch",  action="store_false", default=True,
                   help="Disable eye/lip retargeting in LivePortrait")
    p.add_argument("--scene_threshold",            type=float, default=0.2)
    p.add_argument("--scene_similarity_threshold", type=float, default=2.0)
    p.add_argument("--cache",      action="store_false", default=True,
                   help="Remove temp files after processing")

    # Stage control — useful for reruns / debugging
    p.add_argument("--skip_stage1", action="store_true", help="Skip clip_creation")
    p.add_argument("--skip_stage2", action="store_true", help="Skip inpainting")
    p.add_argument("--skip_stage3", action="store_true", help="Skip motion transfer")

    args = p.parse_args()
    main(args)
