"""
celebvhq_inventory.py — VEIL-Bench v2.0
Builds a filtered, annotated pool of CelebV-HQ clips for VEIL-HQ.

Derives from celebvhq_info.json (no model inference needed here):
  - duration filter  : >= 2.7s (81 frames at 30fps)
  - blurry filter    : appearance[0] == 0
  - face_size_bin    : from bbox height * 256px
  - gender           : from appearance[1]
  - occluded flag    : eyeglasses/sunglasses/hat/mask → Extreme bin candidate
  - file existence   : checks disk efficiently via pre-built index

Does NOT compute yaw here — that runs after window extraction via pose_detect.py.
Instead, pre-samples 600 clips (~3x target of 200) stratified by face_size and
gender so we have enough headroom after pose annotation to hit final quotas.

Outputs:
  celebvhq_pool.json     — all eligible clips with derived annotations
  celebvhq_sample.json   — 600-clip pre-sample for window extraction

Usage:
  python celebvhq_inventory.py
"""

import os
import json
import random
from pathlib import Path
from collections import defaultdict

random.seed(42)

SHARE   = Path(os.environ.get("SHARE_ROOT", "raw_corpora"))
VEIL    = Path(os.environ.get("VEIL_ROOT", "veilbench_data"))

INFO_PATH   = SHARE / "CelebV-HQ/celebvhq_info.json"
VIDEO_ROOT  = SHARE / "CelebV-HQ/path_to_videos_256_for_inf"
POOL_OUT    = VEIL  / "manifests/celebvhq_pool.json"
SAMPLE_OUT  = VEIL  / "manifests/celebvhq_sample.json"

FRAME_HEIGHT = 256   # all videos are 256px tall
FPS          = 30.0
MIN_DURATION = 81 / FPS   # 2.7s

PRE_SAMPLE_N = 600   # 3x the 200 target — headroom for pose annotation dropout

# appearance_mapping indices
IDX_BLURRY     = 0
IDX_MALE       = 1
IDX_EYEGLASSES = 20
IDX_SUNGLASSES = 21
IDX_HAT        = 35
IDX_MASK       = 39


def build_file_index(video_root: Path) -> dict:
    """
    Scan all splits and build {stem: full_path} index.
    Much faster than per-clip disk hits.
    """
    print("Building file index from disk...", flush=True)
    index = {}
    for split in ["train", "test", "val"]:
        split_dir = video_root / split
        if not split_dir.exists():
            continue
        for f in split_dir.glob("*.mp4"):
            index[f.stem] = f
    print(f"  {len(index)} files found on disk", flush=True)
    return index


def face_size_bin(height_px: float) -> str:
    if height_px >= 160:
        return "ge160"
    elif height_px >= 80:
        return "r80_160"
    else:
        return "lt80"


def main():
    print("Loading celebvhq_info.json...", flush=True)
    with open(INFO_PATH) as f:
        data = json.load(f)
    clips = data["clips"]
    print(f"  Total clips in JSON: {len(clips)}", flush=True)

    file_index = build_file_index(VIDEO_ROOT)

    pool = {}
    stats = defaultdict(int)

    for key, meta in clips.items():
        # Expected filename: TALKING_{key}.mp4
        filename_stem = "TALKING_" + key

        # Duration filter
        duration = meta["duration"]["end_sec"] - meta["duration"]["start_sec"]
        if duration < MIN_DURATION:
            stats["filtered_short"] += 1
            continue

        # Blurry filter
        appearance = meta["attributes"]["appearance"]
        if appearance[IDX_BLURRY]:
            stats["filtered_blurry"] += 1
            continue

        # File existence
        if filename_stem not in file_index:
            stats["filtered_missing"] += 1
            continue

        # Derive annotations
        face_height_px = (meta["bbox"]["bottom"] - meta["bbox"]["top"]) * FRAME_HEIGHT
        gender         = "M" if appearance[IDX_MALE] else "F"
        occluded       = bool(
            appearance[IDX_EYEGLASSES] or appearance[IDX_SUNGLASSES] or
            appearance[IDX_HAT]        or appearance[IDX_MASK]
        )

        clip_id = "celebvhq____" + filename_stem

        pool[clip_id] = {
            "clip_id":        clip_id,
            "corpus":         "celebv_hq",
            "subset":         "veil_hq",
            "src_path":       str(file_index[filename_stem]),
            "duration_s":     round(duration, 3),
            "face_height_px": round(face_height_px, 1),
            "face_size_bin":  face_size_bin(face_height_px),
            "gender":         gender,
            "occluded":       occluded,
            # yaw_bin set after pose annotation — placeholder for now
            "yaw_bin":        "extreme" if occluded else None,
        }
        stats["eligible"] += 1

    print(f"\nPool summary:")
    print(f"  eligible       : {stats['eligible']}")
    print(f"  filtered short : {stats['filtered_short']}")
    print(f"  filtered blurry: {stats['filtered_blurry']}")
    print(f"  missing on disk: {stats['filtered_missing']}")

    # Save full pool
    with open(POOL_OUT, "w") as f:
        json.dump(pool, f, indent=2)
    print(f"\nFull pool saved → {POOL_OUT}")

    # -----------------------------------------------------------------------
    # Pre-sample 600 clips stratified by face_size_bin x gender
    # This gives headroom for pose annotation dropout before final 200 sample
    # -----------------------------------------------------------------------
    strata = defaultdict(list)
    for clip_id, meta in pool.items():
        key = (meta["face_size_bin"], meta["gender"])
        strata[key].append(clip_id)

    print(f"\nStrata breakdown (face_size x gender):")
    for k, v in sorted(strata.items()):
        print(f"  {k}: {len(v)} clips")

    # Proportional allocation to PRE_SAMPLE_N
    total = len(pool)
    selected = []
    for key, clip_ids in strata.items():
        n = max(1, round(len(clip_ids) / total * PRE_SAMPLE_N))
        chosen = random.sample(clip_ids, min(n, len(clip_ids)))
        selected.extend(chosen)

    # Trim or top up to exactly PRE_SAMPLE_N
    random.shuffle(selected)
    if len(selected) > PRE_SAMPLE_N:
        selected = selected[:PRE_SAMPLE_N]
    elif len(selected) < PRE_SAMPLE_N:
        remaining = [c for c in pool if c not in set(selected)]
        selected.extend(random.sample(remaining,
                                      min(PRE_SAMPLE_N - len(selected),
                                          len(remaining))))

    sample = {clip_id: pool[clip_id] for clip_id in selected}

    with open(SAMPLE_OUT, "w") as f:
        json.dump(sample, f, indent=2)
    print(f"\nPre-sample ({len(sample)} clips) saved → {SAMPLE_OUT}")

    # Summary of pre-sample strata
    print(f"\nPre-sample strata:")
    sample_strata = defaultdict(int)
    for meta in sample.values():
        sample_strata[(meta["face_size_bin"], meta["gender"])] += 1
    for k, v in sorted(sample_strata.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
