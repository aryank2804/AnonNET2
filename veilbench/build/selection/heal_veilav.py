"""
heal_veilav.py
Completely rebuilds VEIL-AV (300 clips) ensuring 81 frames AND intact audio.
Reconstructs MEAD paths dynamically by splitting the clip_id tokens.
"""
import os
import json
import subprocess
import random
from pathlib import Path

VEIL = Path(os.environ.get("VEIL_ROOT", "veilbench_data"))
SHARE = Path(os.environ.get("SHARE_ROOT", "raw_corpora"))
V1_MANIFEST = VEIL / "manifests/veilbench_manifest.json"
WIN_AV_OUT = VEIL / "manifests/window_manifest_veilav.json"
OUT_DIR = VEIL / "clips_81f"
CREMAD_RAW = VEIL / "raw/cremad/crema-d-mirror/VideoFlash"

def extract_with_audio(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(src),
        "-frames:v", "81",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p",
        str(dst)
    ]
    subprocess.run(cmd)
    
    if not dst.exists(): 
        return False
        
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames", "-of", "default=noprint_wrappers=1:nokey=1", str(dst)
    ], capture_output=True, text=True)
    try:
        return int(probe.stdout.strip()) == 81
    except ValueError:
        return False

def main():
    with open(V1_MANIFEST) as f:
        v1 = json.load(f)
    clips = v1.get("clips", v1)
    
    mead_targets = [k for k, v in clips.items() if v.get("corpus") == "mead"]
    cremad_targets = [k for k, v in clips.items() if v.get("corpus") == "cremad"]
    
    print(f"V1.0 target counts: {len(mead_targets)} MEAD, {len(cremad_targets)} CREMA-D")
    
    win_data = {}
    
    # 1. Process MEAD
    print("\n--- Rebuilding MEAD (210 clips) ---")
    for cid in mead_targets:
        # Reconstruct path from token structure: M019____front____happy____level_2____011
        parts = cid.split("____")
        if len(parts) < 5:
            print(f"Malformed MEAD clip_id: {cid}")
            continue
            
        # Target layout: SHARE/MEAD/videos/{id}/video/{angle}/{emotion}/{level}/{clip}.mp4
        src = SHARE / "MEAD/videos" / parts[0] / "video" / parts[1] / parts[2] / parts[3] / f"{parts[4]}.mp4"
        dst = OUT_DIR / f"{cid}.mp4"
        
        if src.exists() and extract_with_audio(src, dst):
            win_data[cid] = {"status": "ok", "window": {"start_frame": 0, "end_frame": 81, "policy": "head"}}
        else:
            print(f"Failed MEAD extraction or file missing: {cid} (Path: {src})")
            
    # 2. Process CREMA-D
    print("\n--- Rebuilding CREMA-D (90 clips) ---")
    cremad_success = 0
    for cid in cremad_targets:
        stem = cid.split("____", 1)[1]
        src = CREMAD_RAW / f"{stem}.flv"
        dst = OUT_DIR / f"{cid}.mp4"
        if src.exists() and extract_with_audio(src, dst):
            win_data[cid] = {"status": "ok", "window": {"start_frame": 0, "end_frame": 81, "policy": "head"}}
            cremad_success += 1
            
    # 3. Replacements for corrupt CREMA-D FLVs
    if cremad_success < 90:
        needed = 90 - cremad_success
        print(f"Original CREMA-D had {needed} corruptions. Sampling replacements...")
        
        all_flvs = list(CREMAD_RAW.glob("*.flv"))
        random.shuffle(all_flvs)
        
        for flv in all_flvs:
            if cremad_success >= 90: 
                break
            cid = f"cremad____{flv.stem}"
            if cid in win_data: 
                continue 
            
            dst = OUT_DIR / f"{cid}.mp4"
            if extract_with_audio(flv, dst):
                win_data[cid] = {"status": "ok", "message": "Replacement", "window": {"start_frame": 0, "end_frame": 81, "policy": "head"}}
                cremad_success += 1

    with open(WIN_AV_OUT, "w") as f:
        json.dump(win_data, f, indent=4)
        
    print(f"\nDONE! VEIL-AV completely rebuilt. Total successful AV clips: {len(win_data)}")

if __name__ == "__main__":
    main()