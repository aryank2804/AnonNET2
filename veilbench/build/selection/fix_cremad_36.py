import json
import os
import subprocess
import random
from pathlib import Path

# Paths based on Grid5000 Sophia layout
RAW_DIR = os.environ.get("VEIL_ROOT", "veilbench_data") + "/raw/cremad/crema-d-mirror/VideoFlash/"
MANIFEST_V1 = "manifests/veilbench_manifest.json"
WINDOW_MANIFEST = "manifests/window_manifest_veilav.json"
OUT_DIR = "clips_81f"

def main():
    # 1. Identify used CREMA-D clips from v1.0
    with open(MANIFEST_V1, 'r') as f:
        v1_manifest = json.load(f)
    
    used_ids = {k for k, v in v1_manifest['clips'].items() if v.get('corpus') == 'cremad'}

    # 2. Find unused FLVs
    all_flvs = list(Path(RAW_DIR).glob("*.flv"))
    # Match the quadruple underscore format: cremad____[name]
    unused_flvs = [p for p in all_flvs if f"cremad____{p.stem}" not in used_ids]
    
    # Shuffle to ensure diverse random sampling across demographics/emotions
    random.shuffle(unused_flvs)

    # 3. Load the AV window manifest to append our new successes
    with open(WINDOW_MANIFEST, 'r') as f:
        window_data = json.load(f)

    valid_clips = []
    replacement_count = 36
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Scanning for {replacement_count} valid CREMA-D FLVs out of {len(unused_flvs)} available...")

    for flv_path in unused_flvs:
        if len(valid_clips) >= replacement_count:
            break

        clip_id = f"cremad____{flv_path.stem}"
        out_path = os.path.join(OUT_DIR, f"{clip_id}.mp4")

        # Check for corruption using ffprobe
        probe_cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(flv_path)
        ]
        
        try:
            probe = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=5)
            if probe.returncode != 0 or not probe.stdout.strip():
                continue # Corrupted, skip
        except subprocess.TimeoutExpired:
            continue

        # Extract exactly 81 frames at 25fps
        ext_cmd = [
            "ffmpeg", "-y", "-i", str(flv_path),
            "-r", "25", "-frames:v", "81",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            "-pix_fmt", "yuv420p", out_path
        ]
        
        ext = subprocess.run(ext_cmd, capture_output=True)
        if ext.returncode == 0:
            valid_clips.append(clip_id)
            window_data[clip_id] = {
                "status": "ok",
                "message": "Extracted as replacement",
                "window": {
                    "start_frame": 0,
                    "end_frame": 81,
                    "policy": "head"
                }
            }
            print(f"[{len(valid_clips)}/{replacement_count}] Extracted {clip_id}")

    # 4. Save updated window manifest
    with open(WINDOW_MANIFEST, 'w') as f:
        json.dump(window_data, f, indent=4)

    print(f"\nDone. Successfully replaced {len(valid_clips)} CREMA-D clips.")
    print(f"Output saved to {OUT_DIR} and {WINDOW_MANIFEST} updated.")

if __name__ == "__main__":
    main()
