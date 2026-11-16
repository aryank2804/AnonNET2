"""
sample_veilhq_200.py
Samples exactly 200 VEIL-HQ clips from the 530 successful extractions.
Matches the quotas.yaml constraints (64% M / 36% F, 50% ge160 / 50% r80_160).
Outputs a master list of the final 1100 active benchmark clips.
"""

import os
import json
import random
from collections import defaultdict
from pathlib import Path

random.seed(42)

VEIL = Path(os.environ.get("VEIL_ROOT", "veilbench_data"))
WIN_HQ    = VEIL / "manifests/window_manifest_veilhq.json"
SAMPLE_HQ = VEIL / "manifests/celebvhq_sample.json"
WIN_ID    = VEIL / "manifests/window_manifest_veilid.json"
WIN_AV    = VEIL / "manifests/window_manifest_veilav.json"

OUT_HQ_200   = VEIL / "manifests/veilhq_final_200.json"
ACTIVE_CLIPS = VEIL / "manifests/active_1100_clips.txt"

def get_meta(clip_id, pool_dict):
    """Safely extracts gender and face_size_bin, computing from appearance/bbox if needed."""
    item = pool_dict.get(clip_id, {})
    
    # Gender
    gender = item.get("gender")
    if not gender:
        app = item.get("appearance", [])
        gender = "M" if (len(app) > 1 and app[1] == 1) else "F"

    # Face Size Bin
    face_size = item.get("face_size_bin")
    if not face_size:
        bbox = item.get("bbox", {})
        if bbox:
            h = (bbox.get("bottom", 1.0) - bbox.get("top", 0.0)) * 256
            if h >= 160: face_size = "ge160"
            elif h >= 80: face_size = "r80_160"
            else: face_size = "lt80"
        else:
            face_size = "ge160" # Fallback
            
    return gender, face_size

def main():
    # 1. Load successful HQ clips
    with open(WIN_HQ) as f:
        win_hq = json.load(f)
    ok_hq_clips = [k for k, v in win_hq.items() if v.get("status") == "ok"]

    # 2. Load metadata pool
    with open(SAMPLE_HQ) as f:
        pool_data = json.load(f)
    pool_dict = {item["clip_id"]: item for item in pool_data} if isinstance(pool_data, list) else pool_data

    # 3. Bucket candidates
    buckets = defaultdict(list)
    for cid in ok_hq_clips:
        g, fs = get_meta(cid, pool_dict)
        if fs == "lt80": continue # quotas.yaml sets lt80 to 0 for veil_hq
        buckets[(g, fs)].append(cid)

    for b in buckets.values():
        random.shuffle(b)

    # 4. Target distribution for 200 clips
    # M: 128 (64%), F: 72 (36%) | ge160: 100 (50%), r80_160: 100 (50%)
    targets = {
        ("M", "ge160"): 64,
        ("M", "r80_160"): 64,
        ("F", "ge160"): 36,
        ("F", "r80_160"): 36
    }

    selected_hq = []
    
    # 5. Sample from buckets
    for strata, target_count in targets.items():
        available = buckets[strata]
        taken = available[:target_count]
        selected_hq.extend(taken)
        
        if len(taken) < target_count:
            print(f"Warning: Shortfall in {strata}. Needed {target_count}, found {len(taken)}.")
            # In a full pipeline we'd borrow from another bucket here, 
            # but with 530 OK clips vs 200 needed, we should have plenty of margin.

    print(f"Sampled exactly {len(selected_hq)} VEIL-HQ clips.")

    # Save final HQ selection
    with open(OUT_HQ_200, "w") as f:
        json.dump(selected_hq, f, indent=4)

    # 6. Generate Master Active List (1100 clips)
    active_master = []
    active_master.extend(selected_hq)
    
    for mf in [WIN_ID, WIN_AV]:
        with open(mf) as f:
            data = json.load(f)
            active_master.extend([k for k, v in data.items() if v.get("status") == "ok"])

    with open(ACTIVE_CLIPS, "w") as f:
        f.write("\n".join(active_master))

    print(f"Total active benchmark clips written to active_1100_clips.txt: {len(active_master)}")

if __name__ == "__main__":
    main()
