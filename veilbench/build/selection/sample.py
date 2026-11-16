#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path
from collections import defaultdict
import yaml

COVARIATE_DEFS = [
    ("_len_bin",  "clip_length_s"),
    ("_yaw_bin",  "yaw_deg"),
    ("_expr_bin", "expression"),
    ("_snr_bin",  "snr"),
    ("_face_bin", "face_size_px"),
    ("_gender",   "gender"),
    ("_ethnicity","ethnicity"),
]

MEAD_ANGLE_YAW = {
    "front":    "le15",
    "left_30":  "r15_40",
    "right_30": "r15_40",
    "left_60":  "gt40",
    "right_60": "gt40",
    "top":      None,
    "down":     None,
}

def bin_len(d):
    if d <= 5:  return "le5"
    if d <= 15: return "r5_15"
    return "gt15"

def load_json(path):
    if not path.exists(): return {}
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("clips", data)

def gather_and_enrich_subset(subset_name, corpora, ann_dir):
    clips = []
    for corpus in corpora:
        inv_path = ann_dir / corpus / "inventory.json"
        if not inv_path.exists(): continue
        
        raw_clips = load_json(inv_path)
        snr_data  = load_json(ann_dir / corpus / "snr.json")
        face_data = load_json(ann_dir / corpus / "face_detect.json")
        pose_data = load_json(ann_dir / corpus / "pose.json")
        demo_data = load_json(ann_dir / corpus / "demographics.json")

        for c in raw_clips:
            cid = c.get("clip_id")
            if face_data and (cid not in face_data or not face_data[cid].get("detected")):
                continue

            d = c.get("duration_s", 0)
            if d <= 0.5: continue

            enriched = {
                "clip_id": cid,
                "_corpus": corpus,
                "_len_bin": bin_len(d),
                "_snr_bin": snr_data.get(cid, {}).get("snr_bin", "PENDING"),
                "_face_bin": face_data.get(cid, {}).get("face_size_bin", "PENDING"),
                "_yaw_bin": pose_data.get(cid, {}).get("yaw_bin", "PENDING"),
                "_expr_bin": "PENDING",
                "_gender": demo_data.get(cid, {}).get("gender", "PENDING"),
                "_ethnicity": demo_data.get(cid, {}).get("ethnicity", "PENDING")
            }
            
            if corpus == "mead":
                enriched["_yaw_bin"] = MEAD_ANGLE_YAW.get(c.get("angle_label", "front"))
                emotion = c.get("emotion", "neutral")
                intensity = c.get("intensity", "level_1")
                enriched["_expr_bin"] = "neutral_low" if emotion == "neutral" or intensity == "level_1" else "high"
            elif corpus == "cremad":
                emotion = c.get("emotion", "NEU")
                intensity = c.get("intensity", "LO")
                enriched["_expr_bin"] = "neutral_low" if emotion == "NEU" or intensity == "LO" else "high"

            clips.append(enriched)
    return clips

def run_stratified_sampling(pool, total_target, quotas, rng):
    cov_targets = quotas.get("covariates", {})
    
    flattened_mins = {}
    for field_key, cov_name in COVARIATE_DEFS:
        if cov_name in cov_targets:
            for b_name, b_min in cov_targets[cov_name].items():
                flattened_mins[(field_key, b_name)] = b_min
                
    selected_clips = []
    selected_ids = set()
    current_counts = defaultdict(int)
    
    remaining_pool = list(pool)
    rng.shuffle(remaining_pool)
    
    while len(selected_clips) < total_target and remaining_pool:
        # Find the bin lagging furthest behind its minimum quota
        best_deficit = 0
        best_bin = None
        
        for (f_key, b_name), b_min in flattened_mins.items():
            deficit = b_min - current_counts[(f_key, b_name)]
            if deficit > best_deficit:
                best_deficit = deficit
                best_bin = (f_key, b_name)
                
        chosen_idx = -1
        if best_bin is not None:
            # Hunt for a clip that fulfills the needed bin
            fk, bn = best_bin
            for i, c in enumerate(remaining_pool):
                if c.get(fk) == bn:
                    chosen_idx = i
                    break
        
        # If no clip fits the deficit (or no deficits exist), grab the first random available
        if chosen_idx == -1:
            chosen_idx = 0
            
        c = remaining_pool.pop(chosen_idx)
        selected_clips.append(c)
        selected_ids.add(c["clip_id"])
        
        for fk, _ in COVARIATE_DEFS:
            val = c.get(fk)
            if val:
                current_counts[(fk, val)] += 1
                
    return list(selected_ids)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    workdir = Path(args.workdir)
    ann_dir = workdir / "annotations"
    out_dir = workdir / "manifests"
    out_dir.mkdir(exist_ok=True)

    rng = random.Random(args.seed)

    with open(workdir / "quotas.yaml") as f:
        quotas = yaml.safe_load(f)

    subset_map = {
        "veil_id": ["voxceleb2_test"],
        "veil_av": ["mead", "cremad"],
        "veil_hq": ["hdtf"]
    }

    targets = {
        "veil_id": {"full": 600, "lite": 75},
        "veil_av": {"full": 300, "lite": 38},
        "veil_hq": {"full": 100, "lite": 12}
    }

    final_manifest_pools = {}

    for subset, corpora in subset_map.items():
        print(f"Processing stratified draw for {subset}...")
        pool = gather_and_enrich_subset(subset, corpora, ann_dir)
        
        subset_quotas = quotas.get("subsets", {}).get(subset, {})
        
        # Full track exactly hits target
        full_count = targets[subset]["full"]
        full_sampled = run_stratified_sampling(pool, full_count, subset_quotas, rng)
        
        # Lite track sampled strictly from within Full track
        lite_count = targets[subset]["lite"]
        full_pool_enriched = [c for c in pool if c["clip_id"] in full_sampled]
        lite_sampled = run_stratified_sampling(full_pool_enriched, lite_count, subset_quotas, rng)

        final_manifest_pools[subset] = {
            "full": full_sampled,
            "lite": lite_sampled
        }
        print(f"  -> Selected Full: {len(full_sampled)} / Lite: {len(lite_sampled)}")

    with open(out_dir / "selected_clips.json", "w") as f:
        json.dump(final_manifest_pools, f, indent=2)
    print(f"\n[SUCCESS] Saved sampling targets to {out_dir / 'selected_clips.json'}")

if __name__ == "__main__":
    main()
