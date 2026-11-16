import os
import json
import csv
import argparse
from pathlib import Path

def age_to_bin(age_str):
    try:
        age = int(age_str)
        if age <= 2: return "0-2"
        elif age <= 9: return "3-9"
        elif age <= 19: return "10-19"
        elif age <= 29: return "20-29"
        elif age <= 39: return "30-39"
        elif age <= 49: return "40-49"
        elif age <= 59: return "50-59"
        elif age <= 69: return "60-69"
        else: return "70+"
    except ValueError:
        return "Unknown"

def map_race(race, ethnicity):
    r = str(race).lower().strip()
    e = str(ethnicity).lower().strip()
    
    # In CREMA-D, Hispanic is usually split into the Ethnicity column
    if "hispanic" in e and "non" not in e:
        return "Latino_Hispanic"
    if "caucasian" in r or "white" in r:
        return "White"
    if "african" in r or "black" in r:
        return "Black"
    if "asian" in r:
        return "East Asian" # Standardizing to FairFace bin
    return "Unknown"

def map_gender(sex):
    s = str(sex).lower().strip()
    if s == "male" or s == "m": return "Male"
    if s == "female" or s == "f": return "Female"
    return "Unknown"

def main(args):
    workdir = Path(args.workdir)
    csv_path = workdir / "raw/cremad/crema-d-mirror/VideoDemographics.csv"
    anno_dir = workdir / "annotations/cremad"
    face_detect_path = anno_dir / "face_detect.json"
    demo_path = anno_dir / "demographics.json"

    if not csv_path.exists():
        print(f"Error: Could not find {csv_path}")
        return

    # Parse CSV with resilient column matching (handles repos with 'Ethicity' typo)
    actor_metadata = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        
        id_col = next((h for h in headers if "actor" in h.lower()), headers[0])
        age_col = next((h for h in headers if "age" in h.lower()), headers[1])
        sex_col = next((h for h in headers if "sex" in h.lower() or "gender" in h.lower()), headers[2])
        race_col = next((h for h in headers if "race" in h.lower()), headers[3])
        eth_col = next((h for h in headers if "eth" in h.lower()), None)

        for row in reader:
            actor_id = str(row[id_col]).strip()
            age_bin = age_to_bin(row[age_col])
            gender = map_gender(row[sex_col])
            eth_val = row[eth_col] if eth_col else ""
            race = map_race(row[race_col], eth_val)
            
            actor_metadata[actor_id] = {
                "age_group": age_bin,
                "gender": gender,
                "ethnicity": race,
                "demo_source": "cremad_metadata"
            }

    # Load Face Detect
    with open(face_detect_path, "r") as f:
        face_data = json.load(f)

    # Idempotency
    demo_results = {}
    if demo_path.exists():
        with open(demo_path, "r") as f:
            demo_results = json.load(f)
        print(f"Loaded {len(demo_results)} existing annotations.")

    clips_to_process = [cid for cid in face_data.keys() if cid not in demo_results and face_data[cid].get("detected", False)]
    print(f"Clips remaining: {len(clips_to_process)}")

    processed_count = 0
    for clip_id in clips_to_process:
        # clip_id format: cremad____1065_IEO_DIS_MD
        stem = clip_id.split("____", 1)[1]
        actor_id = stem.split("_")[0]

        if actor_id in actor_metadata:
            demo_results[clip_id] = actor_metadata[actor_id]
            processed_count += 1
        else:
            print(f"Warning: Actor {actor_id} not found in CSV for clip {clip_id}")

    with open(demo_path, "w") as f:
        json.dump(demo_results, f, indent=2)
    
    print(f"Finished cremad. Total saved: {len(demo_results)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=str, default=os.environ.get("VEIL_ROOT", "veilbench_data") + "/")
    args = parser.parse_args()
    main(args)
