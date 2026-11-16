#!/usr/bin/env python3
"""
cremad_demo_merge.py — Merge CREMA-D demographics from VideoDemographics.csv
into v2_demographics.json, matching the FaceXFormer output schema.
"""
import os
import json, csv
from pathlib import Path

WORKDIR  = Path(os.environ.get("VEIL_ROOT", "veilbench_data"))
CSV_PATH = WORKDIR / "raw/cremad/crema-d-mirror/VideoDemographics.csv"
DEMO_JSON = WORKDIR / "annotations/v2_demographics.json"
CLIPS_DIR = WORKDIR / "clips_81f"

def age_to_group(age):
    age = int(age)
    if age < 20:  return "10-19"
    elif age < 30: return "20-29"
    elif age < 40: return "30-39"
    elif age < 50: return "40-49"
    elif age < 60: return "50-59"
    elif age < 70: return "60-69"
    else:          return "70+"

def map_ethnicity(race, ethnicity):
    # Ethnicity column flags Hispanic origin regardless of race
    if "hispanic" in ethnicity.lower():
        return "Latino"
    r = race.strip().lower()
    if "caucasian" in r:           return "White"
    elif "african" in r or "black" in r: return "Black"
    elif "asian" in r:             return "Asian"
    else:                          return "Other"

def main():
    # Load CSV → actor_id → demographics
    actor_demo = {}
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            actor_id = int(row["ActorID"])
            actor_demo[actor_id] = {
                "age_group":   age_to_group(row["Age"]),
                "gender":      row["Sex"].strip(),
                "ethnicity":   map_ethnicity(row["Race"], row["Ethnicity"]),
                "demo_source": "videographics_csv",
            }

    # Load existing demographics JSON
    demo_data = {}
    if DEMO_JSON.exists():
        with open(DEMO_JSON) as f:
            demo_data = json.load(f)

    cremad_clips = sorted(CLIPS_DIR.glob("cremad____*.mp4"))
    added = skipped = missing = 0

    for clip_path in cremad_clips:
        cid = clip_path.stem
        if cid in demo_data:
            skipped += 1
            continue
        try:
            stem     = cid.split("____")[1]   # e.g. 1065_IEO_DIS_MD
            actor_id = int(stem.split("_")[0]) # 1065
        except (IndexError, ValueError):
            print(f"WARNING: cannot parse actor ID from {cid}")
            missing += 1
            continue
        if actor_id not in actor_demo:
            print(f"WARNING: actor {actor_id} not in CSV ({cid})")
            missing += 1
            continue
        demo_data[cid] = actor_demo[actor_id]
        added += 1

    with open(DEMO_JSON, "w") as f:
        json.dump(demo_data, f, indent=2)

    print(f"Added:   {added} CREMA-D clips")
    print(f"Skipped: {skipped} (already present)")
    print(f"Missing: {missing}")
    print(f"Total in v2_demographics.json: {len(demo_data)}")

if __name__ == "__main__":
    main()
