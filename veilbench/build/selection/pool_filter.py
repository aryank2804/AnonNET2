#!/usr/bin/env python3
"""
pool_filter.py — VEIL-Bench pre-annotation pool filter
Uses clip_length_s and snr_bin to reduce each corpus to a
coverage-safe candidate pool before expensive GPU annotation.

Target pool sizes (10x quota):
  voxceleb2_test: 6000 (from 36237)
  mead:           3000 (from 207912) — run after snr.json complete
  cremad:          900 (from 7442)
  hdtf:            all 409 (too small to filter)

Output: annotations/{corpus}/pool.json  — list of clip_ids in pool

Usage:
  conda activate datasetsdown2
  python pool_filter.py --workdir . --corpus voxceleb2_test --seed 42
  python pool_filter.py --workdir . --corpus cremad --seed 42
  python pool_filter.py --workdir . --corpus hdtf --seed 42
  # run mead after snr.json finishes
"""

import argparse, json, random
from collections import defaultdict
from pathlib import Path

DURATION_FLOOR = 0.5

POOL_SIZES = {
    "voxceleb2_test": 6000,
    "mead":           3000,
    "cremad":          900,
    "hdtf":           None,   # take all
}

# Minimum per stratum within the pool — ensures coverage cells are filled
# These are pool-level floors, not final quota floors
STRATUM_FLOORS = {
    "voxceleb2_test": {
        "_len_bin":  {"le5": 400, "r5_15": 400, "gt15": 100},
        "_snr_bin":  {"clean": 800, "noisy": 300},
    },
    "cremad": {
        "_len_bin":  {"le5": 300, "r5_15": 50, "gt15": 10},
        "_snr_bin":  {"clean": 200, "noisy": 500},
    },
    "mead": {
        "_len_bin":  {"le5": 1500, "r5_15": 500, "gt15": 20},
        "_snr_bin":  {"clean": 2000, "noisy": 500},
    },
    "hdtf": {},
}


def bin_len(d):
    if d <= 5:  return "le5"
    if d <= 15: return "r5_15"
    return "gt15"


def load_snr(ann_dir, corpus):
    p = ann_dir / corpus / "snr.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--corpus",  required=True, choices=list(POOL_SIZES))
    ap.add_argument("--seed",    type=int, default=42)
    args = ap.parse_args()

    workdir = Path(args.workdir)
    ann_dir = workdir / "annotations"
    rng     = random.Random(args.seed)

    inv_path = ann_dir / args.corpus / "inventory.json"
    with open(inv_path) as f:
        clips = json.load(f)["clips"]

    snr_data = load_snr(ann_dir, args.corpus)
    if not snr_data and args.corpus != "hdtf":
        print(f"[WARN] No snr.json found for {args.corpus} — SNR stratum floors will be skipped")

    # Enrich clips
    enriched = []
    for c in clips:
        d = c.get("duration_s", 0)
        if d <= DURATION_FLOOR:
            continue
        cid = c["clip_id"]
        snr_entry = snr_data.get(cid, {})
        enriched.append({
            "clip_id":    cid,
            "speaker_id": c.get("speaker_id"),
            "session_id": c.get("session_id"),
            "rel_path":   c.get("rel_path"),
            "duration_s": d,
            "_len_bin":   bin_len(d),
            "_snr_bin":   snr_entry.get("snr_bin", "unknown"),
        })

    print(f"[{args.corpus}] eligible clips: {len(enriched)}")

    target = POOL_SIZES[args.corpus]
    if target is None or len(enriched) <= target:
        pool = enriched
        print(f"  Pool = all {len(pool)} clips (no filtering needed)")
    else:
        floors = STRATUM_FLOORS.get(args.corpus, {})

        # Stratified floor sampling
        strata = defaultdict(list)
        for c in enriched:
            key = (c["_len_bin"], c["_snr_bin"])
            strata[key].append(c)

        selected   = {}
        guaranteed = []

        # First pass: guarantee stratum floors
        for field, bins in floors.items():
            field_groups = defaultdict(list)
            for c in enriched:
                field_groups[c.get(field, "unknown")].append(c)
            for bin_name, floor in bins.items():
                candidates = [c for c in field_groups[bin_name]
                              if c["clip_id"] not in selected]
                take = rng.sample(candidates, min(floor, len(candidates)))
                for c in take:
                    selected[c["clip_id"]] = c
                    guaranteed.append(c)

        print(f"  Guaranteed by stratum floors: {len(selected)}")

        # Second pass: random fill to target
        remaining = [c for c in enriched if c["clip_id"] not in selected]
        rng.shuffle(remaining)
        fill = remaining[:max(0, target - len(selected))]
        for c in fill:
            selected[c["clip_id"]] = c

        pool = list(selected.values())
        print(f"  Pool size after fill: {len(pool)}")

    # Write pool
    out_path = ann_dir / args.corpus / "pool.json"
    with open(out_path, "w") as f:
        json.dump([c["clip_id"] for c in pool], f, indent=2)
    print(f"  Written: {out_path}")

    # Summary
    len_counts = defaultdict(int)
    snr_counts = defaultdict(int)
    for c in pool:
        len_counts[c["_len_bin"]] += 1
        snr_counts[c["_snr_bin"]] += 1
    print(f"  len_bin: {dict(len_counts)}")
    print(f"  snr_bin: {dict(snr_counts)}")


if __name__ == "__main__":
    main()
