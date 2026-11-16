#!/usr/bin/env python3
import argparse, json, random, sys
from collections import defaultdict
from pathlib import Path
import yaml

DURATION_FLOOR = 0.5

MEAD_ANGLE_YAW = {
    "front":    "le15",
    "left_30":  "r15_40",
    "right_30": "r15_40",
    "left_60":  "gt40",
    "right_60": "gt40",
    "top":      None,
    "down":     None,
}

def load_inv(path):
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if "clips" in data:
        return data["clips"]
    raise ValueError(f"Cannot find clip list in {path}")

def bin_len(d):
    if d <= 5:  return "le5"
    if d <= 15: return "r5_15"
    return "gt15"

def enrich_mead(c):
    d = c.get("duration_s", 0)
    if d <= DURATION_FLOOR: return None
    emotion   = c.get("emotion", "neutral")
    intensity = c.get("intensity", "level_1")
    return {**c,
        "_corpus":   "mead",
        "_len_bin":  bin_len(d),
        "_yaw_bin":  MEAD_ANGLE_YAW.get(c.get("angle_label", "front")),
        "_expr_bin": "neutral_low" if emotion == "neutral" or intensity == "level_1" else "high",
    }

def enrich_cremad(c):
    d = c.get("duration_s", 0)
    if d <= DURATION_FLOOR: return None
    emotion   = c.get("emotion", "NEU")
    intensity = c.get("intensity", "LO")
    return {**c,
        "_corpus":   "cremad",
        "_len_bin":  bin_len(d),
        "_expr_bin": "neutral_low" if emotion == "NEU" or intensity == "LO" else "high",
    }

def enrich_generic(corpus):
    def fn(c):
        d = c.get("duration_s", 0)
        if d <= DURATION_FLOOR: return None
        return {**c, "_corpus": corpus, "_len_bin": bin_len(d)}
    return fn

def count_field(clips, field):
    counts = defaultdict(int)
    for c in clips:
        v = c.get(field)
        if v and v != "PENDING":
            counts[v] += 1
    return dict(counts)

def all_pending(clips, field):
    vals = {c.get(field) for c in clips if c.get(field) is not None}
    return not vals or vals == {"PENDING"}

def random_draw_exp(clips, field, bins, bin_mins, n_draw, n_trials, rng):
    target_bins = [b for b in bins if bin_mins.get(b, 0) > 0]
    if not target_bins: return None
        
    eligible = [c for c in clips if c.get(field) not in (None, "PENDING")]
    if not eligible: return None
        
    empty_counts = []
    for _ in range(n_trials):
        sample = rng.sample(eligible, min(n_draw, len(eligible)))
        found  = {c[field] for c in sample}
        empty_counts.append(len(set(target_bins) - found))
        
    return {
        "n_draw":           n_draw,
        "n_trials":         n_trials,
        "n_eligible":       len(eligible),
        "mean_empty_bins":  round(sum(empty_counts) / len(empty_counts), 2),
        "worst_empty_bins": max(empty_counts),
    }

def write_report(results, path):
    lines = ["VEIL-Bench Coverage Report", "=" * 62, ""]
    for subset, covs in results.items():
        lines.append(f"── {subset} " + "─" * (50 - len(subset)))
        for cov_name, info in covs.items():
            lines.append(f"  {cov_name}:")
            if info.get("pending"):
                lines.append("    [PENDING — requires Step 2 annotations]")
                continue
            for bname, bd in info["cells"].items():
                status = "OK " if bd["count"] >= bd["min"] else "GAP"
                lines.append(f"    {bname:<15} count={bd['count']:>6}  min={bd['min']:>4}  {status}"
                             + (f"  short by {bd['gap']}" if bd["gap"] else ""))
            rd = info.get("random_draw")
            if rd:
                lines.append(
                    f"    [random draw] {rd['n_draw']} clips × {rd['n_trials']} trials → "
                    f"mean {rd['mean_empty_bins']} empty bins, worst {rd['worst_empty_bins']}"
                )
        lines.append("")
    text = "\n".join(lines)
    path.write_text(text)
    return text

COVARIATE_DEFS = [
    ("_len_bin",  "clip_length_s", ["le5", "r5_15", "gt15"]),
    ("_yaw_bin",  "yaw_deg",       ["le15", "r15_40", "gt40"]),
    ("_expr_bin", "expression",    ["neutral_low", "high"]),
    ("_snr_bin",  "snr",           ["clean", "noisy"]),
    ("_face_bin", "face_size_px",  ["ge160", "r80_160", "lt80"]),
    ("_gender",   "gender",        ["Male", "Female"]),
    ("_ethnicity","ethnicity",     ["White", "Black", "Latino_Hispanic", "East Asian", "Indian", "Middle Eastern"]),
]

CORPUS_ENRICHERS = {
    "mead":           enrich_mead,
    "cremad":         enrich_cremad,
    "voxceleb2_test": enrich_generic("voxceleb2_test"),
    "vidtimit":       enrich_generic("vidtimit"),
    "hdtf":           enrich_generic("hdtf"),
}

SUBSET_CORPORA = {
    "veil_id":  ["voxceleb2_test"],
    "veil_av":  ["mead", "cremad"],
    "veil_hq":  ["hdtf"],
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir",       required=True)
    ap.add_argument("--seed",          type=int, default=42)
    ap.add_argument("--random-draws",  type=int, default=2000)
    ap.add_argument("--random-trials", type=int, default=500)
    args = ap.parse_args()

    workdir  = Path(args.workdir)
    ann_dir  = workdir / "annotations"
    out_dir  = workdir / "coverage"
    out_dir.mkdir(exist_ok=True)

    rng = random.Random(args.seed)
    quota_path = workdir / "quotas.yaml"
    if not quota_path.exists():
        sys.exit(f"[ERROR] quotas.yaml not found at {quota_path}")
    with open(quota_path) as f:
        quotas = yaml.safe_load(f)

    corpus_clips = {}
    for corpus, fn in CORPUS_ENRICHERS.items():
        inv = ann_dir / corpus / "inventory.json"
        if not inv.exists():
            continue
            
        print(f"[load] {corpus} ...", end=" ", flush=True)
        raw = load_inv(inv)
        enriched = [fn(c) for c in raw]
        enriched = [c for c in enriched if c is not None]
        
        def _ld(name):
            p = ann_dir / corpus / name
            if p.exists():
                with open(p) as f: return json.load(f)
            return {}
            
        snr_data  = _ld("snr.json")
        face_data = _ld("face_detect.json")
        pose_data = _ld("pose.json")
        demo_data = _ld("demographics.json")
        
        final_enriched = []
        for c in enriched:
            cid = c.get("clip_id")
            
            if face_data and (cid not in face_data or not face_data[cid].get("detected")):
                continue
                
            c["_snr_bin"]  = snr_data.get(cid, {}).get("snr_bin", "PENDING")
            c["_face_bin"] = face_data.get(cid, {}).get("face_size_bin", "PENDING")
            c["_yaw_bin"]  = pose_data.get(cid, {}).get("yaw_bin", c.get("_yaw_bin", "PENDING"))
            c["_gender"]   = demo_data.get(cid, {}).get("gender", "PENDING")
            c["_ethnicity"]= demo_data.get(cid, {}).get("ethnicity", "PENDING")
            c["_age"]      = demo_data.get(cid, {}).get("age_group", "PENDING")
            
            final_enriched.append(c)

        print(f"{len(final_enriched)} valid clips.")
        corpus_clips[corpus] = final_enriched

    subset_clips = {
        name: sum([corpus_clips.get(c, []) for c in corpora], [])
        for name, corpora in SUBSET_CORPORA.items()
    }

    results    = {}
    cells_out  = {}

    for subset_name, clips in subset_clips.items():
        subset_quota = quotas.get("subsets", {}).get(subset_name, {})
        cov_quotas   = subset_quota.get("covariates", {})

        results[subset_name]   = {}
        cells_out[subset_name] = {}

        for field, cov_name, bins in COVARIATE_DEFS:
            bin_mins = cov_quotas.get(cov_name, {})
            if all_pending(clips, field):
                results[subset_name][cov_name] = {"pending": True}
                continue

            counts    = count_field(clips, field)
            cell_data = {
                b: {"count": counts.get(b, 0),
                    "min":   bin_mins.get(b, 0),
                    "gap":   max(0, bin_mins.get(b, 0) - counts.get(b, 0))}
                for b in bins
            }
            
            rd = random_draw_exp(clips, field, bins, bin_mins,
                                 args.random_draws, args.random_trials, rng)

            results[subset_name][cov_name]   = {"cells": cell_data, "random_draw": rd}
            cells_out[subset_name][cov_name] = cell_data

    cells_path = out_dir / "coverage_cells.json"
    with open(cells_path, "w") as f:
        json.dump(cells_out, f, indent=2)
    print(f"\n[write] {cells_path}")

    report_path = out_dir / "coverage_report.txt"
    report = write_report(results, report_path)
    print(f"[write] {report_path}\n")
    print(report)

if __name__ == "__main__":
    main()
