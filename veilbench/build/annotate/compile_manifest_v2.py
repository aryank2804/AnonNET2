#!/usr/bin/env python3
"""
compile_manifest_v2.py — Compile the V2.0 benchmark manifest.
Handles VEIL-ID (VoxCeleb2), VEIL-HQ (CelebV-HQ), VEIL-AV (MEAD + CREMA-D).
"""
import os
import json
from pathlib import Path

WORKDIR      = Path(os.environ.get("VEIL_ROOT", "veilbench_data"))
ACTIVE_LIST  = Path(os.environ.get("ACTIVE_LIST", WORKDIR / "manifests/clips_756.txt"))
FACE_JSON    = WORKDIR / "annotations/v2_face_detect.json"
POSE_JSON    = WORKDIR / "annotations/v2_pose.json"
DEMO_JSON    = WORKDIR / "annotations/v2_demographics.json"
SNR_JSON     = WORKDIR / "annotations/v2_snr.json"
OUT_MANIFEST = WORKDIR / "manifests/veilbench_manifest_v2.json"

MEAD_YAW_BIN = {
    "front":    "frontal",
    "left_30":  "profile",  "right_30": "profile",
    "left_60":  "extreme",  "right_60": "extreme",
    "top":      "extreme",  "down":     "extreme",
}

CREMAD_EXPR_BIN = {
    "NEU": "neutral_low",
    "ANG": "high", "DIS": "high", "FEA": "high",
    "HAP": "high", "SAD": "high",
}

def get_corpus_and_subset(cid):
    if cid.startswith("id"):       return "voxceleb2_test", "veil_id"
    if cid.startswith("celebvhq"): return "celebv_hq",      "veil_hq"
    if cid.startswith("cremad") or cid.startswith("M") or cid.startswith("W"):
        return "av_subset", "veil_av"
    return "unknown", "unknown"

def get_identity(cid, subset):
    if subset == "veil_id":
        return cid.split("__")[0]
    if subset == "veil_av":
        if cid.startswith("cremad"):
            return cid.split("____")[1].split("_")[0]   # actor number e.g. "1065"
        else:
            return cid.split("____")[0]                  # e.g. "M019"
    return None   # VEIL-HQ has no identity labels

def mead_annotations(cid):
    """Ground-truth yaw_bin and expression_bin from MEAD clip_id."""
    parts    = cid.split("____")
    yaw_bin  = MEAD_YAW_BIN.get(parts[1], "frontal")
    expr_bin = "neutral_low" if parts[2] == "neutral" else "high"
    return yaw_bin, expr_bin

def cremad_expr_bin(cid):
    try:
        code = cid.split("____")[1].split("_")[2]
        return CREMAD_EXPR_BIN.get(code, "high")
    except IndexError:
        return "high"

def main():
    print("Loading annotations...")
    with open(FACE_JSON) as f: face_data = json.load(f)
    with open(POSE_JSON) as f: pose_data = json.load(f)
    with open(DEMO_JSON) as f: demo_data = json.load(f)
    snr_data = json.load(open(SNR_JSON)) if SNR_JSON.exists() else {}   # SNR is only annotated for audio-visual clips

    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(ACTIVE_LIST) as f:
        active_clips = [l.strip() for l in f if l.strip()]

    manifest = {
        "version":     "2.0",
        "description": "VEIL-Bench v2.0 (DreamID-V, short-clip friendly)",
        "clips":       {}
    }
    counts = {"veil_id": 0, "veil_hq": 0, "veil_av": 0, "skipped": 0}

    for cid in active_clips:
        corpus, subset = get_corpus_and_subset(cid)
        if subset == "unknown":
            counts["skipped"] += 1
            continue

        # Must have a detected face
        f_info = face_data.get(cid, {})
        if f_info.get("detected") is not True:
            counts["skipped"] += 1
            continue

        # Must have demographics
        d_info = demo_data.get(cid)
        if d_info is None:
            counts["skipped"] += 1
            continue

        p_info = pose_data.get(cid, {})
        s_info = snr_data.get(cid, {})

        # Pose + expression + SNR — handled differently per sub-corpus
        is_mead = subset == "veil_av" and (cid.startswith("M") or cid.startswith("W"))
        is_cremad = subset == "veil_av" and cid.startswith("cremad")

        if is_mead:
            yaw_bin, expr_bin = mead_annotations(cid)
            yaw = pitch = roll = None
            snr_db  = s_info.get("snr_db")
            snr_bin = s_info.get("snr_bin", "clean")  # studio-clean fallback
        elif is_cremad:
            yaw     = p_info.get("yaw")
            yaw_bin = p_info.get("yaw_bin")
            pitch   = p_info.get("pitch")
            roll    = p_info.get("roll")
            expr_bin = cremad_expr_bin(cid)
            snr_db  = s_info.get("snr_db")
            snr_bin = s_info.get("snr_bin")
        else:
            yaw     = p_info.get("yaw")
            yaw_bin = p_info.get("yaw_bin")
            pitch   = p_info.get("pitch")
            roll    = p_info.get("roll")
            expr_bin = None
            snr_db = snr_bin = None

        manifest["clips"][cid] = {
            "clip_id":  cid,
            "corpus":   corpus,
            "subset":   subset,
            "identity": get_identity(cid, subset),

            "window": {
                "start_frame":  0,
                "end_frame":    80,
                "mean_abs_yaw": abs(yaw) if yaw is not None else None,
                "policy":       "head",
            },

            "face_size_median": f_info.get("face_size_median"),
            "face_size_bin":    f_info.get("face_size_bin"),
            "best_frame_idx":   f_info.get("best_frame_idx"),
            "yaw":     yaw,
            "yaw_bin": yaw_bin,
            "pitch":   pitch,
            "roll":    roll,

            "gender":    d_info.get("gender"),
            "age_group": d_info.get("age_group"),
            "ethnicity": d_info.get("ethnicity"),

            "expression_bin": expr_bin,
            "snr_db":         snr_db,
            "snr_bin":        snr_bin,
        }
        counts[subset] += 1

    with open(OUT_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=4)

    print("\n--- VEIL-Bench V2.0 Manifest ---")
    print(f"VEIL-ID  : {counts['veil_id']}")
    print(f"VEIL-HQ  : {counts['veil_hq']}")
    print(f"VEIL-AV  : {counts['veil_av']}")
    print(f"Skipped  : {counts['skipped']}")
    print(f"Total    : {len(manifest['clips'])}")
    print(f"Saved to : {OUT_MANIFEST.name}")

if __name__ == "__main__":
    main()
