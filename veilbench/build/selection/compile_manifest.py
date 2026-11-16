"""
compile_manifest.py — build veilbench_manifest.json from all annotation files
"""
import os
import json, pathlib, re

WORKDIR = pathlib.Path(os.environ.get("VEIL_ROOT", "veilbench_data"))

def load(corpus, fname):
    p = WORKDIR / "annotations" / corpus / f"{fname}.json"
    return json.load(open(p)) if p.exists() else {}

def corpus_of(clip_id):
    if clip_id.startswith("id"):       return "voxceleb2_test"
    if re.match(r'^[MW]\d+', clip_id): return "mead"
    if clip_id.startswith("cremad"):   return "cremad"
    if clip_id.startswith("hdtf"):     return "hdtf"
    raise ValueError(f"Unknown corpus for clip_id: {clip_id}")

def main():
    clips = json.load(open(WORKDIR / "manifests/selected_clips.json"))

    # Build tier lookup: clip_id -> list of tiers it appears in
    tier_map = {}
    for subset in ["veil_id", "veil_av", "veil_hq"]:
        for tier in ["full", "lite"]:
            for clip_id in clips[subset][tier]:
                tier_map.setdefault(clip_id, {"subset": subset, "tiers": []})
                tier_map[clip_id]["tiers"].append(tier)

    # Load all annotation files per corpus
    annots = {}
    for corpus in ["voxceleb2_test", "mead", "hdtf", "cremad"]:
        annots[corpus] = {
            "face_detect":  load(corpus, "face_detect"),
            "pose":         load(corpus, "pose"),
            "demographics": load(corpus, "demographics"),
            "transcripts":  load(corpus, "transcripts"),  # empty for non-vox
        }

    manifest = {}
    missing_annots = []

    for clip_id, meta in tier_map.items():
        corpus = corpus_of(clip_id)
        a = annots[corpus]

        face = a["face_detect"].get(clip_id)
        pose = a["pose"].get(clip_id)
        demo = a["demographics"].get(clip_id)
        trans = a["transcripts"].get(clip_id)  # None for non-vox

        if face is None or demo is None:
            missing_annots.append(clip_id)

        # Flatten pose — set to None if errored
        if pose and "error" in pose:
            pose = None

        record = {
            "subset":   meta["subset"],
            "tiers":    meta["tiers"],
            "corpus":   corpus,
            "face":     face,
            "pose":     pose,
            "demographics": demo,
        }
        if trans is not None:
            record["transcript"] = trans

        manifest[clip_id] = record

    out = WORKDIR / "manifests/veilbench_manifest.json"
    json.dump({"version": "1.0", "clips": manifest}, open(out, "w"), indent=2)

    print(f"Total clips:        {len(manifest)}")
    print(f"Missing any annot:  {len(missing_annots)}")
    if missing_annots:
        for c in missing_annots[:10]:
            print(f"  {c}")
    print(f"Written to {out}")

    # Summary stats
    for subset in ["veil_id", "veil_av", "veil_hq"]:
        subset_clips = [v for v in manifest.values() if v["subset"] == subset]
        full = sum(1 for v in subset_clips if "full" in v["tiers"])
        lite = sum(1 for v in subset_clips if "lite" in v["tiers"])
        no_pose = sum(1 for v in subset_clips if v["pose"] is None)
        print(f"{subset}: full={full}, lite={lite}, pose_errors={no_pose}")

if __name__ == "__main__":
    main()
