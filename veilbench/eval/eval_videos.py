#!/usr/bin/env python3
import re
"""
AnonNET Table 3 video evaluation. Run twice with different envs:

  # 1. DINO temporal identity consistency  (use env_dino)
  conda activate env_dino
  python eval_videos.py --metric dino --orig /orig --anon /anon --out dino.json

  # 2. Q-Align quality + aesthetics        (use env_qalign)
  conda activate env_qalign
  python eval_videos.py --metric qalign --orig /orig --anon /anon --out qalign.json
"""

import argparse, json, sys
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ── helpers ───────────────────────────────────────────────────────────────────

def find_pairs(orig_dir, anon_dir):
    """Match videos by stem. Strips __scene_000 style suffixes from anon filenames."""
    def norm(stem):
        return re.sub(r'__scene_\d+$', '', stem)
    anon = {norm(p.stem): p for p in Path(anon_dir).rglob("*") if p.suffix.lower() in _EXTS}
    return [(p, anon[norm(p.stem)]) for p in sorted(Path(orig_dir).rglob("*"))
            if p.suffix.lower() in _EXTS and norm(p.stem) in anon]


def read_frames(path, step=1):
    cap = cv2.VideoCapture(str(path))
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok: break
        if i % step == 0:
            frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
        i += 1
    cap.release()
    return frames


def sample_frames(path, n=8):
    cap = cv2.VideoCapture(str(path))
    total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    cap.release()
    idxs = set(np.linspace(0, total - 1, n, dtype=int).tolist())
    frames, i = [], 0
    cap = cv2.VideoCapture(str(path))
    while True:
        ok, f = cap.read()
        if not ok: break
        if i in idxs:
            frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
        i += 1
    cap.release()
    return frames


def agg(records, key):
    vals = [r[key] for r in records if key in r and not np.isnan(r[key])]
    return float(np.mean(vals)) if vals else float("nan")


# ── DINO metric ───────────────────────────────────────────────────────────────
# Paper §4.2: "average cosine distance of DINO [5] embeddings between
# consecutive frames". Lower = more temporally consistent.

def run_dino(args, pairs):
    import torch.nn.functional as F
    from torchvision import transforms

    tfm = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    print("Loading DINO ViT-B/16 ...")
    dino = torch.hub.load("facebookresearch/dino:main", "dino_vitb16",
                          pretrained=True, verbose=False).eval().to(args.device)

    @torch.no_grad()
    def embed(frames, bs=64):
        t = torch.stack([tfm(f) for f in frames])
        out = []
        for i in range(0, len(t), bs):
            out.append(F.normalize(dino(t[i:i+bs].to(args.device)), dim=-1).cpu())
        return torch.cat(out)

    def id_pres(path):
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        step = max(1, round(fps / args.sample_fps)) if args.sample_fps else 1
        frames = read_frames(path, step=step)
        if len(frames) < 2: return float("nan")
        e = embed(frames)
        return float((1 - (e[:-1] * e[1:]).sum(dim=-1)).mean())

    records = []
    for orig_p, anon_p in tqdm(pairs, desc="DINO"):
        records.append({
            "video":        orig_p.name,
            "orig_id_pres": id_pres(orig_p),
            "anon_id_pres": id_pres(anon_p),
        })
    return records


# ── Q-Align metric ────────────────────────────────────────────────────────────
# Official HF page (q-future/one-align): use AutoModelForCausalLM with
# transformers==4.36.1 and attn_implementation="eager".
# model.score(frames, task_="quality"|"aesthetics", input_="image"|"video")

def run_qalign(args, pairs):
    from transformers import AutoModelForCausalLM

    print("Loading one-align ...")
    model = AutoModelForCausalLM.from_pretrained(
        "q-future/one-align",
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval()

    def score_video(path):
        frames = sample_frames(path, n=args.n_frames)
        if not frames: return float("nan"), float("nan")
        # score each frame as an image then average
        qual = model.score(frames, task_="quality",    input_="image")
        aes  = model.score(frames, task_="aesthetics", input_="image")
        qual = float(qual.mean()) if hasattr(qual, "mean") else float(sum(qual) / len(qual))
        aes  = float(aes.mean())  if hasattr(aes,  "mean") else float(sum(aes)  / len(aes))
        return qual, aes

    records = []
    for orig_p, anon_p in tqdm(pairs, desc="Q-Align"):
        oq, oa = score_video(orig_p)
        aq, aa = score_video(anon_p)
        records.append({
            "video":     orig_p.name,
            "orig_qual": oq, "orig_aes": oa,
            "anon_qual": aq, "anon_aes": aa,
        })
    return records


# ── main ──────────────────────────────────────────────────────────────────────

def print_table(records, dataset, metric):
    label = f"  [{dataset}]" if dataset else ""
    W = 54
    print(f"\n{'='*W}")
    print(f"  Table 3{label} — {metric.upper()}")
    print(f"  {'':15s}  {'GT (orig)':>12}  {'Anon (yours)':>12}")
    print(f"  {'-'*(W-4)}")
    if metric == "dino":
        print(f"  {'id_pres ↓':<15}  {agg(records,'orig_id_pres'):>12.3f}  {agg(records,'anon_id_pres'):>12.3f}")
    else:
        print(f"  {'qual ↑':<15}  {agg(records,'orig_qual'):>12.3f}  {agg(records,'anon_qual'):>12.3f}")
        print(f"  {'aes ↑':<15}  {agg(records,'orig_aes'):>12.3f}  {agg(records,'anon_aes'):>12.3f}")
    print(f"{'='*W}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric",     required=True, choices=["dino", "qalign"])
    ap.add_argument("--orig",       required=True)
    ap.add_argument("--anon",       required=True)
    ap.add_argument("--dataset",    default="")
    ap.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sample_fps", type=float, default=None,
                    help="[dino] FPS for frame sampling. None=every frame (paper-faithful).")
    ap.add_argument("--n_frames",   type=int,   default=8,
                    help="[qalign] Frames sampled per video.")
    ap.add_argument("--max_videos", type=int, default=None,
                    help="Cap number of videos for testing (e.g. 5)")
    ap.add_argument("--out",        default="results.json")
    args = ap.parse_args()

    pairs = find_pairs(args.orig, args.anon)
    if not pairs:
        sys.exit("No matched video pairs found. Check --orig / --anon.")
    if args.max_videos:
        pairs = pairs[:args.max_videos]
    print(f"Found {len(pairs)} pairs.")

    if args.metric == "dino":
        records = run_dino(args, pairs)
    else:
        records = run_qalign(args, pairs)

    print_table(records, args.dataset, args.metric)

    with open(args.out, "w") as f:
        json.dump({"dataset": args.dataset, "metric": args.metric,
                   "n_videos": len(records), "per_video": records}, f, indent=2)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
