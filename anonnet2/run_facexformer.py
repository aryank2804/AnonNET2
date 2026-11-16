#!/usr/bin/env python3
"""
run_facexformer.py  —  Standalone FaceXFormer attribute extraction
Env:  facex
Called as a subprocess from anonymize.py (a2 env).

Usage:
    python run_facexformer.py \
        --face-crop  /path/to/face_crop.png \
        --output     /path/to/attributes.json \
        --repo       /path/to/facexformer \
        --ckpt       /path/to/facexformer/ckpts/model.pt

Writes attributes.json with: age_bin, age_label, gender, race
"""

import argparse
import json
import sys

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--face-crop", required=True)
ap.add_argument("--output",    required=True)
ap.add_argument("--repo",      required=True)
ap.add_argument("--ckpt",      required=True)
args = ap.parse_args()

# ── labels ───────────────────────────────────────────────────────────────────
# Age: 8 UTKFace decade bins → semantic FLUX-friendly labels
AGE_BIN_LABELS = [
    "child",        # 0-9
    "young adult",  # 10-19  (teens read as young adult for prompt purposes)
    "young adult",  # 20-29
    "young adult",  # 30-39
    "middle-aged",  # 40-49
    "middle-aged",  # 50-59
    "elderly",      # 60-69
    "elderly",      # 70+
]
AGE_BIN_RANGES = ["0-9","10-19","20-29","30-39","40-49","50-59","60-69","70+"]

# Race: 5-class UTKFace
RACE_LABELS   = ["White", "Black", "Asian", "Indian", "Others"]

# Gender: binary
GENDER_LABELS = ["Male", "Female"]

# ── load model ────────────────────────────────────────────────────────────────
sys.path.insert(0, args.repo)
from network import FaceXFormer  # noqa

model = FaceXFormer()
ckpt  = torch.load(args.ckpt, map_location="cpu")
model.load_state_dict(ckpt["state_dict_backbone"])
model = model.cuda().eval()

# ── preprocess ────────────────────────────────────────────────────────────────
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
img = Image.open(args.face_crop).convert("RGB")
x   = transform(img).unsqueeze(0).cuda()

# ── inference ─────────────────────────────────────────────────────────────────
# tasks tensor: 4 = age/gender/race
tasks = torch.tensor([4]).cuda()

with torch.no_grad():
    (_, _, _, _, age_out, gender_out, race_out, _) = model(x, None, tasks)

# ── decode ────────────────────────────────────────────────────────────────────
age_idx    = int(age_out[0].argmax().cpu().item())
gender_idx = int(gender_out[0].argmax().cpu().item())
race_idx   = int(race_out[0].argmax().cpu().item())

# softmax confidences for logging
age_probs = F.softmax(age_out[0], dim=-1).cpu().tolist()

attrs = {
    "age_bin":       age_idx,
    "age_range":     AGE_BIN_RANGES[age_idx],
    "age_label":     AGE_BIN_LABELS[age_idx],
    "age_probs":     {r: round(p, 3) for r, p in zip(AGE_BIN_RANGES, age_probs)},
    "gender":        GENDER_LABELS[gender_idx],
    "race":          RACE_LABELS[race_idx],
    "source":        "facexformer",
}

with open(args.output, "w") as f:
    json.dump(attrs, f, indent=2)

print(f"[facex] age={attrs['age_range']} ({attrs['age_label']})  "
      f"gender={attrs['gender']}  race={attrs['race']}")
print(f"[facex] done → {args.output}")