#!/usr/bin/env python3
"""
VEIL-Bench Step 1 — download_missing.py
Downloads CREMA-D and VidTIMIT via kagglehub into workdir/raw/.
Run after placing ~/.kaggle/kaggle.json.
"""

import os
import shutil
from pathlib import Path

WORKDIR = Path(os.environ.get("VEIL_ROOT", "veilbench_data"))
RAW     = WORKDIR / "raw"

# Direct kagglehub cache into our raw dir
os.environ["KAGGLE_CACHE_DIR"] = str(RAW)

import kagglehub  # import AFTER setting env var

DATASETS = {
    "cremad":   "ejlok1/cremad",
    "vidtimit": "crazyt/vidtimit-audiovideo-dataset",
}

for name, slug in DATASETS.items():
    target = RAW / name
    print(f"\n[download] {name} ← kaggle:{slug}")
    dl_path = Path(kagglehub.dataset_download(slug))
    print(f"[download] kagglehub landed at: {dl_path}")

    # kagglehub nests under datasets/{slug}/versions/{n}/
    # move contents to our flat raw/{name}/ dir
    target.mkdir(parents=True, exist_ok=True)
    if dl_path != target and dl_path.exists():
        for item in dl_path.iterdir():
            dest = target / item.name
            if not dest.exists():
                shutil.move(str(item), str(dest))
                print(f"[download]   moved {item.name} → {dest}")
            else:
                print(f"[download]   already exists, skipping: {item.name}")

    print(f"[download] ✓ {name} → {target}")
    print(f"[download]   contents: {[p.name for p in sorted(target.iterdir())]}")