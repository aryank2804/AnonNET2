#!/usr/bin/env python3
"""
snr_annotate_v2.py — Stage 1: V2.0 audio SNR annotation.
Computes audio SNR directly from the 81-frame MP4 clips in clips_81f/.
Runs on MEAD and CREMA-D clips only (VoxCeleb2 and CelebV-HQ do not require SNR).
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
WORKDIR = Path(os.environ.get("VEIL_ROOT", "veilbench_data"))
CLIPS_DIR = WORKDIR / "clips_81f"
ACTIVE_LIST = Path(os.environ.get("ACTIVE_LIST", WORKDIR / "manifests/clips_756.txt"))
OUT_PATH = WORKDIR / "annotations/v2_snr.json"

SNR_CLEAN_DB = 20.0

# ── Audio Extraction ───────────────────────────────────────────────────────────
def load_wav_scipy(wav_path):
    try:
        from scipy.io import wavfile
        sr, data = wavfile.read(str(wav_path))
        if data.ndim > 1:
            data = data[:, 0]
        data = data.astype(np.float32)
        # Normalise int16 range
        if np.abs(data).max() > 1.0:
            data = data / 32768.0
        return data, int(sr)
    except Exception:
        return None, None

def extract_mp4_audio(mp4_path):
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", str(mp4_path),
             "-ac", "1", "-ar", "16000", "-vn", tmp, "-loglevel", "error"],
            capture_output=True, timeout=60
        )
        if r.returncode != 0:
            if os.path.exists(tmp): os.unlink(tmp)
            return None, None
        data, sr = load_wav_scipy(tmp)
        if os.path.exists(tmp): os.unlink(tmp)
        return data, sr
    except Exception:
        return None, None

# ── SNR Math ───────────────────────────────────────────────────────────────────
def compute_snr(wav, sr):
    if wav is None or len(wav) == 0:
        return None
    wav = wav.astype(np.float32)
    if np.abs(wav).max() < 1e-10:
        return None
    
    frame_len = max(1, int(sr * 0.02))          # 20ms frames
    n_frames  = len(wav) // frame_len
    if n_frames < 5:
        return 60.0                             # Very short -> assume clean
        
    frames     = wav[:n_frames * frame_len].reshape(n_frames, frame_len)
    frame_rms  = np.sqrt(np.mean(frames ** 2, axis=1))
    noise_rms  = float(np.percentile(frame_rms, 10))
    signal_rms = float(np.sqrt(np.mean(wav ** 2)))
    
    if noise_rms < 1e-10:
        return 60.0
    return float(np.clip(20.0 * np.log10(signal_rms / noise_rms), -10.0, 60.0))

def snr_bin(snr_db):
    if snr_db is None:
        return "no_audio"
    return "clean" if snr_db >= SNR_CLEAN_DB else "noisy"

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not ACTIVE_LIST.exists():
        print(f"ERROR: {ACTIVE_LIST} not found. Run sample script first.")
        return

    with open(ACTIVE_LIST) as f:
        all_clips = [line.strip() for line in f if line.strip()]

    # Filter out VoxCeleb2 (starts with 'id') and CelebV-HQ (starts with 'celebvhq')
    av_clips = [
        cid for cid in all_clips 
        if not cid.startswith("id") and not cid.startswith("celebvhq")
    ]
    print(f"Total audio-visual clips to annotate (MEAD + CREMA-D): {len(av_clips)}")

    results = {}
    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            results = json.load(f)
        print(f"Resuming: {len(results)} clips already done.")

    pending = [cid for cid in av_clips if cid not in results]
    if not pending:
        print("All audio-visual clips processed.")
        return

    for i, cid in enumerate(pending):
        vid_path = CLIPS_DIR / f"{cid}.mp4"
        
        if not vid_path.exists():
            results[cid] = {"snr_db": None, "snr_bin": "no_audio", "snr_source": "missing_file"}
            continue

        wav, sr = extract_mp4_audio(vid_path)
        
        if wav is None:
            results[cid] = {"snr_db": None, "snr_bin": "no_audio", "snr_source": "extraction_failed"}
        else:
            db = compute_snr(wav, sr)
            results[cid] = {
                "snr_db": round(db, 2) if db is not None else None,
                "snr_bin": snr_bin(db),
                "snr_source": "ffmpeg_mp4",
            }

        if (i + 1) % 100 == 0:
            with open(OUT_PATH, "w") as f:
                json.dump(results, f)
            print(f"[{i+1}/{len(pending)}] Checkpoint saved.")

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=4)

    # Print summary breakdown
    counts = {}
    for v in results.values():
        counts[v["snr_bin"]] = counts.get(v["snr_bin"], 0) + 1
    print(f"\nDONE. clean={counts.get('clean', 0)} noisy={counts.get('noisy', 0)} no_audio={counts.get('no_audio', 0)}")

if __name__ == "__main__":
    main()
