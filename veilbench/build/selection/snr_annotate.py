#!/usr/bin/env python3
"""
snr_annotate.py — VEIL-Bench Step 2, Stage 1
Compute audio SNR for all clips. Runs on CPU/login node.
Idempotent — resumes from checkpoint if preempted.

Method: percentile-based
  - Frame audio into 20ms windows
  - Noise floor = 10th percentile of per-frame RMS
  - SNR = 20*log10(signal_rms / noise_rms), clamped [-10, 60]
  - clean >= 20 dB, noisy < 20 dB

Output: annotations/{corpus}/snr.json
  {clip_id: {"snr_db": float|null, "snr_bin": "clean"|"noisy"|"no_audio", "snr_source": str}}

Usage:
  conda activate datasetsdown2
  python snr_annotate.py --workdir . --corpus vidtimit    # smallest, start here
  python snr_annotate.py --workdir . --corpus hdtf
  python snr_annotate.py --workdir . --corpus cremad
  python snr_annotate.py --workdir . --corpus voxceleb2_test
  python snr_annotate.py --workdir . --corpus mead        # largest — run in tmux
"""

import argparse, json, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

DURATION_FLOOR    = 0.5
SNR_CLEAN_DB      = 20.0
CHECKPOINT_EVERY  = 500

CORPUS_ROOTS = {
    "voxceleb2_test": os.environ.get("SHARE_ROOT", "raw_corpora") + "/vox-celeb-dataset/vox-celeb-raw/mp4",
    "mead":           os.environ.get("SHARE_ROOT", "raw_corpora") + "/MEAD/videos",
    "cremad":         os.environ.get("VEIL_ROOT", "veilbench_data") + "/raw/cremad/crema-d-mirror",
    "vidtimit":       os.environ.get("VEIL_ROOT", "veilbench_data") + "/raw/vidtimit",
    "hdtf":           os.environ.get("SHARE_ROOT", "raw_corpora") + "/HDTF-dataset/HDTF-RGB",
}


# ── audio loading ─────────────────────────────────────────────────────────────

def load_wav_scipy(wav_path):
    try:
        from scipy.io import wavfile
        sr, data = wavfile.read(str(wav_path))
        if data.ndim > 1:
            data = data[:, 0]
        data = data.astype(np.float32)
        # normalise int16 range
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
            os.unlink(tmp)
            return None, None
        data, sr = load_wav_scipy(tmp)
        os.unlink(tmp)
        return data, sr
    except Exception:
        return None, None


# ── SNR computation ───────────────────────────────────────────────────────────

def compute_snr(wav, sr):
    if wav is None or len(wav) == 0:
        return None
    wav = wav.astype(np.float32)
    if np.abs(wav).max() < 1e-10:
        return None
    frame_len = max(1, int(sr * 0.02))          # 20ms frames
    n_frames  = len(wav) // frame_len
    if n_frames < 5:
        return 60.0                              # very short → assume clean
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


# ── per-corpus audio resolvers ────────────────────────────────────────────────

def resolve_audio(corpus, clip, root):
    """Returns (wav_array, sr, source_label) or (None, None, reason)."""

    if corpus == "hdtf":
        return None, None, "corpus_no_audio"

    if corpus == "vidtimit":
        if not clip.get("has_audio", False):
            return None, None, "no_audio_track"
        speaker = clip["speaker_id"]
        seq     = clip["sequence_name"]
        wav_path = Path(root) / speaker / speaker / "audio" / f"{seq}.wav"
        if not wav_path.exists():
            return None, None, "wav_missing"
        d, sr = load_wav_scipy(wav_path)
        return d, sr, "vidtimit_wav"

    if corpus == "cremad":
        stem     = Path(clip["rel_path"]).stem
        wav_path = Path(root) / "AudioWAV" / f"{stem}.wav"
        if not wav_path.exists():
            return None, None, "wav_missing"
        d, sr = load_wav_scipy(wav_path)
        return d, sr, "cremad_wav"

    # voxceleb2_test, mead — mp4 files
    mp4_path = Path(root) / clip["rel_path"]
    if not mp4_path.exists():
        return None, None, "mp4_missing"
    d, sr = extract_mp4_audio(mp4_path)
    return d, sr, "ffmpeg_mp4"


# ── main loop ─────────────────────────────────────────────────────────────────

def process_corpus(corpus, clips, root, out_path):
    results = {}
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
        print(f"  Resuming: {len(results)} already done")

    total = len(clips)
    new   = 0

    for i, clip in enumerate(clips):
        cid = clip["clip_id"]
        if cid in results:
            continue

        if clip.get("duration_s", 0) <= DURATION_FLOOR:
            results[cid] = {"snr_db": None, "snr_bin": "no_audio", "snr_source": "floor_filtered"}
            continue

        wav, sr, source = resolve_audio(corpus, clip, root)

        if wav is None:
            results[cid] = {"snr_db": None, "snr_bin": "no_audio", "snr_source": source}
        else:
            db = compute_snr(wav, sr)
            results[cid] = {
                "snr_db":     round(db, 2) if db is not None else None,
                "snr_bin":    snr_bin(db),
                "snr_source": source,
            }
        new += 1

        if (i + 1) % CHECKPOINT_EVERY == 0:
            with open(out_path, "w") as f:
                json.dump(results, f)
            pct = 100 * (i + 1) / total
            print(f"  [{i+1:>6}/{total}] {pct:.1f}%  new={new}  checkpoint saved")

    with open(out_path, "w") as f:
        json.dump(results, f)

    counts = {}
    for v in results.values():
        counts[v["snr_bin"]] = counts.get(v["snr_bin"], 0) + 1
    print(f"  DONE  clean={counts.get('clean',0)}  "
          f"noisy={counts.get('noisy',0)}  no_audio={counts.get('no_audio',0)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--corpus",  required=True,
                    choices=list(CORPUS_ROOTS) + ["all"])
    ap.add_argument("--seed", type=int, default=42)   # kept for interface consistency
    args = ap.parse_args()

    workdir = Path(args.workdir)
    corpora = list(CORPUS_ROOTS) if args.corpus == "all" else [args.corpus]

    for corpus in corpora:
        inv_path = workdir / "annotations" / corpus / "inventory.json"
        if not inv_path.exists():
            print(f"[SKIP] {inv_path} not found")
            continue
        with open(inv_path) as f:
            clips = json.load(f)["clips"]
        root    = CORPUS_ROOTS[corpus]
        out_path = workdir / "annotations" / corpus / "snr.json"
        print(f"\n[{corpus}]  {len(clips)} clips")
        process_corpus(corpus, clips, root, out_path)


if __name__ == "__main__":
    main()