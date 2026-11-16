#!/usr/bin/env python3
"""
VEIL-Bench Step 1 — inventory.py

Inventories VoxCeleb2 TEST, MEAD, HDTF, CREMA-D, and VidTIMIT in-place on shared storage.
Writes structured JSON outputs to {workdir}/annotations/{corpus}/.

Usage:
    conda activate bench
    python inventory.py --corpus voxceleb2_test
    python inventory.py --corpus mead [--workers 16]
    python inventory.py --corpus hdtf
    python inventory.py --corpus cremad
    python inventory.py --corpus vidtimit

    # Dry-run (skip checksums, first 100 clips only):
    python inventory.py --corpus mead --limit 100 --no-checksum

Done-criterion (Step 1):
    inventory.json exists for all corpora, ffprobe_errors == 0 (or understood),
    checksums.json committed, spot-check of 20 random clips per corpus passes.

Idempotent: safe to kill and re-run after besteffort preemption.
Checksum cache keyed by (path, mtime) — unmodified files are never re-hashed.
"""

import os
import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Optional
import subprocess

from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────────────────────────────────────

SHARE   = Path(os.environ.get("SHARE_ROOT", "raw_corpora"))
WORKDIR = Path(os.environ.get("VEIL_ROOT", "veilbench_data"))

CORPUS_ROOTS: dict[str, Path] = {
    "voxceleb2_test": SHARE / "vox-celeb-dataset/vox-celeb-raw/mp4",
    "mead":           SHARE / "MEAD/videos",
    "hdtf":           SHARE / "HDTF-dataset/HDTF-RGB",
    "cremad":         WORKDIR / "raw/cremad/crema-d-mirror",
    "vidtimit":       WORKDIR / "raw/vidtimit",
}

LICENSE_TABLE: dict[str, dict] = {
    "voxceleb2_test": {
        "name":           "VoxCeleb2 (TEST split only)",
        "url":            "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html",
        "redistribution": "identifiers_and_scripts_only",
        "notes": (
            "Academic non-commercial. Raw MP4s not redistributable. "
            "TEST split is disjoint from ECAPA-TDNN / VoxCeleb2-dev training data — "
            "attacker-evaluation safe (RQ0-c). "
            "Distribute speaker IDs + download scripts only."
        ),
    },
    "mead": {
        "name":           "MEAD",
        "url":            "https://wywu.github.io/projects/MEAD/MEAD.html",
        "redistribution": "academic_non_commercial_check_license",
        "notes": (
            "Signed MEAD research license required. "
            "Verify redistribution clause before releasing media or derived features."
        ),
    },
    "hdtf": {
        "name":           "HDTF",
        "url":            "https://github.com/MRzzm/HDTF",
        "redistribution": "scripts_and_ids_only",
        "notes": (
            "Source videos from YouTube. "
            "Redistribute download scripts + IDs only."
        ),
    },
    "cremad": {
        "name":           "CREMA-D",
        "url":            "https://github.com/CheyneyComputerScience/CREMA-D",
        "redistribution": "open_odbl_1.0",
        "notes": (
            "Open Database License (ODbL). Fully redistributable with attribution. "
            "91 actors, 7442 clips, 6 emotions, 4 intensities. "
            "Demographics available in VideoDemographics.csv — skips FairFace annotation."
        ),
    },
    "vidtimit": {
        "name":           "VidTIMIT",
        "url":            "https://conradsanderson.id.au/vidtimit/",
        "redistribution": "check_eula",
        "notes": (
            "EULA required from Conrad Sanderson. Academic non-commercial. "
            "43 subjects x 3 head-rotation sessions — primary T4/T6 multi-session source."
        ),
    },
}

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".flv", ".mkv", ".webm", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# MEAD controlled vocabulary (from dataset spec)
MEAD_EMOTIONS    = {"angry","contempt","disgusted","fear","happy","neutral","sad","surprised"}
MEAD_INTENSITIES = {"level_1","level_2","level_3"}
MEAD_ANGLES = {"front", "down", "top", "left_30", "right_30", "left_60", "right_60"}

# CREMA-D controlled vocabulary (from filename encoding)
CREMAD_EMOTIONS = {
    "ANG": "angry", "DIS": "disgusted", "FEA": "fear",
    "HAP": "happy", "NEU": "neutral",   "SAD": "sad",
}
CREMAD_INTENSITIES = {
    "LO": "low", "MD": "medium", "HI": "high", "XX": "unspecified",
}

# VidTIMIT session keys (head rotation sequences = T4 session split)
VIDTIMIT_HEAD_SESSIONS = {"head", "head2", "head3"}


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_partial(p: Path, limit_mb: int = 10) -> str:
    """SHA-256 of first `limit_mb` MB. Appends '_partial' suffix if file was truncated."""
    h = hashlib.sha256()
    limit = limit_mb * 1024 * 1024
    read  = 0
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
            read += len(chunk)
            if read >= limit:
                return h.hexdigest() + "_partial"
    return h.hexdigest()


def ffprobe(p: Path) -> Optional[dict]:
    """Basic A/V metadata via ffprobe. Returns None on any failure."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(p),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return None
        data = json.loads(r.stdout)
    except Exception:
        return None

    fmt      = data.get("format", {})
    duration = float(fmt.get("duration") or 0)
    width = height = fps = None
    has_audio = False

    for s in data.get("streams", []):
        ct = s.get("codec_type")
        if ct == "video" and width is None:
            width, height = s.get("width"), s.get("height")
            fr = s.get("r_frame_rate", "0/1")
            try:
                n, d = fr.split("/")
                fps = round(int(n) / int(d), 3) if int(d) else None
            except Exception:
                pass
        if ct == "audio":
            has_audio = True

    return dict(
        duration_s=round(duration, 3),
        fps=fps,
        width=width,
        height=height,
        has_audio=has_audio,
    )


def _stats(vals: list[float]) -> dict:
    if not vals:
        return {}
    return {
        "count":   len(vals),
        "min":     round(min(vals), 3),
        "max":     round(max(vals), 3),
        "mean":    round(mean(vals), 3),
        "median":  round(median(vals), 3),
        "stdev":   round(stdev(vals), 3) if len(vals) > 1 else 0.0,
        "total_h": round(sum(vals) / 3600, 4),
    }


def _freq(vals: list) -> dict:
    t: dict[str, int] = defaultdict(int)
    for v in vals:
        t[str(v)] += 1
    return dict(sorted(t.items(), key=lambda x: -x[1]))


# ─────────────────────────────────────────────────────────────────────────────
# Checksum cache — mtime-keyed, survives preemption restarts
# ─────────────────────────────────────────────────────────────────────────────

class ChecksumCache:
    def __init__(self, path: Path):
        self.path = path
        self._d: dict = {}
        if path.exists():
            try:
                self._d = json.loads(path.read_text())
            except Exception:
                pass

    def get(self, p: Path) -> Optional[str]:
        e = self._d.get(str(p))
        if e and abs(e.get("mtime", 0) - p.stat().st_mtime) < 0.01:
            return e["sha256"]
        return None

    def put(self, p: Path, digest: str):
        self._d[str(p)] = {"sha256": digest, "mtime": p.stat().st_mtime}

    def compute(self, p: Path) -> str:
        cached = self.get(p)
        if cached:
            return cached
        digest = sha256_partial(p)
        self.put(p, digest)
        return digest

    def save(self):
        self.path.write_text(json.dumps(self._d, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Crawlers — path discovery ONLY, no ffprobe
# Each yields plain dicts; ffprobe is applied in a separate parallel phase.
#
# Internal-only keys prefixed with "_" are stripped before writing JSON.
# ─────────────────────────────────────────────────────────────────────────────

def crawl_voxceleb2_test(root: Path):
    """
    Layout: root/id{N}/{video_id}/{utterance}.mp4

    speaker_id  = id{N}           (118 test-split identities)
    session_id  = video_id        (each YouTube video = one independent recording)
    utterance_id = mp4 stem       (segment within that video)

    session_id is the T4 gallery split key: enrollment from session A,
    probe from session B. No further processing needed here.
    """
    for spk_dir in sorted(root.iterdir()):
        if not spk_dir.is_dir() or not spk_dir.name.startswith("id"):
            continue
        speaker_id = spk_dir.name
        for vid_dir in sorted(spk_dir.iterdir()):
            if not vid_dir.is_dir():
                continue
            session_id = vid_dir.name
            for mp4 in sorted(vid_dir.glob("*.mp4")):
                yield {
                    "clip_id":      f"{speaker_id}____{session_id}____{mp4.stem}",
                    "rel_path":     str(mp4.relative_to(root)),
                    "speaker_id":   speaker_id,
                    "session_id":   session_id,
                    "utterance_id": mp4.stem,
                    "file_type":    "video",
                }


def crawl_mead(root: Path):
    """
    Actual layout: root/{actor}/video/{angle}/{emotion}/{intensity}/*.mp4

    All stratification covariates free from path — no Step 2 annotation needed.
    Angle names as stored: front, down, top, left_30, right_30, left_60, right_60
    """
    for actor_dir in sorted(root.iterdir()):
        if not actor_dir.is_dir():
            continue
        actor_id = actor_dir.name          # M003, W011, ...

        video_dir = actor_dir / "video"
        if not video_dir.is_dir():
            continue                        # skip actors without video/ subdir

        for angle_dir in sorted(video_dir.iterdir()):
            if not angle_dir.is_dir() or angle_dir.name not in MEAD_ANGLES:
                continue
            angle = angle_dir.name

            for emo_dir in sorted(angle_dir.iterdir()):
                if not emo_dir.is_dir() or emo_dir.name not in MEAD_EMOTIONS:
                    continue
                emotion = emo_dir.name

                for inten_dir in sorted(emo_dir.iterdir()):
                    if not inten_dir.is_dir():
                        continue
                    # neutral sometimes skips intensity — detect by checking
                    # whether child is already a video file
                    if inten_dir.name in MEAD_INTENSITIES:
                        intensity = inten_dir.name
                        clip_dir  = inten_dir
                    else:
                        continue           # unexpected subdir, skip

                    for mp4 in sorted(clip_dir.glob("*.mp4")):
                        yield {
                            "clip_id":      (
                                f"{actor_id}____{angle}____{emotion}"
                                f"____{intensity}____{mp4.stem}"
                            ),
                            "rel_path":     str(mp4.relative_to(root)),
                            "speaker_id":   actor_id,
                            "session_id":   f"{actor_id}____{angle}____{emotion}____{intensity}",
                            "utterance_id": mp4.stem,
                            "emotion":      emotion,
                            "intensity":    intensity,
                            "angle_label":  angle,
                            "file_type":    "video",
                        }


def crawl_hdtf(root: Path):
    """
    HDTF-RGB has two possible layouts:
      (a) flat .mp4 files at root level
      (b) per-clip subdirectories of JPEG frames (no audio)

    Detected dynamically. Frame sequences get duration = n_frames / 25 fps.
    We store the first frame path in _probe_path for dimension lookup in
    Phase 2; this key is stripped before JSON output.
    """
    top = list(root.iterdir())
    has_videos = any(f.is_file() and f.suffix.lower() in VIDEO_EXTS for f in top)

    if has_videos:
        # Layout (a): flat MP4s
        for f in sorted(root.glob("*")):
            if f.suffix.lower() not in VIDEO_EXTS:
                continue
            yield {
                "clip_id":     f"hdtf____{f.stem}",
                "rel_path":    str(f.relative_to(root)),
                "speaker_id":  f.stem,
                "session_id":  f.stem,
                "file_type":   "video",
                "hdtf_format": "mp4",
            }
    else:
        # Layout (b): frame-sequence directories
        for clip_dir in sorted(root.iterdir()):
            if not clip_dir.is_dir():
                continue
            frames = sorted(
                f for f in clip_dir.iterdir()
                if f.suffix.lower() in IMAGE_EXTS
            )
            if not frames:
                continue
            n = len(frames)
            yield {
                "clip_id":     f"hdtf____{clip_dir.name}",
                "rel_path":    str(clip_dir.relative_to(root)),
                "speaker_id":  clip_dir.name,
                "session_id":  clip_dir.name,
                "file_type":   "frames",
                "hdtf_format": "frames",
                "n_frames":    n,
                "assumed_fps": 25.0,
                "duration_s":  round(n / 25.0, 3),
                "has_audio":   False,
                # Internal: path for dimension probe in Phase 2
                "_probe_path": str(frames[0]),
            }


def crawl_cremad(root: Path):
    """
    Layout: root/VideoFlash/{actor}_{sentence}_{emotion}_{intensity}.flv
            root/AudioWAV/{same_stem}.wav
    All metadata from filename — no path traversal complexity.
    Demographics from root/VideoDemographics.csv — skips FairFace entirely.
    """
    # Load demographics once
    demo: dict[str, dict] = {}
    demo_csv = root / "VideoDemographics.csv"
    if demo_csv.exists():
        with open(demo_csv, newline="") as f:
            for row in csv.DictReader(f):
                actor = str(row.get("ActorID", "")).strip()
                demo[actor] = {
                    "demo_age":       row.get("Age", "").strip(),
                    "demo_sex":       row.get("Sex", "").strip(),
                    "demo_race":      row.get("Race", "").strip(),
                    "demo_ethnicity": row.get("Ethnicity", "").strip(),
                }

    video_dir = root / "VideoFlash"
    audio_dir = root / "AudioWAV"

    if not video_dir.exists():
        print(f"[cremad] WARNING: VideoFlash/ not found at {root}")
        return

    for vf in sorted(f for f in video_dir.iterdir() if f.suffix.lower() in {".mp4", ".flv"}):
        parts = vf.stem.split("_")
        if len(parts) != 4:
            continue
            
        actor_id, sentence_id, emotion_code, intensity_code = parts
        wav = audio_dir / f"{vf.stem}.wav"
        
        rec = {
            "clip_id":          f"cremad____{vf.stem}",
            "rel_path":         str(vf.relative_to(root)),
            "speaker_id":       actor_id,
            "session_id":       actor_id,   # one session per actor in CREMA-D
            "utterance_id":     vf.stem,
            "sentence_id":      sentence_id,
            "emotion":          CREMAD_EMOTIONS.get(emotion_code, emotion_code.lower()),
            "intensity":        CREMAD_INTENSITIES.get(intensity_code, intensity_code.lower()),
            "has_paired_audio": wav.exists(),
            "file_type":        "video",
        }
        if actor_id in demo:
            rec.update(demo[actor_id])
            
        yield rec


def crawl_vidtimit(root: Path):
    """
    Layout: root/{subject}/{subject}/video/{sequence}/001 002 003 ...
            root/{subject}/{subject}/audio/{sequence}.wav
    head / head2 / head3 = 3 independent recording sessions → T4/T6 split key.
    Sentence sequences (sa1, sa2, si*, sx*) = scripted speech with paired audio.
    Frames are numbered files (no extension); ffprobe reads by content not extension.
    """
    for subj_outer in sorted(root.iterdir()):
        if not subj_outer.is_dir():
            continue
            
        subject_id = subj_outer.name
        # Handle double-nesting: {subject}/{subject}/
        subj_inner = subj_outer / subject_id
        if not subj_inner.is_dir():
            subj_inner = subj_outer
            
        video_dir = subj_inner / "video"
        audio_dir = subj_inner / "audio"
        
        if not video_dir.is_dir():
            continue
            
        for seq_dir in sorted(video_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
                
            seq_name = seq_dir.name
            is_head  = seq_name in VIDTIMIT_HEAD_SESSIONS
            
            # two-pass: grab first frame cheap, count the rest without sorting
            frames_iter = (f for f in seq_dir.iterdir() if f.is_file())
            first_frame = next(frames_iter, None)
            
            if not first_frame:
                continue
                
            n_frames = 1 + sum(1 for _ in frames_iter)
            wav      = audio_dir / f"{seq_name}.wav"
            
            yield {
                "clip_id":       f"vidtimit____{subject_id}____{seq_name}",
                "rel_path":      str(seq_dir.relative_to(root)),
                "speaker_id":    subject_id,
                # head/head2/head3 get their own session_id → enables T4 session-disjoint split
                # sentence clips share subject session_id (not used for gallery split)
                "session_id":    seq_name if is_head else subject_id,
                "sequence_name": seq_name,
                "sequence_type": "head_rotation" if is_head else "sentence",
                "n_frames":      n_frames,
                "assumed_fps":   25.0,
                "duration_s":    round(n_frames / 25.0, 3),
                "has_audio":     wav.exists(),
                "file_type":     "frames",
                "_probe_path":   str(first_frame),  # stripped before JSON output
            }


CRAWLERS = {
    "voxceleb2_test": crawl_voxceleb2_test,
    "mead":           crawl_mead,
    "hdtf":           crawl_hdtf,
    "cremad":         crawl_cremad,
    "vidtimit":       crawl_vidtimit,
}

PRIVATE_KEYS = {"_probe_path"}  # stripped before JSON output


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate stats
# ─────────────────────────────────────────────────────────────────────────────

def build_stats(clips: list[dict], corpus: str) -> dict:
    spk_sessions: dict[str, set] = defaultdict(set)
    spk_clips:    dict[str, int] = defaultdict(int)
    for c in clips:
        sid = c.get("speaker_id", "?")
        spk_sessions[sid].add(c.get("session_id", "?"))
        spk_clips[sid] += 1

    session_counts = [len(v) for v in spk_sessions.values()]
    clips_per_id   = list(spk_clips.values())
    durations      = [c["duration_s"] for c in clips if c.get("duration_s") is not None]
    fpss           = [c["fps"]        for c in clips if c.get("fps")        is not None]
    widths         = [c["width"]      for c in clips if c.get("width")      is not None]

    stats: dict[str, Any] = {
        "n_clips":               len(clips),
        "n_identities":          len(spk_sessions),
        "clips_per_identity":    _stats([float(x) for x in clips_per_id]),
        "sessions_per_identity": _stats([float(x) for x in session_counts]),
        "duration_s":            _stats(durations),
        "fps_distribution":      _freq(fpss),
        "width_px_distribution": _freq(widths),
        "ffprobe_errors":        sum(1 for c in clips if c.get("ffprobe_error")),
    }

    if corpus == "mead":
        stats["emotion_distribution"]   = _freq([c.get("emotion")     for c in clips if c.get("emotion")])
        stats["intensity_distribution"] = _freq([c.get("intensity")   for c in clips if c.get("intensity")])
        stats["angle_distribution"]     = _freq([c.get("angle_label") for c in clips if c.get("angle_label")])

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Worker functions (top-level so ThreadPoolExecutor can pickle them)
# ─────────────────────────────────────────────────────────────────────────────

def _ffprobe_worker(args: tuple[int, dict, Path]) -> tuple[int, Optional[dict]]:
    """Returns (original_index, ffprobe_result_or_None)."""
    idx, clip, root = args
    if clip.get("file_type") == "frames":
        probe_path = clip.get("_probe_path")
        if probe_path:
            m = ffprobe(Path(probe_path))
            if m:
                return idx, {"width": m.get("width"), "height": m.get("height")}
        return idx, None
    full = root / clip["rel_path"]
    return idx, ffprobe(full)


def _checksum_worker(args: tuple[int, dict, Path]) -> tuple[int, str]:
    """Returns (original_index, digest). Skips frame dirs."""
    idx, clip, root = args
    if clip.get("file_type") == "frames":
        # Checksum first frame only; note it in the digest string
        probe = clip.get("_probe_path")
        if probe and Path(probe).is_file():
            digest = sha256_partial(Path(probe)) + "_first_frame_only"
            return idx, digest
        return idx, "skipped_frame_dir"
    full = root / clip["rel_path"]
    if not full.is_file():
        return idx, "file_not_found"
    return idx, sha256_partial(full)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="VEIL-Bench Step 1: per-corpus inventory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--corpus",      required=True, choices=list(CORPUS_ROOTS),
                    help="Which corpus to inventory")
    ap.add_argument("--workdir",     default=str(WORKDIR),
                    help="VEIL-Bench root directory")
    ap.add_argument("--workers",     type=int, default=8,
                    help="Parallel workers for ffprobe + checksum phases")
    ap.add_argument("--no-checksum", action="store_true",
                    help="Skip checksumming (useful for dry runs)")
    ap.add_argument("--limit",       type=int, default=None,
                    help="Process only first N clips (testing)")
    args = ap.parse_args()

    corpus  = args.corpus
    root    = CORPUS_ROOTS[corpus]
    out_dir = Path(args.workdir) / "annotations" / corpus
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = ChecksumCache(out_dir / ".checksum_cache.json")

    print(f"[inventory] corpus  = {corpus}")
    print(f"[inventory] root    = {root}")
    print(f"[inventory] out_dir = {out_dir}")
    print(f"[inventory] workers = {args.workers}")

    if not root.exists():
        sys.exit(f"ERROR: corpus root not found: {root}")

    # ── Phase 1: path discovery (single-threaded, very fast) ─────────────────
    print("\n[Phase 1] path discovery ...")
    clips: list[dict] = []
    for rec in tqdm(CRAWLERS[corpus](root), desc="discover", unit="clip"):
        clips.append(rec)
        if args.limit and len(clips) >= args.limit:
            print(f"[Phase 1] --limit {args.limit} reached")
            break
    print(f"[Phase 1] found {len(clips):,} clips")

    # ── Phase 2: ffprobe (parallel) ───────────────────────────────────────────
    print(f"\n[Phase 2] ffprobe ({len(clips):,} clips × {args.workers} workers) ...")
    probe_args = [(i, c, root) for i, c in enumerate(clips)]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_ffprobe_worker, a): a[0] for a in probe_args}
        for fut in tqdm(as_completed(futs), total=len(clips),
                        desc="ffprobe", unit="clip"):
            idx, meta = fut.result()
            if meta:
                if clips[idx].get("file_type") == "frames":
                    # frame dirs: only apply width/height; duration already set
                    clips[idx].update({k: v for k, v in meta.items()
                                       if k in ("width", "height")})
                else:
                    clips[idx].update(meta)
            else:
                if clips[idx].get("file_type") != "frames":
                    clips[idx].update({
                        "duration_s": None, "fps": None,
                        "width": None, "height": None,
                        "has_audio": None, "ffprobe_error": True,
                    })

    # ── Intermediate save after Phase 2 (survives Phase 3 preemption) ────────
    inv_path = out_dir / "inventory.json"
    inv_path.write_text(json.dumps({
        "corpus":       corpus,
        "generated_at": now_utc(),
        "version":      "1.0",
        "root_path":    str(root),
        "license":      LICENSE_TABLE[corpus],
        "stats":        build_stats(clips, corpus),
        "clips":        clips,
    }, indent=2))
    print(f"[inventory] ✓ intermediate inventory saved ({len(clips):,} clips, no checksums yet)")

    # ── Phase 3: checksums (parallel, cache-backed) ───────────────────────────
    checksums: dict[str, str] = {}
    if not args.no_checksum:
        print(f"\n[Phase 3] checksums ({args.workers} workers) ...")
        ck_args = [(i, c, root) for i, c in enumerate(clips)]

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_checksum_worker, a): a[0] for a in ck_args}
            for fut in tqdm(as_completed(futs), total=len(clips),
                            desc="checksum", unit="clip"):
                idx, digest = fut.result()
                rel = clips[idx]["rel_path"]
                checksums[rel] = digest
                clips[idx]["sha256_partial"] = digest
                # Update cache for file-backed clips
                if clips[idx].get("file_type") != "frames":
                    full = root / rel
                    if full.is_file():
                        cache.put(full, digest.replace("_partial", ""))
                
                # ── periodic cache flush ──────────────────────────────
                if len(checksums) % 2000 == 0:
                    cache.save()

        cache.save()
    else:
        print("\n[Phase 3] checksums skipped (--no-checksum)")

    # ── Strip private keys before serialisation ────────────────────────────────
    for c in clips:
        for k in PRIVATE_KEYS:
            c.pop(k, None)

    # ── Build inventory ────────────────────────────────────────────────────────
    inventory = {
        "corpus":       corpus,
        "generated_at": now_utc(),
        "version":      "1.0",
        "root_path":    str(root),
        "license":      LICENSE_TABLE[corpus],
        "stats":        build_stats(clips, corpus),
        "clips":        clips,
    }

    # ── Write outputs ──────────────────────────────────────────────────────────
    inv_path = out_dir / "inventory.json"
    inv_path.write_text(json.dumps(inventory, indent=2))
    print(f"\n[inventory] ✓ {inv_path}  ({len(clips):,} clips)")

    ck_path = out_dir / "checksums.json"
    ck_path.write_text(json.dumps({
        "corpus":       corpus,
        "generated_at": now_utc(),
        "algorithm":    "sha256_partial_10mb",
        "files":        checksums,
    }, indent=2))
    print(f"[inventory] ✓ {ck_path}  ({len(checksums):,} entries)")

    # ── Console summary ────────────────────────────────────────────────────────
    s = inventory["stats"]
    print(f"\n── {corpus} summary {'─'*40}")
    print(f"  clips       : {s['n_clips']:,}")
    print(f"  identities  : {s['n_identities']}")
    sp = s.get("sessions_per_identity", {})
    if sp.get("mean"):
        print(f"  sessions/id : {sp['min']} – {sp['max']} (mean {sp['mean']})")
    d = s.get("duration_s", {})
    if d.get("mean"):
        print(f"  duration    : {d['min']}s – {d['max']}s  "
              f"(mean {d['mean']}s, total {d['total_h']}h)")
    fps = s.get("fps_distribution", {})
    if fps:
        print(f"  fps         : {dict(list(fps.items())[:4])}")
    if s.get("ffprobe_errors"):
        print(f"  ⚠  ffprobe errors : {s['ffprobe_errors']}  "
              f"(check annotations/{corpus}/inventory.json for clip_ids)")
    if corpus == "mead":
        print(f"  emotions    : {list(s['emotion_distribution'].keys())}")
        print(f"  intensities : {list(s['intensity_distribution'].keys())}")
        print(f"  angles      : {list(s['angle_distribution'].keys())}")
    print()


if __name__ == "__main__":
    main()