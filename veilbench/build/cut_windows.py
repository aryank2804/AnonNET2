#!/usr/bin/env python3
"""
cut_windows.py — materialise the 81-frame VEIL-Bench clips from the raw corpora.

Reads the released clip list and the per-clip start frames (manifests/windows_756.json) and cuts exactly the same
81-frame windows that the benchmark was built with. No pose model or sampling is involved (that is what
build/selection/window_extract.py does when deriving windows from scratch).

Needs the raw corpora under $SHARE_ROOT (see README) and writes $VEIL_ROOT/clips_81f/<clip_id>.mp4.

    python veilbench/build/cut_windows.py --clips veilbench/manifests/clips_756.txt \
                                          --windows veilbench/manifests/windows_756.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "selection"))
import window_extract as we   # source-path resolver + ffmpeg cutter shared with the from-scratch pipeline


def corpus_of(clip_id: str) -> str:
    if clip_id.startswith("id"):
        return "voxceleb2_test"
    if clip_id.startswith("celebvhq"):
        return "celebv_hq"
    raise ValueError(f"Cannot infer corpus for {clip_id}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", required=True, help="text file, one clip id per line")
    ap.add_argument("--windows", required=True, help="json {clip_id: start_frame}")
    ap.add_argument("--max-clips", type=int, default=None, help="only the first N clips (smoke test)")
    args = ap.parse_args()

    ids = [l.strip() for l in open(args.clips) if l.strip()]
    if args.max_clips:
        ids = ids[:args.max_clips]
    starts = json.load(open(args.windows))
    out_dir = we.WORKDIR / "clips_81f"
    out_dir.mkdir(parents=True, exist_ok=True)

    done = skipped = 0
    errors = []
    for i, cid in enumerate(ids, 1):
        dst = out_dir / f"{cid}.mp4"
        if dst.exists():
            skipped += 1
            continue
        try:
            src, src_type = we.resolve_src(cid, corpus_of(cid))
            if not src.exists():
                raise FileNotFoundError(src)
            total, fps = we.get_video_info(src, src_type)
            if total <= we.WINDOW:                     # short source: keep it whole
                start, n = 0, total
            else:
                start, n = int(starts[cid]), we.WINDOW
            if src_type == "jpeg_dir":
                we.extract_from_jpeg_dir(src, dst, start, fps)
            else:
                we.extract_from_video(src, dst, start, fps, n_frames=n)
            done += 1
        except Exception as e:                         # keep going; report at the end
            errors.append((cid, str(e).splitlines()[0] if str(e) else repr(e)))
        if i % 50 == 0:
            print(f"[{i}/{len(ids)}] cut={done} skipped={skipped} errors={len(errors)}", flush=True)

    print(f"done: cut={done} skipped(existing)={skipped} errors={len(errors)}")
    for cid, msg in errors[:20]:
        print(f"  {cid}: {msg}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
