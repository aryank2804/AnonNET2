#!/usr/bin/env python3
"""
run_batch.py  —  3-phase batch anonymization
Env:  a2

Phase 1 — Cluster  (fast, per video subprocess)
    YOLO + ArcFace clustering, frontal selection, FaceXFormer attributes
    Calls: anonymize.py --stage cluster

Phase 2 — FLUX  (FLUX.1-dev loaded ONCE for entire batch)
    Generates anonymous faces + ArcFace privacy gate for all identities
    Runs inline in this process

Phase 3 — Swap  (DreamID-V loaded ONCE via a single subprocess)
    Extracts scene clips, calls run_dreamidv_worker.py once with all jobs,
    then reassembles each video

Usage:
    python run_batch.py --input-dir /path/to/clips --output-dir /path/to/results \\
        --yolo-model ... --facexformer-repo ... --facexformer-ckpt ... \\
        --dreamidv-repo ... --wan-dir ... --dreamidv-ckpt ...

Flags:
    --phase1-only   stop after clustering
    --phase2-only   stop after FLUX (skip DreamID-V)
    --skip-phase1   skip clustering (resume after it's done)
    --skip-phase2   skip FLUX (resume after it's done)
    --resume        skip videos/identities already completed
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)
import numpy as np
import torch

ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Paths
ap.add_argument("--input-dir",        required=True)
ap.add_argument("--output-dir",       required=True)
ap.add_argument("--yolo-model",       required=True)
ap.add_argument("--facexformer-repo", required=True)
ap.add_argument("--facexformer-ckpt", required=True)
ap.add_argument("--dreamidv-repo",    required=True)
ap.add_argument("--wan-dir",          required=True)
ap.add_argument("--dreamidv-ckpt",    required=True)

# Envs
ap.add_argument("--facex-env",    default="facex")
ap.add_argument("--dreamidv-env", default="dreamidv")
ap.add_argument("--facex-script", default=None)
ap.add_argument("--dreamidv-worker", default=None)

# Control
ap.add_argument("--extensions",       nargs="+", default=["mp4","avi","mov","mkv"])
ap.add_argument("--checkpoint-every", type=int,  default=10)
ap.add_argument("--resume",           action="store_true")
ap.add_argument("--phase1-only",      action="store_true")
ap.add_argument("--phase2-only",      action="store_true")
ap.add_argument("--skip-phase1",      action="store_true")
ap.add_argument("--skip-phase2",      action="store_true")

# Cluster params
ap.add_argument("--sample-fps",          type=float, default=2.0)
ap.add_argument("--min-identity-frames", type=int,   default=3)
ap.add_argument("--max-candidates",      type=int,   default=10)
ap.add_argument("--min-conf",            type=float, default=0.5)
ap.add_argument("--crop-margin",         type=float, default=0.35)

# FLUX params
ap.add_argument("--flux-model",    default="black-forest-labs/FLUX.1-dev")
ap.add_argument("--flux-steps",    type=int,   default=28)
ap.add_argument("--flux-size",     type=int,   default=1024)
ap.add_argument("--flux-guidance", type=float, default=3.5)
ap.add_argument("--flux-suffix",
                default=(", soft even indoor lighting, plain light background, "
                         "natural skin texture, candid amateur photo, looking at the camera"))
ap.add_argument("--arcface-threshold", type=float, default=0.3)
ap.add_argument("--max-retries",       type=int,   default=5)
ap.add_argument("--seed",              type=int,   default=42)

# Swap params
ap.add_argument("--swap-steps", type=int, default=16)
ap.add_argument("--swap-size",  default="832*480")

args = ap.parse_args()

# ── setup ─────────────────────────────────────────────────────────────────────

out_root     = Path(args.output_dir)
workdir_root = out_root / "workdirs"
anon_root    = out_root / "anonymized"
log_path     = out_root / "batch_log.jsonl"

for d in (workdir_root, anon_root):
    d.mkdir(parents=True, exist_ok=True)

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ANONYMIZE    = os.path.join(SCRIPT_DIR, "anonymize.py")
DREAMIDV_W   = args.dreamidv_worker or os.path.join(SCRIPT_DIR, "run_dreamidv_worker.py")
FACEX_SCRIPT = args.facex_script    or os.path.join(SCRIPT_DIR, "run_facexformer.py")

exts   = {f".{e.lstrip('.')}" for e in args.extensions}
videos = sorted(p for p in Path(args.input_dir).rglob("*") if p.suffix.lower() in exts)
print(f"[batch] {len(videos)} videos  output → {out_root}", flush=True)

def log(phase, msg): print(f"[{phase}] {msg}", flush=True)

def append_log(rec):
    with open(log_path, "a") as f:
        f.write(json.dumps(rec) + "\n")

def get_manifests():
    """Return list of (stem, wdir, manifest) for all workdirs with a manifest."""
    result = []
    for v in videos:
        wdir = workdir_root / v.stem
        mp   = wdir / "scene_manifest.json"
        if mp.exists():
            # with open(mp) as f:
            #     result.append((v.stem, wdir, json.load(f)))
            try:
                with open(mp) as f:
                    content = f.read().strip()
                if not content:
                    continue
                result.append((v.stem, wdir, json.loads(content)))
            except json.JSONDecodeError:
                continue
    return result

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — CLUSTER
# ═════════════════════════════════════════════════════════════════════════════

if not args.skip_phase1:
    print(f"\n{'═'*60}", flush=True)
    print(f"  PHASE 1 — CLUSTER  ({len(videos)} videos)", flush=True)
    print(f"{'═'*60}\n", flush=True)

    done = set()
    if args.resume:
        for v in videos:
            rp = workdir_root / v.stem / "pipeline_result.json"
            if rp.exists():
                with open(rp) as f:
                    r = json.load(f)
                if r.get("status") in ("cluster_done","no_faces","no_valid_identities"):
                    done.add(v.stem)
        print(f"[phase1] resume: {len(done)} already done", flush=True)

    counts = {"cluster_done":0,"no_faces":0,"no_valid_identities":0,"failed":0}
    t0 = time.time()

    for idx, video_path in enumerate(videos, 1):
        stem    = video_path.stem
        wdir    = workdir_root / stem
        out_vid = anon_root / f"{stem}.mp4"
        wdir.mkdir(parents=True, exist_ok=True)

        if args.resume and stem in done:
            print(f"[{idx}/{len(videos)}] SKIP {stem}", flush=True); continue

        cmd = [
            sys.executable, ANONYMIZE,
            "--stage",       "cluster",
            "--video",       str(video_path),
            "--workdir",     str(wdir),
            "--output",      str(out_vid),
            "--yolo-model",       args.yolo_model,
            "--facexformer-repo", args.facexformer_repo,
            "--facexformer-ckpt", args.facexformer_ckpt,
            "--facex-env",        args.facex_env,
            "--facex-script",     FACEX_SCRIPT,
            "--sample-fps",       str(args.sample_fps),
            "--min-identity-frames", str(args.min_identity_frames),
            "--max-candidates",   str(args.max_candidates),
            "--min-conf",         str(args.min_conf),
            "--crop-margin",      str(args.crop_margin),
        ]

        t_start = time.time()
        log_file = wdir / "run_cluster.log"
        with open(log_file, "w") as lf:
            proc = subprocess.run(cmd, stdout=lf, stderr=lf)
        elapsed = time.time() - t_start

        rp = wdir / "pipeline_result.json"
        status = "failed"
        if rp.exists():
            with open(rp) as f:
                status = json.load(f).get("status", "failed")
        elif proc.returncode != 0:
            status = "failed"

        tag = {"cluster_done":"OK","no_faces":"NO-FACE",
               "no_valid_identities":"NO-ID","failed":"FAIL"}.get(status,"???")
        print(f"[{idx}/{len(videos)}] {tag}  {stem}  ({elapsed:.0f}s)", flush=True)
        if status == "failed":
            print(f"  log: {log_file}", flush=True)

        counts[status if status in counts else "failed"] += 1
        append_log({"phase":1,"idx":idx,"stem":stem,"status":status,"elapsed":round(elapsed,1)})

        if idx % args.checkpoint_every == 0:
            rate = idx / (time.time()-t0) * 60
            print(f"\n── checkpoint {idx}/{len(videos)} (phase 1) ─────────────", flush=True)
            print(f"   ok={counts['cluster_done']}  no_face={counts['no_faces']}  "
                  f"no_id={counts['no_valid_identities']}  fail={counts['failed']}", flush=True)
            print(f"   rate ≈ {rate:.1f} videos/min\n", flush=True)

    print(f"\n── PHASE 1 DONE ─────────────────────────────────────────", flush=True)
    print(f"   ok={counts['cluster_done']}  no_face={counts['no_faces']}  "
          f"no_id={counts['no_valid_identities']}  fail={counts['failed']}", flush=True)

if args.phase1_only:
    print("[batch] --phase1-only set, stopping."); sys.exit(0)

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — FLUX  (loaded once for entire batch)
# ═════════════════════════════════════════════════════════════════════════════

if not args.skip_phase2:
    print(f"\n{'═'*60}", flush=True)
    print(f"  PHASE 2 — FLUX (loading model once)", flush=True)
    print(f"{'═'*60}\n", flush=True)

    # collect all identities that need FLUX
    flux_jobs = []  # list of (stem, wdir, ident_info, idir)
    for stem, wdir, manifest in get_manifests():
        for ident_info in manifest.get("identities", []):
            uid  = ident_info["id"]
            idir = Path(wdir) / f"identity_{uid}"
            if ident_info.get("status") in ("no_frontal_frame","facexformer_failed"):
                continue
            if not (idir / "face_crop.png").exists(): continue
            if not (idir / "attributes.json").exists(): continue
            if args.resume:
                arc = idir / "arcface_result.json"
                gen = idir / "generated_face.png"
                if arc.exists() and gen.exists():
                    with open(arc) as f:
                        if json.load(f).get("passed"): continue
            flux_jobs.append((stem, wdir, manifest, ident_info, str(idir)))

    print(f"[phase2] {len(flux_jobs)} identities to generate across "
          f"{len(set(j[0] for j in flux_jobs))} videos", flush=True)

    if flux_jobs:
        # load models once
        log("phase2", "loading YOLO11 + ArcFace...")
        from ultralytics import YOLO
        yolo = YOLO(args.yolo_model)
        from insightface.app import FaceAnalysis
        arcface = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
        arcface.prepare(ctx_id=0, det_size=(640,640))

        log("phase2", f"loading FLUX.1-dev from {args.flux_model}...")
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(args.flux_model, torch_dtype=torch.bfloat16)
        pipe.to("cuda")
        log("phase2", f"FLUX loaded — GPU: {torch.cuda.get_device_name(0)}  "
            f"({torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB)")

        def get_emb(img_path):
            img = cv2.imread(img_path)
            if img is None: return None
            res = yolo.predict(img, verbose=False)[0]
            if len(res.boxes) > 0:
                bi = int(res.boxes.conf.cpu().numpy().argmax())
                xyxy = res.boxes.xyxy[bi].cpu().numpy().tolist()
                x1,y1,x2,y2 = [int(v) for v in xyxy]
                crop = img[max(0,y1):min(img.shape[0],y2), max(0,x1):min(img.shape[1],x2)]
                if crop.size > 0:
                    pad = int(max(x2-x1,y2-y1)*1.5)
                    padded = cv2.copyMakeBorder(crop,pad,pad,pad,pad,
                                               cv2.BORDER_CONSTANT,value=(0,0,0))
                    faces = arcface.get(padded)
                    if faces: return max(faces,key=lambda f:f.det_score).normed_embedding
            faces = arcface.get(img)
            return max(faces,key=lambda f:f.det_score).normed_embedding if faces else None

        def gen_face(idir, seed, prompt):
            g = torch.Generator(device="cuda").manual_seed(seed)
            img = pipe(prompt, height=args.flux_size, width=args.flux_size,
                       guidance_scale=args.flux_guidance,
                       num_inference_steps=args.flux_steps, generator=g).images[0]
            out = os.path.join(idir, "generated_face.png")
            img.save(out); return out

        flux_counts = {"passed":0,"failed":0,"no_face":0}

        for i, (stem, wdir, manifest, ident_info, idir) in enumerate(flux_jobs, 1):
            uid   = ident_info["id"]
            label = f"{stem}/id{uid}"

            with open(os.path.join(idir,"attributes.json")) as f:
                attrs = json.load(f)

            prompt = (f"A photorealistic portrait of a {attrs.get('age_label','adult')} "
                      f"{attrs.get('race','person')} {attrs.get('gender','person')}"
                      + args.flux_suffix)
            with open(os.path.join(idir,"flux_prompt.txt"),"w") as f: f.write(prompt)

            log(f"flux {i}/{len(flux_jobs)}", f"{label}")

            orig_emb = get_emb(os.path.join(idir,"face_crop.png"))
            if orig_emb is None:
                log(f"flux {i}/{len(flux_jobs)}", "  no ArcFace embedding, skipping")
                ident_info["status"] = "arcface_no_face"
                flux_counts["no_face"] += 1; continue

            passed=False; final_sim=None; seed_used=args.seed+uid*100
            for attempt in range(args.max_retries):
                gen_path = gen_face(idir, seed_used, prompt)
                gen_emb  = get_emb(gen_path)
                if gen_emb is None:
                    log(f"flux {i}/{len(flux_jobs)}", f"  attempt {attempt+1}: no face in generated image")
                    seed_used+=1; continue
                sim = float(np.dot(orig_emb,gen_emb)/
                            (np.linalg.norm(orig_emb)*np.linalg.norm(gen_emb)+1e-8))
                status_str = "PASS" if sim < args.arcface_threshold else "FAIL"
                log(f"flux {i}/{len(flux_jobs)}", f"  attempt {attempt+1}  sim={sim:.4f} → {status_str}")
                final_sim=sim
                if sim < args.arcface_threshold: passed=True; break
                seed_used+=1

            arc_result = {"similarity":final_sim,"threshold":args.arcface_threshold,
                          "passed":passed,"seed_used":seed_used,"attempts":attempt+1}
            with open(os.path.join(idir,"arcface_result.json"),"w") as f:
                json.dump(arc_result,f,indent=2)

            if passed:
                ident_info["arcface"] = arc_result
                flux_counts["passed"] += 1

                # YOLO crop the FLUX output → tight face crop for DreamID-V
                gen_img = cv2.imread(gen_path)
                crop_res = yolo.predict(gen_img, verbose=False)[0]
                crop_path = os.path.join(idir, "generated_face_crop.png")
                if len(crop_res.boxes) > 0:
                    bi = int(crop_res.boxes.conf.cpu().numpy().argmax())
                    x1,y1,x2,y2 = [int(v) for v in crop_res.boxes.xyxy[bi].cpu().numpy().tolist()]
                    m = 0.3
                    h_img, w_img = gen_img.shape[:2]
                    px1 = max(0, int(x1 - (x2-x1)*m))
                    py1 = max(0, int(y1 - (y2-y1)*m))
                    px2 = min(w_img, int(x2 + (x2-x1)*m))
                    py2 = min(h_img, int(y2 + (y2-y1)*m))
                    cv2.imwrite(crop_path, gen_img[py1:py2, px1:px2])
                    log(f"flux {i}/{len(flux_jobs)}", f"  YOLO crop saved -> {crop_path}")
                else:
                    log(f"flux {i}/{len(flux_jobs)}", "  WARNING: no face in FLUX output, using full image")
                    cv2.imwrite(crop_path, gen_img)
            else:
                ident_info["status"] = "arcface_failed"
                flux_counts["failed"] += 1

            # update manifest after each identity
            with open(os.path.join(wdir,"scene_manifest.json"),"w") as f:
                json.dump(manifest, f, indent=2, cls=NumpyEncoder)

        # unload FLUX
        import gc
        log("phase2", "unloading FLUX...")
        del pipe; gc.collect(); torch.cuda.empty_cache()

        print(f"\n── PHASE 2 DONE ─────────────────────────────────────────", flush=True)
        print(f"   passed={flux_counts['passed']}  failed={flux_counts['failed']}  "
              f"no_face={flux_counts['no_face']}", flush=True)

    else:
        print("[phase2] nothing to do", flush=True)

if args.phase2_only:
    print("[batch] --phase2-only set, stopping."); sys.exit(0)

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 3 — SWAP  (DreamID-V loaded once via single subprocess)
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*60}", flush=True)
print(f"  PHASE 3 — DREAMID-V SWAP", flush=True)
print(f"{'═'*60}\n", flush=True)

# step 3a: extract all scene clips first (fast ffmpeg, no GPU needed)
log("phase3", "extracting scene clips...")

def extract_clip(video_in, t0, t1, out_path):
    if os.path.exists(out_path): return
    subprocess.run(
        ["ffmpeg","-y","-i",video_in,"-ss",str(t0),"-to",str(t1),
         "-c:v","libx264","-crf","18","-preset","fast",out_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

class NumpyEncoder(json.JSONEncoder):
    def default(self,obj):
        if isinstance(obj,np.floating): return float(obj)
        if isinstance(obj,np.integer):  return int(obj)
        if isinstance(obj,np.ndarray):  return obj.tolist()
        return super().default(obj)

all_swap_jobs = []  # list of dicts for dreamidv worker

for stem, wdir, manifest in get_manifests():
    video_path = None
    for v in videos:
        if v.stem == stem: video_path = str(v); break
    if video_path is None: continue

    for ident_info in manifest.get("identities",[]):
        uid  = ident_info["id"]
        idir = os.path.join(wdir, f"identity_{uid}")
        gen_face = os.path.join(idir, "generated_face.png")

        if not os.path.exists(gen_face): continue
        arc_path = os.path.join(idir,"arcface_result.json")
        if os.path.exists(arc_path):
            with open(arc_path) as f:
                if not json.load(f).get("passed"): continue

        scenes_dir = os.path.join(idir,"scenes")
        os.makedirs(scenes_dir, exist_ok=True)

        for si, scene in enumerate(ident_info["scenes"]):
            orig_clip = os.path.join(scenes_dir, f"scene_{si:03d}_orig.mp4")
            anon_clip = os.path.join(scenes_dir, f"scene_{si:03d}_anon.mp4")

            if args.resume and os.path.exists(anon_clip): continue

            extract_clip(video_path, scene["start_time"], scene["end_time"], orig_clip)
            crop_img = os.path.join(idir, "generated_face_crop.png")
            all_swap_jobs.append({
                "ref_video": orig_clip,
                "ref_image": crop_img if os.path.exists(crop_img) else gen_face,
                "out_path":  anon_clip,
            })

log("phase3", f"{len(all_swap_jobs)} swap jobs collected")

# step 3b: write jobs file and call DreamID-V worker ONCE
if all_swap_jobs:
    jobs_file = str(out_root / "dreamidv_jobs.json")
    with open(jobs_file,"w") as f:
        json.dump(all_swap_jobs, f, indent=2)

    log("phase3", f"calling DreamID-V worker for all {len(all_swap_jobs)} clips...")
    cmd = [
        "conda","run","--no-capture-output","-n", args.dreamidv_env,
        "python", DREAMIDV_W,
        "--jobs",       jobs_file,
        "--repo",       args.dreamidv_repo,
        "--wan-dir",    args.wan_dir,
        "--ckpt",       args.dreamidv_ckpt,
        "--size",       args.swap_size,
        "--steps",      str(args.swap_steps),
        "--seed",       str(args.seed),
    ]
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        log("phase3", "WARNING: DreamID-V worker exited with errors — check output")
else:
    log("phase3", "no swap jobs — skipping DreamID-V")

# step 3c: reassemble each video
log("phase3", "reassembling videos...")
reasm_counts = {"success":0,"failed":0,"skipped":0}
t0 = time.time()

for idx, video_path in enumerate(videos, 1):
    stem    = video_path.stem
    wdir    = workdir_root / stem
    out_vid = anon_root / f"{stem}.mp4"
    rp      = wdir / "pipeline_result.json"

    # skip videos with no faces
    if rp.exists():
        with open(rp) as f:
            s = json.load(f).get("status","")
        if s in ("no_faces","no_valid_identities","no_clips"):
            reasm_counts["skipped"]+=1; continue

    if args.resume and out_vid.exists():
        reasm_counts["skipped"]+=1; continue

    cmd = [
        sys.executable, ANONYMIZE,
        "--stage",  "reassemble",
        "--video",  str(video_path),
        "--workdir",str(wdir),
        "--output", str(out_vid),
    ]
    t_start = time.time()
    log_file = wdir / "run_reassemble.log"
    with open(log_file,"w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=lf)
    elapsed = time.time()-t_start

    if proc.returncode == 0:
        reasm_counts["success"]+=1
        print(f"[{idx}/{len(videos)}] OK  {stem}  ({elapsed:.0f}s)", flush=True)
    else:
        reasm_counts["failed"]+=1
        print(f"[{idx}/{len(videos)}] FAIL  {stem}  ({elapsed:.0f}s)  log:{log_file}", flush=True)

    append_log({"phase":3,"idx":idx,"stem":stem,
                "status":"success" if proc.returncode==0 else "failed",
                "elapsed":round(elapsed,1)})

print(f"\n{'═'*60}", flush=True)
print(f"  BATCH COMPLETE", flush=True)
print(f"  success={reasm_counts['success']}  "
      f"skipped={reasm_counts['skipped']}  "
      f"failed={reasm_counts['failed']}", flush=True)
print(f"  results → {anon_root}", flush=True)
print(f"{'═'*60}", flush=True)