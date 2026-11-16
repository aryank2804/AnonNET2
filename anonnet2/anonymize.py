#!/usr/bin/env python3
"""
anonymize.py  —  Single-video helper called by run_batch.py
Env:  a2
FaceXFormer subprocess: facex env

Stages:
  --stage cluster    extract frames, cluster identities, frontal selection,
                     FaceXFormer attributes. Writes workdir structure.
  --stage reassemble reads workdir + clip list, runs ffmpeg concat → output.mp4

FLUX and DreamID-V are NOT handled here — run_batch.py manages those
across the whole batch with a single model load each.
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import cv2
import numpy as np

# ─────────────────────────── CLI ─────────────────────────────────────────────

ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("--video",   required=True)
ap.add_argument("--workdir", required=True)
ap.add_argument("--output",  default=None)
ap.add_argument("--stage",   required=True, choices=["cluster", "reassemble"])

# Model paths (cluster stage)
ap.add_argument("--yolo-model",       default="yolo11n-face.pt")
ap.add_argument("--facexformer-repo", default=None)
ap.add_argument("--facexformer-ckpt", default=None)
ap.add_argument("--facex-env",        default="facex")
ap.add_argument("--facex-script",     default=None)

# Clustering
ap.add_argument("--sample-fps",          type=float, default=2.0)
ap.add_argument("--cluster-threshold",   type=float, default=0.45)
ap.add_argument("--min-identity-frames", type=int,   default=3)
ap.add_argument("--gap-tolerance",       type=float, default=1.5)
ap.add_argument("--min-scene-duration",  type=float, default=0.5)

# Frontal selection
ap.add_argument("--max-candidates", type=int,   default=10)
ap.add_argument("--min-conf",       type=float, default=0.5)
ap.add_argument("--crop-margin",    type=float, default=0.35)

args = ap.parse_args()

os.makedirs(args.workdir, exist_ok=True)
output_path = args.output or os.path.join(args.workdir, "output.mp4")

if args.facex_script is None:
    args.facex_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "run_facexformer.py")

# ─────────────────────────── helpers ─────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)

def log(step, msg): print(f"[{step}] {msg}", flush=True)
def die(step, msg): print(f"[{step}] FAILED: {msg}", flush=True); sys.exit(1)
def run(cmd, **kw): return subprocess.run(cmd, **kw)

def write_result(status, extra=None):
    r = {"status": status, "video_in": os.path.abspath(args.video),
         "video_out": os.path.abspath(output_path)}
    if extra: r.update(extra)
    with open(os.path.join(args.workdir, "pipeline_result.json"), "w") as f:
        json.dump(r, f, indent=2, cls=NumpyEncoder)

# ═════════════════════════════════════════════════════════════════════════════
#  CLUSTER STAGE
# ═════════════════════════════════════════════════════════════════════════════

if args.stage == "cluster":

    # ── extract frames ────────────────────────────────────────────────────────
    log("setup", f"video={args.video}")
    frames_dir = os.path.join(args.workdir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    log("frames", "extracting via ffmpeg...")
    run(["ffmpeg", "-y", "-i", args.video, "-q:v", "2",
         os.path.join(frames_dir, "%06d.png")],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    if not frame_paths:
        die("frames", "ffmpeg produced no frames")

    probe = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height,r_frame_rate",
                 "-of", "json", args.video],
                capture_output=True, text=True, check=True)
    info    = json.loads(probe.stdout)["streams"][0]
    frame_w = int(info["width"])
    frame_h = int(info["height"])
    num, den = info["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    log("frames", f"{len(frame_paths)} frames @ {fps:.2f}fps  {frame_w}x{frame_h}")

    # ── load detection models ─────────────────────────────────────────────────
    log("models", "loading YOLO11 + face_alignment + ArcFace...")
    from ultralytics import YOLO
    yolo = YOLO(args.yolo_model)

    import face_alignment as fa_lib
    fa = fa_lib.FaceAlignment(fa_lib.LandmarksType.TWO_D, flip_input=False, device="cuda")

    from insightface.app import FaceAnalysis
    arcface = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
    arcface.prepare(ctx_id=0, det_size=(640, 640))
    log("models", "loaded")

    # ── cluster identities ────────────────────────────────────────────────────
    log("cluster", f"sampling at {args.sample_fps}fps...")
    step           = max(1, int(fps / args.sample_fps))
    sample_indices = list(range(0, len(frame_paths), step))
    all_detections = []

    for idx in sample_indices:
        img = cv2.imread(frame_paths[idx])
        res = yolo.predict(img, verbose=False)[0]
        for bi in range(len(res.boxes)):
            conf = float(res.boxes.conf[bi].cpu().item())
            if conf < args.min_conf: continue
            xyxy = res.boxes.xyxy[bi].cpu().numpy().tolist()
            x1,y1,x2,y2 = [int(v) for v in xyxy]
            crop = img[max(0,y1):min(frame_h,y2), max(0,x1):min(frame_w,x2)]
            if crop.size == 0: continue
            pad    = int(max(x2-x1, y2-y1) * 1.5)
            padded = cv2.copyMakeBorder(crop, pad, pad, pad, pad,
                                        cv2.BORDER_CONSTANT, value=(0,0,0))
            faces = arcface.get(padded)
            if not faces: continue
            emb = max(faces, key=lambda f: f.det_score).normed_embedding.astype(np.float32)
            all_detections.append({"frame_idx": idx, "bbox": xyxy, "embedding": emb})

    log("cluster", f"{len(all_detections)} detections across {len(sample_indices)} frames")

    if not all_detections:
        run(["cp", args.video, output_path], check=True)
        write_result("no_faces")
        sys.exit(0)

    # greedy clustering
    identities = []
    for det in all_detections:
        emb = det["embedding"]
        best_sim, best_id = -1.0, -1
        for i, ident in enumerate(identities):
            sim = float(np.dot(emb, ident["centroid"]))
            if sim > best_sim: best_sim, best_id = sim, i
        if best_sim >= args.cluster_threshold:
            identities[best_id]["members"].append(det)
            n = len(identities[best_id]["members"])
            c = identities[best_id]["centroid"]
            c = c * ((n-1)/n) + emb/n
            c /= np.linalg.norm(c) + 1e-8
            identities[best_id]["centroid"] = c
        else:
            identities.append({"centroid": emb.copy(), "members": [det]})

    identities = [i for i in identities if len(i["members"]) >= args.min_identity_frames]
    log("cluster", f"{len(identities)} valid identity cluster(s)")

    if not identities:
        run(["cp", args.video, output_path], check=True)
        write_result("no_valid_identities")
        sys.exit(0)

    def group_scenes(frame_indices):
        gap = int(args.gap_tolerance * fps)
        sf  = sorted(set(frame_indices))
        scenes, s, e = [], sf[0], sf[0]
        for f in sf[1:]:
            if f - e <= gap: e = f
            else: scenes.append((s, e)); s = e = f
        scenes.append((s, e))
        return scenes

    min_scene_frames = int(args.min_scene_duration * fps)
    manifest = {"fps": fps, "frame_w": frame_w, "frame_h": frame_h,
                "num_frames": len(frame_paths), "identities": []}

    for uid, ident in enumerate(identities):
        scenes = [(s,e) for s,e in group_scenes([d["frame_idx"] for d in ident["members"]])
                  if (e-s) >= min_scene_frames]
        if not scenes: continue
        manifest["identities"].append({
            "id": uid, "n_detections": len(ident["members"]),
            "scenes": [{"start_frame": s, "end_frame": e,
                        "start_time": round(s/fps,3), "end_time": round(e/fps,3)}
                       for s,e in scenes],
        })
        log("cluster", f"  identity {uid}: {len(ident['members'])} detections, {len(scenes)} scenes")

    with open(os.path.join(args.workdir, "scene_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, cls=NumpyEncoder)

    if not manifest["identities"]:
        die("cluster", "no processable identities after scene filtering")

    # ── frontal selection helpers ─────────────────────────────────────────────

    def symmetry_score(lm68):
        el,er,n = lm68[36],lm68[45],lm68[30]
        ml,mr   = lm68[48],lm68[54]
        d = abs(er[0]-el[0])+1e-8
        cx = (el[0]+er[0])/2
        return abs(n[0]-cx)/d + abs(el[1]-er[1])/d + abs((ml[0]+mr[0])/2-n[0])/d

    def is_occluded(img, lm68):
        path = os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml")
        if not os.path.exists(path): return False
        cas = cv2.CascadeClassifier(path)
        xs,ys = lm68[:,0],lm68[:,1]
        x1,y1,x2,y2 = int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())
        fh = y2-y1
        if fh<=0 or x2<=x1: return False
        h,w = img.shape[:2]
        crop = img[max(0,y1):min(h,y1+int(fh*0.65)), max(0,x1):min(w,x2)]
        if crop.size==0: return False
        eyes = cas.detectMultiScale(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),
                                    1.1,5,minSize=(15,15))
        return len(eyes)<2

    def best_frontal(frame_indices):
        results = []
        for idx in frame_indices:
            img = cv2.imread(frame_paths[idx])
            res = yolo.predict(img, verbose=False)[0]
            if not len(res.boxes): continue
            confs = res.boxes.conf.cpu().numpy()
            bi = int(np.argmax(confs))
            conf = float(confs[bi])
            if conf < args.min_conf: continue
            xyxy = res.boxes.xyxy[bi].cpu().numpy().tolist()
            try:
                preds = fa.get_landmarks(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),
                                         detected_faces=[[*xyxy,conf]])
            except: continue
            if not preds: continue
            lm68 = preds[0]
            xs,ys = lm68[:,0],lm68[:,1]
            if ((xs>=0)&(xs<frame_w)&(ys>=0)&(ys<frame_h)).mean()<0.8: continue
            results.append({"idx":idx,"conf":conf,"bbox":xyxy,
                            "lm68":lm68.tolist(),"score":symmetry_score(lm68),
                            "occluded":is_occluded(img,lm68)})
        clear = [r for r in results if not r["occluded"]]
        pool  = clear if clear else results
        return min(pool, key=lambda r: r["score"]) if pool else None

    # ── per-identity: frontal crop + FaceXFormer ──────────────────────────────

    for ident_info in manifest["identities"]:
        uid  = ident_info["id"]
        idir = os.path.join(args.workdir, f"identity_{uid}")
        os.makedirs(idir, exist_ok=True)

        all_frames = sorted(set(
            f for sc in ident_info["scenes"]
            for f in range(sc["start_frame"], sc["end_frame"]+1, max(1,step))
        ))

        if len(all_frames) > args.max_candidates:
            idx_s = np.linspace(0, len(all_frames)-1, args.max_candidates).astype(int)
            cands = [all_frames[i] for i in idx_s]
        else:
            cands = all_frames

        best = best_frontal(cands)
        if best is None and len(all_frames) > args.max_candidates:
            best = best_frontal(all_frames)
        if best is None:
            log(f"id{uid}", "WARNING: no frontal frame, skipping")
            ident_info["status"] = "no_frontal_frame"; continue

        log(f"id{uid}", f"frame {best['idx']}  sym={best['score']:.4f}  conf={best['conf']:.2f}")

        img = cv2.imread(frame_paths[best["idx"]])
        cv2.imwrite(os.path.join(idir, "best_frame.png"), img)

        x1,y1,x2,y2 = best["bbox"]
        m=args.crop_margin
        px1=max(0,int(x1-(x2-x1)*m)); py1=max(0,int(y1-(y2-y1)*m))
        px2=min(frame_w,int(x2+(x2-x1)*m)); py2=min(frame_h,int(y2+(y2-y1)*m))
        face_crop_path = os.path.join(idir, "face_crop.png")
        cv2.imwrite(face_crop_path, img[py1:py2, px1:px2])

        with open(os.path.join(idir,"landmarks.json"),"w") as f:
            json.dump({"frame_idx":best["idx"],"bbox":best["bbox"],
                       "symmetry_score":best["score"],"yolo_conf":best["conf"]},
                      f, indent=2, cls=NumpyEncoder)

        log(f"id{uid}", "FaceXFormer...")
        attrs_path = os.path.join(idir, "attributes.json")
        proc = run(["conda","run","--no-capture-output","-n",args.facex_env,
                    "python", args.facex_script,
                    "--face-crop", face_crop_path,
                    "--output",    attrs_path,
                    "--repo",      args.facexformer_repo,
                    "--ckpt",      args.facexformer_ckpt])
        if proc.returncode != 0:
            log(f"id{uid}", "WARNING: FaceXFormer failed, skipping")
            ident_info["status"] = "facexformer_failed"; continue

        with open(attrs_path) as f:
            attrs = json.load(f)
        ident_info["attributes"] = attrs
        log(f"id{uid}", f"age={attrs['age_label']}  gender={attrs['gender']}  race={attrs['race']}")

    with open(os.path.join(args.workdir, "scene_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, cls=NumpyEncoder)

    write_result("cluster_done", {"manifest": manifest})
    n = len([i for i in manifest["identities"]
             if i.get("status") not in ("no_frontal_frame","facexformer_failed")])
    log("done", f"cluster stage complete — {n} identities ready for FLUX")
    sys.exit(0)

# ═════════════════════════════════════════════════════════════════════════════
#  REASSEMBLE STAGE
# ═════════════════════════════════════════════════════════════════════════════

if args.stage == "reassemble":

    manifest_path = os.path.join(args.workdir, "scene_manifest.json")
    if not os.path.exists(manifest_path):
        die("reassemble", "scene_manifest.json not found — run cluster stage first")

    with open(manifest_path) as f:
        manifest = json.load(f)

    fps            = manifest["fps"]
    video_duration = manifest["num_frames"] / fps

    # collect timeline segments from per-identity scene dirs
    timeline = []
    for ident_info in manifest["identities"]:
        uid  = ident_info["id"]
        idir = os.path.join(args.workdir, f"identity_{uid}")
        for si, scene in enumerate(ident_info["scenes"]):
            anon_clip = os.path.join(idir, "scenes", f"scene_{si:03d}_anon.mp4")
            orig_clip = os.path.join(idir, "scenes", f"scene_{si:03d}_orig.mp4")
            if os.path.exists(anon_clip):
                timeline.append({"start_time": scene["start_time"],
                                  "end_time":   scene["end_time"],
                                  "clip":        anon_clip})
            elif os.path.exists(orig_clip):
                log("reassemble", f"WARNING: using original clip for id{uid} scene{si}")
                timeline.append({"start_time": scene["start_time"],
                                  "end_time":   scene["end_time"],
                                  "clip":        orig_clip})

    if not timeline:
        log("reassemble", "no clips found — copying original video")
        run(["cp", args.video, output_path], check=True)
        write_result("no_clips")
        sys.exit(0)

    timeline.sort(key=lambda s: s["start_time"])

    def extract_clip(video_in, t0, t1, out):
        run(["ffmpeg","-y","-i",video_in,"-ss",str(t0),"-to",str(t1),
             "-c:v","libx264","-crf","18","-preset","fast",out],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    full_timeline = []
    cursor = 0.0
    for seg in timeline:
        if seg["start_time"] > cursor + 0.1:
            gap = os.path.join(args.workdir, f"gap_{cursor:.3f}.mp4")
            extract_clip(args.video, cursor, seg["start_time"], gap)
            full_timeline.append(gap)
        full_timeline.append(seg["clip"])
        cursor = seg["end_time"]

    if video_duration - cursor > 0.1:
        tail = os.path.join(args.workdir, "gap_tail.mp4")
        extract_clip(args.video, cursor, video_duration, tail)
        full_timeline.append(tail)

    concat_path = os.path.join(args.workdir, "concat_list.txt")
    with open(concat_path, "w") as f:
        for clip in full_timeline:
            f.write(f"file '{os.path.abspath(clip)}'\n")

    log("reassemble", f"concatenating {len(full_timeline)} clips → {output_path}")
    run(["ffmpeg","-y","-f","concat","-safe","0","-i",concat_path,
         "-c","copy", output_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    write_result("success")
    log("done", f"→ {output_path}")