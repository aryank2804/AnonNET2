#!/usr/bin/env python3
"""
run_eval_v2.py — Attacker evaluation + results compilation for VEIL-Bench v2.0

Env: bench   (numpy, scikit-learn — no heavy deps)

Runs T1/T2/T3/T4a/T6 on pre-computed ArcFace embeddings, then compiles
everything (including separately-computed quality_metrics.json and
fvd_results.json if present) into the paper's headline CSVs.

Attackers
─────────
  T1   mean cosine similarity across per-frame pairs  [primary]
  T2   max  cosine similarity across per-frame pairs  [worst-frame]
  T3   cosine similarity of mean-pooled embeddings    [appendix only]
  T4a  cross-session gallery                          [primary]
  T6   unlinkability: anon-vs-anon                    [primary]

Usage:
    python run_eval_v2.py \
        --manifest      $VEIL_ROOT/manifests/veilbench_manifest_v2.json \
        --emb-orig      $VEIL_ROOT/embeddings/arcface_original.json \
        --emb-anon      $VEIL_ROOT/embeddings/arcface_dreamidv.json \
        --out-dir       $VEIL_ROOT/results/ \
        [--quality-json $VEIL_ROOT/results/quality_metrics.json] \
        [--fvd-json     $VEIL_ROOT/results/fvd_results.json]

Outputs (all in --out-dir):
    attacker_results.json    raw per-attacker EER / minDCF
    final_table.csv          headline table (metric × stratum × model)
    attacker_table.csv       EER / minDCF stratified by yaw_bin and face_size_bin
    clip_level.csv           per-clip scores for supplementary
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import roc_curve  # type: ignore

logging.basicConfig(format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)
log = logging.getLogger(__name__)

P_TARGET = 0.01   # NIST speaker-verification DCF parameters
C_MISS   = 1.0
C_FA     = 1.0

YAW_BINS = ("frontal", "profile", "extreme")


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float(fpr[idx])


def compute_min_dcf(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr  = 1.0 - tpr
    dcf  = C_MISS * P_TARGET * fnr + C_FA * (1.0 - P_TARGET) * fpr
    norm = min(C_MISS * P_TARGET, C_FA * (1.0 - P_TARGET))
    return float(dcf.min() / norm)


def bundle(labels: np.ndarray, scores: np.ndarray) -> dict:
    return {"eer": compute_eer(labels, scores),
            "min_dcf": compute_min_dcf(labels, scores),
            "n_trials": int(len(scores))}


# ══════════════════════════════════════════════════════════════════════════════
#  IDENTITY / SESSION HELPERS  (VEIL-ID clip_id format: id03347__sessID__00306)
# ══════════════════════════════════════════════════════════════════════════════

def identity(clip_id: str) -> str:
    return clip_id.split("__")[0]

def session(clip_id: str) -> str:
    return clip_id.split("__")[1]


# ══════════════════════════════════════════════════════════════════════════════
#  EMBEDDING MATRIX BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def mean_matrix(emb_data: dict, clip_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        means : (N, 512) — per-clip mean embedding (zero row for missing clips)
        valid : (N,) bool
    """
    M = np.zeros((len(clip_ids), 512), dtype=np.float32)
    V = np.zeros(len(clip_ids), dtype=bool)
    for i, cid in enumerate(clip_ids):
        embs = emb_data.get(cid, {}).get("embeddings", [])
        if embs:
            M[i] = np.mean(embs, axis=0)
            V[i] = True
    return M, V


def frame_blocks(emb_data: dict, clip_ids: list[str]) -> tuple[np.ndarray, list[tuple[int,int]]]:
    """
    Stack all per-frame embeddings and return (all_frames, clip_ranges).
    clip_ranges[i] = (start, end) into all_frames for clip clip_ids[i].
    """
    arrs, ranges, cursor = [], [], 0
    for cid in clip_ids:
        embs = emb_data.get(cid, {}).get("embeddings", [])
        arr  = np.array(embs, dtype=np.float32) if embs else np.empty((0, 512), dtype=np.float32)
        ranges.append((cursor, cursor + len(arr)))
        cursor += len(arr)
        if len(arr):
            arrs.append(arr)
    all_frames = np.vstack(arrs) if arrs else np.empty((0, 512), dtype=np.float32)
    return all_frames, ranges


# ══════════════════════════════════════════════════════════════════════════════
#  SCORE MATRICES
# ══════════════════════════════════════════════════════════════════════════════

def score_t1(probe_means: np.ndarray, gallery_means: np.ndarray) -> np.ndarray:
    """
    Mean pairwise cosine similarity.
    For L2-normalised embeddings: mean_{i,j}[dot(p_i, g_j)] = dot(mean(P), mean(G))
    so T1 reduces to a single matmul.
    """
    return probe_means @ gallery_means.T   # (Np, Ng)


def score_t2(pf: np.ndarray, pr: list[tuple[int,int]],
             gf: np.ndarray, gr: list[tuple[int,int]]) -> np.ndarray:
    """Max pairwise cosine similarity (worst-frame attacker)."""
    Np, Ng = len(pr), len(gr)
    out = np.full((Np, Ng), -np.inf, dtype=np.float32)
    if pf.shape[0] == 0 or gf.shape[0] == 0:
        return out
    sims = pf @ gf.T   # all frame-level similarities in one shot
    for i, (ps, pe) in enumerate(pr):
        if ps == pe:
            continue
        for j, (gs, ge) in enumerate(gr):
            if gs == ge:
                continue
            out[i, j] = sims[ps:pe, gs:ge].max()
    return out


def score_t3(probe_means: np.ndarray, gallery_means: np.ndarray) -> np.ndarray:
    """Cosine similarity of normalised mean embeddings (T3 ≠ T1)."""
    def norm(m):
        n = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.where(n > 1e-8, n, 1.0)
    return norm(probe_means) @ norm(gallery_means).T


# ══════════════════════════════════════════════════════════════════════════════
#  LABEL MATRIX  (anon-vs-orig or anon-vs-anon)
# ══════════════════════════════════════════════════════════════════════════════

def label_matrix(probe_ids: list[str], gallery_ids: list[str],
                 exclude_diagonal: bool = False) -> np.ndarray:
    P = np.array([identity(c) for c in probe_ids])
    G = np.array([identity(c) for c in gallery_ids])
    L = (P[:, None] == G[None, :]).astype(np.int32)
    if exclude_diagonal and len(probe_ids) == len(gallery_ids):
        np.fill_diagonal(L, 0)
    return L


# ══════════════════════════════════════════════════════════════════════════════
#  STRATIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def stratify(scores: np.ndarray, labels: np.ndarray,
             probe_ids: list[str], gallery_valid: np.ndarray,
             meta: dict) -> dict:
    """
    Flatten the (Np × Ng) matrices, drop non-finite entries,
    then compute EER/minDCF overall and by probe yaw_bin / face_size_bin.
    """
    sc, lb, yaw_s, sz_s = [], [], [], []

    for i, pid in enumerate(probe_ids):
        pm       = meta.get(pid, {})
        yaw_bin  = pm.get("yaw_bin", "unknown")
        size_bin = pm.get("face_size_bin", "unknown")
        for j in range(scores.shape[1]):
            if not gallery_valid[j]:
                continue
            v = float(scores[i, j])
            if not np.isfinite(v):
                continue
            sc.append(v);          lb.append(int(labels[i, j]))
            yaw_s.append(yaw_bin); sz_s.append(size_bin)

    sc = np.array(sc, dtype=np.float32)
    lb = np.array(lb, dtype=np.int32)
    ya = np.array(yaw_s)
    sz = np.array(sz_s)

    result: dict = {"overall": bundle(lb, sc)}

    result["by_yaw_bin"] = {
        y: bundle(lb[ya==y], sc[ya==y])
        for y in YAW_BINS if (ya==y).sum() >= 20
    }
    result["by_face_size_bin"] = {
        s: bundle(lb[sz==s], sc[sz==s])
        for s in np.unique(sz) if (sz==s).sum() >= 20
    }
    return result


def clip_rank1(scores: np.ndarray, labels: np.ndarray,
               probe_ids: list[str]) -> dict:
    """Per-probe clip: score and label of the nearest gallery match."""
    out = {}
    for i, pid in enumerate(probe_ids):
        row = scores[i]
        if not np.any(np.isfinite(row)):
            out[pid] = {"score": None, "label": None}
        else:
            j = int(np.nanargmax(row))
            out[pid] = {"score": float(row[j]), "label": int(labels[i, j])}
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  T4a  — cross-session gallery
# ══════════════════════════════════════════════════════════════════════════════

def run_t4a(clip_ids: list[str], orig_data: dict, anon_data: dict,
            clips_meta: dict) -> dict:
    """
    Enroll: original clips from each identity's FIRST session (lex order).
    Probe:  anonymized clips from all remaining sessions.
    Scoring: T1 (mean cosine sim).
    """
    # Group by identity → session → clips
    by_id: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for cid in clip_ids:
        by_id[identity(cid)][session(cid)].append(cid)

    enroll_clips, probe_clips = [], []
    for ident, sessions in by_id.items():
        skeys = sorted(sessions)
        enroll_clips.extend(sessions[skeys[0]])
        for s in skeys[1:]:
            probe_clips.extend(sessions[s])

    if not enroll_clips or not probe_clips:
        log.warning("T4a: could not build enroll/probe split")
        return {}

    enroll_means, enroll_valid = mean_matrix(orig_data, enroll_clips)
    probe_means,  probe_valid  = mean_matrix(anon_data, probe_clips)

    scores = probe_means @ enroll_means.T   # T1
    scores[~probe_valid, :]  = np.nan
    scores[:, ~enroll_valid] = np.nan

    labels = label_matrix(probe_clips, enroll_clips)

    log.info(f"T4a: {len(enroll_clips)} enroll, {len(probe_clips)} probe clips")

    result = stratify(scores, labels, probe_clips, enroll_valid, clips_meta)
    result["clip_scores"] = clip_rank1(scores, labels, probe_clips)
    result["description"] = ("Cross-session gallery: enroll on 1st session (original), "
                             "probe on remainder (DreamID-V)")
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ATTACKER RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_attackers(manifest: dict, orig_data: dict, anon_data: dict) -> dict:
    clips_meta = manifest["clips"]
    veil_id = sorted(c for c, m in clips_meta.items() if m["subset"] == "veil_id")
    log.info(f"VEIL-ID clips: {len(veil_id)}")

    orig_means, orig_valid = mean_matrix(orig_data, veil_id)
    anon_means, anon_valid = mean_matrix(anon_data, veil_id)
    orig_frames, orig_ranges = frame_blocks(orig_data, veil_id)
    anon_frames, anon_ranges = frame_blocks(anon_data, veil_id)

    L = label_matrix(veil_id, veil_id)    # anon probe × orig gallery

    def run(tag: str, scores: np.ndarray, labels: np.ndarray,
            pv: np.ndarray, gv: np.ndarray, desc: str) -> dict:
        scores[~pv, :] = np.nan
        scores[:, ~gv] = np.nan
        result = stratify(scores, labels, veil_id, gv, clips_meta)
        result["clip_scores"] = clip_rank1(scores, labels, veil_id)
        result["description"] = desc
        log.info(f"{tag} overall EER = {result['overall']['eer']:.4f}")
        return result

    results: dict = {}

    log.info("T1 …")
    results["T1"] = run(
        "T1", score_t1(anon_means, orig_means).copy(), L,
        anon_valid, orig_valid,
        "Per-frame ArcFace, mean cosine sim; probe=DreamID-V, gallery=original")

    log.info("T2 …")
    results["T2"] = run(
        "T2", score_t2(anon_frames, anon_ranges, orig_frames, orig_ranges), L,
        anon_valid, orig_valid,
        "Per-frame ArcFace, max cosine sim (worst-frame); probe=DreamID-V, gallery=original")

    log.info("T3 (appendix) …")
    results["T3"] = run(
        "T3", score_t3(anon_means, orig_means), L,
        anon_valid, orig_valid,
        "Clip-mean ArcFace cosine sim (DEMOTED — appendix only)")

    log.info("T4a …")
    results["T4a"] = run_t4a(veil_id, orig_data, anon_data, clips_meta)

    log.info("T6 (unlinkability) …")
    L6 = label_matrix(veil_id, veil_id, exclude_diagonal=True)
    t6 = score_t1(anon_means, anon_means).copy()
    np.fill_diagonal(t6, np.nan)
    results["T6"] = run(
        "T6", t6, L6, anon_valid, anon_valid,
        "Unlinkability: anon-vs-anon. EER near 50% = strong unlinkability.")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  CSV COMPILATION
# ══════════════════════════════════════════════════════════════════════════════

def fmt(v, d=4) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "NA"
    return f"{v:.{d}f}" if isinstance(v, float) else str(v)


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    headers = list(rows[0].keys())
    with open(path, "w") as fh:
        fh.write(",".join(headers) + "\n")
        for r in rows:
            fh.write(",".join(str(r.get(h, "NA")) for h in headers) + "\n")
    log.info(f"Wrote {len(rows)} rows → {path}")


# ── attacker_table.csv ────────────────────────────────────────────────────────

def make_attacker_table(attacker_results: dict) -> list[dict]:
    rows = []
    for att_name, att in attacker_results.items():
        strata = [("overall", att.get("overall"))]
        for yaw in YAW_BINS:
            strata.append((f"yaw:{yaw}", att.get("by_yaw_bin", {}).get(yaw)))
        for sz, v in att.get("by_face_size_bin", {}).items():
            strata.append((f"size:{sz}", v))
        for stratum_name, s in strata:
            if not s:
                continue
            rows.append({
                "attacker":  att_name,
                "stratum":   stratum_name,
                "eer":       fmt(s.get("eer")),
                "min_dcf":   fmt(s.get("min_dcf")),
                "n_trials":  s.get("n_trials", "NA"),
                "desc":      att.get("description", ""),
            })
    return rows


# ── final_table.csv (quality + fvd, stratified by yaw_bin) ───────────────────

def agg_quality(quality_data: dict, clip_ids: list[str],
                metric: str, model_key: str) -> Optional[float]:
    vals = [
        float((quality_data.get(cid) or {}).get(model_key, {}).get(metric))
        for cid in clip_ids
        if (quality_data.get(cid) or {}).get(model_key, {}).get(metric) is not None
    ]
    return float(np.mean(vals)) if vals else None


def make_headline_table(manifest: dict,
                        quality_data: Optional[dict],
                        fvd_data: Optional[dict]) -> list[dict]:
    clips_meta = manifest["clips"]
    all_ids    = list(clips_meta)

    # Clip groups by yaw_bin
    by_yaw: dict[str, list[str]] = {y: [] for y in YAW_BINS}
    for cid, m in clips_meta.items():
        yb = m.get("yaw_bin")
        if yb in by_yaw:
            by_yaw[yb].append(cid)

    strata = [("all", all_ids)] + [(y, by_yaw[y]) for y in YAW_BINS]
    rows: list[dict] = []

    # Quality metrics (id_pres, qual, aes) from separately-computed JSON
    if quality_data:
        for metric in ("id_pres", "qual", "aes"):
            for stratum, ids in strata:
                if not ids:
                    continue
                orig = agg_quality(quality_data, ids, metric, "original")
                dv   = agg_quality(quality_data, ids, metric, "dreamidv")
                an   = agg_quality(quality_data, ids, metric, "anonnet")
                rows.append({
                    "metric":   metric,
                    "stratum":  stratum,
                    "n_clips":  len(ids),
                    "original": fmt(orig),
                    "dreamidv": fmt(dv),
                    "anonnet":  fmt(an),
                    "delta_dv": fmt((float(dv) - float(orig))
                                   if dv is not None and orig is not None else None),
                    "delta_an": fmt((float(an) - float(orig))
                                   if an is not None and orig is not None else None),
                })

    # FVD (population-level, already stratified)
    if fvd_data:
        fvd_strata = [
            ("all",     fvd_data.get("dreamidv", {}).get("overall"),
                        fvd_data.get("anonnet",  {}).get("overall")),
        ] + [
            (y,         fvd_data.get("dreamidv", {}).get("by_yaw_bin", {}).get(y),
                        fvd_data.get("anonnet",  {}).get("by_yaw_bin", {}).get(y))
            for y in YAW_BINS
        ]
        for stratum, dv_entry, an_entry in fvd_strata:
            dv_fvd = (dv_entry or {}).get("fvd")
            an_fvd = (an_entry or {}).get("fvd")
            n      = (dv_entry or {}).get("n_clips", "NA")
            rows.append({
                "metric":   "fvd",
                "stratum":  stratum,
                "n_clips":  n,
                "original": "0 (reference)",
                "dreamidv": fmt(dv_fvd, 1),
                "anonnet":  fmt(an_fvd, 1),
                "delta_dv": fmt(dv_fvd, 1),
                "delta_an": fmt(an_fvd, 1),
            })

    return rows


# ── clip_level.csv ────────────────────────────────────────────────────────────

def make_clip_level(manifest: dict, quality_data: Optional[dict],
                    attacker_results: dict) -> list[dict]:
    clips_meta = manifest["clips"]
    t1_scores  = attacker_results.get("T1", {}).get("clip_scores", {})
    rows = []
    for cid, m in clips_meta.items():
        row: dict = {
            "clip_id":       cid,
            "subset":        m.get("subset", ""),
            "yaw_bin":       m.get("yaw_bin", ""),
            "yaw":           fmt(m.get("yaw"), 1),
            "face_size_bin": m.get("face_size_bin", ""),
            "age_group":     m.get("demographics", {}).get("age_group", ""),
            "gender":        m.get("demographics", {}).get("gender", ""),
        }
        for model_key in ("original", "dreamidv", "anonnet"):
            entry = (quality_data.get(cid) if quality_data else None) or {}
            me    = (entry.get(model_key) or {}) if isinstance(entry.get(model_key), dict) else {}
            for metric in ("id_pres", "qual", "aes"):
                row[f"{model_key}_{metric}"] = fmt(me.get(metric))
        t1 = t1_scores.get(cid, {})
        row["t1_score"]        = fmt(t1.get("score"))
        row["t1_rank1_correct"] = str(t1.get("label", "NA"))
        rows.append(row)
    return rows


# ── pretty-print summary ──────────────────────────────────────────────────────

def print_attacker_summary(attacker_results: dict) -> None:
    print("\n" + "=" * 70)
    print("ATTACKER SUMMARY")
    print(f"{'Attacker':<8}  {'Stratum':<20}  {'EER':>7}  {'minDCF':>8}")
    print("-" * 70)
    for att, data in attacker_results.items():
        ov = data.get("overall", {})
        print(f"{att:<8}  {'overall':<20}  {fmt(ov.get('eer')):>7}  {fmt(ov.get('min_dcf')):>8}")
        for yaw in YAW_BINS:
            s = data.get("by_yaw_bin", {}).get(yaw)
            if s:
                print(f"{'':8}  {('yaw:'+yaw):<20}  {fmt(s.get('eer')):>7}  {fmt(s.get('min_dcf')):>8}")
    print("=" * 70 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest",      required=True)
    p.add_argument("--emb-orig",      required=True,
                   help="embeddings/arcface_original.json")
    p.add_argument("--emb-anon",      required=True,
                   help="embeddings/arcface_dreamidv.json")
    p.add_argument("--out-dir",       default="results")
    p.add_argument("--quality-json",  default=None,
                   help="quality_metrics.json produced separately (optional)")
    p.add_argument("--max-clips", type=int, default=None, help="Truncate to N clips (smoke-test)")
    p.add_argument("--fvd-json",      default=None,
                   help="fvd_results.json produced separately (optional)")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.manifest) as fh:
        manifest = json.load(fh)

    if args.max_clips:
        manifest["clips"] = dict(list(manifest["clips"].items())[:args.max_clips])
    with open(args.emb_orig) as fh:
        orig_data = json.load(fh)
    with open(args.emb_anon) as fh:
        anon_data = json.load(fh)

    quality_data: Optional[dict] = None
    if args.quality_json and Path(args.quality_json).exists():
        with open(args.quality_json) as fh:
            quality_data = json.load(fh)
        log.info(f"Loaded quality metrics from {args.quality_json}")
    else:
        log.info("No quality_metrics.json — quality columns will be NA")

    fvd_data: Optional[dict] = None
    if args.fvd_json and Path(args.fvd_json).exists():
        with open(args.fvd_json) as fh:
            fvd_data = json.load(fh)
        log.info(f"Loaded FVD results from {args.fvd_json}")
    else:
        log.info("No fvd_results.json — FVD rows will be omitted")

    # ── run attackers ─────────────────────────────────────────────────────────
    attacker_results = run_attackers(manifest, orig_data, anon_data)
    att_path = out / "attacker_results.json"
    att_path.write_text(json.dumps(attacker_results, indent=2))
    log.info(f"Attacker results saved → {att_path}")
    print_attacker_summary(attacker_results)

    # ── compile CSVs ─────────────────────────────────────────────────────────
    write_csv(make_attacker_table(attacker_results),
              out / "attacker_table.csv")

    write_csv(make_headline_table(manifest, quality_data, fvd_data),
              out / "final_table.csv")

    write_csv(make_clip_level(manifest, quality_data, attacker_results),
              out / "clip_level.csv")

    log.info("Done.")


if __name__ == "__main__":
    main()
