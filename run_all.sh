#!/usr/bin/env bash
# run_all.sh — run the whole pipeline in one go.
#
#   ./run_all.sh                 all stages, in order: setup benchmark anonymize evaluate
#   ./run_all.sh benchmark       only one (or several) stage(s):  ./run_all.sh anonymize evaluate
#   DRY_RUN=1 ./run_all.sh       print every command without running anything
#   FORCE=1 ./run_all.sh ...     ignore the "done" markers and redo the steps
#
# Finished steps are recorded in $VEIL_ROOT/.done/, so a re-run (or a re-submission after a job was killed)
# continues where it stopped. Needs a GPU node for `anonymize` and `evaluate`.
# On OAR:  oarsub -l gpu=1,walltime=48:00:00 -S "./run_all.sh"   (or `oarsub ... "./run_all.sh anonymize"`)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
[ -f config.sh ] || { echo "config.sh not found: run  cp config.example.sh config.sh  and edit it."; exit 1; }
source ./config.sh
: "${VEIL_ROOT:?}" "${SHARE_ROOT:?}" "${WORK:?}" "${ANON_DIR:?}"
DRY=${DRY_RUN:-0}
DONE="$VEIL_ROOT/.done"
[ "$DRY" = 1 ] || mkdir -p "$DONE"

CLIP_LIST=$ANONNET_REPO/veilbench/manifests/clips_756.txt
WINDOWS=$ANONNET_REPO/veilbench/manifests/windows_756.json
MANIFEST=$VEIL_ROOT/manifests/veilbench_manifest_v2.json
CLIPS=$VEIL_ROOT/clips_756
export ACTIVE_LIST=$CLIP_LIST           # the annotation scripts read their clip list from here

run()   { if [ "$DRY" = 1 ]; then echo "  + $*"; else "$@"; fi; }
inenv() { local e=$1; shift; run "$ANONNET_REPO/scripts/with_env.sh" "$e" "$@"; }
step()  {  # step <name> <function>
  if [ -z "${FORCE:-}" ] && [ -f "$DONE/$1" ]; then echo "[skip] $1"; return; fi
  echo "[run ] $1"; "$2"; [ "$DRY" = 1 ] || touch "$DONE/$1"
}
BASE=(); [ -n "${BASELINE_DIR:-}" ] && BASE=(--anon-an "$BASELINE_DIR")
base_args() { [ -n "${BASELINE_DIR:-}" ] && echo "--anon-an $BASELINE_DIR" || true; }

# ───────────────────────────── stage 1: setup ─────────────────────────────
s_envs()    { run bash environments/create_envs.sh; }
s_facex()   { inenv a2 bash anonnet2/setup_facexformer.sh "$WORK"; }
s_models()  { run bash scripts/download_models.sh; }

# ───────────────────────────── stage 2: benchmark ─────────────────────────
b_cut()     { inenv bench python veilbench/build/cut_windows.py --clips "$CLIP_LIST" --windows "$WINDOWS"; }
b_link()    { run bash veilbench/build/link_eval_clips.sh "$CLIP_LIST"; }
b_face()    { inenv bench python veilbench/build/annotate/face_detect_v2.py; }
b_pose()    { inenv bench python veilbench/build/annotate/pose_detect_v2.py; }
b_demo()    { inenv a2    python veilbench/build/annotate/demo_annotate_v2.py; }
b_manifest(){ inenv bench python veilbench/build/annotate/compile_manifest_v2.py; }

# ───────────────────────────── stage 3: anonymize ─────────────────────────
a_phase12() {
  inenv a2 python anonnet2/run_batch.py --input-dir "$CLIPS" --output-dir "$VEIL_ROOT/anonymized/dreamidv" \
    --yolo-model "$YOLO_MODEL" --facexformer-repo "$FACEXFORMER_REPO" --facexformer-ckpt "$FACEXFORMER_CKPT" \
    --dreamidv-repo "$DREAMIDV_REPO" --wan-dir "$WAN_DIR" --dreamidv-ckpt "$DREAMIDV_CKPT" --phase2-only --resume; }
a_phase3() {
  inenv a2 python anonnet2/run_phase3_pad100.py --dreamidv-dir "$VEIL_ROOT/anonymized/dreamidv" \
    --main1-dir "$ANONNET_REPO/anonnet2" --dreamidv-repo "$DREAMIDV_REPO" --wan-dir "$WAN_DIR" \
    --dreamidv-ckpt "$DREAMIDV_CKPT" --dreamidv-env dreamidv --batch-size 50; }

# ───────────────────────────── stage 4: evaluate ──────────────────────────
E=veilbench/eval
R=$VEIL_ROOT/results
e_embed_a() { inenv embed python $E/embed_extract_v2.py --manifest "$MANIFEST" --clips-dir "$CLIPS" --anon-dv "$ANON_DIR" --out-dir "$VEIL_ROOT/embeddings/anonnet2" --subset all; }
e_attack_a(){ inenv bench python $E/run_eval_v2.py --manifest "$MANIFEST" --emb-orig "$VEIL_ROOT/embeddings/anonnet2/arcface_original.json" --emb-anon "$VEIL_ROOT/embeddings/anonnet2/arcface_dreamidv.json" --out-dir "$R/anonnet2"; }
e_embed_b() { inenv embed python $E/embed_extract_v2.py --manifest "$MANIFEST" --clips-dir "$CLIPS" --anon-dv "$BASELINE_DIR" --out-dir "$VEIL_ROOT/embeddings/baseline" --subset all; }
e_attack_b(){ inenv bench python $E/run_eval_v2.py --manifest "$MANIFEST" --emb-orig "$VEIL_ROOT/embeddings/baseline/arcface_original.json" --emb-anon "$VEIL_ROOT/embeddings/baseline/arcface_dreamidv.json" --out-dir "$R/baseline"; }
e_pose()    { inenv bench python $E/metrics_pose_expr_v2.py --manifest "$MANIFEST" --clips-dir "$CLIPS" --anon-dv "$ANON_DIR" ${BASE[@]+"${BASE[@]}"} --sixd-weights "$SIXDREPNET_CKPT" --out "$R/pose_expr_results.json"; }
e_fvd()     { inenv fvd python $E/metrics_fvd_v2.py --manifest "$MANIFEST" --clips-dir "$CLIPS" --anon-dv "$ANON_DIR" ${BASE[@]+"${BASE[@]}"} --out "$R/fvd_results.json" --save-features "$R/features/"; }
e_vbench()  { inenv vbench python $E/metrics_vbench_v2.py --manifest "$MANIFEST" --anon-dv "$ANON_DIR" ${BASE[@]+"${BASE[@]}"} --out "$R/vbench_results.json"; }
e_detect()  { inenv embed python $E/metrics_detection_rate_v2.py --manifest "$MANIFEST" --anon-dv "$ANON_DIR" ${BASE[@]+"${BASE[@]}"} --out "$R/detection_rate.json"; }
e_dino_a()  { inenv env_dino python $E/eval_videos.py --metric dino --orig "$CLIPS" --anon "$ANON_DIR" --sample_fps 2 --dataset AnonNET2 --out "$R/dino_anonnet2.json"; }
e_qalign_a(){ inenv env_qalign python $E/eval_videos.py --metric qalign --orig "$CLIPS" --anon "$ANON_DIR" --dataset AnonNET2 --out "$R/qalign_anonnet2.json"; }
e_dino_b()  { inenv env_dino python $E/eval_videos.py --metric dino --orig "$CLIPS" --anon "$BASELINE_DIR" --sample_fps 2 --dataset baseline --out "$R/dino_baseline.json"; }
e_qalign_b(){ inenv env_qalign python $E/eval_videos.py --metric qalign --orig "$CLIPS" --anon "$BASELINE_DIR" --dataset baseline --out "$R/qalign_baseline.json"; }

stage_setup()     { step envs s_envs; step facexformer s_facex; step models s_models; }
stage_benchmark() { step cut_windows b_cut; step link_clips b_link; step annotate_face b_face; step annotate_pose b_pose
                    step annotate_demographics b_demo; step compile_manifest b_manifest; }
stage_anonymize() { step anonymize_phase12 a_phase12; step anonymize_phase3 a_phase3; }
stage_evaluate()  {
  step embed_anonnet2 e_embed_a; step attackers_anonnet2 e_attack_a
  if [ -n "${BASELINE_DIR:-}" ]; then step embed_baseline e_embed_b; step attackers_baseline e_attack_b; fi
  step pose_expression e_pose; step fvd e_fvd; step vbench e_vbench; step detection_rate e_detect
  step dino_anonnet2 e_dino_a; step qalign_anonnet2 e_qalign_a
  if [ -n "${BASELINE_DIR:-}" ]; then step dino_baseline e_dino_b; step qalign_baseline e_qalign_b; fi
  echo "Results are in $R"
}

STAGES=("$@"); [ ${#STAGES[@]} -eq 0 ] && STAGES=(setup benchmark anonymize evaluate)
for s in "${STAGES[@]}"; do
  case $s in
    setup|benchmark|anonymize|evaluate) echo "=== stage: $s"; "stage_$s" ;;
    *) echo "unknown stage '$s' (setup | benchmark | anonymize | evaluate)"; exit 2 ;;
  esac
done
echo "done."
