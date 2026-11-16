# AnonNET2 & VEIL-Bench

Official code for **AnonNET2: Robust Video Face Anonymization with Standardized Privacy Evaluation**.

* **AnonNET2** replaces every face in a video with a synthetic identity while keeping pose, expression and background.
  Pipeline: YOLO11-face + ArcFace clustering → frontal-frame selection → FaceXFormer attributes → FLUX.1-dev identity
  generation with an ArcFace privacy gate → DreamID-V video face swap → re-assembly.
* **VEIL-Bench** is the evaluation benchmark: 756 81-frame clips (600 VEIL-ID from VoxCeleb2, 156 VEIL-HQ from CelebV-HQ),
  re-identification attackers (EER) and quality / consistency metrics (pose, expression, FVD, VBench, DINO, Q-Align,
  face-detection rate).

```
run_all.sh          one-command entry point (setup → benchmark → anonymize → evaluate)
config.example.sh   all paths in one place
anonnet2/           the anonymization pipeline
veilbench/
  manifests/        clips_756.txt (the evaluated clip IDs) and windows_756.json (start frame of each 81-frame window)
  build/            cut the clips + annotate them (annotate/); selection/ = how the clips were originally sampled
  eval/             embeddings, attackers, metrics
baselines/          code used to run AnonNET (modified) and the blur / pixelation / DeepPrivacy2 baselines
environments/       conda specs (create_envs.sh builds them all)
scripts/            helpers used by run_all.sh
```

## Quick start

Needs a Linux machine with an NVIDIA GPU (≥ 40 GB for DreamID-V), conda, ffmpeg and git.

```bash
git clone <this repo> && cd <this repo>
cp config.example.sh config.sh && nano config.sh      # 1. set the paths (see "Raw data" and "Manual prerequisites")
./run_all.sh                                          # 2. run everything
```

`./run_all.sh` runs four stages in order and skips every step that is already finished, so it is safe to re-run after an
interruption (`FORCE=1` redoes everything, `DRY_RUN=1` only prints the commands):

| stage | what it does | env |
|---|---|---|
| `setup` | creates the conda environments, installs FaceXFormer, downloads Wan2.1 / FLUX / DINO / Q-Align | – |
| `benchmark` | cuts the 756 clips from the raw corpora, annotates them (face, pose, demographics), writes `veilbench_manifest_v2.json` | `bench`, `a2` |
| `anonymize` | AnonNET2 on all clips: clustering + FLUX identities, then DreamID-V swap (pad100) | `a2`, `dreamidv` |
| `evaluate` | embeddings, attackers/EER, pose + expression, FVD, VBench, detection rate, DINO, Q-Align | several |

Run a single stage with `./run_all.sh anonymize` (or several: `./run_all.sh anonymize evaluate`).
On an OAR cluster: `oarsub -l gpu=1,walltime=48:00:00 "./run_all.sh"`.
Outputs go to `$VEIL_ROOT`: anonymized clips in `anonymized/`, metrics in `results/`.

**To evaluate another method side by side** (e.g. AnonNET, a blur baseline), put its output clips in a folder with the same
file names as `clips_756/` and set `BASELINE_DIR=/that/folder` in `config.sh`; the evaluate stage then scores both.

### Raw data (not redistributed)

Place the original datasets under `$SHARE_ROOT`, laid out as `window_extract.py` expects:

```
$SHARE_ROOT/vox-celeb-dataset/vox-celeb-raw/mp4/<idXXXXX>/<video id>/<clip>.mp4      VoxCeleb2 (test)
$SHARE_ROOT/CelebV-HQ/path_to_videos_256_for_inf/{train,val,test}/<name>.mp4          CelebV-HQ
```

### Manual prerequisites (cannot be downloaded automatically)

* **FLUX.1-dev**: accept the licence on Hugging Face and run `huggingface-cli login` before `setup`.
* **DreamID-V**: the official inference code at `$DREAMIDV_REPO` and the checkpoint `dreamidv_faster.pth` at `$DREAMIDV_CKPT`.
* **YOLO11 face detector** weights at `$YOLO_MODEL`, and the **SixDRepNet** checkpoint `6DRepNet_300W_LP_AFLW2000.pth` at `$SIXDREPNET_CKPT`.

Exact versions and licences of all third-party components are listed in `THIRD_PARTY.md`.

## Running the pieces by hand

Every step of `run_all.sh` is a plain command; read the script for the exact arguments (`DRY_RUN=1 ./run_all.sh` prints them
all with your paths filled in). Each script also has `--help`.

* Anonymize one folder of videos: `python anonnet2/run_batch.py --help` (phases: cluster, FLUX identities, swap), then
  `python anonnet2/run_phase3_pad100.py --help` (padded swap used in the paper).
* Attackers: `veilbench/eval/run_eval_v2.py`. Attackers T1 (mean similarity), T2 (worst frame), T3 (clip mean),
  T4a (cross-session gallery; "T3a" in the paper), T6 (unlinkability). EER is threshold-free; 0.5 = chance.
* Metrics: `metrics_pose_expr_v2.py`, `metrics_fvd_v2.py`, `metrics_vbench_v2.py`, `metrics_detection_rate_v2.py`,
  `eval_videos.py --metric dino|qalign`.

## Rebuilding VEIL-Bench from scratch (optional)

`run_all.sh benchmark` rebuilds the *released* benchmark from the clip list and window offsets. To reproduce how the
candidates were sampled (inventory → annotation → stratified sampling → window selection), see `veilbench/build/selection/`:
`inventory/inventory.py`, `face_detect.py`, `pose_detect.py`, `demo_annotate.py`, `snr_annotate.py`, `pool_filter.py`,
`sample.py` (quotas in `quotas.yaml`, seed 42), `window_extract.py` (81-frame windows, `--policy max_yaw`).

## Notes

* Clip lengths: the benchmark clips have 81 frames; the DreamID-V outputs have 69 frames.
* The privacy gate `--arcface-threshold` (default 0.3) accepts a generated face when its ArcFace similarity to the source is
  below the threshold; `--cluster-threshold` (0.45, in `anonymize.py`) only controls identity clustering.
* In result files, the JSON key `dreamidv` denotes AnonNET2 and `anonnet` denotes the AnonNET baseline.
* The scripts were run on Grid5000 (OAR); `run_all.sh` has been checked in dry-run mode, and individual stages can be run separately.

## License

MIT for the code in this repository (see `LICENSE`). Third-party models and datasets keep their own licences (`THIRD_PARTY.md`);
FLUX.1-dev is non-commercial.
