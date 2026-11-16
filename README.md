# AnonNET2 & VEIL-Bench

Code for **AnonNET2: Robust Video Face Anonymization with Standardized Privacy Evaluation**.

- **AnonNET2**: video face anonymization (YOLO11-face + ArcFace clustering, FaceXFormer attributes, FLUX.1-dev identity generation, DreamID-V face swap).
- **VEIL-Bench**: benchmark of 756 81-frame clips with re-identification attackers and quality metrics.

## Run

Requires Linux, an NVIDIA GPU, conda, ffmpeg and git.

```bash
cp config.example.sh config.sh     # set the paths
./run_all.sh                       # setup -> benchmark -> anonymize -> evaluate
```

Stages can be run separately: `./run_all.sh <setup|benchmark|anonymize|evaluate>`.
Finished steps are skipped on re-run. `DRY_RUN=1 ./run_all.sh` prints the commands without running them; `FORCE=1` redoes everything.

| stage | what it does |
|---|---|
| `setup` | creates the conda environments, installs FaceXFormer, downloads Wan2.1, FLUX, DINO and Q-Align |
| `benchmark` | cuts the clips from the raw corpora, annotates them, writes the benchmark manifest |
| `anonymize` | runs AnonNET2 on all clips |
| `evaluate` | embeddings, attackers (EER), pose, expression, FVD, VBench, detection rate, DINO, Q-Align |

Outputs are written to `$VEIL_ROOT` (`anonymized/`, `results/`).
To score another method alongside AnonNET2, set `BASELINE_DIR` in `config.sh` to a folder of its output clips (same file names as `clips_756/`).

### Prerequisites

- Raw data under `$SHARE_ROOT`:
  - `vox-celeb-dataset/vox-celeb-raw/mp4/<id>/<video>/<clip>.mp4` (VoxCeleb2 test)
  - `CelebV-HQ/path_to_videos_256_for_inf/{train,val,test}/*.mp4`
- Models to place manually (paths set in `config.sh`): DreamID-V code and checkpoint, YOLO11-face weights, SixDRepNet checkpoint.
- FLUX.1-dev: accept the licence on Hugging Face and run `huggingface-cli login`.

Third-party components and licences: see `THIRD_PARTY.md`.

## Repository layout

```
run_all.sh            entry point
config.example.sh     paths and model locations
anonnet2/             anonymization pipeline
  run_batch.py          clustering, attributes, FLUX identities
  run_phase3_pad100.py  DreamID-V face swap
  anonymize.py, run_facexformer.py, run_dreamidv_worker.py, setup_facexformer.sh
veilbench/
  manifests/            clips_756.txt, windows_756.json
  build/                cut_windows.py, annotate/ (face, pose, demographics, manifest),
                        selection/ (inventory, sampling, window extraction)
  eval/                 embed_extract_v2.py, run_eval_v2.py (attackers),
                        metrics_*_v2.py, eval_videos.py (DINO, Q-Align)
baselines/            AnonNET runner, blur / pixelation / DeepPrivacy2 runner
environments/         conda specs (create_envs.sh)
scripts/              helpers used by run_all.sh
```

## Individual scripts

Each script has `--help`; `DRY_RUN=1 ./run_all.sh` shows the exact call for every step.

## License

MIT (see `LICENSE`). Third-party models and datasets keep their own licences.
