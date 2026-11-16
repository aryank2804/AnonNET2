# Copy to config.sh and edit.   cp config.example.sh config.sh
export ANONNET_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # root of this repo (do not edit)

# ---- where things live (large; keep outside the repo) ----
export WORK=/path/to/workdir                 # storage for models and outputs
export VEIL_ROOT=$WORK/veilbench             # benchmark working dir: clips_81f/, clips_756/, annotations/, anonymized/, embeddings/, results/
export SHARE_ROOT=/path/to/raw_corpora       # raw corpora, see README "Raw data"

# ---- model locations (README "Manual prerequisites") ----
export FACEXFORMER_REPO=$WORK/facexformer                    # cloned and filled by `./run_all.sh setup`
export FACEXFORMER_CKPT=$FACEXFORMER_REPO/ckpts/model.pt
export DREAMIDV_REPO=$WORK/dreamidv-inference                # official DreamID-V inference code
export DREAMIDV_CKPT=$WORK/models/dreamidv/dreamidv_faster.pth
export WAN_DIR=$WORK/models/wan2.1-1.3B                      # filled by `./run_all.sh setup`
export YOLO_MODEL=$WORK/models/yolo11n-face.pt
export SIXDREPNET_CKPT=$WORK/models/6DRepNet_300W_LP_AFLW2000.pth

# ---- outputs ----
export ANON_DIR=$VEIL_ROOT/anonymized/dreamidv_pad100/anonymized   # AnonNET2 clips (written by the `anonymize` stage)
export BASELINE_DIR=                                               # optional: folder with a baseline's clips (e.g. AnonNET) to evaluate side by side

# ---- cluster (optional) ----
export CUDA_MODULE=                           # e.g. cuda/12.2.1_gcc-10.4.0 ; leave empty if not needed
