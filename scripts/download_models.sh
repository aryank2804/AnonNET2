#!/usr/bin/env bash
# Download the models that can be fetched automatically. Run once on a machine with internet access
# (compute nodes are often offline). The remaining items are listed in the README ("Manual prerequisites").
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source ./config.sh
W=scripts/with_env.sh

echo "[models] Wan2.1-T2V-1.3B -> $WAN_DIR"
$W a2 huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir "$WAN_DIR"

echo "[models] FLUX.1-dev (gated: accept the licence on Hugging Face and run 'huggingface-cli login' first)"
$W a2 huggingface-cli download black-forest-labs/FLUX.1-dev

echo "[models] DINO ViT-B/16 (temporal metric)"
$W env_dino python -c "import torch; torch.hub.load('facebookresearch/dino:main','dino_vitb16')"

echo "[models] Q-Align (one-align)"
$W env_qalign python -c "
import torch
from transformers import AutoModelForCausalLM
AutoModelForCausalLM.from_pretrained('q-future/one-align', trust_remote_code=True, attn_implementation='eager', torch_dtype=torch.float16)"

echo "[models] FVD I3D + InsightFace buffalo_l are fetched on first use by the metric scripts."
