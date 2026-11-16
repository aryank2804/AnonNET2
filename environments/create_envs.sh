#!/usr/bin/env bash
# Create every conda environment used by the repo from the exported specs (skips the ones that already exist).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
[ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh || true
command -v conda >/dev/null 2>&1 || { command -v module >/dev/null 2>&1 && module load conda; }

for e in a2 facex dreamidv bench embed fvd env_dino env_qalign; do
  if conda env list | awk '{print $1}' | grep -qx "$e"; then
    echo "[env] $e exists"
  else
    echo "[env] creating $e"; conda env create -n "$e" -f "$e.yml"
  fi
done
if ! conda env list | awk '{print $1}' | grep -qx vbench; then
  echo "[env] creating vbench"
  conda create -y -n vbench python=3.10
  conda run --no-capture-output -n vbench pip install torch torchvision vbench
fi
