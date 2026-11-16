#!/usr/bin/env bash
# with_env.sh <conda-env> <command...>
# Runs a command inside a conda environment (no `conda activate` needed) with the CUDA pip libraries on LD_LIBRARY_PATH.
set -euo pipefail
env_name=$1; shift
[ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh || true
command -v conda >/dev/null 2>&1 || { command -v module >/dev/null 2>&1 && module load conda; }
[ -n "${CUDA_MODULE:-}" ] && command -v module >/dev/null 2>&1 && module load "$CUDA_MODULE" || true
exec conda run --no-capture-output -n "$env_name" bash -c \
  'export LD_LIBRARY_PATH="$(python "$ANONNET_REPO/scripts/cuda_libs.py"):${LD_LIBRARY_PATH:-}"; exec "$@"' _ "$@"
