#!/usr/bin/env bash
# iota wrapper: deterministic + offline env, GPU0 only, conda env muse
# (/dev_data/wlt/conda/envs/muse, verified 2026-10-18).
# Usage: bash repro_timemmd_common_v1/wrapper_muse_iota.sh <cmd...>
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
CONDA_BASE="${MUSE_CONDA_BASE:-/dev_data/wlt/conda}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate muse
cd "$(dirname "$0")/.."
exec "$@"
