#!/usr/bin/env bash
# Submit a Slurm run on the SJSU coe-hpc3 cluster.
#
# A thin wrapper around `make sbatch` that exports the environment the worker
# needs on nodes with no outbound internet and no system CUDA toolkit. Every
# variable here is forwarded into the sbatch --export by
# inference/framework/run.py (_SLURM_FORWARDED_ENV); nothing else in the
# harness changes.
#
# Usage:  bash scripts/sbatch_coe_hpc3.sh [make-vars...]
#   e.g.  bash scripts/sbatch_coe_hpc3.sh RUN_NAME=q38-effort-low
#
# Run from coe-hpc3 (sbatch lives there). Assumes the shared ~/.cache/uv has
# already been prefetched from the login node -- see the memory note
# "slurm-worker-offline-venv" for the recipe.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# The Makefile exports XDG_CACHE_HOME under .cache/arc3_runtime, and uv honours
# $XDG_CACHE_HOME/uv when UV_CACHE_DIR is unset -- that cache is empty. Pin uv
# to the shared cache that was actually prefetched, and stop it trying PyPI.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
export UV_OFFLINE="${UV_OFFLINE:-1}"

# Weights live in the vLLM container store, not $HOME/.cache/huggingface. Skip
# the huggingface.co freshness HEAD, whose connect timeouts eat the server
# start budget.
export HF_HOME="${HF_HOME:-$PWD/.cache/vllm-container/hf}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# flashinfer JIT-compiles Qwen3.x's GDN linear-attention kernels with nvcc at
# first inference. Without a toolkit vLLM boots, reports ready, and 500s every
# request. The nvhpc module ships CUDA 12.6.
export CUDA_HOME="${CUDA_HOME:-/opt/ohpc/pub/apps/nvidia/nvhpc/24.11/Linux_x86_64/24.11/cuda/12.6}"

# Every h100 node here exposes a single GPU; a 2-GPU gres is unschedulable.
exec make sbatch DEPLOYMENT_SLURM_GPU_COUNT=1 "$@"
