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

# Every h100 node here exposes a single GPU, so a multi-GPU gres is unschedulable.
# With deployment.slurm.gpu_count > 1 the harness instead submits one 1-GPU job
# per GPU and splits the passes (and concurrent_jobs) across them -- see
# _run_split_slurm_local_servers. So gpu_count 2 = 2 jobs x 1 H100.
# vLLM's sequential MTP draft loop writes the CUDA-graph input buffers without a
# barrier, so num_speculative_tokens >= 2 hits a sticky device-side assert (710).
# See vllm-project/vllm#40756. Measured here: the unpatched 20260818 MTP=3 run
# asserted on both passes; the patched 20260820 run logged zero.
export TAAF_PATCH_VLLM_MTP_RACE="${TAAF_PATCH_VLLM_MTP_RACE:-1}"

# `make sbatch` re-runs itself over `ssh $(SBATCH_LOGIN_HOST)` whenever the local
# short hostname differs from it -- and that fresh login shell drops every export
# above. SBATCH_LOGIN_HOST defaults to "coe-hpc3", but the box that name resolves
# to reports itself as "g17", so the hop ALWAYS fired and the run reached the
# compute node without UV_OFFLINE/UV_CACHE_DIR, dying on a wheels.vllm.ai fetch
# that no compute node can reach. When sbatch is already on PATH here there is
# nothing to hop to, so point the check at ourselves and submit in-process.
if command -v sbatch >/dev/null 2>&1; then
  export SBATCH_LOGIN_HOST="${SBATCH_LOGIN_HOST:-$(hostname -s)}"
fi

exec make sbatch "$@"
