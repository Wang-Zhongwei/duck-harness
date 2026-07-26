#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_HOME="${HF_HOME:-$HERE/.cache/vllm-container/hf}"
export HF_HOME
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-0}"

hf download Qwen/Qwen3.5-397B-A17B-GPTQ-Int4 \
  --max-workers "${HF_DOWNLOAD_MAX_WORKERS:-4}"
