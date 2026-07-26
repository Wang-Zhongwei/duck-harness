#!/usr/bin/env bash
set -euo pipefail

ROOT="${DISTILL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

# DISTILL_GAME may be empty on purpose: that means "every official game", and
# configs/distill.games.json decides which of them are trained on.
: "${DISTILL_GAME?DISTILL_GAME must be set (empty means all official games)}"
: "${DISTILL_PASSES:?DISTILL_PASSES is required}"
: "${DISTILL_RESULT_DIR:?DISTILL_RESULT_DIR is required}"

CONFIG_PATH="${CONFIG_PATH:-configs/inference.json}"
JOB_TAG="${SLURM_JOB_ID:-local}-student"
STATE_ROOT="$ROOT/distillation/jobs/$JOB_TAG"
mkdir -p "$STATE_ROOT" "$DISTILL_RESULT_DIR"

export SERVER_STATE_ROOT="$STATE_ROOT/server"
export SERVER_API_KEY_FILE="$SERVER_STATE_ROOT/server-api-key"
export SERVER_PID="$SERVER_STATE_ROOT/server.pid"
export SERVER_LOG="$SERVER_STATE_ROOT/server.log"
export SERVER_NODE_FILE="$SERVER_STATE_ROOT/server-host"
export SERVER_TMP_DIR="/tmp/arc3_distill_${SLURM_JOB_ID:-local}"
export SERVER_CACHE_ROOT="$STATE_ROOT/cache"
export XDG_CACHE_HOME="$SERVER_CACHE_ROOT/xdg"
export TORCHINDUCTOR_CACHE_DIR="$SERVER_CACHE_ROOT/torchinductor"
export TORCHINDUCTOR_FX_GRAPH_CACHE_DIR="$SERVER_CACHE_ROOT/torchinductor-fx"
export TRITON_CACHE_DIR="$SERVER_CACHE_ROOT/triton"

cleanup() {
  make stop-server CONFIG_PATH="$CONFIG_PATH" >/dev/null 2>&1 || true
}
trap cleanup EXIT TERM INT

# Preferred path: fold the adapter into the base weights and serve a plain
# checkpoint. vLLM's runtime-LoRA path has no fast kernels for the GDN
# linear_attn modules and decodes ~17x slower, which stalls every turn past the
# analyzer timeout. Merged bf16 weights leave fewer Mamba cache blocks, hence
# the SERVER_MAX_NUM_SEQS default here.
if [[ -n "${DISTILL_MERGE_ADAPTER_DIR:-}" ]]; then
  : "${DISTILL_MERGED_DIR:?DISTILL_MERGED_DIR is required when merging}"
  BASE_SNAPSHOT="${DISTILL_BASE_SNAPSHOT:-$ROOT/.cache/vllm-container/hf/hub/models--vrfai--Qwen3.6-27B-FP8/snapshots/70462b01826175b3f45fa80065184167ddc973fb}"
  "$ROOT/.venv/bin/python" "$ROOT/inference/distillation/merge_lora.py" \
    --base "$BASE_SNAPSHOT" \
    --adapter "$DISTILL_MERGE_ADAPTER_DIR" \
    --output "$DISTILL_MERGED_DIR"
  export SERVER_MODEL_PATH="$DISTILL_MERGED_DIR"
  export SERVER_MAX_NUM_SEQS="${SERVER_MAX_NUM_SEQS:-64}"
fi

model_args=()
if [[ -n "${DISTILL_ADAPTER_PATH:-}" ]]; then
  if [[ ! -d "$DISTILL_ADAPTER_PATH" ]]; then
    echo "Adapter does not exist: $DISTILL_ADAPTER_PATH" >&2
    exit 1
  fi
  export SERVER_ENABLE_LORA=true
  export SERVER_MAX_LORAS=1
  export SERVER_MAX_LORA_RANK="${SERVER_MAX_LORA_RANK:-32}"
  export SERVER_LORA_ALIAS="${SERVER_LORA_ALIAS:-policy}"
  export SERVER_LORA_PATH="$DISTILL_ADAPTER_PATH"
  export LOCAL_ANALYZER_MODEL_ID="$SERVER_LORA_ALIAS"
  model_args=(MODEL="$SERVER_LORA_ALIAS" LOCAL_ANALYZER_MODEL_ID="$SERVER_LORA_ALIAS")
fi

if [[ -n "${DISTILL_CAPTURE_DIR:-}" ]]; then
  if find "$DISTILL_CAPTURE_DIR" -maxdepth 1 -name '*.jsonl' -print -quit 2>/dev/null | grep -q .; then
    echo "Capture directory already contains rollouts: $DISTILL_CAPTURE_DIR" >&2
    exit 1
  fi
  mkdir -p "$DISTILL_CAPTURE_DIR"
  export DISTILL_ROLLOUT_DIR="$DISTILL_CAPTURE_DIR"
  export DISTILL_POLICY_ID="${DISTILL_POLICY_ID:?DISTILL_POLICY_ID is required for capture}"
fi

make server \
  CONFIG_PATH="$CONFIG_PATH" \
  SERVER_ENABLE_LORA="${SERVER_ENABLE_LORA:-false}" \
  SERVER_MAX_LORA_RANK="${SERVER_MAX_LORA_RANK:-32}" \
  SERVER_LORA_ALIAS="${SERVER_LORA_ALIAS:-policy}" \
  SERVER_LORA_PATH="${SERVER_LORA_PATH:-}"

# The scorer tails DISTILL_CAPTURE_DIR while this runs, so publish a marker on
# the way out -- including on failure, or the scorer would poll until its own
# walltime expires.
finish() {
  if [[ -n "${DISTILL_CAPTURE_DIR:-}" ]]; then
    touch "$DISTILL_CAPTURE_DIR/COLLECT_DONE"
  fi
  cleanup
}
trap finish EXIT TERM INT

make interactive \
  CONFIG_PATH="$CONFIG_PATH" \
  GAME="$DISTILL_GAME" \
  N_PASSES="$DISTILL_PASSES" \
  CONCURRENT_JOBS="${DISTILL_CONCURRENCY:-16}" \
  ANALYZER_TIMEOUT="${DISTILL_ANALYZER_TIMEOUT:-600}" \
  RUN_NAME="${DISTILL_RUN_NAME:-distill-$DISTILL_GAME}" \
  EXPERIMENT_DIR="$DISTILL_RESULT_DIR" \
  SERVER_API_KEY_FILE="$SERVER_API_KEY_FILE" \
  "${model_args[@]}"
