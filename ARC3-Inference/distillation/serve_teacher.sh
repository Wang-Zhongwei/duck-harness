#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="${HF_HOME:-$HERE/.cache/vllm-container/hf}"
STUDENT_REPO="$HF_HOME/hub/models--vrfai--Qwen3.6-27B-FP8"
STUDENT_REVISION="$(<"$STUDENT_REPO/refs/main")"
STUDENT_TEMPLATE="${STUDENT_CHAT_TEMPLATE:-$STUDENT_REPO/snapshots/$STUDENT_REVISION/chat_template.jinja}"

API_KEY_ARGS=()
if [[ -n "${TEACHER_API_KEY:-}" ]]; then
  API_KEY_ARGS=(--api-key "$TEACHER_API_KEY")
fi

exec vllm serve Qwen/Qwen3.5-397B-A17B-GPTQ-Int4 \
  --served-model-name Qwen/Qwen3.5-397B-A17B-GPTQ-Int4 \
  --host "${TEACHER_HOST:-0.0.0.0}" \
  --port "${TEACHER_PORT:-8000}" \
  --tensor-parallel-size "${TEACHER_TP:-4}" \
  --max-model-len "${TEACHER_MAX_MODEL_LEN:-32768}" \
  --max-num-seqs "${TEACHER_MAX_NUM_SEQS:-16}" \
  --gpu-memory-utilization "${TEACHER_GPU_MEMORY_UTILIZATION:-0.92}" \
  --max-logprobs 1 \
  --chat-template "$STUDENT_TEMPLATE" \
  --default-chat-template-kwargs '{"preserve_thinking":true}' \
  --limit-mm-per-prompt "{\"image\": ${TEACHER_MAX_IMAGES:-4}, \"video\": 0}" \
  --generation-config vllm \
  --trust-remote-code \
  "${API_KEY_ARGS[@]}" \
  "$@"
