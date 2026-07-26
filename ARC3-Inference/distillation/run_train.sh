#!/usr/bin/env bash
set -euo pipefail

ROOT="${DISTILL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

: "${DISTILL_SCORED_DIR:?DISTILL_SCORED_DIR is required}"
: "${DISTILL_ADAPTER_DIR:?DISTILL_ADAPTER_DIR is required}"

if ! find "$DISTILL_SCORED_DIR" -maxdepth 1 -name '*.jsonl' -print -quit | grep -q .; then
  echo "No scored rollouts found in $DISTILL_SCORED_DIR" >&2
  exit 1
fi
if [[ -e "$DISTILL_ADAPTER_DIR" ]]; then
  echo "Adapter output already exists: $DISTILL_ADAPTER_DIR" >&2
  exit 1
fi

apptainer exec --nv \
  --env HF_HOME="$ROOT/.cache/vllm-container/hf" \
  --env HF_HUB_OFFLINE=1 \
  --env PYTORCH_ALLOC_CONF=expandable_segments:True \
  "$ROOT/.cache/vllm-container/vllm-openai.sif" \
  "$ROOT/.venv/bin/accelerate" launch \
    --num_processes 1 \
    -m inference.distillation.train \
    --config configs/distill.qwen35-teacher-qwen36-student.json \
    --rollouts "$DISTILL_SCORED_DIR" \
    --output-dir "$DISTILL_ADAPTER_DIR"
