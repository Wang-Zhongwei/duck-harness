#!/usr/bin/env bash
set -euo pipefail

ROOT="${DISTILL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

: "${DISTILL_RAW_DIR:?DISTILL_RAW_DIR is required}"
: "${DISTILL_SCORED_DIR:?DISTILL_SCORED_DIR is required}"

# In watch mode this job starts alongside collect, so the raw directory may not
# exist yet and is expected to keep growing; a one-shot run still requires
# rollouts to already be on disk.
if [[ -n "${DISTILL_WATCH:-}" ]]; then
  mkdir -p "$DISTILL_RAW_DIR"
elif ! find "$DISTILL_RAW_DIR" -maxdepth 1 -name '*.jsonl' -print -quit | grep -q .; then
  echo "No captured rollouts found in $DISTILL_RAW_DIR" >&2
  exit 1
fi
if [[ -e "$DISTILL_SCORED_DIR" ]]; then
  echo "Scored output already exists: $DISTILL_SCORED_DIR" >&2
  exit 1
fi

score_args=(--workers "${DISTILL_SCORE_WORKERS:-8}")
if [[ -n "${DISTILL_GAMES_FILE:-}" ]]; then
  score_args+=(--games-file "$DISTILL_GAMES_FILE")
fi
if [[ -n "${DISTILL_WATCH:-}" ]]; then
  score_args+=(--done-marker "$DISTILL_RAW_DIR/COLLECT_DONE")
fi

SIF="$ROOT/.cache/vllm-container/vllm-openai.sif"
HF_CACHE="$ROOT/.cache/vllm-container/hf"
JOB_DIR="$ROOT/distillation/jobs/${SLURM_JOB_ID:-local}-teacher"
SERVER_LOG="$JOB_DIR/teacher.log"
mkdir -p "$JOB_DIR"

apptainer exec --nv \
  --env HF_HOME="$HF_CACHE" \
  --env HF_HUB_OFFLINE=1 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$SIF" \
  bash "$ROOT/distillation/serve_teacher.sh" \
    --cpu-offload-gb "${TEACHER_CPU_OFFLOAD_GB:-24}" \
    >"$SERVER_LOG" 2>&1 &
teacher_pid=$!

cleanup() {
  kill "$teacher_pid" >/dev/null 2>&1 || true
  wait "$teacher_pid" >/dev/null 2>&1 || true
}
trap cleanup EXIT TERM INT

ready=0
for _ in $(seq 1 "${TEACHER_START_TIMEOUT:-3600}"); do
  if ! kill -0 "$teacher_pid" >/dev/null 2>&1; then
    echo "Teacher server exited before becoming ready" >&2
    tail -n 200 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:${TEACHER_PORT:-8000}/v1/models" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
done
if [[ "$ready" != 1 ]]; then
  echo "Teacher server did not become ready" >&2
  tail -n 200 "$SERVER_LOG" >&2 || true
  exit 1
fi

apptainer exec --nv \
  --env HF_HOME="$HF_CACHE" \
  --env HF_HUB_OFFLINE=1 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$SIF" \
  "$ROOT/.venv/bin/python" \
    -m inference.distillation.score_rollouts \
    --input "$DISTILL_RAW_DIR" \
    --output "$DISTILL_SCORED_DIR" \
    --base-url "http://127.0.0.1:${TEACHER_PORT:-8000}/v1" \
    "${score_args[@]}"
