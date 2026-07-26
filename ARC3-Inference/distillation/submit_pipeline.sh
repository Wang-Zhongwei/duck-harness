#!/usr/bin/env bash
# Submit an on-policy distillation run as a chain of Slurm iterations.
#
# One iteration = rollout -> score -> train. The stages are pipelined where the
# algorithm allows:
#
#   rollout (1xH100, 25 games at DISTILL_CONCURRENCY) -+
#   score   (4xA100, tails the raw dir as it fills) ---+--> train (1xH100) -> next
#
# rollout and score start together and overlap: scoring a turn only needs that
# turn's own record, so the teacher consumes records while the student is still
# playing. train is the only true barrier -- it needs every scored record --
# and iteration k+1 cannot start collecting until it has the new weights,
# because that is what makes the data on-policy.
#
# Held-out games are played in the same rollout job (one extra wave at most) but
# are filtered out of scoring/training by configs/distill.games.json, so the
# held-out score comes for free instead of costing a separate eval stage.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

stage="${1:-all}"
[[ "$stage" == collect ]] && stage=rollout
case "$stage" in
  all|rollout|score|train) ;;
  *)
    echo "Usage: $0 {all|rollout|score|train}" >&2
    exit 2
    ;;
esac

passes="${DISTILL_PASSES:-1}"
iterations="${DISTILL_ITERATIONS:-1}"
start_iteration=$((10#${DISTILL_ITERATION:-0}))
games_file="${DISTILL_GAMES_FILE:-$ROOT/configs/distill.games.json}"
concurrency="${DISTILL_CONCURRENCY:-16}"
log_dir="$ROOT/distillation/logs"
mkdir -p "$log_dir"

# Empty DISTILL_GAME means "all official games"; the games file decides which of
# them are scored and trained on.
game="${DISTILL_GAME:-}"

adapter_dir_for() { printf '%s/artifacts/qwen36-27b-fp8-rkl-lora/iteration-%04d' "$ROOT" "$1"; }
merged_dir_for()  { printf '%s/artifacts/qwen36-27b-merged/iteration-%04d' "$ROOT" "$1"; }
raw_dir_for()     { printf '%s/distillation/rollouts/raw/iteration-%04d' "$ROOT" "$1"; }
scored_dir_for()  { printf '%s/distillation/rollouts/scored/iteration-%04d' "$ROOT" "$1"; }
run_dir_for()     { printf '%s/runs/distill-iteration-%04d' "$ROOT" "$1"; }

submit_rollout() {
  local iteration="$1" dependency="$2"
  local tag; tag="$(printf 'i%04d' "$iteration")"
  local policy; policy="iteration-$(printf '%04d' "$iteration")"
  local exports="ALL,DISTILL_ROOT=$ROOT,DISTILL_GAME=$game,DISTILL_PASSES=$passes"
  exports+=",DISTILL_POLICY_ID=$policy"
  exports+=",DISTILL_CAPTURE_DIR=$(raw_dir_for "$iteration")"
  exports+=",DISTILL_RESULT_DIR=$(run_dir_for "$iteration")"
  exports+=",DISTILL_RUN_NAME=distill-$policy"
  exports+=",DISTILL_CONCURRENCY=$concurrency"
  # Iteration 0 is the untuned base model: no adapter exists yet.
  if (( iteration > 0 )); then
    exports+=",DISTILL_MERGE_ADAPTER_DIR=$(adapter_dir_for "$iteration")"
    exports+=",DISTILL_MERGED_DIR=$(merged_dir_for "$iteration")"
  fi
  sbatch --parsable \
    --job-name="distill-rollout-$tag" \
    --partition=gpuqs --qos=normal --account=default \
    --gres=gpu:h100:1 --cpus-per-task=16 --mem=128G --time=12:00:00 \
    --output="$log_dir/rollout-$tag-%j.out" \
    --error="$log_dir/rollout-$tag-%j.err" \
    ${dependency:+--dependency="afterok:$dependency"} \
    --export="$exports" \
    "$ROOT/distillation/run_student_games.sh"
}

submit_score() {
  local iteration="$1" dependency="$2"
  local tag; tag="$(printf 'i%04d' "$iteration")"
  local exports="ALL,DISTILL_ROOT=$ROOT"
  exports+=",DISTILL_RAW_DIR=$(raw_dir_for "$iteration")"
  exports+=",DISTILL_SCORED_DIR=$(scored_dir_for "$iteration")"
  exports+=",DISTILL_GAMES_FILE=$games_file,DISTILL_WATCH=1"
  exports+=",DISTILL_SCORE_WORKERS=${DISTILL_SCORE_WORKERS:-8}"
  # Scoring re-renders the student's prompt AND its output as one prompt, so a
  # turn that filled the student's 32768 window overflows it by a few tokens.
  exports+=",TEACHER_MAX_MODEL_LEN=${TEACHER_MAX_MODEL_LEN:-40960}"
  exports+=",TEACHER_TP=4,TEACHER_MAX_NUM_SEQS=${TEACHER_MAX_NUM_SEQS:-8}"
  exports+=",TEACHER_GPU_MEMORY_UTILIZATION=0.84"
  exports+=",TEACHER_CPU_OFFLOAD_GB=${TEACHER_CPU_OFFLOAD_GB:-24}"
  sbatch --parsable \
    --job-name="distill-score-$tag" \
    --partition=gpuqs --qos=normal --account=default \
    --nodes=1 --gres=gpu:a100:4 --cpus-per-task=48 --mem=230G --time=16:00:00 \
    --output="$log_dir/score-$tag-%j.out" \
    --error="$log_dir/score-$tag-%j.err" \
    ${dependency:+--dependency="afterok:$dependency"} \
    --export="$exports" \
    "$ROOT/distillation/run_teacher_score.sh"
}

submit_train() {
  local iteration="$1" dependency="$2"
  local tag; tag="$(printf 'i%04d' "$iteration")"
  sbatch --parsable \
    --job-name="distill-train-$tag" \
    --partition=gpuqs --qos=normal --account=default \
    --gres=gpu:h100:1 --cpus-per-task=16 --mem=128G --time=08:00:00 \
    --output="$log_dir/train-$tag-%j.out" \
    --error="$log_dir/train-$tag-%j.err" \
    ${dependency:+--dependency="afterok:$dependency"} \
    --export="ALL,DISTILL_ROOT=$ROOT,DISTILL_SCORED_DIR=$(scored_dir_for "$iteration"),DISTILL_ADAPTER_DIR=$(adapter_dir_for $((iteration + 1)))" \
    "$ROOT/distillation/run_train.sh"
}

if [[ "$stage" != all ]]; then
  "submit_$stage" "$start_iteration" "${DISTILL_DEPENDENCY:-}"
  exit
fi

previous_train="${DISTILL_DEPENDENCY:-}"
echo "On-policy distillation: $iterations iteration(s) starting at $start_iteration"
echo "  games:       ${game:-all official} (split: $games_file)"
echo "  passes:      $passes per game, rollout concurrency $concurrency"
for (( offset = 0; offset < iterations; offset++ )); do
  iteration=$((start_iteration + offset))
  rollout_id="$(submit_rollout "$iteration" "$previous_train")"
  # Same dependency as rollout, not on rollout: the scorer tails the raw
  # directory while the student is still playing.
  score_id="$(submit_score "$iteration" "$previous_train")"
  train_id="$(submit_train "$iteration" "$rollout_id:$score_id")"
  previous_train="$train_id"
  printf '  iteration %04d: rollout=%-8s score=%-8s train=%-8s -> %s\n' \
    "$iteration" "$rollout_id" "$score_id" "$train_id" \
    "$(basename "$(adapter_dir_for $((iteration + 1)))")"
done
echo "  logs: $log_dir"
