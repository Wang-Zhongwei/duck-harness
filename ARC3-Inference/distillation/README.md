# Harness-native on-policy reverse-KL distillation

This pipeline trains a BF16 LoRA adapter over the frozen static-W8A8
`vrfai/Qwen3.6-27B-FP8` student while it plays Duck harness games. It does not
train on standalone prompts.

`configs/inference.json` is authoritative for the student checkpoint, harness
sampling parameters, multimodal context, tool parser, and chat template. The
teacher is `Qwen/Qwen3.5-397B-A17B-GPTQ-Int4`.

## One-command cluster run

Submit a fresh game rollout, teacher scoring, one FP8 LoRA update, and a second
game run with the new adapter:

```bash
make distill-submit DISTILL_GAME=ar25 DISTILL_PASSES=1
```

The command submits four Slurm jobs with `afterok` dependencies and prints every
job ID. Defaults are `ar25`, one pass, and iteration zero. Logs are written under
`distillation/logs/`; the final result is written to
`runs/distill-ar25-eval-iteration-0001/`.

The stages can also be submitted separately:

```bash
make distill-collect DISTILL_GAME=ar25
make distill-score DISTILL_GAME=ar25 DISTILL_DEPENDENCY=<collect-job-id>
make distill-train DISTILL_GAME=ar25 DISTILL_DEPENDENCY=<score-job-id>
make distill-eval DISTILL_GAME=ar25 DISTILL_DEPENDENCY=<train-job-id>
```

Collection, training, and evaluation each use one H100. Teacher scoring uses
one 4×A100 node with CPU weight offload because the 204 GiB teacher does not fit
in its 160 GiB of aggregate GPU memory.

## What one iteration does

1. vLLM serves the current FP8 student plus the current LoRA adapter.
2. The normal Duck harness plays complete games. Capture records every exact
   multimodal model request, tool schema, prompt token ID, sampled assistant
   token ID, and processed behavior-policy logprob.
3. The teacher scores the same assistant tokens under the same messages,
   images, and tools.
4. Token reward is `log p_teacher - log p_student`. Reward-to-go is computed
   backward across all model calls in each game, not reset at turn boundaries.
5. The trainer accumulates gradients over every captured game without changing
   the adapter, makes exactly one optimizer update, and saves the next adapter.
6. Reload that adapter into vLLM and collect fresh games. Do not reuse the old
   batch for another update.

## FP8 training path

The checkpoint contains frozen E4M3 weights plus static per-tensor weight and
input scales. `StaticW8A8Linear` uses those tensors directly. Its forward pass
uses `torch._scaled_mm` on Hopper or newer GPUs. Its custom backward quantizes
the output gradient to FP8 and computes only the input gradient; base-weight
gradients are deliberately omitted. LoRA parameters and optimizer state remain
BF16 and FP32 respectively.

The checkpoint leaves vision and Gated DeltaNet projections in BF16, matching
its `compressed-tensors` configuration.

## Prerequisites

The teacher is already cached under `.cache/vllm-container/hf`. Install the
trainer inside the current GPU container/image:

```bash
cd ARC3-Inference
uv sync --project distillation
```

The first implementation uses one H100 for training. Teacher serving requires a
separate multi-GPU allocation.

## 1. Serve the FP8 student

Start the normal student server from `inference.json`. The configured
`processed_logprobs` mode makes returned logprobs match temperature/top-k/top-p
sampling.

```bash
make server CONFIG_PATH=configs/inference.json \
  SERVER_ENABLE_LORA=true \
  SERVER_MAX_LORA_RANK=32
```

For iteration zero, omit `SERVER_LORA_PATH`. For later iterations:

```bash
make server CONFIG_PATH=configs/inference.json \
  SERVER_ENABLE_LORA=true \
  SERVER_MAX_LORA_RANK=32 \
  SERVER_LORA_ALIAS=policy \
  SERVER_LORA_PATH=artifacts/qwen36-27b-fp8-rkl-lora/iteration-0001
```

Point `analyzer.model_id`/the served model at the adapter alias if required by
the local vLLM deployment.

## 2. Collect real game episodes

Run the ordinary harness, adding only the capture directory and immutable
policy ID:

```bash
DISTILL_ROLLOUT_DIR=distillation/rollouts/raw/iteration-0000 \
DISTILL_POLICY_ID=iteration-0000 \
make run CONFIG_PATH=configs/inference.json
```

After loading a nonzero adapter under the `policy` alias, add
`MODEL=policy LOCAL_ANALYZER_MODEL_ID=policy` to that collection command.

The game list, passes, concurrency, context window, images, and sampling values
all continue to come from `inference.json`. Each game writes one JSONL file.
Capture fails rather than falling back to decoded text if vLLM omits exact token
IDs.

## 3. Serve the teacher

Inside its GPU allocation:

```bash
TEACHER_TP=4 ./distillation/serve_teacher.sh
```

The teacher server intentionally uses the student's chat template so tool calls
and reasoning serialize to the same token IDs.

## 4. Score the captured games

```bash
uv run --project distillation python -m inference.distillation.score_rollouts \
  --input distillation/rollouts/raw/iteration-0000 \
  --output distillation/rollouts/scored/iteration-0000
```

Scoring preserves images and tool context. It rejects any turn whose assistant
token sequence cannot be reconstructed exactly by the teacher chat request.

## 5. Make one on-policy update

```bash
uv run --project distillation accelerate launch --num_processes 1 \
  -m inference.distillation.train \
  --config configs/distill.qwen35-teacher-qwen36-student.json
```

This produces:

```text
artifacts/qwen36-27b-fp8-rkl-lora/iteration-0001
```

Restart/reload the student with that adapter, collect `iteration-0001`, score
it, change the distillation config input/output paths to iteration 1/2, and
repeat.

## Important invariants

- One scored directory contains exactly one `policy_id`.
- One scored directory is consumed by exactly one optimizer update.
- Dropout is zero.
- Student behavior logprobs use the sampling values from `inference.json`:
  temperature 0.6, top-p 0.95, and top-k 20.
- Teacher logprobs are raw probabilities; student logprobs are probabilities
  under the actual truncated behavior policy.
- Reverse-KL reward is applied to every sampled assistant token, including
  reasoning and Python tool-call serialization. Environment transitions affect
  later contexts and therefore later KL reward, but game score is not added to
  this objective.
