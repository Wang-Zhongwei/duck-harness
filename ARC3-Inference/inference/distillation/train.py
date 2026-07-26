from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from inference.distillation.fp8 import load_static_fp8_model
from inference.distillation.objective import token_logprobs
from inference.distillation.rollouts import load_game_episodes


class _CpuGradientAccumulator:
    def __init__(self, named_parameters: list[tuple[str, Any]]) -> None:
        self._parameters = named_parameters
        self._gradients: dict[str, Any] = {}
        self._handles = [
            parameter.register_post_accumulate_grad_hook(
                self._capture_hook(name, parameter)
            )
            for name, parameter in named_parameters
        ]

    def _capture_hook(self, name: str, parameter: Any) -> Any:
        def capture(_: Any) -> None:
            import torch

            if parameter.grad is None:
                return
            gradient = parameter.grad.detach().to(
                device="cpu", dtype=torch.float32
            )
            if name in self._gradients:
                self._gradients[name].add_(gradient)
            else:
                self._gradients[name] = gradient
            parameter.grad = None

        return capture

    def restore(self) -> None:
        for handle in self._handles:
            handle.remove()
        for name, parameter in self._parameters:
            gradient = self._gradients.get(name)
            if gradient is not None:
                parameter.grad = gradient.to(
                    device=parameter.device, dtype=parameter.dtype
                )


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _processed_logprobs(
    logits: Any,
    targets: Any,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> Any:
    import torch

    if top_k > 0:
        # Top-p only needs to inspect the already-truncated top-k candidates.
        # Keeping this sparse avoids materializing and sorting a float32
        # [output_tokens, vocabulary] tensor for every game turn.
        candidate_scores, candidate_ids = logits.topk(
            min(top_k, logits.shape[-1]), dim=-1
        )
        sampled_scores = logits.gather(-1, targets.unsqueeze(-1))
        candidate_scores = candidate_scores.float() / temperature
        sampled_scores = sampled_scores.float() / temperature

        # Append the sampled action and mask its original top-k slot, if present,
        # so it is represented exactly once.
        duplicate = candidate_ids.eq(targets.unsqueeze(-1))
        candidate_scores = candidate_scores.masked_fill(duplicate, -torch.inf)
        candidate_scores = torch.cat((candidate_scores, sampled_scores), dim=-1)

        if top_p < 1.0:
            sorted_scores, sorted_indices = candidate_scores.sort(
                dim=-1, descending=True
            )
            cumulative = sorted_scores.softmax(dim=-1).cumsum(dim=-1)
            remove = cumulative > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            sorted_scores = sorted_scores.masked_fill(remove, -torch.inf)
            candidate_scores = torch.full_like(
                candidate_scores, -torch.inf
            ).scatter(-1, sorted_indices, sorted_scores)

        # The rollout token was in the vLLM behavior policy's support. Tiny
        # kernel and quantization differences can move it across a hard boundary
        # in the training forward, so restore its candidate after truncation.
        candidate_scores = torch.cat(
            (candidate_scores[..., :-1], sampled_scores), dim=-1
        )
        return candidate_scores.log_softmax(dim=-1)[..., -1]

    scores = logits.float() / temperature
    sampled_scores = scores.gather(-1, targets.unsqueeze(-1))
    if top_p < 1.0:
        sorted_scores, sorted_indices = scores.sort(dim=-1, descending=True)
        cumulative = sorted_scores.softmax(dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_scores = sorted_scores.masked_fill(remove, -torch.inf)
        scores = torch.full_like(scores, -torch.inf).scatter(
            -1, sorted_indices, sorted_scores
        )
    scores = scores.scatter(-1, targets.unsqueeze(-1), sampled_scores)
    return token_logprobs(scores, targets)


def _turn_inputs(processor: Any, turn: Any, device: Any) -> dict[str, Any]:
    import torch
    from transformers.image_utils import load_image

    messages = []
    for message in turn.messages:
        normalized = dict(message)
        if isinstance(normalized.get("content"), str):
            normalized["content"] = [{"type": "text", "text": normalized["content"]}]
        elif isinstance(normalized.get("content"), list):
            content = []
            for block in normalized["content"]:
                if block.get("type") == "image_url":
                    image_url = block.get("image_url")
                    url = (
                        image_url.get("url")
                        if isinstance(image_url, dict)
                        else image_url
                    )
                    content.append({"type": "image", "image": load_image(url)})
                else:
                    content.append(block)
            normalized["content"] = content
        if isinstance(normalized.get("tool_calls"), list):
            tool_calls = []
            for tool_call in normalized["tool_calls"]:
                tool_call = dict(tool_call)
                function = dict(tool_call.get("function") or {})
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    function["arguments"] = json.loads(arguments)
                tool_call["function"] = function
                tool_calls.append(tool_call)
            normalized["tool_calls"] = tool_calls
        if (
            isinstance(normalized.get("reasoning"), str)
            and "reasoning_content" not in normalized
        ):
            normalized["reasoning_content"] = normalized["reasoning"]
        messages.append(normalized)
    inputs = processor.apply_chat_template(
        messages,
        tools=turn.tools or None,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        truncation=False,
        max_length=processor.tokenizer.model_max_length,
        enable_thinking=True,
        preserve_thinking=True,
    )
    rendered_prompt = inputs["input_ids"][0].tolist()
    if rendered_prompt != turn.prompt_token_ids:
        raise ValueError(
            f"{turn.episode_id}: Transformers and rollout-server prompt token IDs differ"
        )
    output = torch.tensor([turn.output_token_ids], dtype=torch.long)
    inputs["input_ids"] = torch.cat((inputs["input_ids"], output), dim=1)
    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    if "mm_token_type_ids" in inputs:
        inputs["mm_token_type_ids"] = torch.cat(
            (
                inputs["mm_token_type_ids"],
                torch.zeros_like(output),
            ),
            dim=1,
        )
    return {name: value.to(device) for name, value in inputs.items()}


def _checkpoint_path(model_name: str) -> tuple[str, Path]:
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(model_name, local_files_only=True))
    checkpoint = snapshot / "model.safetensors"
    if not checkpoint.is_file():
        raise ValueError(f"{model_name}: expected one model.safetensors file")
    return str(snapshot), checkpoint


def train(config: dict[str, Any]) -> None:
    import torch
    from accelerate import Accelerator
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor

    accelerator = Accelerator()
    if accelerator.num_processes != 1:
        raise ValueError(
            "the initial static-FP8 implementation supports one training GPU; "
            "use gradient checkpointing and one H100"
        )
    student = config["student"]
    training = config["training"]
    inference = _load_json(config["inference_config"])
    analyzer = inference["analyzer"]
    configured_model = inference["shared"]["model_name"]
    if student["model"] != configured_model:
        raise ValueError(
            "distillation student.model must match inference.json shared.model_name"
        )

    model_path, checkpoint = _checkpoint_path(student["model"])
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model, fp8_modules = load_static_fp8_model(
        model_path,
        checkpoint=checkpoint,
        device=accelerator.device,
        attn_implementation=student.get("attn_implementation", "sdpa"),
    )
    lora = student["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(lora["rank"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora["target_modules"],
        ),
    )
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.dtype != torch.bfloat16:
            parameter.data = parameter.data.to(torch.bfloat16)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.config.use_cache = False
    model.train()

    episodes = load_game_episodes(
        training["rollouts"], gamma=float(training.get("gamma", 1.0))
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(training["learning_rate"]),
        betas=tuple(training.get("betas", [0.9, 0.95])),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    model, optimizer = accelerator.prepare(model, optimizer)
    optimizer.zero_grad(set_to_none=True)
    gradient_accumulator = _CpuGradientAccumulator(
        [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
    )

    episode_count = len(episodes)
    turn_count = sum(map(len, episodes))
    completed_turns = 0
    sampled_cost = 0.0
    sampled_tokens = 0
    for episode in episodes:
        for turn in episode:
            inputs = _turn_inputs(processor, turn, accelerator.device)
            output_width = len(turn.output_token_ids)
            outputs = model(
                **inputs,
                use_cache=False,
                logits_to_keep=output_width + 1,
            )
            targets = inputs["input_ids"][:, -output_width:]
            new_logprobs = _processed_logprobs(
                outputs.logits[:, :-1],
                targets,
                temperature=float(analyzer["temperature"]),
                top_k=int(analyzer["top_k"]),
                top_p=float(analyzer["top_p"]),
            )
            if not torch.isfinite(new_logprobs).all():
                raise ValueError(
                    f"{turn.episode_id}: a captured token is outside the current "
                    "temperature/top-k/top-p policy support"
                )
            advantages = torch.tensor(
                [turn.advantages],
                dtype=new_logprobs.dtype,
                device=new_logprobs.device,
            )
            loss = -(new_logprobs * advantages.detach()).sum() / episode_count
            accelerator.backward(loss)
            sampled_cost += sum(
                p - q
                for p, q in zip(
                    turn.student_logprobs, turn.teacher_logprobs, strict=True
                )
            )
            sampled_tokens += output_width
            completed_turns += 1
            if accelerator.is_main_process:
                print(
                    json.dumps(
                        {
                            "event": "turn_complete",
                            "turn": completed_turns,
                            "turns": turn_count,
                            "tokens": sampled_tokens,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    gradient_accumulator.restore()
    accelerator.clip_grad_norm_(
        model.parameters(), float(training.get("max_grad_norm", 1.0))
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    output_dir = Path(training["output_dir"])
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(
        output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )
    if accelerator.is_main_process:
        processor.save_pretrained(output_dir)
        print(
            json.dumps(
                {
                    "episodes": episode_count,
                    "turns": turn_count,
                    "tokens": sampled_tokens,
                    "reverse_kl_sample": sampled_cost / episode_count,
                    "reverse_kl_per_token": sampled_cost / max(1, sampled_tokens),
                    "fp8_modules": len(fp8_modules),
                    "optimizer_updates": 1,
                    "output_dir": str(output_dir),
                },
                sort_keys=True,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One harness-native on-policy reverse-KL adapter update"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--rollouts")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = _load_json(args.config)
    if args.rollouts:
        config["training"]["rollouts"] = args.rollouts
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir
    train(config)


if __name__ == "__main__":
    main()
