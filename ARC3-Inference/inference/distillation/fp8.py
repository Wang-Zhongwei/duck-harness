from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def safetensors_dtypes(path: str | Path) -> dict[str, str]:
    """Read only the safetensors header and return tensor storage dtypes."""
    with Path(path).open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
    return {
        name: str(metadata["dtype"])
        for name, metadata in header.items()
        if name != "__metadata__"
    }


class StaticW8A8Linear(nn.Linear):
    """Frozen static per-tensor FP8 linear with input-gradient support."""

    def __init__(self, source: nn.Linear) -> None:
        if source.bias is not None:
            raise ValueError("StaticW8A8Linear currently requires bias=False")
        super().__init__(
            source.in_features,
            source.out_features,
            bias=False,
            device=source.weight.device,
            dtype=torch.bfloat16,
        )
        self.weight = nn.Parameter(
            torch.empty(
                self.out_features,
                self.in_features,
                device=source.weight.device,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        self.register_buffer(
            "weight_scale",
            torch.empty(1, device=source.weight.device, dtype=torch.bfloat16),
        )
        self.register_buffer(
            "input_scale",
            torch.empty(1, device=source.weight.device, dtype=torch.bfloat16),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return _StaticW8A8Function.apply(
            inputs, self.weight, self.input_scale, self.weight_scale
        )


class _StaticW8A8Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(weight, weight_scale)
        ctx.input_shape = inputs.shape
        flat = inputs.reshape(-1, inputs.shape[-1])
        if flat.is_cuda and torch.cuda.get_device_capability(flat.device)[0] >= 9:
            limit = torch.finfo(torch.float8_e4m3fn).max
            quantized = (
                (flat / input_scale).clamp(-limit, limit).to(torch.float8_e4m3fn)
            )
            output = torch._scaled_mm(
                quantized,
                weight.t(),
                scale_a=input_scale.float(),
                scale_b=weight_scale.float(),
                out_dtype=inputs.dtype,
            )
        else:
            output = F.linear(
                flat, weight.to(inputs.dtype) * weight_scale.to(inputs.dtype)
            )
        return output.reshape(*inputs.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        weight, weight_scale = ctx.saved_tensors
        flat = grad_output.reshape(-1, grad_output.shape[-1])
        if flat.is_cuda and torch.cuda.get_device_capability(flat.device)[0] >= 9:
            limit = torch.finfo(torch.float8_e4m3fn).max
            grad_scale = (flat.detach().abs().max() / limit).clamp_min(1e-12)
            quantized_grad = (
                (flat / grad_scale).clamp(-limit, limit).to(torch.float8_e4m3fn)
            )
            backward_weight = weight.t().contiguous().t()
            grad_input = torch._scaled_mm(
                quantized_grad,
                backward_weight,
                scale_a=grad_scale.float(),
                scale_b=weight_scale.float(),
                out_dtype=grad_output.dtype,
            )
        else:
            grad_input = flat @ (
                weight.to(grad_output.dtype) * weight_scale.to(grad_output.dtype)
            )
        return grad_input.reshape(ctx.input_shape), None, None, None


def replace_fp8_linears(model: nn.Module, checkpoint: str | Path) -> list[str]:
    """Replace linears whose checkpoint weights use F8_E4M3 storage."""
    dtypes = safetensors_dtypes(checkpoint)
    replaced: list[str] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if dtypes.get(f"{name}.weight") != "F8_E4M3":
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, StaticW8A8Linear(module))
        replaced.append(name)
    if not replaced:
        raise ValueError(f"{checkpoint}: no F8_E4M3 linear weights found")
    return replaced


def load_static_fp8_model(
    model_name_or_path: str,
    *,
    checkpoint: str | Path,
    device: Any,
    attn_implementation: str = "sdpa",
) -> tuple[nn.Module, list[str]]:
    """Construct Qwen on meta, install FP8 layers, then load the compressed file."""
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForImageTextToText

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    config.quantization_config = None
    # Qwen's rotary-frequency buffers are derived from config and are not in
    # the checkpoint, so keep buffers materialized while parameters stay meta.
    with init_empty_weights(include_buffers=False):
        model = AutoModelForImageTextToText.from_config(
            config,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
        )
        replaced = replace_fp8_linears(model, checkpoint)
    expected = set(model.state_dict())
    # Read one memory-mapped CPU tensor at a time, then move it into its final
    # parameter. Opening the monolithic file directly on CUDA can retain a
    # second device-side view and exceed an 80 GB H100 during loading.
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        found = set(handle.keys())
        missing = expected - found
        unexpected = found - expected
        if missing or unexpected:
            raise ValueError(
                f"checkpoint/model key mismatch: missing={len(missing)}, "
                f"unexpected={len(unexpected)}"
            )
        for name in handle.keys():
            value = handle.get_tensor(name)
            set_module_tensor_to_device(model, name, device, value=value)
    # Non-persistent buffers (notably Qwen rotary frequencies) are constructed
    # from config and therefore never appear in the safetensors file.
    target_device = torch.device(device)
    if target_device.type == "cuda" and target_device.index is None:
        target_device = torch.device("cuda", torch.cuda.current_device())
    for name, value in model.named_buffers():
        if value.device != target_device:
            set_module_tensor_to_device(model, name, target_device, value=value)
    return model, replaced
