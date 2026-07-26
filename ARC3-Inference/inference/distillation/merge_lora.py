"""Merge a PEFT LoRA adapter into an FP8 compressed-tensors base checkpoint.

Produces a plain bf16 checkpoint that vLLM serves without any LoRA machinery
(the vLLM LoRA path has no fast kernels for the GDN linear_attn modules and
decodes ~17x slower). FP8 weights are dequantized with their per-tensor
weight_scale; quantization scales are dropped and quantization_config is
removed from config.json.

Usage:
  python merge_lora.py --base <snapshot_dir> --adapter <adapter_dir> --output <out_dir>
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ADAPTER_PREFIX = "base_model.model."
SHARD_BYTES = 5 * 1024**3
COPY_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "processor_config.json",
    "preprocessor_config.json",
]


def tensor_names(path: Path) -> list[str]:
    with open(path, "rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    return [name for name in header if name != "__metadata__"]


def load_adapter(adapter_dir: Path) -> tuple[dict[str, dict[str, torch.Tensor]], float]:
    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    scaling = config["lora_alpha"] / config["r"]
    deltas: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(adapter_dir / "adapter_model.safetensors", framework="pt") as handle:
        for key in handle.keys():
            module, _, ab = key.rpartition(".lora_")
            assert module.startswith(ADAPTER_PREFIX), key
            base_key = module[len(ADAPTER_PREFIX):] + ".weight"
            deltas.setdefault(base_key, {})[ab.removesuffix(".weight")] = handle.get_tensor(key)
    return deltas, scaling


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    out = args.output
    if (out / "model.safetensors.index.json").exists():
        print(f"Merged checkpoint already exists at {out}; skipping.")
        return
    out.mkdir(parents=True, exist_ok=True)

    deltas, scaling = load_adapter(args.adapter)
    merged_modules: set[str] = set()
    dequantized = 0

    base_file = args.base / "model.safetensors"
    names = tensor_names(base_file)
    weight_map: dict[str, str] = {}
    shard: dict[str, torch.Tensor] = {}
    shard_bytes = 0
    shard_index = 0
    shard_names: list[str] = []

    def flush() -> None:
        nonlocal shard, shard_bytes, shard_index
        if not shard:
            return
        shard_index += 1
        name = f"model-{shard_index:05d}.safetensors"
        shard_names.append(name)
        for key in shard:
            weight_map[key] = name
        save_file(shard, out / name, metadata={"format": "pt"})
        shard = {}
        shard_bytes = 0

    with safe_open(base_file, framework="pt") as handle:
        for name in names:
            if name.endswith((".weight_scale", ".input_scale")):
                continue
            tensor = handle.get_tensor(name)
            if tensor.dtype == torch.float8_e4m3fn:
                scale = handle.get_tensor(name + "_scale").float()
                tensor = tensor.to(torch.float32) * scale
                dequantized += 1
            elif name in deltas:
                tensor = tensor.to(torch.float32)
            if name in deltas:
                pair = deltas[name]
                delta = (pair["B"].float() @ pair["A"].float()) * scaling
                assert delta.shape == tensor.shape, (name, delta.shape, tensor.shape)
                tensor = tensor + delta
                merged_modules.add(name)
            if tensor.dtype == torch.float32:
                tensor = tensor.to(torch.bfloat16)
            shard[name] = tensor
            shard_bytes += tensor.numel() * tensor.element_size()
            if shard_bytes >= SHARD_BYTES:
                flush()
    flush()

    missing = set(deltas) - merged_modules
    assert not missing, f"adapter modules with no base weight: {sorted(missing)[:5]}"

    total = sum((out / n).stat().st_size for n in shard_names)
    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    for filename in COPY_FILES:
        source = args.base / filename
        if source.exists():
            shutil.copyfile(source, out / filename)
    config = json.loads((out / "config.json").read_text())
    config.pop("quantization_config", None)
    (out / "config.json").write_text(json.dumps(config, indent=2))

    print(
        f"Merged {len(merged_modules)} modules (scaling={scaling}), "
        f"dequantized {dequantized} fp8 tensors, wrote {shard_index} shards "
        f"({total / 1024**3:.1f} GiB) to {out}"
    )


if __name__ == "__main__":
    main()
