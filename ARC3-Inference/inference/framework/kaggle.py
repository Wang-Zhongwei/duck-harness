"""Kaggle helpers for the ARC3 duck harness."""

from __future__ import annotations

import os
from dataclasses import dataclass

from inference.utils.openai_compat import normalize_provider

DEFAULT_VLLM_WHEELHOUSE_DATASET_SOURCE = "driessmit1/arc3-vllm-h100-wheelhouse-v3"
# SGLang ships as one opaque >1 GiB blob (a tar of a site-packages tree) because Kaggle
# silently fails to create a dataset from an archive of ~100k files.
DEFAULT_SGLANG_BLOB_DATASET_SOURCE = "jonathanwang2022/sglang-0517-sp-blob"
DEFAULT_QWEN_MODEL_DATASET_SOURCE = "jakobbrggen/qwen3-8-27b-fp8-hf-snapshot"
DEFAULT_QWEN_NVFP4_MODEL_DATASET_SOURCE = "jonathanwang2022/qwen38-27b-nvfp4-unsloth"
DEFAULT_SERVED_MODEL_NAME = "Qwen/Qwen3.8-27B-FP8"
DEFAULT_SERVED_MODEL_NAME_NVFP4 = "Qwen/Qwen3.8-27B-NVFP4"
DEFAULT_VLLM_PORT = 1234
DEFAULT_VLLM_MAX_MODEL_LEN = 65536
DEFAULT_VLLM_TENSOR_PARALLEL_SIZE = 1
DEFAULT_WHEELHOUSE_STAMP_TEXT = "vllm==0.19.0 torch==2.10.0 flashinfer==0.6.6\n"

# The 25 official ARC-AGI-3 games. The first 16 are the original Kaggle duck
# validation harness order; the remaining 9 complete the official tag set.
DUCK_HARNESS_PUBLIC_GAME_IDS: tuple[str, ...] = (
    "tn36-ef4dde99",
    "lf52-271a04aa",
    "cn04-2fe56bfb",
    "bp35-0a0ad940",
    "wa30-ee6fef47",
    "lp85-305b61c3",
    "r11l-495a7899",
    "tu93-0768757b",
    "sp80-589a99af",
    "m0r0-492f87ba",
    "vc33-5430563c",
    "ar25-0c556536",
    "ka59-38d34dbb",
    "sc25-635fd71a",
    "sk48-d8078629",
    "dc22-fdcac232",
    "cd82-fb555c5d",
    "ft09-0d8bbf25",
    "g50t-5849a774",
    "ls20-9607627b",
    "re86-8af5384d",
    "s5i5-18d95033",
    "sb26-7fbdac44",
    "su15-1944f8ab",
    "tr87-cd924810",
)


@dataclass(frozen=True)
class DuckKaggleVllmConfig:
    """Kaggle-side vLLM/model configuration declared by ``HarnessSolver``."""

    wheelhouse_dataset_source: str = DEFAULT_VLLM_WHEELHOUSE_DATASET_SOURCE
    model_dataset_source: str = DEFAULT_QWEN_MODEL_DATASET_SOURCE
    served_model_name: str = DEFAULT_SERVED_MODEL_NAME
    vllm_port: int = DEFAULT_VLLM_PORT
    max_model_len: int = DEFAULT_VLLM_MAX_MODEL_LEN
    tensor_parallel_size: int = DEFAULT_VLLM_TENSOR_PARALLEL_SIZE
    wheelhouse_stamp_text: str = DEFAULT_WHEELHOUSE_STAMP_TEXT


def _kaggle_env(name: str, default: str = "") -> str:
    """Read a KAGGLE_* knob, treating an empty export as unset.

    The Makefile exports every variable it defines, so a config key with no value
    arrives as "" rather than absent -- a dict default would never fire.
    """
    return (str(os.environ.get(name, "") or "").strip()) or default


def _kaggle_backend() -> str:
    # KAGGLE_SERVER_BACKEND is deliberately distinct from SERVER_BACKEND: the latter
    # gates the LOCAL `make server` target, which only knows how to launch vLLM.
    return _kaggle_env("KAGGLE_SERVER_BACKEND", "vllm").lower()


def _runtime_dataset_source(cfg: DuckKaggleVllmConfig) -> str:
    """Runtime slot: the vLLM wheelhouse, or the SGLang site-packages blob."""
    if _kaggle_backend() != "sglang":
        return cfg.wheelhouse_dataset_source
    return _kaggle_env("SGLANG_BLOB_DATASET_SOURCE", DEFAULT_SGLANG_BLOB_DATASET_SOURCE)


def _model_dataset_source(cfg: DuckKaggleVllmConfig) -> str:
    # HarnessSolver never populates its kaggle_* fields from JSON, so an env override is
    # the only way to point Kaggle at a different checkpoint without editing this file.
    override = _kaggle_env("KAGGLE_MODEL_DATASET_SOURCE")
    if override:
        return override
    if _kaggle_backend() == "sglang":
        return DEFAULT_QWEN_NVFP4_MODEL_DATASET_SOURCE
    return cfg.model_dataset_source


def _served_model_name(cfg: DuckKaggleVllmConfig) -> str:
    override = _kaggle_env("KAGGLE_SERVED_MODEL_NAME")
    if override:
        return override
    if _kaggle_backend() == "sglang":
        return DEFAULT_SERVED_MODEL_NAME_NVFP4
    return cfg.served_model_name


def duck_kaggle_dataset_sources(
    config: DuckKaggleVllmConfig | None = None,
) -> list[str]:
    cfg = config or DuckKaggleVllmConfig()
    return [_runtime_dataset_source(cfg), _model_dataset_source(cfg)]


def duck_kaggle_setup_command(config: DuckKaggleVllmConfig | None = None) -> str:
    cfg = config or DuckKaggleVllmConfig()
    # These two must agree with duck_kaggle_dataset_sources(), which decides what is
    # actually attached to the notebook.
    wheelhouse_owner, wheelhouse_slug = _split_dataset_source(
        _runtime_dataset_source(cfg),
        option_name="wheelhouse_dataset_source",
    )
    model_owner, model_slug = _split_dataset_source(
        _model_dataset_source(cfg),
        option_name="model_dataset_source",
    )
    served_model_name = _served_model_name(cfg)
    # Base URL / model are pinned to the local server below, so reject a provider that
    # disagrees (e.g. openrouter): build_chat_payload only emits top_k, seed and
    # chat_template_kwargs on the "vllm" branch, and both vLLM and SGLang accept all
    # three. Anything else silently drops them — no error, just worse play.
    analyzer_provider = os.environ.get("LOCAL_ANALYZER_PROVIDER", "vllm")
    if normalize_provider(analyzer_provider) != "vllm":
        raise ValueError(
            f"kaggle-duck talks to a local OpenAI-compatible server (vLLM or SGLang), so "
            f"LOCAL_ANALYZER_PROVIDER must normalize to 'vllm', got {analyzer_provider!r}."
        )
    replacements = {
        "__WHEELHOUSE_OWNER__": repr(wheelhouse_owner),
        "__WHEELHOUSE_SLUG__": repr(wheelhouse_slug),
        "__MODEL_OWNER__": repr(model_owner),
        "__MODEL_SLUG__": repr(model_slug),
        "__SERVED_MODEL_NAME__": repr(served_model_name),
        "__VLLM_PORT__": repr(int(cfg.vllm_port)),
        "__VLLM_MAX_MODEL_LEN__": repr(int(cfg.max_model_len)),
        # The launcher's Makefile exports LOCAL_ANALYZER_CONTEXT_WINDOW from
        # JSON shared.context_window (or analyzer.context_window). Embed it
        # here so the agent's prompt budget on Kaggle is the JSON value, not
        # vllm's max-model-len. Falls back to max_model_len if unset.
        "__ANALYZER_CONTEXT_WINDOW__": repr(
            int(os.environ.get("LOCAL_ANALYZER_CONTEXT_WINDOW") or cfg.max_model_len)
        ),
        # Remaining JSON-driven analyzer/multimodal config: the launcher's
        # Makefile exports each from inference.json; embed the launcher value
        # so the rendered setup_env on Kaggle reflects JSON edits. Fallback
        # equals the historical hardcoded literal so direct kaggle.py callers
        # outside Make are unaffected.
        "__LOCAL_ANALYZER_PROVIDER__": repr(analyzer_provider),
        "__LOCAL_ANALYZER_APP_NAME__": repr(os.environ.get("LOCAL_ANALYZER_APP_NAME", "ARC3 Kaggle Harness")),
        "__LOCAL_ANALYZER_MAX_OUTPUT__": repr(os.environ.get("LOCAL_ANALYZER_MAX_OUTPUT", "0")),
        "__LOCAL_ANALYZER_TOOL_STEPS__": repr(os.environ.get("LOCAL_ANALYZER_TOOL_STEPS", "0")),
        "__LOCAL_ANALYZER_TOOL_TIMEOUT__": repr(os.environ.get("LOCAL_ANALYZER_TOOL_TIMEOUT", "30")),
        "__LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS__": repr(os.environ.get("LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS", "1024")),
        "__LOCAL_ANALYZER_YIELD_SECONDS__": repr(os.environ.get("LOCAL_ANALYZER_YIELD_SECONDS", "60")),
        "__LOCAL_ANALYZER_TEMPERATURE__": repr(os.environ.get("LOCAL_ANALYZER_TEMPERATURE", "0.6")),
        "__LOCAL_ANALYZER_TOP_P__": repr(os.environ.get("LOCAL_ANALYZER_TOP_P", "0.95")),
        "__LOCAL_ANALYZER_TOP_K__": repr(os.environ.get("LOCAL_ANALYZER_TOP_K", "20")),
        "__LOCAL_ANALYZER_ENABLE_THINKING__": repr(os.environ.get("LOCAL_ANALYZER_ENABLE_THINKING", "1")),
        "__MULTIMODAL_CONTEXT__": repr(os.environ.get("MULTIMODAL_CONTEXT", "current_grid")),
        "__MULTIMODAL_UPSCALE__": repr(os.environ.get("MULTIMODAL_UPSCALE", "4")),
        "__VLLM_TENSOR_PARALLEL_SIZE__": repr(int(cfg.tensor_parallel_size)),
        # JSON server.speculative_config, exported by the launcher's Makefile.
        # Empty (the default) omits --speculative-config entirely.
        "__VLLM_SPECULATIVE_CONFIG__": repr(os.environ.get("SERVER_SPECULATIVE_CONFIG", "")),
        "__WHEELHOUSE_STAMP_TEXT__": repr(cfg.wheelhouse_stamp_text),
        # Engine selection. "vllm" (default) renders the historical path
        # byte-identically; "sglang" swaps in the blob-mounted SGLang server.
        "__SERVER_BACKEND__": repr(_kaggle_backend()),
        # Server knobs previously unreachable from JSON: configs/inference.json
        # set "kv_cache_dtype" but nothing read it (a dead key on every run).
        "__ATTENTION_BACKEND__": repr(_kaggle_env("KAGGLE_ATTENTION_BACKEND")),
        "__KV_CACHE_DTYPE__": repr(_kaggle_env("KAGGLE_KV_CACHE_DTYPE")),
        # vLLM takes --speculative-config as one JSON blob; SGLang wants four flags.
        "__SPEC_ALGORITHM__": repr(_kaggle_env("KAGGLE_SPEC_ALGORITHM")),
        "__SPEC_NUM_STEPS__": repr(_kaggle_env("KAGGLE_SPEC_NUM_STEPS")),
        "__SPEC_EAGLE_TOPK__": repr(_kaggle_env("KAGGLE_SPEC_EAGLE_TOPK")),
        "__SPEC_NUM_DRAFT_TOKENS__": repr(_kaggle_env("KAGGLE_SPEC_NUM_DRAFT_TOKENS")),
        "__MAX_RUNNING_REQUESTS__": repr(_kaggle_env("KAGGLE_MAX_RUNNING_REQUESTS")),
        "__LIMIT_MM_PER_REQUEST__": repr(_kaggle_env("KAGGLE_LIMIT_MM_DATA_PER_REQUEST")),
        # The smoke test must sample exactly like the agent: greedy decoding sends this
        # model into a repetition loop that never returns.
        "__SMOKE_TEMPERATURE__": repr(float(os.environ.get("LOCAL_ANALYZER_TEMPERATURE") or 1.0)),
        "__SMOKE_TOP_P__": repr(float(os.environ.get("LOCAL_ANALYZER_TOP_P") or 0.95)),
        "__SMOKE_TOP_K__": repr(int(os.environ.get("LOCAL_ANALYZER_TOP_K") or 20)),
        "__SMOKE_MAX_TOKENS__": repr(int(_kaggle_env("KAGGLE_SMOKE_MAX_TOKENS", "2048"))),
        "__SERVER_START_TIMEOUT__": repr(int(_kaggle_env("KAGGLE_SERVER_START_TIMEOUT", "1800"))),
        # The vLLM argv hardcodes these three; SGLang needs them passed explicitly or the
        # agent silently gets no tool calls and no reasoning_content. Same values, and the
        # fallbacks matter because an unset config key exports as "" rather than absent.
        "__TOOL_CALL_PARSER__": repr(_kaggle_env("SERVER_TOOL_CALL_PARSER", "qwen3_coder")),
        "__REASONING_PARSER__": repr(_kaggle_env("SERVER_REASONING_PARSER", "qwen3")),
        "__DEFAULT_CHAT_TEMPLATE_KWARGS__": repr(
            _kaggle_env(
                "SERVER_DEFAULT_CHAT_TEMPLATE_KWARGS",
                '{"preserve_thinking": true, "reasoning_effort": "xhigh"}',
            )
        ),
        "__SMOKE_TEST_PNG_B64__": repr(_SMOKE_TEST_PNG_B64),
    }
    script = _DUCK_VLLM_SETUP_SCRIPT
    for placeholder, value in replacements.items():
        script = script.replace(placeholder, value)
    return f"\"$PYTHON\" - <<'PYSETUP'\n{script}\nPYSETUP"


def duck_kaggle_teardown_command() -> str:
    return f"\"$PYTHON\" - <<'PYTEARDOWN'\n{_DUCK_VLLM_TEARDOWN_SCRIPT}\nPYTEARDOWN"


# A 64x64 RGB PNG. The agent sends a base64 image part on every turn, so the setup
# script pushes one through the server before the run starts.
_SMOKE_TEST_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAABPklEQVR42u3XEduDYBgF4CAIBoNgEAwGg0EwGAyCQRAMgiAY"
    "BMFgEARBEARBMAgGwWAwGARBEARBEARBMAiCIAiC4PsFLw2/8+hz5KZzHYqiKJqmGYZZLBbL5ZJl2dVqxXEcSzhBEE6nkyiK"
    "kiSdz2dZlhVFUVWVlLdt23Ec13U9z/N9/36/B0HweDxI+TRNsyzL87woirIsq6qq67ppGlKeAgAAAAAA4CcA6bFerzebzXa7"
    "3e12PM/v9/vD4XA8Hkn5y+WiaZqu69fr9Xa7GYZhmqZlWaR8GIbP5/P1er3f78/nE0VRHMdJkpDy3++3bduu6/q+H4ZhHMdp"
    "muZ5BgAAAAAA4DcAmhgAAAAAAHsATQwAAAAAgD2AJgYAAAAAwB5AEwMAAAAAYA+giQEAAAAAsAfQxAAAAAAA/wbwB0gtsMRp"
    "RMEjAAAAAElFTkSuQmCC"
)


def _split_dataset_source(value: str, *, option_name: str) -> tuple[str, str]:
    parts = str(value or "").strip().split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            f"{option_name} must be a Kaggle dataset ref in owner/slug format."
        )
    return parts[0], parts[1]


_DUCK_VLLM_SETUP_SCRIPT = r"""import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

WHEELHOUSE_OWNER = __WHEELHOUSE_OWNER__
WHEELHOUSE_SLUG = __WHEELHOUSE_SLUG__
MODEL_OWNER = __MODEL_OWNER__
MODEL_SLUG = __MODEL_SLUG__
SERVED_MODEL_NAME = __SERVED_MODEL_NAME__
VLLM_HOST = '127.0.0.1'
VLLM_PORT = __VLLM_PORT__
VLLM_BASE_URL = f'http://{VLLM_HOST}:{VLLM_PORT}/v1'
VLLM_MAX_MODEL_LEN = __VLLM_MAX_MODEL_LEN__
ANALYZER_CONTEXT_WINDOW = __ANALYZER_CONTEXT_WINDOW__
VLLM_TENSOR_PARALLEL_SIZE = __VLLM_TENSOR_PARALLEL_SIZE__
# MTP self-speculative decoding, mirrored from JSON server.speculative_config.
# Empty leaves it off (vLLM default), matching the local/slurm path in the Makefile.
VLLM_SPECULATIVE_CONFIG = __VLLM_SPECULATIVE_CONFIG__
WORKING_DIR = Path(os.environ['TAAF_KAGGLE_WORKING_DIR'])
SITE_PACKAGES = WORKING_DIR / 'vllm-site-packages'
VLLM_SERVER_LOG = WORKING_DIR / 'vllm-openai-server.log'
VLLM_SERVER_PID = WORKING_DIR / 'vllm-openai-server.pid'
INSTALL_STAMP = SITE_PACKAGES / f'.{WHEELHOUSE_SLUG}'
STAMP_TEXT = __WHEELHOUSE_STAMP_TEXT__

SERVER_BACKEND = __SERVER_BACKEND__
ATTENTION_BACKEND = __ATTENTION_BACKEND__
KV_CACHE_DTYPE = __KV_CACHE_DTYPE__
SPEC_ALGORITHM = __SPEC_ALGORITHM__
SPEC_NUM_STEPS = __SPEC_NUM_STEPS__
SPEC_EAGLE_TOPK = __SPEC_EAGLE_TOPK__
SPEC_NUM_DRAFT_TOKENS = __SPEC_NUM_DRAFT_TOKENS__
MAX_RUNNING_REQUESTS = __MAX_RUNNING_REQUESTS__
LIMIT_MM_PER_REQUEST = __LIMIT_MM_PER_REQUEST__
SMOKE_TEMPERATURE = __SMOKE_TEMPERATURE__
SMOKE_TOP_P = __SMOKE_TOP_P__
SMOKE_TOP_K = __SMOKE_TOP_K__
SMOKE_MAX_TOKENS = __SMOKE_MAX_TOKENS__
SERVER_START_TIMEOUT = __SERVER_START_TIMEOUT__
TOOL_CALL_PARSER = __TOOL_CALL_PARSER__
REASONING_PARSER = __REASONING_PARSER__
DEFAULT_CHAT_TEMPLATE_KWARGS = __DEFAULT_CHAT_TEMPLATE_KWARGS__
# The blob untars to /tmp and NOT to WORKING_DIR. Two independent reasons: everything
# under /kaggle/working is committed as notebook output against a 20 GiB cap, and the
# CUDA-layout repair below mutates the tree, which /kaggle/input cannot do (read-only).
SGLANG_TREE = Path('/tmp/sglang-sp')
# The SGLang branch deliberately reuses the vLLM log and pid filenames so that
# tail_server_log(), wait_for_vllm_server() and the teardown script -- which take no
# config and hardcode these names -- keep working with no engine knowledge.
SERVER_PGID_PATH = WORKING_DIR / 'inference-server.pgid'
# A 64x64 RGB PNG used to prove the multimodal path end to end during setup. The agent
# sends a base64 image part on EVERY turn, so a server that cannot accept one scores 0
# while looking perfectly healthy on a text-only smoke test.
SMOKE_TEST_PNG_B64 = __SMOKE_TEST_PNG_B64__

GPU_NAME_PATTERNS = {'rtx-pro-6000': ('rtx pro 6000',), 'h100': ('h100',), 'l4': ('l4',)}


def taaf_kaggle_input_paths() -> dict[str, Path]:
    raw = os.getenv('TAAF_KAGGLE_INPUT_PATHS', '').strip()
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise RuntimeError('TAAF_KAGGLE_INPUT_PATHS must contain a JSON object.')
    return {str(ref): Path(str(path)) for ref, path in data.items()}


def resolve_kaggle_dataset_path(owner: str, slug: str) -> Path:
    mapped = taaf_kaggle_input_paths().get(f'{owner}/{slug}')
    if mapped is not None:
        return mapped
    for dataset_path in (Path('/kaggle/input') / slug, Path('/kaggle/input/datasets') / owner / slug):
        if dataset_path.exists():
            return dataset_path
    return Path('/kaggle/input') / slug


# One runtime dataset slot, reinterpreted by SERVER_BACKEND: the vLLM wheelhouse to
# pip-install from, or the SGLang site-packages blob to untar.
WHEELHOUSE = resolve_kaggle_dataset_path(WHEELHOUSE_OWNER, WHEELHOUSE_SLUG)
SGLANG_RUNTIME = WHEELHOUSE
MODEL_ROOT = resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG)


def resolve_model_dir(root: Path) -> Path:
    # Kaggle mounts some snapshots at the dataset root and others one level down. The
    # server needs the directory that actually holds config.json + *.safetensors.
    if (root / 'config.json').exists():
        return root
    for candidate in sorted(root.rglob('config.json')):
        if list(candidate.parent.glob('*.safetensors')):
            return candidate.parent
    return root


MODEL_PATH = resolve_model_dir(MODEL_ROOT) if MODEL_ROOT.exists() else MODEL_ROOT


def assert_expected_cuda_gpu() -> None:
    if not Path('/kaggle/input').exists():
        return
    assert shutil.which('nvidia-smi'), 'CUDA GPU check failed: nvidia-smi is not available.'
    result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True)
    assert result.returncode == 0, f'nvidia-smi failed: {result.stderr.strip()}'
    gpu_names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert gpu_names, 'nvidia-smi did not report any CUDA GPUs.'
    expected_gpu_type = os.getenv('KAGGLE_GPU_TYPE', 'rtx-pro-6000').strip().lower()
    expected_count = os.getenv('KAGGLE_GPU_COUNT', '1')
    if expected_count.isdigit():
        assert len(gpu_names) == int(expected_count), f'Expected {expected_count} CUDA GPU(s), found {gpu_names}'
    patterns = GPU_NAME_PATTERNS.get(expected_gpu_type, (expected_gpu_type.replace('-', ' '),))
    mismatched = [name for name in gpu_names if not any(pattern in name.lower() for pattern in patterns)]
    assert not mismatched, f'Expected GPU type {expected_gpu_type!r}, found {gpu_names}'
    print(f'CUDA GPU check passed for {expected_gpu_type} x{expected_count}: {gpu_names}', flush=True)


def vllm_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = str(SITE_PACKAGES) if not existing else f'{SITE_PACKAGES}{os.pathsep}{existing}'
    env.update(
        {
            'USE_TF': '0',
            'TRANSFORMERS_NO_TF': '1',
            'TRANSFORMERS_NO_TORCHVISION': '1',
            'VLLM_NO_USAGE_STATS': '1',
        }
    )
    return env


def cached_install_is_usable() -> bool:
    if not INSTALL_STAMP.exists() or INSTALL_STAMP.read_text(encoding='utf-8') != STAMP_TEXT:
        return False
    result = subprocess.run(
        [sys.executable, '-c', "import vllm, torch; print(f'Cached vLLM {vllm.__version__}, torch {torch.__version__}')"],
        env=vllm_env(),
        text=True,
    )
    return result.returncode == 0


def install_vllm_wheelhouse() -> None:
    requirements = WHEELHOUSE / 'requirements.lock'
    if not requirements.exists():
        raise FileNotFoundError(f'Missing wheelhouse lock file: {requirements}')
    if cached_install_is_usable():
        print(f'Using cached vLLM target install at {SITE_PACKAGES}', flush=True)
        return
    shutil.rmtree(SITE_PACKAGES, ignore_errors=True)
    SITE_PACKAGES.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        '-m',
        'pip',
        'install',
        '--no-index',
        '--find-links',
        str(WHEELHOUSE),
        '--requirement',
        str(requirements),
        '--target',
        str(SITE_PACKAGES),
        '--upgrade',
        '--ignore-installed',
        '--only-binary',
        ':all:',
        '--no-compile',
        '--disable-pip-version-check',
        '--no-warn-conflicts',
    ]
    print('Installing vLLM wheelhouse into', SITE_PACKAGES, flush=True)
    subprocess.run(cmd, check=True)
    INSTALL_STAMP.write_text(STAMP_TEXT, encoding='utf-8')


def request_json(url: str, payload: dict | None = None, timeout: int = 30) -> dict:
    data = None if payload is None else json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))


def tail_server_log(lines: int = 80) -> str:
    if not VLLM_SERVER_LOG.exists():
        return ''
    return '\n'.join(VLLM_SERVER_LOG.read_text(encoding='utf-8', errors='replace').splitlines()[-lines:])


def wait_for_vllm_server(timeout_seconds: int = 900) -> None:
    deadline = time.monotonic() + timeout_seconds
    url = f'{VLLM_BASE_URL}/models'
    while time.monotonic() < deadline:
        if VLLM_SERVER_PID.exists():
            try:
                os.kill(int(VLLM_SERVER_PID.read_text().strip()), 0)
            except OSError as exc:
                raise RuntimeError(f'vLLM server process is not alive: {exc}\n{tail_server_log()}') from exc
        try:
            models = request_json(url, timeout=5)
            print('vLLM server ready:', models, flush=True)
            return
        except Exception:
            time.sleep(5)
    raise TimeoutError(f'Timed out waiting for vLLM server at {url}.\nLast server log lines:\n{tail_server_log()}')


def sglang_supported_flags(env: dict) -> set:
    # Ask the installed argparse what it accepts instead of trusting a flag list.
    # An unrecognised option makes launch_server SystemExit(2) with no server and no
    # useful log, which on a real submission costs the entire 9-hour run.
    result = subprocess.run(
        [sys.executable, '-m', 'sglang.launch_server', '--help'],
        env=env, cwd='/tmp', capture_output=True, text=True, timeout=600,
    )
    text = (result.stdout or '') + (result.stderr or '')
    flags = set(re.findall(r'--[a-z0-9][a-z0-9-]+', text))
    assert '--model-path' in flags, (
        'Could not read SGLang argument table (is the tree importable?).\n'
        + text[-4000:]
    )
    return flags


def prepare_sglang_tree() -> Path:
    # Untar the blob, then repair a defect in sglang 0.5.17's own dependency pins:
    # <sp>/nvidia/cu13 has lib/ but no lib64/ and ships only versioned sonames, so every
    # JIT link dies with 'ld: cannot find -lcudart'. Verified fatal on this exact image,
    # then fixed, by kernel sglang-matrix-probe. The nvcc/header skew is handled in
    # sglang_env(). Every check here is an assert: it costs seconds, and the alternative
    # is discovering the problem minutes later as an unexplained CUDA failure.
    import stat
    import tarfile
    # The blob is a cp312 site-packages tree. Asserting that torch/_C.cpython-312*.so exists
    # (below) proves something about the BLOB, not about the interpreter the notebook handed
    # us -- taaf_kaggle_run.ipynb sets PYTHON = sys.executable. If Kaggle bumps the image,
    # every extension module fails to import minutes later with no obvious cause.
    assert sys.version_info[:2] == (3, 12), (
        f'The SGLang blob is a cp312 tree but this interpreter is '
        f'{sys.version_info.major}.{sys.version_info.minor}. The Kaggle base image changed; '
        f'rebuild the blob before submitting.'
    )
    blobs = sorted(
        (path for path in SGLANG_RUNTIME.rglob('*') if path.is_file() and path.stat().st_size > 1_000_000_000),
        key=lambda path: path.stat().st_size,
        reverse=True,
    )
    assert blobs, f'No SGLang blob (a file > 1 GiB) under {SGLANG_RUNTIME}. Is the dataset attached?'
    blob = blobs[0]
    existing = sorted(SGLANG_TREE.rglob('sglang/srt')) if SGLANG_TREE.exists() else []
    if existing and (SGLANG_TREE / '.unpacked').exists():
        site_packages = existing[0].parent.parent
        print(f'Reusing unpacked SGLang tree at {site_packages}', flush=True)
    else:
        # Assert free space BEFORE extracting: a partial 5.1 GB untar leaves a tree that
        # passes every existence check and then fails at import or JIT link minutes later.
        free_gib = shutil.disk_usage('/tmp').free / 2 ** 30
        assert free_gib >= 8.0, (
            f'Only {free_gib:.1f} GiB free on /tmp; need >= 8 GiB for the SGLang tree '
            f'plus the triton cache.'
        )
        shutil.rmtree(SGLANG_TREE, ignore_errors=True)
        SGLANG_TREE.mkdir(parents=True, exist_ok=True)
        started = time.time()
        print(f'Untarring SGLang blob {blob} ({blob.stat().st_size} bytes) -> {SGLANG_TREE}', flush=True)
        with tarfile.open(blob, 'r:*') as archive:
            # An explicit filter is required: the default is deprecated and becomes an
            # error on Python 3.14. 'tar' keeps internal symlinks, which the tree needs.
            archive.extractall(SGLANG_TREE, filter='tar')
        print(f'Untar took {time.time() - started:.0f}s', flush=True)
        hits = sorted(SGLANG_TREE.rglob('sglang/srt'))
        assert hits, f'Blob extracted but no sglang/srt found under {SGLANG_TREE}.'
        site_packages = hits[0].parent.parent
        (SGLANG_TREE / '.unpacked').write_text('ok', encoding='utf-8')
    cu13 = site_packages / 'nvidia' / 'cu13'
    lib = cu13 / 'lib'
    assert lib.is_dir(), f'{lib} is missing; the CUDA layout cannot be repaired and every JIT link will fail.'
    lib64 = cu13 / 'lib64'
    if not lib64.exists():
        lib64.symlink_to('lib')
    aliases = 0
    for shared_object in sorted(lib.glob('*.so.*')):
        # libcudart.so.13 -> libcudart.so, which is the name -lcudart actually looks for.
        bare = lib / (shared_object.name.split('.so.')[0] + '.so')
        if not bare.exists():
            try:
                bare.symlink_to(shared_object.name)
                aliases += 1
            except OSError:
                pass
    print(f'Repaired CUDA layout: lib64 -> lib, {aliases} unversioned soname aliases', flush=True)
    # Kaggle datasets can strip the exec bit, and triton shells out to these binaries.
    for binary in list((site_packages / 'triton/backends/nvidia/bin').glob('*')) + list((cu13 / 'bin').glob('*')):
        try:
            binary.chmod(binary.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        except OSError:
            pass
    required = {
        'cp312 torch ext': site_packages / 'torch/_C.cpython-312-x86_64-linux-gnu.so',
        'sglang package': site_packages / 'sglang/srt',
        'libnvrtc.so.13': cu13 / 'lib/libnvrtc.so.13',
        'cutlass DSL': site_packages / 'nvidia_cutlass_dsl/dsl_packages/cutlass',
        'nvcc (cu13)': cu13 / 'bin/nvcc',
        'ptxas': site_packages / 'triton/backends/nvidia/bin/ptxas',
    }
    missing = [f'{name} ({path})' for name, path in required.items() if not path.exists()]
    assert not missing, 'Incomplete SGLang tree, missing: ' + '; '.join(missing)
    print(f'SGLang preflight passed for {site_packages}', flush=True)
    return site_packages


def sglang_env(site_packages: Path) -> dict:
    # Every entry must be set BEFORE spawn: ld.so snapshots LD_LIBRARY_PATH at exec, and
    # SGLang forks scheduler and detokenizer subprocesses that would otherwise import
    # Kaggle's own torch instead of the pinned one in the blob.
    sp = str(site_packages)
    # .pth files do NOT execute for PYTHONPATH entries, and the GDN layers import
    # cutlass.cute, so the CuTe DSL directory has to be added by hand.
    dsl = f'{sp}/nvidia_cutlass_dsl/dsl_packages'
    libs = [f'{sp}/torch/lib', f'{sp}/nvidia/cu13/lib', f'{sp}/nvidia/cudnn/lib',
            f'{sp}/nvidia/nccl/lib', f'{sp}/nvidia/cusparselt/lib', f'{sp}/nvidia/nvshmem/lib',
            '/usr/local/nvidia/lib64', '/usr/lib/x86_64-linux-gnu']
    libs += sorted(str(path) for path in (site_packages / 'nvidia').glob('*/lib'))
    env = os.environ.copy()
    env['PYTHONPATH'] = f'{sp}:{dsl}'
    env['LD_LIBRARY_PATH'] = ':'.join(libs + [env.get('LD_LIBRARY_PATH', '')]).strip(':')
    env['LIBRARY_PATH'] = ':'.join(['/usr/local/nvidia/lib64', env.get('LIBRARY_PATH', '')]).strip(':')
    env['CUDA_HOME'] = f'{sp}/nvidia/cu13'
    env['PATH'] = f'{sp}/nvidia/cu13/bin:' + env.get('PATH', '')
    env['PYTHONNOUSERSITE'] = '1'
    env['TRITON_CACHE_DIR'] = '/tmp/triton'
    env['FLASHINFER_WORKSPACE_BASE'] = '/tmp/fi'
    env['HF_HUB_OFFLINE'] = '1'
    env['TRANSFORMERS_OFFLINE'] = '1'
    # sglang 0.5.17 resolves nvidia_cuda_nvcc 13.4.46rc1 against nvidia_cuda_runtime
    # 13.0.96, so nvcc reports 13.4 while cuda_runtime_api.h says CUDART_VERSION 13000 and
    # CCCL hard-errors on the mismatch. That header documents this exact escape hatch, and
    # a minor skew inside CUDA 13 is covered by minor-version compatibility.
    env['NVCC_APPEND_FLAGS'] = (env.get('NVCC_APPEND_FLAGS', '') + ' -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK').strip()
    env.pop('PYTHONHOME', None)
    return env


def start_sglang_server() -> None:
    site_packages = prepare_sglang_tree()
    env = sglang_env(site_packages)
    supported = sglang_supported_flags(env)
    VLLM_SERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
    VLLM_SERVER_PID.unlink(missing_ok=True)
    SERVER_PGID_PATH.unlink(missing_ok=True)
    log_handle = VLLM_SERVER_LOG.open('w', encoding='utf-8')
    cmd = [
        sys.executable, '-m', 'sglang.launch_server',
        '--model-path', str(MODEL_PATH),
        '--served-model-name', SERVED_MODEL_NAME,
        '--host', VLLM_HOST,
        '--port', str(VLLM_PORT),
        '--tp-size', str(VLLM_TENSOR_PARALLEL_SIZE),
        '--context-length', str(VLLM_MAX_MODEL_LEN),
        '--trust-remote-code',
        '--log-level', 'info',
    ]
    # Deliberately NOT passed, each verified harmful or absent for this model:
    #   --mem-fraction-static  vLLM's 0.92 does not translate. SGLang's fraction excludes
    #                          activations and cudagraph buffers; auto resolves to ~0.793.
    #   --speculative-draft-model-path  shrinks the KV pool ~5x (num_nextn_predict_layers
    #                          is absent from text_config, so it falls back to 64 layers).
    #   --quantization         auto-detected from config.json for both checkpoints.
    #   --enable-auto-tool-choice / --enable-prefix-caching  no SGLang equivalent; setting
    #                          --tool-call-parser is what enables tool parsing, and the
    #                          radix cache is on by default.
    optional = [
        ('--attention-backend', ATTENTION_BACKEND),
        ('--kv-cache-dtype', KV_CACHE_DTYPE),
        ('--max-running-requests', MAX_RUNNING_REQUESTS),
        # SGLang applies NO per-request image cap unless this is passed, where vLLM 400'd
        # past its limit. The agent keeps up to 30 assistant turns of history and attaches
        # an image to each user turn, so requests vLLM rejected now succeed with a much
        # larger multimodal prefill. Left unset by default (a 400 loses the turn outright);
        # set kaggle.limit_mm_data_per_request to '{"image": 4}' to restore vLLM parity.
        ('--limit-mm-data-per-request', LIMIT_MM_PER_REQUEST),
        ('--tool-call-parser', TOOL_CALL_PARSER),
        ('--reasoning-parser', REASONING_PARSER),
        ('--default-chat-template-kwargs', DEFAULT_CHAT_TEMPLATE_KWARGS),
        # vLLM ran with --generation-config vllm, i.e. ignore the checkpoint's
        # generation_config.json. SGLang's equivalent defaults the other way, so pass it
        # explicitly or the checkpoint silently overrides the harness's sampling params.
        ('--sampling-defaults', 'openai'),
    ]
    dropped = []
    for flag, value in optional:
        if not str(value).strip():
            continue
        if flag in supported:
            cmd += [flag, str(value).strip()]
        else:
            dropped.append(flag)
    if SPEC_ALGORITHM and '--speculative-algorithm' in supported:
        cmd += ['--speculative-algorithm', str(SPEC_ALGORITHM).strip()]
        for flag, value in (('--speculative-num-steps', SPEC_NUM_STEPS),
                            ('--speculative-eagle-topk', SPEC_EAGLE_TOPK),
                            ('--speculative-num-draft-tokens', SPEC_NUM_DRAFT_TOKENS)):
            if str(value).strip():
                cmd += [flag, str(value).strip()]
    elif SPEC_ALGORITHM:
        dropped.append('--speculative-algorithm')
    # Every flag in that list changes what the server actually does -- tool parsing, KV
    # dtype, attention backend, MTP depth. Losing any of them either scores ~0 or runs
    # multiples slower while /health stays green for nine hours, and the drop could equally
    # be a miss in the --help regex above rather than a genuinely absent flag. Fail loudly.
    assert not dropped, (
        f'This SGLang build did not report {dropped} in its argument table. Launching '
        f'without them would silently change what is served.'
    )
    print('Starting SGLang server:', ' '.join(cmd), flush=True)
    process = subprocess.Popen(
        cmd, env=env, cwd='/tmp', stdout=log_handle, stderr=subprocess.STDOUT, text=True,
        # Own session, so teardown can signal the scheduler and detokenizer children too.
        # Without this they survive a SIGTERM to the launcher and keep the 96 GB of VRAM.
        start_new_session=True,
    )
    VLLM_SERVER_PID.write_text(str(process.pid), encoding='utf-8')
    SERVER_PGID_PATH.write_text(str(os.getpgid(process.pid)), encoding='utf-8')
    # Explicit rather than the 900 s default: SGLang pays a cold triton JIT on top of
    # loading 22.57 GB of NVFP4 weights off /kaggle/input. Measured ~340 s cold, but the
    # default leaves little room, and dying here is unrecoverable.
    wait_for_vllm_server(SERVER_START_TIMEOUT)
    assert_served_config()


def assert_served_config() -> None:
    # An FP4 misconfiguration does not raise -- it produces fluent garbage. Check the
    # config the server actually resolved, not just that it answers /health.
    # Raising here rather than skipping: this check exists precisely because an FP4 or KV
    # misconfiguration does not announce itself, so an unreadable endpoint is a reason to
    # stop, not a reason to proceed unchecked.
    info = request_json(f'http://{VLLM_HOST}:{VLLM_PORT}/get_server_info', timeout=60)
    interesting = ('attention_backend', 'kv_cache_dtype', 'quantization', 'context_length',
                   'max_total_num_tokens', 'max_running_requests', 'mem_fraction_static',
                   'speculative_algorithm', 'speculative_num_steps')
    print('Served config: ' + json.dumps({k: info[k] for k in interesting if k in info}), flush=True)
    missing = [key for key in ('attention_backend', 'kv_cache_dtype') if key not in info]
    assert not missing, (
        f'/get_server_info did not report {missing}, so the served config cannot be '
        f'verified. Keys present: {sorted(info)}'
    )
    expected = [('attention_backend', ATTENTION_BACKEND), ('kv_cache_dtype', KV_CACHE_DTYPE)]
    if SPEC_NUM_STEPS:
        expected.append(('speculative_num_steps', int(SPEC_NUM_STEPS)))
    mismatched = [
        f'{key}: asked {want!r}, serving {info.get(key)!r}'
        for key, want in expected
        if str(want).strip() and key in info and str(info.get(key)) != str(want)
    ]
    assert not mismatched, 'SGLang resolved a different config than requested: ' + '; '.join(mismatched)
    # NOT asserted: 'quantization' reads back as None even for the NVFP4 checkpoint, and
    # max_running_requests reports the requested value while the mamba state cache caps
    # the effective one at 16.
    if 'context_length' in info and int(info['context_length']) != int(VLLM_MAX_MODEL_LEN):
        raise AssertionError(
            f'context_length {info["context_length"]} != requested {VLLM_MAX_MODEL_LEN}'
        )


def start_inference_server() -> None:
    if SERVER_BACKEND == 'sglang':
        start_sglang_server()
    else:
        start_vllm_server()


def start_vllm_server() -> None:
    install_vllm_wheelhouse()
    VLLM_SERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
    VLLM_SERVER_PID.unlink(missing_ok=True)
    log_handle = VLLM_SERVER_LOG.open('w', encoding='utf-8')
    cmd = [
        sys.executable,
        '-m',
        'vllm.entrypoints.openai.api_server',
        '--model',
        str(MODEL_PATH),
        '--served-model-name',
        SERVED_MODEL_NAME,
        '--host',
        VLLM_HOST,
        '--port',
        str(VLLM_PORT),
        '--tensor-parallel-size',
        str(VLLM_TENSOR_PARALLEL_SIZE),
        '--enable-auto-tool-choice',
        '--tool-call-parser',
        'qwen3_coder',
        '--generation-config',
        'vllm',
        '--enable-prefix-caching',
        '--default-chat-template-kwargs',
        '{"preserve_thinking": true, "reasoning_effort": "xhigh"}',
        '--reasoning-parser',
        'qwen3',
        '--max-model-len',
        str(VLLM_MAX_MODEL_LEN),
    ]
    if str(VLLM_SPECULATIVE_CONFIG).strip():
        cmd += ['--speculative-config', str(VLLM_SPECULATIVE_CONFIG).strip()]
    print('Starting vLLM OpenAI server:', ' '.join(cmd), flush=True)
    process = subprocess.Popen(cmd, env=vllm_env(), stdout=log_handle, stderr=subprocess.STDOUT, text=True)
    VLLM_SERVER_PID.write_text(str(process.pid), encoding='utf-8')
    wait_for_vllm_server()


def smoke_request(messages, tools=None, tool_choice='auto', timeout=300) -> dict:
    # Sampling MUST match what the agent sends. A previous revision hardcoded
    # temperature 0.0 and the very first smoke request -- "what is 17 times 3?" --
    # ran away into a repetition loop: 46,118 generated tokens and still going when the
    # socket timed out, with the speculative decoder reporting accept_rate 1.00 the whole
    # way, which is what perfectly-predictable repeated text looks like. Greedy decoding
    # is a known repetition trigger for this model family (sglang #35723). Production
    # runs temperature 1.0, so the smoke test does too.
    payload = {
        'model': SERVED_MODEL_NAME,
        'messages': messages,
        'stream': False,
        'temperature': SMOKE_TEMPERATURE,
        'top_p': SMOKE_TOP_P,
        'top_k': SMOKE_TOP_K,
        # Bounded, unlike production, purely so a runaway can never hang setup. Truncation
        # is therefore EXPECTED and is treated as inconclusive below, never as failure.
        'max_tokens': SMOKE_MAX_TOKENS,
        'chat_template_kwargs': {'enable_thinking': False},
    }
    if tools:
        payload['tools'] = tools
        payload['tool_choice'] = tool_choice
    response = request_json(f'{VLLM_BASE_URL}/chat/completions', payload=payload, timeout=timeout)
    choice = response['choices'][0]
    message = choice['message']
    usage = response.get('usage') or {}
    return {
        'content': (message.get('content') or '').strip(),
        'reasoning': (message.get('reasoning_content') or '').strip(),
        'tool_calls': message.get('tool_calls') or [],
        'finish_reason': choice.get('finish_reason'),
        'usage': usage,
        'image_tokens': (usage.get('prompt_tokens_details') or {}).get('image_tokens') or 0,
        'truncated': choice.get('finish_reason') == 'length',
    }


def run_api_smoke_test() -> None:
    # Scope, learned the hard way: this guards SERVER HEALTH, not model capability.
    # Two production failures are invisible to /health -- a tool parser that never fires
    # (the agent acts ONLY through tool calls, so every game scores 0) and a dropped image
    # part (the agent plays blind). Those are the only two things worth aborting for.
    #
    # Anything that measures how WELL the model answers belongs in a warning. A previous
    # revision asserted on a trivial arithmetic prompt and killed a submission when the
    # model, with thinking disabled, answered '5' to 17x3 in two tokens -- a capability
    # artifact of a non-production configuration, not a broken server. The NVFP4 weights
    # were independently shown coherent by kernel sglang-matrix-probe row 2.
    print('\n' + '=' * 88, flush=True)
    print('INFERENCE SERVER SMOKE TEST', flush=True)
    result = smoke_request([{'role': 'user', 'content': 'What is 17 times 3? Reply with the number.'}])
    answer = result['content'] or result['reasoning']
    print(f"text   : finish={result['finish_reason']} content={result['content'][:150]!r} "
          f"reasoning_chars={len(result['reasoning'])} usage={result['usage']}", flush=True)
    if '51' not in answer:
        # Reported, never fatal. An FP4 or KV misconfiguration does show up as fluent
        # nonsense, but so does a perfectly healthy model asked to do mental arithmetic
        # with reasoning switched off, and the two are not distinguishable from here.
        print(f'WARNING: arithmetic answer was {answer[:120]!r}, expected 51. Worth a look '
              f'if scores are poor, but not by itself evidence of a misconfigured server.',
              flush=True)
    tools = [{
        'type': 'function',
        'function': {
            'name': 'take_action',
            'description': 'Take one action in the ARC-AGI-3 game.',
            'parameters': {
                'type': 'object',
                'properties': {'action': {'type': 'string', 'description': 'e.g. ACTION3'}},
                'required': ['action'],
            },
        },
    }]
    messages = [
        {'role': 'system', 'content': 'You play ARC-AGI-3. Always act via the take_action tool.'},
        {'role': 'user', 'content': 'The level just started. Take ACTION3 now.'},
    ]
    result = smoke_request(messages, tools=tools)
    print(f"tool   : finish={result['finish_reason']} "
          f"tool_calls={json.dumps(result['tool_calls'])[:260]} usage={result['usage']}", flush=True)
    if not result['tool_calls'] and not result['truncated']:
        # Choosing prose over a tool is the model's prerogative under tool_choice 'auto',
        # so that alone proves nothing. Force the issue before calling it broken.
        print('No tool call under tool_choice=auto; retrying with tool_choice=required.', flush=True)
        try:
            result = smoke_request(messages, tools=tools, tool_choice='required')
            print(f"tool!  : finish={result['finish_reason']} "
                  f"tool_calls={json.dumps(result['tool_calls'])[:260]}", flush=True)
            forced = True
        except Exception as exc:
            print(f'tool_choice=required not usable ({exc!r}); treating as inconclusive.', flush=True)
            forced = False
        if forced and not result['tool_calls'] and not result['truncated']:
            raise AssertionError(
                'Server produced no tool_calls even under tool_choice=required. The agent '
                'drives the game entirely through tool calls, so this would score 0 on '
                f'every game. content={result["content"][:400]!r}'
            )
    result = smoke_request([{'role': 'user', 'content': [
        {'type': 'text', 'text': 'Reply with one word describing this image.'},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + SMOKE_TEST_PNG_B64}},
    ]}])
    print(f"image  : finish={result['finish_reason']} image_tokens={result['image_tokens']} "
          f"content={result['content'][:150]!r} usage={result['usage']}", flush=True)
    # The one hard assert. image_tokens is counted at PREFILL, so it is known before a
    # single token is generated: conclusive regardless of truncation, sampling or model
    # skill. Non-empty content would not be -- a server that dropped the image still
    # answers, just blindly.
    assert result['image_tokens'] > 0, (
        f'The image part produced no image tokens ({result["usage"]}), so the server is not '
        f'actually looking at the grid and the agent would play blind.'
    )
    print('=' * 88 + '\n', flush=True)


print(f'Inference backend: {SERVER_BACKEND}', flush=True)
print(f'Runtime dataset path: {WHEELHOUSE}', flush=True)
print(f'Qwen model path: {MODEL_PATH}', flush=True)
assert_expected_cuda_gpu()
missing = [str(path) for path in (WHEELHOUSE, MODEL_PATH) if not path.exists()]
if missing:
    raise FileNotFoundError('Missing attached dataset path(s): ' + ', '.join(missing))
start_inference_server()
run_api_smoke_test()
setup_env = {
    'LOCAL_ANALYZER_BASE_URL': VLLM_BASE_URL,
    'OPENAI_BASE_URL': VLLM_BASE_URL,
    'LOCAL_ANALYZER_PROVIDER': __LOCAL_ANALYZER_PROVIDER__,
    'OPENAI_PROVIDER': __LOCAL_ANALYZER_PROVIDER__,
    'LOCAL_ANALYZER_MODEL_ID': SERVED_MODEL_NAME,
    'INFERENCE_ANALYZER_MODEL': SERVED_MODEL_NAME,
    'LOCAL_ANALYZER_APP_NAME': __LOCAL_ANALYZER_APP_NAME__,
    'LOCAL_ANALYZER_CONTEXT_WINDOW': str(ANALYZER_CONTEXT_WINDOW),
    'LOCAL_ANALYZER_MAX_OUTPUT': __LOCAL_ANALYZER_MAX_OUTPUT__,
    'LOCAL_ANALYZER_TOOL_STEPS': __LOCAL_ANALYZER_TOOL_STEPS__,
    'LOCAL_ANALYZER_TOOL_TIMEOUT': __LOCAL_ANALYZER_TOOL_TIMEOUT__,
    'LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS': __LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS__,
    'LOCAL_ANALYZER_YIELD_SECONDS': __LOCAL_ANALYZER_YIELD_SECONDS__,
    'LOCAL_ANALYZER_TEMPERATURE': __LOCAL_ANALYZER_TEMPERATURE__,
    'LOCAL_ANALYZER_TOP_P': __LOCAL_ANALYZER_TOP_P__,
    'LOCAL_ANALYZER_TOP_K': __LOCAL_ANALYZER_TOP_K__,
    'LOCAL_ANALYZER_ENABLE_THINKING': __LOCAL_ANALYZER_ENABLE_THINKING__,
    'MULTIMODAL_CONTEXT': __MULTIMODAL_CONTEXT__,
    'MULTIMODAL_UPSCALE': __MULTIMODAL_UPSCALE__,
}
if SERVER_BACKEND != 'sglang':
    # The notebook splices every PYTHONPATH entry into the HARNESS process's own sys.path,
    # so this is only safe for the vLLM tree, which the agent never imports from anyway.
    # Doing it for SGLang would put its pinned torch/numpy/transformers ahead of Kaggle's
    # inside the process that runs the whole benchmark.
    setup_env.update({
        'USE_TF': '0',
        'TRANSFORMERS_NO_TF': '1',
        'TRANSFORMERS_NO_TORCHVISION': '1',
        'VLLM_NO_USAGE_STATS': '1',
        'PYTHONPATH': str(SITE_PACKAGES) + os.pathsep + os.environ.get('PYTHONPATH', ''),
    })
setup_env_path = Path(os.environ['TAAF_KAGGLE_SETUP_ENV'])
existing_setup_env = {}
if setup_env_path.exists():
    existing_setup_env = json.loads(setup_env_path.read_text(encoding='utf-8'))
    if not isinstance(existing_setup_env, dict):
        raise RuntimeError('TAAF_KAGGLE_SETUP_ENV must contain a JSON object.')
existing_setup_env.update(setup_env)
setup_env_path.write_text(json.dumps(existing_setup_env, indent=2), encoding='utf-8')
"""

_DUCK_VLLM_TEARDOWN_SCRIPT = r"""import os
import shutil
import signal
import time
from pathlib import Path

WORKING_DIR = Path(os.environ['TAAF_KAGGLE_WORKING_DIR'])
pid_path = WORKING_DIR / 'vllm-openai-server.pid'
pgid_path = WORKING_DIR / 'inference-server.pgid'
site_packages = WORKING_DIR / 'vllm-site-packages'


def signal_server(pid, sig):
    # SGLang forks scheduler and detokenizer children that survive a SIGTERM to the
    # launcher and keep holding the GPU, so it records its process group and we signal
    # that instead. The pgid file is absent for vLLM, which needs no group signalling
    # (and where the group would be the notebook's own).
    if pgid_path.exists():
        try:
            os.killpg(int(pgid_path.read_text(encoding='utf-8').strip()), sig)
            return
        except Exception:
            pass
    os.kill(pid, sig)


if pid_path.exists():
    try:
        pid = int(pid_path.read_text(encoding='utf-8').strip())
        print('Stopping inference server', flush=True)
        signal_server(pid, signal.SIGTERM)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except OSError:
                break
            time.sleep(1)
        else:
            signal_server(pid, signal.SIGKILL)
    except Exception as exc:
        print(f'Could not stop inference server cleanly: {exc!r}', flush=True)
    pid_path.unlink(missing_ok=True)
    pgid_path.unlink(missing_ok=True)
shutil.rmtree(site_packages, ignore_errors=True)
print(f'Removed temporary vLLM install at {site_packages}', flush=True)
"""
