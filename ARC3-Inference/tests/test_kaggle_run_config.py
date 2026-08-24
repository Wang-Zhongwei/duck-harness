"""Per-run Kaggle settings live in the notebook's RUN_CONFIG cell, not the source dataset.

Before this, context window, concurrency, per-game runtime and every analyzer knob were
baked into setup_commands.json and the pickled solver, so every config-only push
re-uploaded an identical-looking notebook over a silently different dataset version.
The contract now: the rendered setup script and the pickled benchmark are the same
bytes for any configuration; the notebook carries the values and applies them.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

import taaf.deploy_kaggle as deploy_kaggle
from inference.framework import kaggle
from inference.framework.solver import HarnessSolver

_RUN_ENV_A = {
    "KAGGLE_MAX_MODEL_LEN": "65536",
    "LOCAL_ANALYZER_CONTEXT_WINDOW": "32768",
    "LOCAL_ANALYZER_TEMPERATURE": "1.0",
    "KAGGLE_KV_CACHE_DTYPE": "fp8_e4m3",
    "SERVER_SPECULATIVE_CONFIG": '{"method": "mtp", "num_speculative_tokens": 3}',
    "KAGGLE_SERVER_BACKEND": "vllm",
}
_RUN_ENV_B = {
    "KAGGLE_MAX_MODEL_LEN": "131072",
    "LOCAL_ANALYZER_CONTEXT_WINDOW": "65536",
    "LOCAL_ANALYZER_TEMPERATURE": "0.7",
    "KAGGLE_KV_CACHE_DTYPE": "",
    "SERVER_SPECULATIVE_CONFIG": "",
    "KAGGLE_SERVER_BACKEND": "vllm",
}


def _with_env(monkeypatch: pytest.MonkeyPatch, values: dict[str, str]) -> None:
    for name in kaggle.RUN_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_setup_script_is_identical_across_run_configs(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_env(monkeypatch, _RUN_ENV_A)
    script_a = kaggle.duck_kaggle_setup_command()
    _with_env(monkeypatch, _RUN_ENV_B)
    script_b = kaggle.duck_kaggle_setup_command()
    assert script_a == script_b
    # Nothing from the launcher environment leaks in as a literal.
    for value in ("32768", "131072", "num_speculative_tokens", "= 'fp8_e4m3'"):
        assert value not in script_a
    # And no placeholder survived rendering.
    assert not re.findall(r"__[A-Z_]+__", script_a)


def test_run_env_reflects_launcher_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_env(monkeypatch, _RUN_ENV_A)
    env = kaggle.duck_kaggle_run_env()
    assert set(env) == set(kaggle.RUN_ENV_NAMES)
    assert env["KAGGLE_MAX_MODEL_LEN"] == "65536"
    assert env["LOCAL_ANALYZER_CONTEXT_WINDOW"] == "32768"
    assert env["LOCAL_ANALYZER_TEMPERATURE"] == "1.0"
    # Not a launcher knob: it is derived from the solver, so an unpassed call is blank
    # rather than a stale literal claiming a timeout the run never used.
    assert env["ANALYZER_TIMEOUT_HINT"] == ""
    assert kaggle.duck_kaggle_run_env(analyzer_timeout=900.0)["ANALYZER_TIMEOUT_HINT"] == "900"
    # Unset knobs carry the script's own fallback, so the notebook shows the effective value.
    assert env["LOCAL_ANALYZER_TOP_K"] == "20"
    assert env["KAGGLE_ATTENTION_BACKEND"] == ""


def test_run_env_context_window_falls_back_to_max_model_len(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_env(monkeypatch, {"KAGGLE_MAX_MODEL_LEN": "4096"})
    env = kaggle.duck_kaggle_run_env()
    assert env["LOCAL_ANALYZER_CONTEXT_WINDOW"] == "4096"
    _with_env(monkeypatch, {})
    env = kaggle.duck_kaggle_run_env(kaggle.DuckKaggleVllmConfig(max_model_len=2048))
    assert env["KAGGLE_MAX_MODEL_LEN"] == "2048"
    assert env["LOCAL_ANALYZER_CONTEXT_WINDOW"] == "2048"


def test_run_env_rejects_non_local_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_env(monkeypatch, {"LOCAL_ANALYZER_PROVIDER": "openrouter"})
    with pytest.raises(ValueError, match="LOCAL_ANALYZER_PROVIDER"):
        kaggle.duck_kaggle_run_env()


def _exec_setup_script_header(env: dict[str, str]) -> dict[str, Any]:
    """Run the rendered setup script only up to its first function body."""
    command = kaggle.duck_kaggle_setup_command()
    script = command.split("<<'PYSETUP'\n", 1)[1].rsplit("\nPYSETUP", 1)[0]
    header = script.split("\nGPU_NAME_PATTERNS", 1)[0]
    namespace: dict[str, Any] = {}
    saved = {name: os.environ.get(name) for name in [*kaggle.RUN_ENV_NAMES, "TAAF_KAGGLE_WORKING_DIR"]}
    try:
        for name in kaggle.RUN_ENV_NAMES:
            os.environ.pop(name, None)
        os.environ.update(env)
        os.environ["TAAF_KAGGLE_WORKING_DIR"] = "/tmp/taaf-test-working"
        exec(compile(header, "setup-script-header", "exec"), namespace)
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    return namespace


def test_setup_script_reads_run_env_at_runtime() -> None:
    ns = _exec_setup_script_header(_RUN_ENV_A)
    assert ns["VLLM_MAX_MODEL_LEN"] == 65536
    assert ns["ANALYZER_CONTEXT_WINDOW"] == 32768
    assert ns["KV_CACHE_DTYPE"] == "fp8_e4m3"
    assert ns["VLLM_SPECULATIVE_CONFIG"] == _RUN_ENV_A["SERVER_SPECULATIVE_CONFIG"]
    assert ns["SMOKE_TEMPERATURE"] == 1.0
    assert ns["ANALYZER_TIMEOUT_PHRASE"] == "its configured timeout"
    assert (
        _exec_setup_script_header({"ANALYZER_TIMEOUT_HINT": "900"})["ANALYZER_TIMEOUT_PHRASE"]
        == "900s"
    )
    ns = _exec_setup_script_header({})
    assert ns["VLLM_MAX_MODEL_LEN"] == kaggle.DEFAULT_VLLM_MAX_MODEL_LEN
    assert ns["ANALYZER_CONTEXT_WINDOW"] == kaggle.DEFAULT_VLLM_MAX_MODEL_LEN
    assert ns["SERVER_BACKEND"] == "vllm"
    assert ns["TOOL_CALL_PARSER"] == "qwen3_coder"


def test_solver_run_config_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_env(monkeypatch, _RUN_ENV_A)
    solver = HarnessSolver(concurrency=22, max_runtime_s_per_game=6336.0, analyzer_timeout=360.0)
    config = solver.kaggle_run_config
    assert config["solver"] == {
        "concurrency": 22,
        "max_runtime_s_per_game": 6336.0,
        "max_actions_per_game": None,
        "analyzer_timeout": 360.0,
    }
    assert config["env"]["LOCAL_ANALYZER_CONTEXT_WINDOW"] == "32768"
    # The hint beside the solver block must be the solver block's own value.
    assert config["env"]["ANALYZER_TIMEOUT_HINT"] == "360"
    # Survives the JSON round-trip the framework applies.
    assert deploy_kaggle.merge_run_config(config)["solver"]["concurrency"] == 22


def test_merge_run_config_layers_and_rejects_unknown_sections() -> None:
    merged = deploy_kaggle.merge_run_config(
        {"solver": {"concurrency": 16, "analyzer_timeout": 120}, "env": {"A": "1"}},
        {"solver": {"concurrency": 22}},
        None,
    )
    assert merged == {"solver": {"concurrency": 22, "analyzer_timeout": 120}, "env": {"A": "1"}}
    with pytest.raises(ValueError, match="sections"):
        deploy_kaggle.merge_run_config({"games": {}})


@dataclasses.dataclass
class _StubBenchmark:
    solver: Any
    job_dir: Path | None = None


def test_bundle_is_neutral_to_run_config() -> None:
    solver = HarnessSolver(concurrency=22, max_runtime_s_per_game=6336.0, analyzer_timeout=360.0)
    bm = _StubBenchmark(solver=solver, job_dir=Path("runs/20260822_202638_x"))
    run_config = deploy_kaggle.merge_run_config(solver.kaggle_run_config)
    with deploy_kaggle._run_config_neutral(bm, run_config):  # type: ignore[arg-type]
        assert bm.job_dir is None
        assert solver.concurrency == HarnessSolver.concurrency
        assert solver.max_runtime_s_per_game is None
        assert solver.analyzer_timeout == HarnessSolver.analyzer_timeout
    assert bm.job_dir == Path("runs/20260822_202638_x")
    assert solver.concurrency == 22
    assert solver.max_runtime_s_per_game == 6336.0
    assert solver.analyzer_timeout == 360.0


def test_neutralising_unknown_solver_attribute_fails_on_the_launcher() -> None:
    bm = _StubBenchmark(solver=HarnessSolver())
    with pytest.raises(AttributeError, match="no_such_attr"):
        with deploy_kaggle._run_config_neutral(bm, {"solver": {"no_such_attr": 1}}):  # type: ignore[arg-type]
            pass


def _rendered_cells(run_config: dict[str, Any], template: Path) -> list[str]:
    rendered = deploy_kaggle._render_kaggle_notebook(
        run_as_submission=False, dataset_sources=["u/src"], run_config=run_config, template=template
    )
    notebook = json.loads(rendered)
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


@pytest.mark.parametrize(
    "template", [deploy_kaggle._NOTEBOOK_TEMPLATE, deploy_kaggle._SHARE_NOTEBOOK_TEMPLATE], ids=["run", "share"]
)
def test_rendered_notebook_applies_run_config(template: Path, tmp_path: Path) -> None:
    run_config = {
        "solver": {"concurrency": 22, "max_runtime_s_per_game": 6336.0, "max_actions_per_game": None},
        "env": {"LOCAL_ANALYZER_CONTEXT_WINDOW": "32768", "KAGGLE_MAX_MODEL_LEN": "65536"},
    }
    cells = _rendered_cells(run_config, template)
    config_cell = next(cell for cell in cells if cell.startswith("# Run configuration"))
    solver_cell = next(cell for cell in cells if cell.startswith("# Apply RUN_CONFIG"))
    # The values are visible as source, one per line, so a kernel version diff reads as a config diff.
    assert "'concurrency': 22" in config_cell
    assert "'LOCAL_ANALYZER_CONTEXT_WINDOW': '32768'" in config_cell
    assert "__TAAF_RUN_CONFIG__" not in config_cell
    assert "# noqa" not in config_cell
    # Order: config cell before setup commands, solver cell after the pickle load.
    setup_index = next(i for i, cell in enumerate(cells) if 'setup_commands.json' in cell)
    pickle_index = next(i for i, cell in enumerate(cells) if "benchmark_initial.pkl" in cell)
    assert cells.index(config_cell) < setup_index < pickle_index < cells.index(solver_cell)

    # Execute both cells against stubs.
    setup_env_path = tmp_path / "taaf_setup_env.json"
    setup_env_path.write_text("{}")
    written: dict[str, str] = {}
    namespace: dict[str, Any] = {
        "os": os,
        "json": json,
        "SETUP_ENV_PATH": setup_env_path,
        "_write_setup_env_updates": written.update,
    }
    saved = {k: os.environ.get(k) for k in run_config["env"]}
    try:
        exec(compile(config_cell, "run-config-cell", "exec"), namespace)
        assert namespace["RUN_CONFIG"] == run_config
        assert os.environ["LOCAL_ANALYZER_CONTEXT_WINDOW"] == "32768"
        persisted = written or json.loads(setup_env_path.read_text())
        assert persisted["KAGGLE_MAX_MODEL_LEN"] == "65536"
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    solver = HarnessSolver()
    namespace["bm"] = _StubBenchmark(solver=solver)
    exec(compile(solver_cell, "solver-cell", "exec"), namespace)
    assert solver.concurrency == 22
    assert solver.max_runtime_s_per_game == 6336.0
    assert solver.max_actions_per_game is None
    namespace["RUN_CONFIG"] = {"solver": {"bogus": 1}}
    with pytest.raises(AttributeError, match="bogus"):
        exec(compile(solver_cell, "solver-cell", "exec"), namespace)
