"""Parsing vLLM server logs into the score file's serving metrics."""

from pathlib import Path

from inference.tools.server_metrics import (
    build_server_block,
    collect_trial_metrics,
    parse_server_log,
    summarize,
)

_STAT = (
    "(APIServer pid=1) INFO 07-21 22:09:{sec} [loggers.py:273] Engine 000: "
    "Avg prompt throughput: {prompt} tokens/s, Avg generation throughput: {gen} tokens/s, "
    "Running: {running} reqs, Waiting: {waiting} reqs, GPU KV cache usage: 3.8%, "
    "Prefix cache hit rate: {prefix}%, MM cache hit rate: {mm}%"
)

_ARGS_LINE = (
    "(APIServer pid=1) INFO 07-21 22:07:05 [api_utils.py:273] non-default args: "
    "{'model_tag': '/x', 'trust_remote_code': True, 'max_model_len': 49152, "
    "'served_model_name': ['qwen'], 'kv_cache_dtype': 'fp8', 'enable_prefix_caching': True, "
    "'limit_mm_per_prompt': {'image': 12, 'video': 0}, 'max_num_seqs': 512}"
)


def _write_log(tmp_path: Path, *, lines: list[str], name: str = "server.log") -> Path:
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _stat(sec: str, prompt: float, gen: float, running: int, waiting: int, prefix: float, mm: float) -> str:
    return _STAT.format(sec=sec, prompt=prompt, gen=gen, running=running, waiting=waiting, prefix=prefix, mm=mm)


def test_parses_engine_stat_samples(tmp_path: Path) -> None:
    log = _write_log(
        tmp_path,
        lines=[
            _stat("10", 100.0, 10.0, 2, 0, 50.0, 90.0),
            _stat("20", 300.0, 30.0, 4, 1, 60.0, 95.0),
        ],
    )
    metrics = parse_server_log(log)
    assert len(metrics.samples) == 2
    assert metrics.samples[0].prompt_tps == 100.0
    assert metrics.samples[1].waiting == 1
    assert metrics.samples[1].mm_hit == 95.0


def test_throughput_excludes_idle_intervals_but_reports_duty_cycle(tmp_path: Path) -> None:
    """Idle intervals report 0.0 and would drag a naive mean toward zero."""
    log = _write_log(
        tmp_path,
        lines=[
            _stat("10", 0.0, 0.0, 0, 0, 50.0, 90.0),  # idle
            _stat("20", 100.0, 10.0, 2, 0, 50.0, 90.0),
            _stat("30", 300.0, 30.0, 4, 0, 50.0, 90.0),
        ],
    )
    block = summarize([parse_server_log(log)])
    assert block["prompt_throughput_tokens_per_s"]["mean"] == 200.0  # not 133.3
    assert block["active_sample_count"] == 2
    assert block["sample_count"] == 3
    assert block["duty_cycle"] == 2 / 3


def test_cache_rates_report_distribution_and_last(tmp_path: Path) -> None:
    log = _write_log(
        tmp_path,
        lines=[
            _stat("10", 100.0, 10.0, 1, 0, 40.0, 80.0),
            _stat("20", 100.0, 10.0, 1, 0, 80.0, 90.0),
            _stat("30", 100.0, 10.0, 1, 0, 60.0, 100.0),
        ],
    )
    block = summarize([parse_server_log(log)])
    prefix = block["prefix_cache_hit_rate_pct"]
    assert prefix["mean"] == 60.0
    assert prefix["min"] == 40.0
    assert prefix["max"] == 80.0
    assert prefix["last"] == 60.0  # final window, not the max
    assert block["mm_cache_hit_rate_pct"]["last"] == 100.0


def test_saturation_and_request_counts(tmp_path: Path) -> None:
    log = _write_log(
        tmp_path,
        lines=[
            _stat("10", 100.0, 10.0, 4, 0, 50.0, 90.0),
            _stat("20", 100.0, 10.0, 4, 3, 50.0, 90.0),
            '(APIServer pid=1) INFO:     127.0.0.1:1 - "POST /v1/chat/completions HTTP/1.1" 200 OK',
            '(APIServer pid=1) INFO:     127.0.0.1:2 - "POST /v1/chat/completions HTTP/1.1" 400 Bad Request',
        ],
    )
    block = summarize([parse_server_log(log)])
    assert block["saturated_sample_fraction"] == 0.5
    assert block["waiting_requests"]["max"] == 3.0
    assert block["requests"] == {"total": 2, "by_status": {"200": 1, "400": 1}, "non_2xx": 1}


def test_engine_config_captures_served_settings(tmp_path: Path) -> None:
    metrics = parse_server_log(_write_log(tmp_path, lines=[_ARGS_LINE]))
    assert metrics.engine_config["max_model_len"] == 49152
    assert metrics.engine_config["limit_mm_per_prompt"] == {"image": 12, "video": 0}
    assert metrics.engine_config["kv_cache_dtype"] == "fp8"
    # Paths are noise; they must not leak into the score file.
    assert "model_tag" not in metrics.engine_config


def test_engine_config_only_kept_when_trials_agree(tmp_path: Path) -> None:
    same = parse_server_log(_write_log(tmp_path, lines=[_ARGS_LINE]))
    other = parse_server_log(
        _write_log(tmp_path, lines=[_ARGS_LINE.replace("49152", "32768")], name="other.log")
    )
    assert summarize([same, same])["engine_config"]["max_model_len"] == 49152
    assert "max_model_len" not in summarize([same, other])["engine_config"]


def test_startup_seconds_uses_last_timestamp_before_untimestamped_marker(tmp_path: Path) -> None:
    log = _write_log(
        tmp_path,
        lines=[
            "(APIServer pid=1) INFO 07-21 22:07:05 [api_utils.py:339] starting",
            "(APIServer pid=1) INFO 07-21 22:09:45 [api_utils.py:339] loaded",
            "(APIServer pid=1) INFO:     Application startup complete.",
        ],
    )
    assert parse_server_log(log).startup_seconds == 160.0


def test_client_errors_come_from_stderr(tmp_path: Path) -> None:
    (tmp_path / "server.log").write_text(_stat("10", 1.0, 1.0, 1, 0, 50.0, 90.0) + "\n", encoding="utf-8")
    (tmp_path / "stderr.log").write_text(
        "analyzer request failed at action 5: HTTPConnectionPool(...): Read timed out.\n"
        "analyzer request failed at action 9: HTTPConnectionPool(...): Read timed out.\n",
        encoding="utf-8",
    )
    metrics = collect_trial_metrics(tmp_path)
    assert metrics.read_timeouts == 2
    assert metrics.analyzer_failures == 2
    assert summarize([metrics])["client_errors"] == {
        "read_timeouts": 2,
        "analyzer_request_failures": 2,
    }


def test_missing_log_is_reported_not_fatal(tmp_path: Path) -> None:
    block = summarize([collect_trial_metrics(tmp_path)])
    assert block["logs_found"] == 0
    assert block["logs_expected"] == 1
    assert block["sample_count"] == 0
    assert block["prompt_throughput_tokens_per_s"] is None
    assert block["duty_cycle"] is None


def test_build_server_block_summarizes_trials(tmp_path: Path) -> None:
    trial = tmp_path / "0"
    trial.mkdir()
    _write_log(trial, lines=[_stat("10", 100.0, 10.0, 1, 0, 50.0, 90.0), _ARGS_LINE])
    block = build_server_block([trial])

    assert "per_trial" not in block
    assert block["sample_count"] == 1
    assert block["logs_found"] == 1
    assert "notes" in block
