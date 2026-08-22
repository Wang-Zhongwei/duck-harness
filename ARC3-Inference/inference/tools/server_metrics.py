"""Parse vLLM server logs into serving metrics for the score file.

Every trial directory (``runs/<run>/passes/<group>``) holds the ``server.log``
of the vLLM instance that served that group's games, so serving behavior is
recoverable after the fact without instrumenting the harness.

Two shapes are parsed:

- The periodic engine stat line, emitted roughly every 10s::

    Engine 000: Avg prompt throughput: 691.4 tokens/s, Avg generation
    throughput: 42.9 tokens/s, Running: 4 reqs, Waiting: 0 reqs, GPU KV cache
    usage: 3.8%, Prefix cache hit rate: 54.2%, MM cache hit rate: 88.9%

- The ``non-default args`` line written once at startup, which records the
  engine configuration the run actually served with (context length, image
  cap, cache settings) rather than what the config file asked for.

Aggregation semantics, which the numbers are meaningless without:

- Throughput fields are *averages over the preceding logging interval*, not
  cumulative counters. Idle intervals report 0.0 and would drag a naive mean
  toward zero, so throughput is aggregated over **active** samples only --
  those with a running/waiting request or non-zero throughput. ``duty_cycle``
  reports what fraction of samples were active, so an active-only mean is
  never read without knowing how much of the run it covers.
- Throughput and queue depth are **fleet** figures: a multi-GPU run has one
  server per trial directory, all on one node clock, so samples are binned by
  timestamp into 10 s intervals and **summed across servers** within each
  interval. A 2-GPU run therefore reports ~2x a single server, and
  ``running_requests.max`` reads as the whole run's concurrency, not one
  GPU's share. ``per_server`` keeps each log's own mean so the split is still
  visible, including throughput, cache, KV, and queue distributions. Run-to-run
  comparisons with different server counts must equal-weight the per-server
  means rather than compare fleet totals. Earlier score files (no ``aggregation`` key) pooled per-server
  samples into one distribution instead, which reads as a single-server
  figure and is not comparable to this one on multi-GPU runs.
- Cache hit rates are vLLM sliding-window metrics over recent queries, not
  run-cumulative. They rise and fall during a run, so ``mean``/``min``/``max``
  describe the distribution and ``last`` gives the final window. There is no
  single "the" hit rate for a run.
- Queue depth (``running``/``waiting``) is instantaneous per sample. Sustained
  ``waiting > 0`` means the server was the bottleneck, which is the main thing
  that invalidates a throughput comparison between runs.
"""
from __future__ import annotations

import ast
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

SERVER_LOG_NAME = "server.log"
STDERR_LOG_NAME = "stderr.log"

_ENGINE_STAT_RE = re.compile(
    r"Avg prompt throughput:\s*(?P<prompt_tps>[0-9.]+)\s*tokens/s.*?"
    r"Avg generation throughput:\s*(?P<gen_tps>[0-9.]+)\s*tokens/s.*?"
    r"Running:\s*(?P<running>\d+)\s*reqs.*?"
    r"Waiting:\s*(?P<waiting>\d+)\s*reqs"
    r"(?:.*?GPU KV cache usage:\s*(?P<kv>[0-9.]+)\s*%)?"
    r"(?:.*?Prefix cache hit rate:\s*(?P<prefix>[0-9.]+)\s*%)?"
    r"(?:.*?MM cache hit rate:\s*(?P<mm>[0-9.]+)\s*%)?"
)
_TIMESTAMP_RE = re.compile(r"\b(?P<ts>\d{2}-\d{2} \d{2}:\d{2}:\d{2})\b")
_NON_DEFAULT_ARGS_RE = re.compile(r"non-default args:\s*(?P<args>\{.*\})\s*$")
_HTTP_STATUS_RE = re.compile(r'"(?:POST|GET) [^"]*" (?P<status>\d{3})')
_STARTUP_COMPLETE_RE = re.compile(r"Application startup complete")
_PREEMPTION_RE = re.compile(r"preempt", re.IGNORECASE)
_READ_TIMEOUT_RE = re.compile(r"Read timed out")
_ANALYZER_FAILED_RE = re.compile(r"analyzer request failed|analyzer failed")

# Engine-config keys worth carrying into the score file. Everything else in
# the startup line is either a path or a default that never varies here.
_ENGINE_CONFIG_KEYS = (
    "max_model_len",
    "limit_mm_per_prompt",
    "enable_prefix_caching",
    "kv_cache_dtype",
    "max_num_seqs",
    "served_model_name",
    "tensor_parallel_size",
    "gpu_memory_utilization",
    "reasoning_parser",
    "tool_call_parser",
)


@dataclass
class _Sample:
    prompt_tps: float
    gen_tps: float
    running: int
    waiting: int
    kv_usage: float | None
    prefix_hit: float | None
    mm_hit: float | None
    timestamp: datetime | None = None

    @property
    def active(self) -> bool:
        return bool(self.running or self.waiting or self.prompt_tps or self.gen_tps)


@dataclass
class TrialServerMetrics:
    """Serving metrics parsed from one trial's ``server.log``."""

    log_path: Path | None = None
    samples: list[_Sample] = field(default_factory=list)
    engine_config: dict[str, Any] = field(default_factory=dict)
    status_counts: dict[str, int] = field(default_factory=dict)
    preemptions: int = 0
    startup_seconds: float | None = None
    read_timeouts: int = 0
    analyzer_failures: int = 0

    @property
    def found(self) -> bool:
        return self.log_path is not None


def _stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _rate_stats(values: list[float]) -> dict[str, float] | None:
    stats = _stats(values)
    if stats is None:
        return None
    # ``last`` is the final sliding window, i.e. the rate as the run ended.
    stats["last"] = values[-1]
    return stats


def _parse_timestamp(line: str) -> datetime | None:
    match = _TIMESTAMP_RE.search(line)
    if match is None:
        return None
    try:
        # vLLM omits the year; only deltas within one log are ever taken.
        return datetime.strptime(f"1900-{match.group('ts')}", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _parse_engine_config(line: str) -> dict[str, Any]:
    match = _NON_DEFAULT_ARGS_RE.search(line.rstrip())
    if match is None:
        return {}
    try:
        parsed = ast.literal_eval(match.group("args"))
    except (ValueError, SyntaxError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {key: parsed[key] for key in _ENGINE_CONFIG_KEYS if key in parsed}


def parse_server_log(path: Path) -> TrialServerMetrics:
    metrics = TrialServerMetrics(log_path=path)
    first_timestamp: datetime | None = None
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        metrics.log_path = None
        return metrics

    last_timestamp: datetime | None = None
    for line in lines:
        timestamp = _parse_timestamp(line)
        if timestamp is not None:
            last_timestamp = timestamp
            if first_timestamp is None:
                first_timestamp = timestamp

        if not metrics.engine_config and "non-default args:" in line:
            metrics.engine_config = _parse_engine_config(line)

        if metrics.startup_seconds is None and _STARTUP_COMPLETE_RE.search(line):
            # uvicorn's "Application startup complete." carries no timestamp,
            # so fall back to the most recent timestamped line -- accurate to
            # within one log line.
            if last_timestamp is not None and first_timestamp is not None:
                delta = (last_timestamp - first_timestamp).total_seconds()
                if delta >= 0:
                    metrics.startup_seconds = delta

        status_match = _HTTP_STATUS_RE.search(line)
        if status_match is not None:
            status = status_match.group("status")
            metrics.status_counts[status] = metrics.status_counts.get(status, 0) + 1

        if _PREEMPTION_RE.search(line):
            metrics.preemptions += 1

        stat_match = _ENGINE_STAT_RE.search(line)
        if stat_match is not None:
            metrics.samples.append(
                _Sample(
                    prompt_tps=float(stat_match.group("prompt_tps")),
                    gen_tps=float(stat_match.group("gen_tps")),
                    running=int(stat_match.group("running")),
                    waiting=int(stat_match.group("waiting")),
                    kv_usage=_opt_float(stat_match.group("kv")),
                    prefix_hit=_opt_float(stat_match.group("prefix")),
                    mm_hit=_opt_float(stat_match.group("mm")),
                    timestamp=timestamp,
                )
            )
    return metrics


def _opt_float(raw: str | None) -> float | None:
    return None if raw is None else float(raw)


def parse_client_errors(path: Path) -> tuple[int, int]:
    """``(read_timeouts, analyzer_failures)`` from a trial's ``stderr.log``.

    Client-side, but the clearest signal that the server was saturated: a
    read timeout means the analyzer gave up waiting on a completion.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0, 0
    return len(_READ_TIMEOUT_RE.findall(text)), len(_ANALYZER_FAILED_RE.findall(text))


def collect_trial_metrics(trial_dir: Path) -> TrialServerMetrics:
    log_path = trial_dir / SERVER_LOG_NAME
    metrics = parse_server_log(log_path) if log_path.exists() else TrialServerMetrics()
    metrics.read_timeouts, metrics.analyzer_failures = parse_client_errors(trial_dir / STDERR_LOG_NAME)
    return metrics


_INTERVAL_SECONDS = 10


@dataclass
class _Interval:
    """One logging interval with every server's contribution summed."""

    prompt_tps: float = 0.0
    gen_tps: float = 0.0
    running: int = 0
    waiting: int = 0


def _fleet_intervals(found: list[TrialServerMetrics]) -> list[_Interval] | None:
    """Bin active samples by timestamp and sum across servers per bin.

    Within one bin a server contributes at most once (its latest sample), so
    cadence jitter that lands two of one server's lines in a bin cannot double
    count. Returns None when any sample is untimestamped, since the bins would
    then be meaningless.
    """
    bins: dict[int, dict[int, _Sample]] = {}
    for server_index, item in enumerate(found):
        for sample in item.samples:
            if not sample.active:
                continue
            if sample.timestamp is None:
                return None
            key = int(sample.timestamp.timestamp()) // _INTERVAL_SECONDS
            bins.setdefault(key, {})[server_index] = sample
    intervals: list[_Interval] = []
    for key in sorted(bins):
        interval = _Interval()
        for sample in bins[key].values():
            interval.prompt_tps += sample.prompt_tps
            interval.gen_tps += sample.gen_tps
            interval.running += sample.running
            interval.waiting += sample.waiting
        intervals.append(interval)
    return intervals


def _summarize_server(item: TrialServerMetrics) -> dict[str, Any]:
    """Build one server's complete metric block for normalized comparisons."""
    samples = item.samples
    active = [sample for sample in samples if sample.active]
    return {
        "log": str(item.log_path),
        "sample_count": len(samples),
        "active_sample_count": len(active),
        "duty_cycle": (len(active) / len(samples)) if samples else None,
        "prompt_throughput_tokens_per_s": _stats([sample.prompt_tps for sample in active]),
        "generation_throughput_tokens_per_s": _stats([sample.gen_tps for sample in active]),
        "prefix_cache_hit_rate_pct": _rate_stats(
            [sample.prefix_hit for sample in samples if sample.prefix_hit is not None]
        ),
        "mm_cache_hit_rate_pct": _rate_stats(
            [sample.mm_hit for sample in samples if sample.mm_hit is not None]
        ),
        "gpu_kv_cache_usage_pct": _stats(
            [sample.kv_usage for sample in samples if sample.kv_usage is not None]
        ),
        "running_requests": _stats([float(sample.running) for sample in active]),
        "waiting_requests": _stats([float(sample.waiting) for sample in active]),
        "saturated_sample_fraction": (
            sum(1 for sample in active if sample.waiting > 0) / len(active)
            if active
            else None
        ),
        "startup_seconds": item.startup_seconds,
    }


def summarize(metrics: list[TrialServerMetrics]) -> dict[str, Any]:
    """Fold per-trial metrics into one payload block."""
    found = [item for item in metrics if item.found]
    samples = [sample for item in found for sample in item.samples]
    active = [sample for sample in samples if sample.active]
    intervals = _fleet_intervals(found)
    if intervals is None:
        # No timestamps to align on: fall back to treating each sample as its own
        # interval, which is exact for one server and a per-server pool otherwise.
        intervals = [
            _Interval(sample.prompt_tps, sample.gen_tps, sample.running, sample.waiting)
            for sample in active
        ]
        aggregation = "pooled"
    else:
        aggregation = "fleet"

    status_counts: dict[str, int] = {}
    for item in found:
        for status, count in item.status_counts.items():
            status_counts[status] = status_counts.get(status, 0) + count
    total_requests = sum(status_counts.values())
    non_2xx = sum(count for status, count in status_counts.items() if not status.startswith("2"))

    startup = [item.startup_seconds for item in found if item.startup_seconds is not None]

    payload: dict[str, Any] = {
        "logs_found": len(found),
        "logs_expected": len(metrics),
        "sample_count": len(samples),
        "active_sample_count": len(active),
        "duty_cycle": (len(active) / len(samples)) if samples else None,
        "aggregation": aggregation,
        "interval_count": len(intervals),
        "prompt_throughput_tokens_per_s": _stats([interval.prompt_tps for interval in intervals]),
        "generation_throughput_tokens_per_s": _stats([interval.gen_tps for interval in intervals]),
        "per_server": [_summarize_server(item) for item in found],
        "prefix_cache_hit_rate_pct": _rate_stats(
            [sample.prefix_hit for sample in samples if sample.prefix_hit is not None]
        ),
        "mm_cache_hit_rate_pct": _rate_stats(
            [sample.mm_hit for sample in samples if sample.mm_hit is not None]
        ),
        "gpu_kv_cache_usage_pct": _stats(
            [sample.kv_usage for sample in samples if sample.kv_usage is not None]
        ),
        "running_requests": _stats([float(interval.running) for interval in intervals]),
        "waiting_requests": _stats([float(interval.waiting) for interval in intervals]),
        "saturated_sample_fraction": (
            sum(1 for interval in intervals if interval.waiting > 0) / len(intervals)
            if intervals
            else None
        ),
        "requests": {
            "total": total_requests,
            "by_status": dict(sorted(status_counts.items())),
            "non_2xx": non_2xx,
        },
        "client_errors": {
            "read_timeouts": sum(item.read_timeouts for item in metrics),
            "analyzer_request_failures": sum(item.analyzer_failures for item in metrics),
        },
        "preemptions": sum(item.preemptions for item in found),
        "startup_seconds": _stats([float(value) for value in startup]),
    }

    engine_configs = [item.engine_config for item in found if item.engine_config]
    if engine_configs:
        shared = {
            key: value
            for key, value in engine_configs[0].items()
            if all(config.get(key) == value for config in engine_configs)
        }
        payload["engine_config"] = shared
    return payload


def build_server_block(trial_dirs: list[Path]) -> dict[str, Any]:
    """Aggregate server metrics for the score file."""
    per_trial_metrics = [collect_trial_metrics(trial_dir) for trial_dir in trial_dirs]
    block = summarize(per_trial_metrics)
    block["notes"] = (
        "Throughput and queue depth are fleet totals: per-server samples are binned by "
        "timestamp into 10 s intervals and summed across servers (see per_server for the "
        "split). Averaged over active intervals only (see duty_cycle); "
        "run-to-run serving comparisons should equal-weight per_server means when server "
        "counts differ. "
        "cache hit rates are vLLM sliding-window values, so mean/min/max describe the "
        "distribution and last is the final window."
    )
    return block
