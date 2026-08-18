"""Maintain the full score payload registry consumed by the comparison viewer."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REGISTRY_VERSION = 2
DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "inference_score_comparison.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _resolve_source_path(raw_path: str, *, registry_path: Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else registry_path.parent / path


def _display_score_path(score_path: Path, *, registry_path: Path) -> str:
    try:
        return str(score_path.resolve().relative_to(registry_path.parent.resolve()))
    except ValueError:
        return str(score_path.resolve())


def _run_id(score_payload: dict[str, Any], *, score_path: Path) -> str:
    metadata = score_payload.get("metadata")
    if isinstance(metadata, dict):
        experiment_dirs = metadata.get("experiment_dirs")
        if isinstance(experiment_dirs, list) and len(experiment_dirs) == 1:
            name = Path(str(experiment_dirs[0])).name
            if name:
                return name
    if score_path.name == "score.json" and score_path.parent.name:
        return score_path.parent.name
    return score_path.stem


def _empty_registry() -> dict[str, Any]:
    return {
        "version": REGISTRY_VERSION,
        "updated_at": _utc_now(),
        "run_order": [],
        "runs": {},
    }


def _load_score(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) and "score" in payload else None


def _put_run(
    registry: dict[str, Any],
    *,
    score_payload: dict[str, Any],
    score_path: Path,
    registry_path: Path,
    timestamp: str,
) -> str:
    run_id = _run_id(score_payload, score_path=score_path)
    runs = registry.setdefault("runs", {})
    run_order = registry.setdefault("run_order", [])
    existing = runs.get(run_id)
    added_at = existing.get("added_at") if isinstance(existing, dict) else timestamp
    runs[run_id] = {
        "run_id": run_id,
        "score_path": _display_score_path(score_path, registry_path=registry_path),
        "added_at": added_at,
        "updated_at": timestamp,
        "score": score_payload,
    }
    if run_id not in run_order:
        run_order.append(run_id)
    return run_id


def _migrate_v1(payload: dict[str, Any], *, registry_path: Path) -> dict[str, Any]:
    registry = _empty_registry()
    timestamp = _utc_now()
    loaded_paths: set[Path] = set()

    old_runs = payload.get("runs")
    if isinstance(old_runs, dict):
        for side in ("baseline", "candidate"):
            entry = old_runs.get(side)
            if not isinstance(entry, dict):
                continue
            raw_path = entry.get("source_score_json")
            if not raw_path:
                continue
            path = _resolve_source_path(str(raw_path), registry_path=registry_path)
            score_payload = _load_score(path)
            if score_payload is None:
                continue
            _put_run(
                registry,
                score_payload=score_payload,
                score_path=path,
                registry_path=registry_path,
                timestamp=timestamp,
            )
            loaded_paths.add(path.resolve())

    for key in ("baseline_run", "candidate_run"):
        run_id = payload.get(key)
        if not run_id:
            continue
        path = registry_path.parent / "runs" / str(run_id) / "score.json"
        if path.resolve() in loaded_paths:
            continue
        score_payload = _load_score(path)
        if score_payload is None:
            continue
        _put_run(
            registry,
            score_payload=score_payload,
            score_path=path,
            registry_path=registry_path,
            timestamp=timestamp,
        )

    registry["migrated_from_version"] = payload.get("version", 1)
    return registry


def load_score_registry(path: str | Path = DEFAULT_REGISTRY_PATH) -> dict[str, Any]:
    registry_path = Path(path)
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _empty_registry()
    if not isinstance(payload, dict):
        raise ValueError(f"Score registry must contain a JSON object: {registry_path}")
    if payload.get("version") == REGISTRY_VERSION:
        payload.setdefault("run_order", list(payload.get("runs", {})))
        payload.setdefault("runs", {})
        return payload
    return _migrate_v1(payload, registry_path=registry_path)


def save_score_registry(registry: dict[str, Any], path: str | Path = DEFAULT_REGISTRY_PATH) -> Path:
    registry_path = Path(path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry["version"] = REGISTRY_VERSION
    registry["updated_at"] = _utc_now()
    temp_path = registry_path.with_name(f".{registry_path.name}.tmp")
    temp_path.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(registry_path)
    return registry_path


def update_score_registry(
    *,
    score_payload: dict[str, Any],
    score_path: str | Path,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> tuple[Path, str]:
    path = Path(registry_path)
    registry = load_score_registry(path)
    timestamp = _utc_now()
    run_id = _put_run(
        registry,
        score_payload=score_payload,
        score_path=Path(score_path),
        registry_path=path,
        timestamp=timestamp,
    )
    return save_score_registry(registry, path), run_id
