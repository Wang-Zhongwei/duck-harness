"""Maintain the scored-run registry consumed by the comparison viewer."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REGISTRY_VERSION = 3
DEFAULT_MINIMUM_RUN_DATE = "20260810"
PUBLIC_SCORE_SET = "public"
SEMI_PRIVATE_SCORE_SET = "semi_private"
SCORE_SETS = (PUBLIC_SCORE_SET, SEMI_PRIVATE_SCORE_SET)
DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "runs" / "inference_score_comparison.json"
_RUN_DATE_PATTERN = re.compile(r"^(\d{8})")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _resolve_source_path(raw_path: str, *, registry_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidate = registry_path.parent / path
    if candidate.exists():
        return candidate
    # Version-1/2 registries lived one directory above runs/ and stored paths
    # such as runs/<run>/score.json.
    return registry_path.parent.parent / path


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
    if score_path.parent.name:
        return score_path.parent.name
    return score_path.stem


def _empty_registry(*, minimum_run_date: str = DEFAULT_MINIMUM_RUN_DATE) -> dict[str, Any]:
    return {
        "version": REGISTRY_VERSION,
        "updated_at": _utc_now(),
        "minimum_run_date": minimum_run_date,
        "score_sets": {
            PUBLIC_SCORE_SET: {"label": "Public set"},
            SEMI_PRIVATE_SCORE_SET: {"label": "Semi-private set"},
        },
        "run_order": [],
        "runs": {},
        "semi_private_scores": [],
    }


def _load_score(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) and "score" in payload else None


def _payload_date(score_payload: dict[str, Any]) -> str | None:
    metadata = score_payload.get("metadata")
    if not isinstance(metadata, dict):
        return None
    created_at = str(metadata.get("created_at") or metadata.get("generated_at") or "")
    compact = created_at[:10].replace("-", "")
    return compact if len(compact) == 8 and compact.isdigit() else None


def run_is_included(run_id: str, score_payload: dict[str, Any], *, minimum_run_date: str) -> bool:
    """Return whether a score belongs to the configured Qwen 3.8-era window."""
    match = _RUN_DATE_PATTERN.match(run_id)
    run_date = match.group(1) if match else _payload_date(score_payload)
    return run_date is not None and run_date >= minimum_run_date


def _put_score(
    registry: dict[str, Any],
    *,
    score_payload: dict[str, Any],
    score_path: Path,
    registry_path: Path,
    score_set: str,
    timestamp: str,
    run_id: str | None = None,
) -> str:
    if score_set not in SCORE_SETS:
        raise ValueError(f"Unknown score set {score_set!r}; expected one of {SCORE_SETS}.")
    resolved_run_id = run_id or _run_id(score_payload, score_path=score_path)
    runs = registry.setdefault("runs", {})
    run_order = registry.setdefault("run_order", [])
    existing_run = runs.get(resolved_run_id)
    run_added_at = existing_run.get("added_at") if isinstance(existing_run, dict) else timestamp
    scores = dict(existing_run.get("scores", {})) if isinstance(existing_run, dict) else {}
    existing_score = scores.get(score_set)
    score_added_at = existing_score.get("added_at") if isinstance(existing_score, dict) else timestamp
    scores[score_set] = {
        "score_path": _display_score_path(score_path, registry_path=registry_path),
        "added_at": score_added_at,
        "updated_at": timestamp,
        "score": score_payload,
    }
    runs[resolved_run_id] = {
        "run_id": resolved_run_id,
        "added_at": run_added_at,
        "updated_at": timestamp,
        "scores": scores,
    }
    if resolved_run_id not in run_order:
        run_order.append(resolved_run_id)
    return resolved_run_id


def _put_if_included(
    registry: dict[str, Any],
    *,
    score_payload: dict[str, Any],
    score_path: Path,
    registry_path: Path,
    score_set: str,
    timestamp: str,
    run_id: str | None = None,
) -> str:
    resolved_run_id = run_id or _run_id(score_payload, score_path=score_path)
    minimum_run_date = str(registry.get("minimum_run_date") or DEFAULT_MINIMUM_RUN_DATE)
    if run_is_included(resolved_run_id, score_payload, minimum_run_date=minimum_run_date):
        _put_score(
            registry,
            score_payload=score_payload,
            score_path=score_path,
            registry_path=registry_path,
            score_set=score_set,
            timestamp=timestamp,
            run_id=resolved_run_id,
        )
    return resolved_run_id


def _migrate_legacy(payload: dict[str, Any], *, registry_path: Path) -> dict[str, Any]:
    registry = _empty_registry(
        minimum_run_date=str(payload.get("minimum_run_date") or DEFAULT_MINIMUM_RUN_DATE)
    )
    timestamp = _utc_now()

    old_runs = payload.get("runs")
    if isinstance(old_runs, dict):
        for legacy_run_id, entry in old_runs.items():
            if not isinstance(entry, dict):
                continue
            embedded = entry.get("score")
            raw_path = entry.get("score_path") or entry.get("source_score_json")
            score_path = (
                _resolve_source_path(str(raw_path), registry_path=registry_path)
                if raw_path
                else registry_path.parent / str(legacy_run_id) / "score.json"
            )
            score_payload = embedded if isinstance(embedded, dict) else _load_score(score_path)
            if isinstance(score_payload, dict) and "score" in score_payload:
                _put_if_included(
                    registry,
                    score_payload=score_payload,
                    score_path=score_path,
                    registry_path=registry_path,
                    score_set=PUBLIC_SCORE_SET,
                    timestamp=timestamp,
                    run_id=str(entry.get("run_id") or legacy_run_id),
                )

    # The oldest pair-specific format used baseline/candidate slots plus IDs.
    for key in ("baseline_run", "candidate_run"):
        legacy_run_id = payload.get(key)
        if not legacy_run_id or str(legacy_run_id) in registry["runs"]:
            continue
        score_path = registry_path.parent / str(legacy_run_id) / "score.json"
        score_payload = _load_score(score_path)
        if score_payload is not None:
            _put_if_included(
                registry,
                score_payload=score_payload,
                score_path=score_path,
                registry_path=registry_path,
                score_set=PUBLIC_SCORE_SET,
                timestamp=timestamp,
                run_id=str(legacy_run_id),
            )

    registry["migrated_from_version"] = payload.get("version", 1)
    return registry


def _prune_runs(registry: dict[str, Any]) -> None:
    runs = registry.setdefault("runs", {})
    minimum_run_date = str(registry.get("minimum_run_date") or DEFAULT_MINIMUM_RUN_DATE)
    retained: dict[str, Any] = {}
    for run_id, entry in runs.items():
        if not isinstance(entry, dict):
            continue
        scores = entry.get("scores")
        if not isinstance(scores, dict) or not scores:
            continue
        sample = next(
            (
                score_entry.get("score")
                for score_entry in scores.values()
                if isinstance(score_entry, dict) and isinstance(score_entry.get("score"), dict)
            ),
            {},
        )
        if run_is_included(str(run_id), sample, minimum_run_date=minimum_run_date):
            retained[str(run_id)] = entry
    registry["runs"] = retained
    existing_order = [str(run_id) for run_id in registry.get("run_order", [])]
    registry["run_order"] = [run_id for run_id in existing_order if run_id in retained]
    registry["run_order"].extend(sorted(set(retained) - set(registry["run_order"])))


def load_score_registry(path: str | Path = DEFAULT_REGISTRY_PATH) -> dict[str, Any]:
    registry_path = Path(path)
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _empty_registry()
    if not isinstance(payload, dict):
        raise ValueError(f"Score registry must contain a JSON object: {registry_path}")
    if payload.get("version") != REGISTRY_VERSION:
        return _migrate_legacy(payload, registry_path=registry_path)
    payload.setdefault("minimum_run_date", DEFAULT_MINIMUM_RUN_DATE)
    payload.setdefault("score_sets", _empty_registry()["score_sets"])
    payload.setdefault("run_order", list(payload.get("runs", {})))
    payload.setdefault("runs", {})
    payload.setdefault("semi_private_scores", [])
    _prune_runs(payload)
    return payload


def save_score_registry(registry: dict[str, Any], path: str | Path = DEFAULT_REGISTRY_PATH) -> Path:
    registry_path = Path(path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry["version"] = REGISTRY_VERSION
    registry.setdefault("minimum_run_date", DEFAULT_MINIMUM_RUN_DATE)
    registry.setdefault("score_sets", _empty_registry()["score_sets"])
    registry.setdefault("semi_private_scores", [])
    _prune_runs(registry)
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
    score_set: str = PUBLIC_SCORE_SET,
    run_id: str | None = None,
) -> tuple[Path, str]:
    path = Path(registry_path)
    registry = load_score_registry(path)
    resolved_run_id = _put_if_included(
        registry,
        score_payload=score_payload,
        score_path=Path(score_path),
        registry_path=path,
        score_set=score_set,
        timestamp=_utc_now(),
        run_id=run_id,
    )
    return save_score_registry(registry, path), resolved_run_id


def update_frontier_registry(
    frontier: dict[str, Any],
    *,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> Path:
    path = Path(registry_path)
    registry = load_score_registry(path)
    registry["public_frontier"] = frontier
    return save_score_registry(registry, path)



def append_semi_private_score(
    *,
    score_payload: dict[str, Any],
    score_path: str | Path,
    source_label: str,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    local_run_path: str = "",
) -> tuple[Path, str]:
    """Append one unlinked Kaggle score without replacing earlier imports."""
    import hashlib

    path = Path(registry_path)
    registry = load_score_registry(path)
    timestamp = _utc_now()
    digest = hashlib.sha256(
        json.dumps(score_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:10]
    stem = timestamp.replace("-", "").replace(":", "").replace(".", "")
    base_id = f"{stem}-{digest}"
    existing_ids = {
        str(entry.get("id"))
        for entry in registry.setdefault("semi_private_scores", [])
        if isinstance(entry, dict)
    }
    item_id = base_id
    suffix = 2
    while item_id in existing_ids:
        item_id = f"{base_id}-{suffix}"
        suffix += 1
    registry["semi_private_scores"].append(
        {
            "id": item_id,
            "pulled_at": timestamp,
            "source_label": source_label,
            "local_run_path": local_run_path,
            "score_path": _display_score_path(Path(score_path), registry_path=path),
            "score": score_payload,
        }
    )
    return save_score_registry(registry, path), item_id
