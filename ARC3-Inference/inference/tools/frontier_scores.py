"""Refresh current ARC Prize public frontier scores in the viewer registry."""
from __future__ import annotations

import argparse
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from inference.tools.score_registry import DEFAULT_REGISTRY_PATH, update_frontier_registry

ARC_PRIZE_BASE_URL = "https://arcprize.org"
TASKS_URL = f"{ARC_PRIZE_BASE_URL}/tasks"
PUBLIC_SCORECARD_URL = f"{ARC_PRIZE_BASE_URL}/api/public-scorecard/open"
GAMES_URL = f"{ARC_PRIZE_BASE_URL}/api/games"
CONFIGS_URL = f"{ARC_PRIZE_BASE_URL}/api/models/configs"
SCORES_URL = f"{ARC_PRIZE_BASE_URL}/api/models"
MODELS_URL = f"{ARC_PRIZE_BASE_URL}/media/data/models.json"
PROVIDERS_URL = f"{ARC_PRIZE_BASE_URL}/media/data/providers.json"
REQUEST_TIMEOUT_SECONDS = 30


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _request_json(
    session: requests.Session,
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    json_body: dict[str, Any] | None = None,
) -> Any:
    response = session.request(
        method,
        url,
        headers=headers,
        json=json_body,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def fetch_public_frontier_data(
    *,
    session: requests.Session | None = None,
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch games, selected configs, metadata, and public score rows."""
    client = session or requests.Session()
    card = _request_json(
        client,
        "POST",
        PUBLIC_SCORECARD_URL,
        json_body={"source_url": TASKS_URL, "tags": ["human", "anonymous"]},
    )
    api_key = str(card.get("api_key") or "") if isinstance(card, dict) else ""
    if not api_key:
        raise RuntimeError("ARC Prize did not return an anonymous API key.")
    headers = {"X-API-Key": api_key}

    games = _request_json(client, "GET", GAMES_URL, headers=headers)
    available_configs = _request_json(client, "GET", CONFIGS_URL, headers=headers)
    models = _request_json(client, "GET", MODELS_URL)
    providers = _request_json(client, "GET", PROVIDERS_URL)
    if not all(isinstance(value, list) for value in (games, available_configs, models, providers)):
        raise RuntimeError("ARC Prize returned an unexpected public metadata schema.")

    available = {str(config) for config in available_configs}
    featured_configs = [
        str(model.get("id"))
        for model in models
        if isinstance(model, dict)
        and model.get("featured") is True
        and str(model.get("id") or "") in available
    ]
    selected_configs = featured_configs or sorted(available)
    rows = _request_json(
        client,
        "POST",
        SCORES_URL,
        headers=headers,
        json_body={"game_id": "", "models": [], "runners": [], "configs": selected_configs},
    )
    if not isinstance(rows, list):
        raise RuntimeError("ARC Prize returned an unexpected public score schema.")
    return games, selected_configs, models, providers, rows


def build_frontier_snapshot(
    *,
    games: list[dict[str, Any]],
    selected_configs: list[str],
    models: list[dict[str, Any]],
    providers: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    fetched_at: str | None = None,
) -> dict[str, Any]:
    """Build a compact latest-per-config and best-per-game frontier snapshot."""
    provider_lookup = {
        str(provider.get("id")): provider
        for provider in providers
        if isinstance(provider, dict) and provider.get("id")
    }
    model_lookup = {
        str(model.get("id")): model
        for model in models
        if isinstance(model, dict) and model.get("id")
    }
    config_metadata: dict[str, Any] = {}
    for config_id in selected_configs:
        model = model_lookup.get(config_id, {})
        provider = provider_lookup.get(str(model.get("providerId") or ""), {})
        config_metadata[config_id] = {
            "display_name": str(model.get("displayName") or config_id),
            "provider": str(provider.get("displayName") or model.get("providerId") or ""),
            "model_type": str(model.get("modelType") or ""),
        }

    current_games: dict[tuple[str, str], dict[str, Any]] = {}
    game_versions: dict[str, str] = {}
    for game in games:
        if not isinstance(game, dict):
            continue
        combined_id = str(game.get("game_id") or "")
        base_id, separator, version = combined_id.partition("-")
        if not separator or not base_id or not version:
            continue
        current_games[(base_id, version)] = game
        game_versions[base_id] = version

    latest: dict[tuple[str, str], dict[str, Any]] = {}
    selected = set(selected_configs)
    for row in rows:
        if not isinstance(row, dict) or row.get("verified_publish") is not True:
            continue
        base_id = str(row.get("game_id") or "")
        version = str(row.get("game_version") or "")
        config_id = str(row.get("config") or "")
        if config_id not in selected or (base_id, version) not in current_games:
            continue
        key = (base_id, config_id)
        previous = latest.get(key)
        if previous is None or str(row.get("published_at") or "") >= str(previous.get("published_at") or ""):
            latest[key] = row

    snapshot_games: dict[str, Any] = {}
    scores_by_config: dict[str, list[float]] = {config_id: [] for config_id in selected_configs}
    for base_id in sorted(game_versions):
        version = game_versions[base_id]
        game = current_games[(base_id, version)]
        model_scores: dict[str, Any] = {}
        for config_id in selected_configs:
            row = latest.get((base_id, config_id))
            if row is None:
                continue
            score = float(row.get("score") or 0.0)
            scores_by_config[config_id].append(score)
            session_id = str(row.get("session_id") or "")
            model_scores[config_id] = {
                "score": score,
                "published_at": row.get("published_at"),
                "session_id": session_id,
                "replay_url": f"{ARC_PRIZE_BASE_URL}/replay/{session_id}" if session_id else None,
                "levels_completed": row.get("levels_completed"),
                "actions": row.get("actions"),
                "state": row.get("state"),
            }
        best_config = max(
            model_scores,
            key=lambda config_id: (float(model_scores[config_id]["score"]), config_id),
            default=None,
        )
        best = None
        if best_config is not None:
            best = {
                "config_id": best_config,
                "display_name": config_metadata[best_config]["display_name"],
                **model_scores[best_config],
            }
        snapshot_games[base_id] = {
            "game_id": f"{base_id}-{version}",
            "title": str(game.get("title") or base_id.upper()),
            "task_url": f"{TASKS_URL}/{base_id}",
            "models": model_scores,
            "best": best,
        }

    for config_id, metadata in config_metadata.items():
        config_scores = scores_by_config.get(config_id, [])
        metadata["overall_score"] = statistics.fmean(config_scores) if config_scores else None
        metadata["game_count"] = len(config_scores)

    best_scores = [
        float(game["best"]["score"])
        for game in snapshot_games.values()
        if isinstance(game.get("best"), dict)
    ]
    return {
        "source_name": "ARC Prize public task explorer",
        "source_url": TASKS_URL,
        "task_url_template": f"{TASKS_URL}/{{game_id}}",
        "fetched_at": fetched_at or _utc_now(),
        "selection": "ARC Prize featured model configurations",
        "config_order": selected_configs,
        "configs": config_metadata,
        "game_count": len(snapshot_games),
        "covered_game_count": len(best_scores),
        "oracle_score": statistics.fmean(best_scores) if best_scores else None,
        "games": snapshot_games,
    }


def refresh_frontier_registry(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    session: requests.Session | None = None,
) -> Path:
    games, configs, models, providers, rows = fetch_public_frontier_data(session=session)
    snapshot = build_frontier_snapshot(
        games=games,
        selected_configs=configs,
        models=models,
        providers=providers,
        rows=rows,
    )
    return update_frontier_registry(snapshot, registry_path=registry_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh ARC Prize featured-model public scores in the comparison registry."
    )
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    args = parser.parse_args()
    saved = refresh_frontier_registry(registry_path=args.registry)
    print(f"Updated public frontier scores: {saved}")


if __name__ == "__main__":
    main()
