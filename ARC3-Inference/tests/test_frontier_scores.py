from __future__ import annotations

from inference.tools.frontier_scores import build_frontier_snapshot


def test_build_frontier_snapshot_uses_latest_verified_current_game_scores() -> None:
    snapshot = build_frontier_snapshot(
        games=[
            {"game_id": "ar25-aaaa1111", "title": "AR25"},
            {"game_id": "ft09-bbbb2222", "title": "FT09"},
        ],
        selected_configs=["model-a", "model-b"],
        models=[
            {"id": "model-a", "displayName": "Model A", "providerId": "A", "modelType": "CoT"},
            {"id": "model-b", "displayName": "Model B", "providerId": "B", "modelType": "CoT"},
        ],
        providers=[
            {"id": "A", "displayName": "Provider A"},
            {"id": "B", "displayName": "Provider B"},
        ],
        rows=[
            {
                "game_id": "ar25",
                "game_version": "aaaa1111",
                "config": "model-a",
                "score": 20.0,
                "published_at": "2026-08-01T00:00:00Z",
                "session_id": "old",
                "verified_publish": True,
            },
            {
                "game_id": "ar25",
                "game_version": "aaaa1111",
                "config": "model-a",
                "score": 30.0,
                "published_at": "2026-08-02T00:00:00Z",
                "session_id": "new",
                "verified_publish": True,
            },
            {
                "game_id": "ar25",
                "game_version": "aaaa1111",
                "config": "model-b",
                "score": 40.0,
                "published_at": "2026-08-02T00:00:00Z",
                "session_id": "best",
                "verified_publish": True,
            },
            {
                "game_id": "ft09",
                "game_version": "bbbb2222",
                "config": "model-a",
                "score": 10.0,
                "published_at": "2026-08-02T00:00:00Z",
                "session_id": "ft",
                "verified_publish": True,
            },
            {
                "game_id": "ft09",
                "game_version": "wrong",
                "config": "model-b",
                "score": 100.0,
                "published_at": "2026-08-03T00:00:00Z",
                "session_id": "wrong-version",
                "verified_publish": True,
            },
        ],
        fetched_at="2026-08-19T00:00:00Z",
    )

    ar25 = snapshot["games"]["ar25"]
    assert ar25["task_url"] == "https://arcprize.org/tasks/ar25"
    assert ar25["models"]["model-a"]["score"] == 30.0
    assert ar25["best"]["display_name"] == "Model B"
    assert ar25["best"]["replay_url"] == "https://arcprize.org/replay/best"
    assert snapshot["configs"]["model-a"]["overall_score"] == 20.0
    assert snapshot["configs"]["model-b"]["overall_score"] == 40.0
    assert snapshot["oracle_score"] == 25.0
    assert snapshot["covered_game_count"] == 2
