"""Append Kaggle semi-private leaderboard scores to the viewer registry."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from inference.tools.score_registry import (
    append_semi_private_score,
    load_score_registry,
)

DEFAULT_KAGGLE_COMPETITION = "arc-prize-2026-arc-agi-3"
DEFAULT_MINIMUM_SUBMISSION_DATE = "2026-08-10"
SEMI_PRIVATE_SCORES_DIRNAME = "semi_private_scores"


def _utc_filename_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _validate_payload(payload: Any, *, source: str) -> dict[str, Any]:
    if not isinstance(payload, dict) or not isinstance(payload.get("score"), (int, float)):
        raise ValueError(f"{source}: expected a score JSON object with a numeric 'score' field.")
    games = payload.get("games")
    if games is not None and not isinstance(games, dict):
        raise ValueError(f"{source}: 'games' must be an object when present.")
    return payload


def _load_json_bytes(content: bytes, *, source: str) -> dict[str, Any]:
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{source}: invalid JSON ({exc})") from exc
    return _validate_payload(payload, source=source)


def pull_score_payload(source: str) -> dict[str, Any]:
    """Load one score from a local path, HTTPS URL, or scp-style remote path."""
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        headers: dict[str, str] = {}
        token = os.environ.get("SEMI_PRIVATE_SCORE_TOKEN", "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        response = requests.get(source, headers=headers, timeout=30)
        response.raise_for_status()
        return _load_json_bytes(response.content, source=source)

    local_path = Path(source).expanduser()
    if local_path.is_file():
        return _load_json_bytes(local_path.read_bytes(), source=source)

    if ":" not in source:
        raise FileNotFoundError(f"Semi-private score source not found: {source}")

    with tempfile.TemporaryDirectory(prefix="arc3-semi-private-score-") as temp_dir:
        fetched = Path(temp_dir) / "score.json"
        subprocess.run(
            ["scp", "--", source, str(fetched)],
            check=True,
            stdin=subprocess.DEVNULL,
        )
        return _load_json_bytes(fetched.read_bytes(), source=source)


def _source_label(source: str) -> str:
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        return Path(parsed.path).name or parsed.netloc
    remote_path = source.rsplit(":", 1)[-1] if ":" in source else source
    return Path(remote_path).name or "score.json"


def _unique_payload_path(payload: dict[str, Any], *, runs_dir: Path) -> Path:
    destination_dir = runs_dir / SEMI_PRIVATE_SCORES_DIRNAME
    destination_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:10]
    filename_base = f"{_utc_filename_timestamp()}-{digest}"
    destination = destination_dir / f"{filename_base}.json"
    suffix = 2
    while destination.exists():
        destination = destination_dir / f"{filename_base}-{suffix}.json"
        suffix += 1
    return destination


def _write_new_payload(payload: dict[str, Any], destination: Path) -> None:
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_semi_private_score(
    *,
    payload: dict[str, Any],
    source_label: str,
    runs_dir: str | Path,
    registry_path: str | Path | None = None,
    local_run_path: str = "",
) -> tuple[Path, Path, str]:
    """Write a uniquely named score file and append, never replace, its registry item."""
    runs_path = Path(runs_dir)
    destination = _unique_payload_path(payload, runs_dir=runs_path)
    _write_new_payload(payload, destination)
    registry = Path(registry_path) if registry_path else runs_path / "inference_score_comparison.json"
    saved_registry, item_id = append_semi_private_score(
        score_payload=payload,
        score_path=destination,
        source_label=source_label,
        registry_path=registry,
        local_run_path=local_run_path,
    )
    return destination, saved_registry, item_id


def fetch_kaggle_submissions(
    *,
    competition: str = DEFAULT_KAGGLE_COMPETITION,
    kaggle_executable: str | None = None,
) -> list[dict[str, str]]:
    executable = kaggle_executable or shutil.which("kaggle")
    if not executable:
        raise RuntimeError("Kaggle CLI not found; install/configure kaggle before pulling scores.")
    completed = subprocess.run(
        [
            executable,
            "competitions",
            "submissions",
            competition,
            "--csv",
            "--page-size",
            "200",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_kaggle_submissions_csv(completed.stdout)


def parse_kaggle_submissions_csv(output: str) -> list[dict[str, str]]:
    """Parse Kaggle CSV while ignoring version-warning preamble lines."""
    lines = output.splitlines()
    header_index = next(
        (index for index, line in enumerate(lines) if line.startswith("ref,")),
        None,
    )
    if header_index is None:
        raise RuntimeError("Kaggle submission output did not contain a CSV header.")
    csv_text = "\n".join(lines[header_index:])
    return [dict(row) for row in csv.DictReader(io.StringIO(csv_text))]


def _submission_payload(row: dict[str, str], *, competition: str) -> dict[str, Any]:
    return {
        "version": 1,
        "score": float(row["publicScore"]),
        "games": {},
        "metadata": {
            "score_set": "semi_private",
            "source": "kaggle_competition_submission",
            "competition": competition,
            "kaggle_submission_ref": str(row["ref"]),
            "kaggle_score_field": "publicScore",
            "submitted_at": row.get("date") or None,
            "description": row.get("description") or "",
            "status": row.get("status") or "",
            "file_name": row.get("fileName") or "",
        },
    }


def _existing_kaggle_refs(registry: dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    for item in registry.get("semi_private_scores", []):
        if not isinstance(item, dict):
            continue
        score = item.get("score")
        metadata = score.get("metadata") if isinstance(score, dict) else None
        ref = metadata.get("kaggle_submission_ref") if isinstance(metadata, dict) else None
        if ref:
            refs.add(str(ref))
    return refs


def append_kaggle_submissions(
    *,
    rows: list[dict[str, str]],
    competition: str,
    minimum_date: str,
    runs_dir: str | Path,
    registry_path: str | Path | None = None,
) -> tuple[list[str], list[str]]:
    """Append unseen completed Kaggle submissions; never modify existing imports."""
    runs_path = Path(runs_dir)
    registry = Path(registry_path) if registry_path else runs_path / "inference_score_comparison.json"
    existing_refs = _existing_kaggle_refs(load_score_registry(registry))
    added: list[str] = []
    skipped: list[str] = []
    eligible = sorted(rows, key=lambda row: str(row.get("date") or ""))
    for row in eligible:
        ref = str(row.get("ref") or "").strip()
        date = str(row.get("date") or "")
        score = str(row.get("publicScore") or "").strip()
        status = str(row.get("status") or "")
        if (
            not ref
            or date[:10] < minimum_date
            or "COMPLETE" not in status.upper()
            or not score
        ):
            continue
        try:
            float(score)
        except ValueError:
            continue
        if ref in existing_refs:
            skipped.append(ref)
            continue

        payload = _submission_payload(row, competition=competition)
        destination_dir = runs_path / SEMI_PRIVATE_SCORES_DIRNAME
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"kaggle-{ref}.json"
        if not destination.exists():
            _write_new_payload(payload, destination)
        _, item_id = append_semi_private_score(
            score_payload=payload,
            score_path=destination,
            source_label=f"Kaggle submission {ref}",
            registry_path=registry,
            local_run_path="",
        )
        added.append(item_id)
        existing_refs.add(ref)
    return added, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append unseen Kaggle semi-private scores without overwriting past imports."
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Optional path, HTTPS URL, or user@host:/path/score.json instead of Kaggle.",
    )
    parser.add_argument("--competition", default=DEFAULT_KAGGLE_COMPETITION)
    parser.add_argument("--minimum-date", default=DEFAULT_MINIMUM_SUBMISSION_DATE)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--registry", default=None)
    args = parser.parse_args()

    if args.source:
        payload = pull_score_payload(args.source)
        score_path, registry_path, item_id = save_semi_private_score(
            payload=payload,
            source_label=_source_label(args.source),
            runs_dir=args.runs_dir,
            registry_path=args.registry,
            local_run_path="",
        )
        print(f"Appended semi-private score {item_id}: {score_path}")
        print(f"Local run path: <blank; edit it in {registry_path}>")
        return

    rows = fetch_kaggle_submissions(competition=args.competition)
    added, skipped = append_kaggle_submissions(
        rows=rows,
        competition=args.competition,
        minimum_date=args.minimum_date,
        runs_dir=args.runs_dir,
        registry_path=args.registry,
    )
    print(f"Appended {len(added)} new Kaggle semi-private score(s).")
    print(f"Preserved {len(skipped)} previously imported submission(s).")
    print("New local_run_path fields are blank for manual linking.")


if __name__ == "__main__":
    main()
