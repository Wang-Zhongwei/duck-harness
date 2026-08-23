"""Simple HTTP server for browsing the latest ARC3 run."""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import mimetypes
import re
import subprocess
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from inference.tools.score_registry import load_score_registry
from inference.utils.run_artifacts import is_selectable_run_dir_name, run_dir_sort_key
from viewer.data import list_run_dirs
from viewer.data import load_game_payload, load_game_shell_payload, load_game_step_payload, load_run_summary
from viewer.thumbnails import game_thumbnail_png


log = logging.getLogger(__name__)
_GZIP_MIN_BYTES = 1024
_STATIC_SUBDIRS = {"solver_analysis", "movies"}
_SPLIT_STATIC_SUBDIRS = {"passes", "seeds"}
_COMMIT_RE = re.compile(r"(?m)^commit:\s*([0-9a-fA-F]{40})\s*$")
_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")


@dataclass(frozen=True)
class _ResponseBody:
    body: bytes
    is_gzipped: bool


def _index_html_path() -> Path:
    return Path(__file__).resolve().parent / "index.html"


def _comparison_html_path() -> Path:
    return Path(__file__).resolve().parent / "comparison.html"


def _game_peek_js_path() -> Path:
    return Path(__file__).resolve().parent / "game_peek.js"


def _comparison_json_path(runs_dir: Path) -> Path:
    return runs_dir / "inference_score_comparison.json"


def _apply_unified_file_diff(base_text: str, diff_text: str, file_path: str) -> str:
    """Apply one file's unified diff from a run's git_info.txt snapshot."""
    marker = f"diff --git a/{file_path} b/{file_path}"
    start = diff_text.find(marker)
    if start < 0:
        return base_text
    end = diff_text.find("\ndiff --git ", start + len(marker))
    section = diff_text[start : end if end >= 0 else None].splitlines()
    base_lines = base_text.splitlines()
    output: list[str] = []
    cursor = 0
    index = 0

    while index < len(section):
        match = _HUNK_RE.match(section[index])
        if match is None:
            index += 1
            continue
        old_start = int(match.group(1)) - 1
        if old_start < cursor:
            raise ValueError(f"Overlapping diff hunks for {file_path}")
        output.extend(base_lines[cursor:old_start])
        cursor = old_start
        index += 1
        while index < len(section) and not section[index].startswith("@@ "):
            line = section[index]
            if line.startswith("diff --git "):
                break
            if line == "\\ No newline at end of file":
                index += 1
                continue
            if not line or line[0] not in {" ", "+", "-"}:
                index += 1
                continue
            content = line[1:]
            if line[0] in {" ", "-"}:
                if cursor >= len(base_lines) or base_lines[cursor] != content:
                    raise ValueError(f"Diff context does not match {file_path}")
                cursor += 1
            if line[0] in {" ", "+"}:
                output.append(content)
            index += 1

    output.extend(base_lines[cursor:])
    return "\n".join(output) + ("\n" if base_text.endswith("\n") else "")


def _load_run_inference_config(run_dir: Path) -> dict | None:
    """Load a bundled inference.json, falling back to run git provenance."""
    bundled_relative_path = Path("src/ARC3-Inference/configs/inference.json")
    snapshot_paths = [
        run_dir / "inference.json",
        run_dir / "configs" / "inference.json",
        run_dir / bundled_relative_path,
        *sorted(run_dir.glob(f"passes/*/{bundled_relative_path}")),
        *sorted(run_dir.glob(f"seeds/*/{bundled_relative_path}")),
        *sorted(run_dir.glob(f"*/{bundled_relative_path}")),
    ]
    seen: set[Path] = set()
    for snapshot_path in snapshot_paths:
        if snapshot_path in seen or not snapshot_path.is_file():
            continue
        seen.add(snapshot_path)
        try:
            payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            log.warning("Could not load inference config snapshot %s", snapshot_path, exc_info=True)
            continue
        if isinstance(payload, dict):
            return payload

    git_info_path = run_dir / "git_info.txt"
    if not git_info_path.is_file():
        return None
    git_info = git_info_path.read_text(encoding="utf-8")
    match = _COMMIT_RE.search(git_info)
    if match is None:
        return None

    project_root = Path(__file__).resolve().parents[1]
    try:
        repository_root = Path(
            subprocess.run(
                ["git", "-C", str(project_root), "rev-parse", "--show-toplevel"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        )
        config_path = (project_root / "configs" / "inference.json").relative_to(repository_root).as_posix()
        committed = subprocess.run(
            ["git", "-C", str(repository_root), "show", f"{match.group(1)}:{config_path}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        reconstructed = _apply_unified_file_diff(committed, git_info, config_path)
        payload = json.loads(reconstructed)
    except (OSError, subprocess.SubprocessError, UnicodeError, ValueError, json.JSONDecodeError):
        log.warning("Could not recover inference config for %s", run_dir, exc_info=True)
        return None
    return payload if isinstance(payload, dict) else None


def _load_evaluation(evaluation_path: Path) -> dict | None:
    """Return a run's evaluation.json as a dict, or None when it is unusable."""
    try:
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        log.warning("Skipping unreadable evaluation %s", evaluation_path, exc_info=True)
        return None
    return evaluation if isinstance(evaluation, dict) else None


def _load_comparison_payload(runs_dir: Path) -> dict:
    """Build the comparison payload from every evaluated run on disk.

    ``<run>/evaluation.json`` is the source of truth for which runs can be
    compared and for their public scores, so a run shows up as soon as it is
    evaluated, regardless of whether (or under which name) it was registered.
    ``inference_score_comparison.json`` only contributes what evaluation.json
    cannot hold: frontier references, semi-private imports, and score-set labels.
    """
    payload = load_score_registry(_comparison_json_path(runs_dir))
    registered = payload.get("runs") if isinstance(payload.get("runs"), dict) else {}
    runs: dict[str, dict] = {}
    for run_dir in list_run_dirs(runs_dir):
        evaluation_path = run_dir / "evaluation.json"
        if not evaluation_path.is_file():
            continue
        evaluation = _load_evaluation(evaluation_path)
        if evaluation is None:
            continue
        run_id = run_dir.name
        registered_entry = registered.get(run_id)
        entry = {
            key: value
            for key, value in (registered_entry.items() if isinstance(registered_entry, dict) else ())
            if key not in {"scores", "inference_config"}
        }
        metadata = evaluation.get("metadata") if isinstance(evaluation.get("metadata"), dict) else {}
        entry["run_id"] = run_id
        entry["scores"] = {
            "public": {
                "score": evaluation,
                "score_path": evaluation_path.relative_to(runs_dir).as_posix(),
                "updated_at": metadata.get("created_at"),
            }
        }
        entry["inference_config"] = _load_run_inference_config(run_dir)
        runs[run_id] = entry
    payload["runs"] = runs
    payload["run_order"] = sorted(runs, key=run_dir_sort_key, reverse=True)
    return payload


def _load_index_html() -> str:
    return _index_html_path().read_text(encoding="utf-8")


def _load_comparison_html() -> str:
    return _comparison_html_path().read_text(encoding="utf-8")


def _index_html_version() -> int:
    return max(
        _index_html_path().stat().st_mtime_ns,
        _comparison_html_path().stat().st_mtime_ns,
        _game_peek_js_path().stat().st_mtime_ns,
    )


def _requested_run_dir(*, runs_dir: Path, default_run_dir: Path | None, requested_run: str | None) -> Path | None:
    requested_name = str(requested_run or "").strip()
    if not requested_name:
        return default_run_dir

    candidate = Path(requested_name)
    if candidate.is_absolute():
        return candidate

    if runs_dir.is_dir() and is_selectable_run_dir_name(runs_dir.name) and requested_name == runs_dir.name:
        return runs_dir

    if default_run_dir is not None and requested_name == default_run_dir.name:
        return default_run_dir

    return runs_dir / requested_name


def _resolve_static_file(run_dir: Path | None, rel_path: Path) -> Path | None:
    """Find *rel_path* inside *run_dir* or any of its pass/seed sub-dirs."""
    if run_dir is None:
        return None
    candidate = run_dir / rel_path
    if candidate.is_file():
        return candidate
    for sub in ("passes", "seeds"):
        sub_dir = run_dir / sub
        if not sub_dir.is_dir():
            continue
        for child in sub_dir.iterdir():
            if not child.is_dir():
                continue
            candidate = child / rel_path
            if candidate.is_file():
                return candidate
    return None


class _ViewerHandler(BaseHTTPRequestHandler):
    """Serve the viewer shell and run payload API."""

    runs_dir: Path
    run_dir: Path | None
    environments_dir: Path | None = None

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._send_html(_load_index_html())
            return
        if parsed.path in {"/comparison", "/comparison.html"}:
            self._send_html(_load_comparison_html())
            return
        if parsed.path == "/game-peek.js":
            self._send_bytes(_game_peek_js_path().read_bytes(), "application/javascript; charset=utf-8")
            return
        if parsed.path == "/api/thumbnail":
            self._handle_thumbnail_api(parsed.query)
            return
        if parsed.path == "/api/viewer-version":
            self._send_json({"version": _index_html_version()})
            return
        if parsed.path == "/api/run":
            self._handle_run_api(parsed.query)
            return
        if parsed.path == "/api/comparison":
            self._handle_comparison_api()
            return
        if parsed.path == "/api/game":
            self._handle_game_api(parsed.query)
            return
        if parsed.path == "/api/game-step":
            self._handle_game_step_api(parsed.query)
            return
        if self._try_serve_static(parsed.path, parsed.query):
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def _try_serve_static(self, path: str, query: str) -> bool:
        rel = path.lstrip("/")
        parts = Path(rel).parts
        if not parts or ".." in parts:
            return False
        if parts[0] in _STATIC_SUBDIRS:
            safe_rel = Path(*parts)
        elif len(parts) >= 3 and parts[0] in _SPLIT_STATIC_SUBDIRS and parts[2] in _STATIC_SUBDIRS:
            safe_rel = Path(*parts)
        else:
            return False
        params = parse_qs(query)
        requested_run = params.get("run", [None])[0]
        run_dir = _requested_run_dir(
            runs_dir=self.runs_dir,
            default_run_dir=self.run_dir,
            requested_run=requested_run,
        )
        resolved = _resolve_static_file(run_dir, safe_rel)
        if resolved is None:
            return False
        content_type = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
        body = resolved.read_bytes()
        body = self._maybe_gzip(body)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        if body.is_gzipped:
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
        self.send_header("Content-Length", str(len(body.body)))
        self.end_headers()
        self.wfile.write(body.body)
        return True

    def log_message(self, fmt: str, *args) -> None:
        log.info("%s - %s", self.address_string(), fmt % args)

    def _handle_thumbnail_api(self, query: str) -> None:
        params = parse_qs(query)
        game_id = params.get("game", [""])[0]
        try:
            png = game_thumbnail_png(game_id, environments_dir=self.environments_dir)
        except Exception:
            log.warning("Thumbnail rendering failed for %r", game_id, exc_info=True)
            png = None
        if png is None:
            self.send_error(HTTPStatus.NOT_FOUND, "No thumbnail for this game")
            return
        self._send_bytes(png, "image/png", cache_control="public, max-age=86400")

    def _send_bytes(self, body: bytes, content_type: str, *, cache_control: str = "no-store") -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", cache_control)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_comparison_api(self) -> None:
        try:
            payload = _load_comparison_payload(self.runs_dir)
        except (json.JSONDecodeError, ValueError) as exc:
            self._send_json({"error": f"Invalid comparison JSON: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self._send_json(payload)

    def _handle_run_api(self, query: str) -> None:
        params = parse_qs(query)
        requested_run = params.get("run", [None])[0]
        try:
            payload = load_run_summary(
                runs_dir=self.runs_dir,
                run_dir=_requested_run_dir(
                    runs_dir=self.runs_dir,
                    default_run_dir=self.run_dir,
                    requested_run=requested_run,
                ),
            )
        except FileNotFoundError as exc:
            self._send_json({"error": str(exc), "games": []}, status=HTTPStatus.NOT_FOUND)
            return
        except json.JSONDecodeError as exc:
            self._send_json({"error": f"Invalid viewer artifact JSON: {exc}", "games": []}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self._send_json(payload)

    def _handle_game_api(self, query: str) -> None:
        params = parse_qs(query)
        requested_run = params.get("run", [None])[0]
        raw_index = params.get("index", [None])[0]
        try:
            game_index = int(str(raw_index))
        except (TypeError, ValueError):
            self._send_json({"error": "Missing or invalid game index."}, status=HTTPStatus.BAD_REQUEST)
            return

        try:
            full_payload = params.get("full", ["false"])[0].lower() in {"1", "true", "yes", "on"}
            loader = load_game_payload if full_payload else load_game_shell_payload
            payload = loader(
                runs_dir=self.runs_dir,
                run_dir=_requested_run_dir(
                    runs_dir=self.runs_dir,
                    default_run_dir=self.run_dir,
                    requested_run=requested_run,
                ),
                game_index=game_index,
            )
        except FileNotFoundError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
            return
        except json.JSONDecodeError as exc:
            self._send_json({"error": f"Invalid viewer artifact JSON: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self._send_json(payload)

    def _handle_game_step_api(self, query: str) -> None:
        params = parse_qs(query)
        requested_run = params.get("run", [None])[0]
        raw_game_index = params.get("index", [None])[0]
        raw_step_index = params.get("step", [None])[0]
        try:
            game_index = int(str(raw_game_index))
            step_index = int(str(raw_step_index))
        except (TypeError, ValueError):
            self._send_json({"error": "Missing or invalid game/step index."}, status=HTTPStatus.BAD_REQUEST)
            return

        try:
            payload = load_game_step_payload(
                runs_dir=self.runs_dir,
                run_dir=_requested_run_dir(
                    runs_dir=self.runs_dir,
                    default_run_dir=self.run_dir,
                    requested_run=requested_run,
                ),
                game_index=game_index,
                step_index=step_index,
            )
        except FileNotFoundError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
            return
        except json.JSONDecodeError as exc:
            self._send_json({"error": f"Invalid viewer artifact JSON: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self._send_json(payload)

    def _send_html(self, html: str) -> None:
        content = html.encode("utf-8")
        content = self._maybe_gzip(content)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        if content.is_gzipped:
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
        self.send_header("Content-Length", str(len(content.body)))
        self.end_headers()
        self.wfile.write(content.body)

    def _send_json(self, payload: dict, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        content = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        content = self._maybe_gzip(content)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        if content.is_gzipped:
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
        self.send_header("Content-Length", str(len(content.body)))
        self.end_headers()
        self.wfile.write(content.body)

    def _maybe_gzip(self, content: bytes) -> "_ResponseBody":
        accept_encoding = self.headers.get("Accept-Encoding", "")
        if len(content) < _GZIP_MIN_BYTES or "gzip" not in accept_encoding.lower():
            return _ResponseBody(body=content, is_gzipped=False)
        return _ResponseBody(body=gzip.compress(content), is_gzipped=True)


def build_handler(
    *, runs_dir: Path, run_dir: Path | None, environments_dir: Path | None = None
) -> type[_ViewerHandler]:
    """Bind configuration into the request handler class."""
    handler_cls = type("ViewerHandler", (_ViewerHandler,), {})
    handler_cls.runs_dir = runs_dir
    handler_cls.run_dir = run_dir
    handler_cls.environments_dir = environments_dir
    return handler_cls


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the ARC3 viewer for the latest run.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--run-dir", default=None, help="Optional explicit run directory to view.")
    parser.add_argument(
        "--environments-dir",
        default=None,
        help="Offline ARC env files used for game thumbnails (default: configs/inference.json environment.environments_dir).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    runs_dir = Path(args.runs_dir)
    run_dir = Path(args.run_dir) if args.run_dir else None
    environments_dir = Path(args.environments_dir) if args.environments_dir else None
    handler = build_handler(runs_dir=runs_dir, run_dir=run_dir, environments_dir=environments_dir)
    server = ThreadingHTTPServer((args.host, args.port), handler)

    target = run_dir if run_dir is not None else runs_dir
    log.info("Viewer serving %s at http://%s:%d", target, args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Viewer stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
