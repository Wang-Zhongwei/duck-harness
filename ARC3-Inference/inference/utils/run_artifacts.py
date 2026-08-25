"""Helpers for per-run artifact directories, git metadata, and file logging."""
from __future__ import annotations

import logging
import re
import subprocess
from datetime import datetime
from pathlib import Path


log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RUN_DIR_NAME_RE = re.compile(
    r"^(?P<timestamp>\d{8}_\d{6})(?:_(?P<suffix>\d{2}))?(?:(?P<legacy>_run)|_(?P<label>[A-Za-z0-9][A-Za-z0-9._-]*))?$"
)
# Kaggle runs live at runs/kaggle/<kernel-slug>/v<NN>/ and are named by kernel version.
_KAGGLE_VERSION_DIR_RE = re.compile(r"^v(?P<version>\d{2,})$")

# Directory names that are containers or run internals, never runs themselves.
RESERVED_DIR_NAMES = frozenset({
    "local", "kaggle", "analysis", "aggregates", "semi_private_scores", "traces",
    "dry-runs", "rejected-pushes", "_pending",
    "passes", "seeds", "src", "artifacts", "transcripts", "movies", "prompts",
    "solver_analysis", "failed_passes", "kaggle-output", "kaggle_output", ".venv",
})



def _match_run_dir_name(name: str) -> re.Match[str] | None:
    match = _RUN_DIR_NAME_RE.fullmatch(name)
    if match is None:
        return None
    if str(match.group("label") or "").lower() == "run":
        return None
    return match



def is_run_dir_name(name: str) -> bool:
    """Return whether a directory name matches the current run format."""
    return _match_run_dir_name(name) is not None


def is_kaggle_version_dir_name(name: str) -> bool:
    """Return whether a directory name is a Kaggle kernel version dir (``v07``)."""
    return _KAGGLE_VERSION_DIR_RE.fullmatch(name) is not None


def is_selectable_run_dir_name(name: str) -> bool:
    """Return whether a run directory should participate in automatic discovery."""
    if is_kaggle_version_dir_name(name):
        return True
    match = _match_run_dir_name(name)
    return match is not None and match.group("legacy") is None


def run_dir_sort_key(path_or_name: str | Path) -> tuple[int, str, int, int, str]:
    """Return a deterministic sort key for run directory names.

    Total by construction -- it must never raise, because the viewer sorts whatever
    discovery hands it and a ValueError there surfaces as an opaque HTTP 500. Ordering:
    timestamped runs (rank 2) sort above Kaggle version dirs (rank 1) above anything
    else (rank 0); every branch returns the same tuple shape so comparisons are safe.
    """
    name = Path(path_or_name).name
    match = _match_run_dir_name(name)
    if match is not None:
        return (
            2,
            match.group("timestamp"),
            int(match.group("suffix") or 0),
            0 if match.group("legacy") else 1,
            match.group("label") or "",
        )
    version = _KAGGLE_VERSION_DIR_RE.fullmatch(name)
    if version is not None:
        return (1, "", int(version.group("version")), 0, name)
    return (0, "", 0, 0, name)


def iter_run_dirs(root: str | Path, *, max_depth: int = 3) -> list[Path]:
    """Find run directories under ``root``, descending through the local/ and
    kaggle/<slug>/ container layout.

    A directory counts as a run when its name is selectable AND it holds run data;
    discovery does not descend into a directory once it has been classified as a run.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    found: list[Path] = []

    def walk(directory: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(p for p in directory.iterdir() if p.is_dir())
        except OSError:
            return
        for entry in entries:
            if entry.name in RESERVED_DIR_NAMES or entry.name.startswith("."):
                if entry.name in {"local", "kaggle", "analysis"}:
                    walk(entry, depth + 1)
                continue
            if is_selectable_run_dir_name(entry.name) and _holds_run_data(entry):
                found.append(entry)
                continue
            walk(entry, depth + 1)

    walk(root, 0)
    return found


def _holds_run_data(directory: Path) -> bool:
    """Whether a directory holds run results.

    Viewer payloads live in ``<run>/artifacts/`` or under per-pass / per-seed artifact
    directories -- never at the run root -- so a top-level glob alone would classify
    every viewer-only run as "not a run".
    """
    for marker in ("evaluation.json", "run_config.json", "benchmark.json"):
        if (directory / marker).exists():
            return True
    for pattern in (
        "*_viewer_data.json",
        "artifacts/*viewer_data.json",
        "passes/*/artifacts/*viewer_data.json",
        "seeds/*/artifacts/*viewer_data.json",
    ):
        if any(directory.glob(pattern)):
            return True
    return False


def sanitize_run_name(name: str | None) -> str:
    """Normalize a user-provided run name for filesystem-safe run directories."""
    raw = str(name or "").strip()
    if not raw:
        return ""
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("._-")
    normalized = re.sub(r"-{2,}", "-", normalized)
    normalized = re.sub(r"_{2,}", "_", normalized)
    normalized = re.sub(r"\.{2,}", ".", normalized)
    if not normalized:
        return ""
    if normalized.lower() == "run":
        return "named-run"
    return normalized


def get_git_info() -> tuple[str, str]:
    """Return the current git commit hash and working tree diff."""
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=_PROJECT_ROOT,
        text=True,
    ).strip()
    try:
        diff = subprocess.check_output(
            ["git", "diff"],
            cwd=_PROJECT_ROOT,
            text=True,
        )
    except subprocess.CalledProcessError:
        diff = ""
    return commit, diff


def save_git_info(base_dir: Path) -> Path:
    """Save git commit and diff to a file inside the run directory."""
    path = base_dir / "git_info.txt"
    try:
        commit, diff = get_git_info()
        path.write_text(f"commit: {commit}\n\n{diff}", encoding="utf-8")
    except Exception as exc:
        log.warning("failed to capture git info: %s", exc)
        path.write_text(f"git info unavailable: {exc}\n", encoding="utf-8")
    return path


def setup_experiment_directory(base_output_dir: str | Path = "runs", *, run_name: str | None = None) -> tuple[Path, Path]:
    """Create a timestamped run directory and save git metadata."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = sanitize_run_name(run_name)
    root_dir = Path(base_output_dir)
    attempt_index = 0
    while True:
        if label:
            candidate_name = f"{timestamp}_{label}" if attempt_index == 0 else f"{timestamp}_{attempt_index:02d}_{label}"
        else:
            candidate_name = timestamp if attempt_index == 0 else f"{timestamp}_{attempt_index:02d}"
        if (root_dir / f"{candidate_name}_run").exists():
            attempt_index += 1
            continue

        base_dir = root_dir / candidate_name
        try:
            base_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            attempt_index += 1
            continue

    log_file = base_dir / "logs.log"
    save_git_info(base_dir)
    return base_dir, log_file


def setup_logging_for_experiment(log_file_path: str | Path, fmt: str) -> Path:
    """Attach a file handler for the current run's log file."""
    log_path = Path(log_file_path)
    root_logger = logging.getLogger()

    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(fmt)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(root_logger.level or logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    return log_path
