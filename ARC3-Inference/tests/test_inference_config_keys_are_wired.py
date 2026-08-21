"""Every key in configs/inference.json must reach a real run.

A dead key is worse than a missing one: it reads as a setting the harness
honours. `server.max_images: 4` sat unread on this branch for weeks, and was
then copied onto the Kaggle SGLang argv "for parity with the vLLM runs" -- a
parity that never existed, because no local run ever passed the flag. That cap
rejected 82,135 requests and scored the submission 0.

The local chain is JSON -> `$(CONFIG_VALUE) <dotted.key>` in the Makefile ->
exported variable -> vLLM argv or agent env var. The Kaggle chain is the same
Makefile export -> `KAGGLE_*` -> `_kaggle_env` in inference/framework/kaggle.py.
Either counts as wired; a key on neither path is dead.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _REPO_ROOT / "configs" / "inference.json"
_MAKEFILE_PATH = _REPO_ROOT / "Makefile"

# Sections whose members are values of one setting rather than settings
# themselves, so the audit compares the section, not each member.
_VALUE_BLOB_SECTIONS = {
    "environment",
    "deployment",
    "experiments",
}


def _dotted_keys(config: dict) -> list[str]:
    keys: list[str] = []
    for section, value in config.items():
        if not isinstance(value, dict) or section in _VALUE_BLOB_SECTIONS:
            keys.append(section)
            continue
        keys.extend(f"{section}.{name}" for name in value)
    return keys


def _is_wired(dotted_key: str, makefile: str) -> bool:
    return re.search(rf"CONFIG_VALUE\)\s*{re.escape(dotted_key)}\b", makefile) is not None


def _python_sources() -> str:
    roots = (
        _REPO_ROOT / "inference",
        _REPO_ROOT / "viewer",
        _REPO_ROOT.parent / "tufa-arc-agi-framework" / "src" / "taaf",
    )
    return "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for root in roots
        if root.is_dir()
        for path in root.rglob("*.py")
    )


def test_every_inference_config_key_is_read_by_the_makefile() -> None:
    config = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    makefile = _MAKEFILE_PATH.read_text(encoding="utf-8")

    dead = [key for key in _dotted_keys(config) if not _is_wired(key, makefile)]

    assert not dead, (
        "configs/inference.json declares settings nothing reads: "
        + ", ".join(sorted(dead))
        + ". Wire each into the Makefile as $(CONFIG_VALUE) <dotted.key>, or "
        "delete it. A key the harness does not honour will eventually be "
        "mistaken for one it does."
    )


def test_every_config_backed_makefile_variable_reaches_a_run() -> None:
    """Reading a key into a variable nothing uses is the same bug, one level down.

    A variable is live if the Makefile expands it into a recipe, or exports it
    to the environment where the agent or the Kaggle launcher reads it back.
    """
    makefile = _MAKEFILE_PATH.read_text(encoding="utf-8")
    python_sources = _python_sources()

    definitions = re.findall(
        r"^([A-Z_][A-Z0-9_]*)\s*\??=\s*\$\(shell \$\(CONFIG_VALUE\)\s*([a-z_.]+)",
        makefile,
        re.M,
    )
    assert definitions, "no $(CONFIG_VALUE) reads found -- the audit regex has rotted"

    orphans = [
        f"{variable} (from {key})"
        for variable, key in definitions
        if not re.search(rf"\$[({{]{re.escape(variable)}[)}}]", makefile)
        and not (
            re.search(rf"^export\s+{re.escape(variable)}\s*$", makefile, re.M)
            and variable in python_sources
        )
    ]

    assert not orphans, (
        "the Makefile reads config into variables that never reach a run: "
        + ", ".join(sorted(orphans))
        + ". Expand each into a recipe, or export it and read it in Python."
    )
