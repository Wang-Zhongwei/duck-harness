#!/usr/bin/env python3
"""Patch vLLM's MTP drafter against the CUDA-graph buffer race that kills MTP>=2 runs.

Background
----------
Long runs with `speculative_config.num_speculative_tokens >= 2` die with a sticky
CUDA device-side assert (error 710). Ours died twice in one run:
GPU1 at 26 min (`Failed to initialize the TMA descriptor 710` ->
`flash_fwd_launch_template.h:199: device-side assert`) and GPU0 at 10.8 h in the
Triton GDN kernel.

Root cause (vllm-project/vllm#40756, brasrox 2026-08-17, which RETRACTS that
author's earlier July claim): the sequential MTP draft loop writes the
CUDA-graph input buffers and the draft model reads them without an intervening
barrier, so a slow write can be observed stale. The loop runs K times per step,
which is why k=3 loses the race more often than k=2.

Fix, validated independently by maqifrnswa (2026-08-20) on Qwen3.8-27B-NVFP4 /
sm_120 / vLLM 0.27.1 with a 4096-request stress at concurrency 256, zero errors:
insert a stream synchronize immediately after the buffer writes.

Why a script and not a vendored file
------------------------------------
There is no persistent vLLM install in this repo: each run materialises its own
venv under runs/<run>/passes/<n>/src/ARC3-Inference/.venv, so the patch must be
applied at run-setup time, after `uv sync`, against whatever tree exists then.

Usage
-----
    python scripts/patch_vllm_mtp_race.py [--venv PATH | --site-packages PATH]
                                          [--check] [--revert]

Idempotent. Exit 0 = patched or already patched; 1 = no target found, or no target
carried the anchor (reported as MTP UNPROTECTED); 2 = check failed (not patched).
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

ANCHOR = (
    "            self.input_ids[:batch_size] = input_ids\n"
    "            self.hidden_states[:batch_size] = hidden_states\n"
)
INJECT = "            torch.accelerator.current_stream().synchronize()\n"
MARKER = "torch.accelerator.current_stream().synchronize()"


# vLLM moved the draft loop between releases, so the file to patch depends on the
# version in the tree:
#   0.19.0  -- the buffer writes are in eagle.py
#   0.27.1  -- EagleProposer is a 22-line subclass; the loop moved to
#              llm_base_proposer.py (anchor at lines 713-714, and no synchronize()
#              anywhere in its 1,886 lines)
# Patching only eagle.py against 0.27.1 reports "anchor absent" and protects NOTHING.
TARGET_RELPATHS = (
    "vllm/v1/spec_decode/llm_base_proposer.py",
    "vllm/v1/spec_decode/eagle.py",
)


def find_targets(venv: Path | None, site_packages: Path | None = None) -> list[Path]:
    roots: list[str] = []
    if site_packages:
        # Kaggle installs vLLM with `pip install --target <dir>`, which produces a FLAT
        # site-packages with no lib/python3.*/ component, so the venv globs below miss it
        # entirely and the patch silently applies to nothing.
        roots.append(str(site_packages))
    if venv:
        roots.append(f"{venv}/lib/python3.*/site-packages")
    if not roots:
        roots += [
            ".venv/lib/python3.*/site-packages",
            "runs/*/passes/*/src/ARC3-Inference/.venv/lib/python3.*/site-packages",
        ]
    hits: list[Path] = []
    for r in roots:
        for rel in TARGET_RELPATHS:
            hits += [Path(p) for p in glob.glob(f"{r}/{rel}")]
    return sorted(set(hits))


def apply(path: Path, revert: bool, check: bool) -> int:
    text = path.read_text()
    patched = MARKER in text

    if check:
        print(f"{'PATCHED  ' if patched else 'UNPATCHED'} {path}")
        return 0 if patched else 2

    if revert:
        if not patched:
            print(f"already clean  {path}")
            return 0
        path.write_text(text.replace(ANCHOR + INJECT, ANCHOR))
        print(f"REVERTED       {path}")
        return 0

    if patched:
        print(f"already patched {path}")
        return 0
    if ANCHOR not in text:
        # Not fatal on its own: with two candidate files per tree, the release that does
        # not host the loop legitimately lacks the anchor. main() decides, based on
        # whether ANY file in the tree ended up patched.
        print(f"anchor absent  {path}")
        return 3
    new = text.replace(ANCHOR, ANCHOR + INJECT, 1)
    path.write_text(new)
    # verify what we actually wrote, do not trust the in-memory string
    if MARKER not in path.read_text():
        print(f"VERIFY FAILED  {path}", file=sys.stderr)
        return 1
    print(f"PATCHED        {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--venv", type=Path, default=None,
                    help="venv root; default: search .venv and runs/*/passes/*/src/**/.venv")
    ap.add_argument("--site-packages", type=Path, default=None,
                    help="flat site-packages dir, as produced by `pip install --target` "
                         "(the Kaggle layout); searched directly, not under lib/python3.*")
    ap.add_argument("--check", action="store_true", help="report status, change nothing")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()

    targets = find_targets(a.venv, a.site_packages)
    if not targets:
        print("MTP UNPROTECTED: none of " + ", ".join(TARGET_RELPATHS) + " found",
              file=sys.stderr)
        return 1
    codes = [apply(t, a.revert, a.check) for t in targets]
    if a.revert:
        return max(codes)
    if a.check:
        # 0 from apply() means that file is patched. One patched file is enough: only
        # one of the two hosts the loop in any given release.
        return 0 if 0 in codes else 2
    if 0 in codes:
        return 0
    # Every target failed. Refusing to guess at a patch site is right, but it must NOT
    # read as success -- a reassuring exit code here is what leaves MTP>=2 exposed to the
    # race that cost a 10.8 h run.
    print("MTP UNPROTECTED: no target in this tree carried the anchor "
          "(vLLM layout changed; refusing to guess)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
