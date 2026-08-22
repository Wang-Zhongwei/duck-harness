#!/usr/bin/env bash
# Resolve the wheelhouse lock EXACTLY as Kaggle will, without a GPU minute.
#
# Two things must both be right or this proves nothing:
#   1. python 3.12 interpreter  -> environment markers evaluate as Kaggle evaluates them
#      (this is what v2 got wrong: 3.11 dropped six + vLLM's setuptools<81 pin)
#   2. --platform overrides     -> this login node is glibc 2.17, so manylinux_2_28 wheels
#      are not candidates locally and every one of them reads as "no matching version"
set -u
PY=/home/018270239/.claude/jobs/92269300/tmp/py312/bin/python
D=/home/018270239/arc-agi-3/model-staging/vllm-0271-wheelhouse

PLATS=()
for v in $(seq 17 40); do
  PLATS+=(--platform "manylinux_2_${v}_x86_64")
done
PLATS+=(--platform manylinux2014_x86_64 --platform linux_x86_64 --platform any)

"$PY" -m pip install --no-index \
  --find-links "$D" \
  --requirement "$D/requirements.lock" \
  --target /tmp/dryrun-vllm \
  --upgrade --ignore-installed --only-binary :all: --no-compile \
  --disable-pip-version-check --no-warn-conflicts \
  --python-version 3.12 \
  "${PLATS[@]}" \
  --dry-run 2>&1 | tail -14
