#!/usr/bin/env bash
# Rebuild the vLLM 0.27.1 wheelhouse with CORRECT environment markers.
#
# WHY THE FIRST BUILD WAS BROKEN: it resolved with the login node's python 3.11.13.
# vLLM 0.27.1 gates two requirements on `python_version > "3.11"`:
#     six>=1.16.0;              python_version > "3.11"
#     setuptools<81.0.0,>=77.0.3; python_version > "3.11"
# Both evaluated FALSE on 3.11, so pip silently dropped them. `--python-version 3.12`
# only steers wheel-TAG selection; it does not drive marker evaluation. Result on Kaggle
# (python 3.12, where the markers are TRUE):
#   1. "No matching distribution found for six>=1.16.0" -> kernel v2 died at install.
#   2. setuptools resolved to 84.0.0 via another package's unconstrained requirement,
#      which VIOLATES vLLM's <81.0.0 -- so fixing six alone would have failed next.
# Fix: resolve with a real python 3.12 interpreter so every marker is evaluated as Kaggle
# will evaluate it.
set -u
PY=/home/018270239/.claude/jobs/92269300/tmp/py312/bin/python
D=/home/018270239/arc-agi-3/model-staging/vllm-0271-wheelhouse
rm -rf "$D"
mkdir -p "$D"

PLATS=()
for v in $(seq 17 40); do
  PLATS+=(--platform "manylinux_2_${v}_x86_64")
done
PLATS+=(--platform manylinux2014_x86_64 --platform linux_x86_64 --platform any)

"$PY" -m pip download "vllm==0.27.1" \
  -d "$D" \
  --only-binary=:all: \
  "${PLATS[@]}" \
  2>&1 | tail -12

echo "--- wheel count / size ---"
ls -1 "$D"/*.whl | wc -l
du -sh "$D"
echo "--- marker-gated packages that v2 was missing ---"
ls "$D" | grep -iE "^(six|setuptools)-" || echo "STILL MISSING"
