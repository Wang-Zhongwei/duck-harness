#!/usr/bin/env bash
# Submit a Kaggle notebook version to the code competition once its run completes.
#
#   scripts/kaggle_submit_when_complete.sh <kernel> <version> <expected-commit> <message> [log-dir]
#
# Polls `kaggle kernels status` every 2 minutes. On COMPLETE it downloads the output and
# submits ONLY if the run actually played: submission.parquet present, the bundled commit
# matches <expected-commit>, and summary.txt shows actions > 0 and a non-zero mean score.
# Kernel taaf-qwen38-nvfp4-vllm v4 "completed" with a parquet after every game crashed in
# 7.8 s (KeyError at the first turn) -- submitting that would have spent the single daily
# submission on a 0.00. Every outcome is one line on stdout; exit 0 = submitted,
# 1 = kernel failed, 2 = completed but refused, 3 = submit command failed.
set -u
KERNEL=$1; VERSION=$2; EXPECTED_COMMIT=$3; MESSAGE=$4; LOGDIR=${5:-.}
COMPETITION=${KAGGLE_COMPETITION:-arc-prize-2026-arc-agi-3}
KAGGLE=${KAGGLE_BIN:-$(dirname "$0")/../.venv/bin/kaggle}
MARKER="$LOGDIR/submitted.$KERNEL.v$VERSION"; MARKER=${MARKER//\//_}; MARKER="$LOGDIR/$(basename "$MARKER")"
say() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

if [ -e "$MARKER" ]; then say "already submitted (marker $MARKER); nothing to do"; exit 0; fi
prev=""
while :; do
  s=$("$KAGGLE" kernels status "$KERNEL" 2>/dev/null | grep -o 'KernelWorkerStatus\.[A-Z_]*' || true); s=${s:-QUERY_FAILED}
  [ "$s" != "$prev" ] && say "$KERNEL v$VERSION: $s"; prev=$s
  case "$s" in
    *RUNNING|*QUEUED|QUERY_FAILED) sleep 120; continue ;;
    *COMPLETE) break ;;
    *) say "kernel ended in $s -- NOT submitting"; exit 1 ;;
  esac
done

OUT=$(mktemp -d "${TMPDIR:-/tmp}/kaggle-out.XXXXXX")
"$KAGGLE" kernels output "$KERNEL" -p "$OUT" >/dev/null 2>&1 || { say "could not download output -- NOT submitting"; exit 2; }
[ -s "$OUT/submission.parquet" ] || { say "no submission.parquet in output -- NOT submitting"; exit 2; }
grep -q "$EXPECTED_COMMIT" "$OUT/git_status.txt" 2>/dev/null || { say "output is not from commit $EXPECTED_COMMIT ($(head -3 "$OUT/git_status.txt" 2>/dev/null | tr '\n' ' ')) -- NOT submitting"; exit 2; }
actions=$(grep -o 'total actions:\s*[0-9]*' "$OUT/summary.txt" 2>/dev/null | grep -o '[0-9]*$'); actions=${actions:-0}
score=$(grep -o 'mean score:\s*[0-9.]*' "$OUT/summary.txt" 2>/dev/null | grep -o '[0-9.]*$'); score=${score:-0}
say "run summary: actions=$actions mean_score=$score $(grep -o 'total wallclock:.*' "$OUT/summary.txt" 2>/dev/null)"
[ "$actions" -gt 0 ] || { say "run played 0 actions (games crashed) -- NOT submitting"; exit 2; }
awk "BEGIN{exit !($score > 0)}" || { say "mean score is 0.00 -- NOT submitting (override by hand if you really want the slot spent)"; exit 2; }

say "submitting $KERNEL v$VERSION to $COMPETITION"
# Exit status of the SUBMIT, not of the grep: `grep -v` returns 1 when it filters every
# line, which reported v5's successful submission as FAILED.
"$KAGGLE" competitions submit "$COMPETITION" -k "$KERNEL" -v "$VERSION" -f submission.parquet -m "$MESSAGE" 2>&1 | grep -v outdated
if [ "${PIPESTATUS[0]}" -eq 0 ]; then
  touch "$MARKER"
  sleep 20
  say "SUBMITTED. newest submission row: $("$KAGGLE" competitions submissions -c "$COMPETITION" 2>/dev/null | sed -n '3p')"
  exit 0
fi
say "submit command FAILED -- check 'kaggle competitions submissions -c $COMPETITION'"; exit 3
