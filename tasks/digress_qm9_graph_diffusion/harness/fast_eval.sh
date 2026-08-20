#!/usr/bin/env bash
# Sample 2000 molecules from a checkpoint and score them, on one GPU.
#
# Same pinned tree and metric code as final, but an independent unseeded draw at a
# smaller sample count. Time the first call rather than assuming a fixed cost.
#
# Read the NLL as well as the score. `score` here is
# validity x uniqueness x novelty and is MAXIMISED; the task is ranked on NLL,
# which is MINIMISED. They can move in opposite directions.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: fast_eval.sh <checkpoint-dir-or-file> [--samples N] [--seed S]" >&2
  exit 2
fi

CHECKPOINT=$1
shift

# /tmp is a 256 MiB tmpfs. RDKit, matplotlib, torch and hydra all cache below it,
# and the dataset copy plus sampling output will not fit.
OUT_ROOT=${OUT_DIR:-/out}
resolved_out=$(readlink -m "${OUT_ROOT}")
case "${resolved_out}" in
  /out|/out/*) ;;
  *)
    echo "fast_eval.sh: OUT_DIR must be /out or below it, got '${OUT_ROOT}'." >&2
    exit 78
    ;;
esac
RUN_ID=${FAST_EVAL_RUN_ID:-$(date -u +%Y%m%dT%H%M%S%N)-$$}
EVAL_DIR=${FAST_EVAL_DIR:-${resolved_out}/fast-eval-runs/${RUN_ID}}
EVAL_DIR=$(readlink -m "${EVAL_DIR}")
case "${EVAL_DIR}" in
  /out/*) ;;
  *) echo "fast_eval.sh: FAST_EVAL_DIR must stay below /out, got '${EVAL_DIR}'." >&2; exit 78 ;;
esac
mkdir -p "$(dirname "${EVAL_DIR}")"
if ! mkdir "${EVAL_DIR}" 2>/dev/null; then
  echo "fast_eval.sh: refusing to reuse existing evaluation directory ${EVAL_DIR}" >&2
  exit 73
fi
export TMPDIR=${EVAL_DIR}/tmp
# Override HOME because the image default points at the size-limited tmpfs.
export HOME=${EVAL_DIR}/cache/home
export XDG_CACHE_HOME=${EVAL_DIR}/cache/xdg
export MPLCONFIGDIR=${EVAL_DIR}/cache/matplotlib
mkdir -p "${TMPDIR}" "${HOME}" "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"

exec python3 /opt/harness/gpu_phase_lock.py \
  --lock "${AI4AI_GPU_PHASE_LOCK:-/out/.ai4ai-gpu-phase.lock}" \
  --label "digress-fast-eval:${RUN_ID}" \
  -- python3 /opt/harness/fast_eval.py \
  --checkpoint "${CHECKPOINT}" \
  --data "${QM9_DATA:-/assets/data/qm9_no_h}" \
  --out "${EVAL_DIR}/summary.json" \
  "$@"
