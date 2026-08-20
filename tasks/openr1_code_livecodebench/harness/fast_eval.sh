#!/usr/bin/env bash
# Score a checkpoint on public LiveCodeBench v4/v5 rows, on one GPU.
#
# Two canonical settings:
#
#   fast_eval.sh <ckpt>            64-row proxy    health check only
#   fast_eval.sh <ckpt> --confirm  204-row slice   what to compare candidates on
#
# The 64-row result is a health signal. Use the disjoint 204-row tier for a
# candidate comparison. Neither slice overlaps the v6 hidden final.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: fast_eval.sh <checkpoint-dir> [--confirm] [--rows N] [--offset N]" >&2
  exit 2
fi

CHECKPOINT=$1
shift

# /tmp is a 256 MiB tmpfs. Model loading and the evaluator's per-worker
# temporaries do not fit in it; the grading pool moves itself to /dev/shm, and
# everything else goes to the output mount.
OUT_DIR=${OUT_DIR:-/out}
export TMPDIR=${TMPDIR:-${OUT_DIR}/tmp}
export HF_HOME=${HF_HOME:-${OUT_DIR}/tmp/hf}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-${OUT_DIR}/tmp/inductor}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${OUT_DIR}/tmp/triton}
export AI4AI_LOCK_ROOT=${AI4AI_LOCK_ROOT:-${OUT_DIR}}
mkdir -p "${TMPDIR}" "${HF_HOME}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

exec python3 /opt/harness/fast_eval.py \
  --checkpoint "${CHECKPOINT}" \
  --data "${FAST_EVAL_DATA:-/assets/data/livecodebench_public}" \
  --out "${OUT_DIR}/fast_eval-$(date +%s%N)-$$.json" \
  "$@"
