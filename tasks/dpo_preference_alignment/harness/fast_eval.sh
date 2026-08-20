#!/usr/bin/env bash
# Score a checkpoint on the 128-row IFEval proxy: greedy, one GPU.
#
# The payload reports 128 fixed rows, actual elapsed time, descriptive binomial
# stderr, generated-token counts and the exact count reaching the 1280-token cap.
# Use row-level matched repeats for candidate comparisons; stderr is not seed or
# paired uncertainty.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: fast_eval.sh <checkpoint-dir> [--rows N] [--gpus N]" >&2
  exit 2
fi

CHECKPOINT=$1
shift

# Caches go on /out, not /tmp. /tmp is a tmpfs and docker adds noexec to every
# --tmpfs, so a kernel a library JIT-compiles under /tmp can be written and then
# never loaded -- dlopen fails with "failed to map segment from shared object".
# /out is the writable bind every phase has. See orchestrator/container.py, where
# this bit the project four times.
OUT_DIR=${OUT_DIR:-/out}
export AI4AI_LOCK_ROOT=${AI4AI_LOCK_ROOT:-/out}
export TMPDIR=${TMPDIR:-${OUT_DIR}/tmp}
export HF_HOME=${HF_HOME:-${OUT_DIR}/tmp/hf}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-${OUT_DIR}/tmp/inductor}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${OUT_DIR}/tmp/triton}
mkdir -p "${TMPDIR}" "${HF_HOME}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

exec python3 /opt/harness/fast_eval.py \
  --checkpoint "${CHECKPOINT}" \
  --reference "${POLICY_START:-/assets/models/policy_start}" \
  --data "${FAST_EVAL_DATA:-/assets/data/ifeval_proxy}" \
  --out "${OUT_DIR}/fast_eval-$(date +%s%N)-$$.json" \
  "$@"
