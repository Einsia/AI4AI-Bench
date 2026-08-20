#!/usr/bin/env bash
# Score a checkpoint on 256 Sokoban boards: four banks at seeds 4242-4245.
#
# Same environment, same reward, same rollout code and same sampling as the hidden
# final. Different boards, on purpose -- see the overlap note in the task's
# task.toml.
#
# Read the stderr field. Each board is one Bernoulli draw, so 256 boards puts the
# error on the mean near 0.021 at the baseline's rate. The recorded maintainer gain
# over the training start is +0.0205 on the 512-board final, which is about one of
# those. A gap inside stderr is not an improvement.
#
# The cost of this call is NOT recorded, on any device. Time your first one.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: fast_eval.sh <checkpoint-dir> [--environment-seeds S...] [--boards N]" >&2
  echo "                    [--engine-seed S]" >&2
  exit 2
fi

CHECKPOINT=$1
shift

# /tmp is a 256 MiB tmpfs and vLLM's compile cache does not fit in it. Triton's
# failure mode reads as an ImportError rather than a full filesystem, so redirect
# before anything imports torch.
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
export TMPDIR=${EVAL_DIR}/cache/tmp
export RAY_TMPDIR=${EVAL_DIR}/cache/ray
export TORCHINDUCTOR_CACHE_DIR=${EVAL_DIR}/cache/inductor
export TRITON_CACHE_DIR=${EVAL_DIR}/cache/triton
export VLLM_CACHE_ROOT=${EVAL_DIR}/cache/vllm
export HF_HOME=${EVAL_DIR}/cache/huggingface
mkdir -p "${TMPDIR}" "${RAY_TMPDIR}" "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}" "${VLLM_CACHE_ROOT}" "${HF_HOME}"

exec python3 /opt/harness/gpu_phase_lock.py \
  --lock "${AI4AI_GPU_PHASE_LOCK:-/out/.ai4ai-gpu-phase.lock}" \
  --label "ragen-fast-eval:${RUN_ID}" \
  -- python3 /opt/harness/fast_eval.py \
  --checkpoint "${CHECKPOINT}" \
  --out "${EVAL_DIR}/summary.json" \
  "$@"
