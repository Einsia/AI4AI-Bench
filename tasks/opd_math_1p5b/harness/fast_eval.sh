#!/usr/bin/env bash
# Score a checkpoint on all 500 MATH-500 questions x 4 samples, on one GPU.
#
# Same family as the hidden final (AIME24/25@32) and the same grader, at a
# resolution you can afford inside 4 h: about 10.3 min per call on B300.
#
# Read the stderr field. Measured on this task: 92 training steps moved the old
# proxy 0.0128 while re-scoring one checkpoint under four different sampling
# seeds moved it 0.0110. A gap inside stderr is not an improvement.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: fast_eval.sh <checkpoint-dir> [--questions N] [--samples K] [--seed S]" >&2
  exit 2
fi

CHECKPOINT=$1
shift

# /tmp is a 256 MiB tmpfs and vLLM's compile cache does not fit in it.
#
# HOME is the general case: it is /tmp in this image, so any library defaulting to
# ~/.cache/<name> lands in the tmpfs. Four have now hit that -- Triton, inductor, vLLM,
# and flashinfer, which JIT-compiles on Blackwell for want of a prebuilt sm103a cubin.
# Setting HOME and XDG_CACHE_HOME covers the fifth before it happens.
OUT_DIR=${OUT_DIR:-/out}
export HOME=${HOME_OVERRIDE:-${OUT_DIR}/tmp/home}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${OUT_DIR}/tmp/cache}
export TMPDIR=${TMPDIR:-${OUT_DIR}/tmp}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-${OUT_DIR}/tmp/inductor}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${OUT_DIR}/tmp/triton}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-${OUT_DIR}/tmp/vllm}
export FLASHINFER_CACHE_DIR=${FLASHINFER_CACHE_DIR:-${OUT_DIR}/tmp/flashinfer}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-${OUT_DIR}/tmp/flashinfer}
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${TMPDIR}" "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}" "${VLLM_CACHE_ROOT}" "${FLASHINFER_CACHE_DIR}"

# Evaluation sizes vLLM for the full assigned GPU. It must not overlap training
# or another evaluation. Training holds this lock in shared mode; evaluation is
# exclusive. The nvidia-smi check also catches an orphan or a process that did
# not enter through run.sh.
GPU_WORKLOAD_LOCK=${GPU_WORKLOAD_LOCK:-/out/.gpu-workload.lock}
exec 9>"${GPU_WORKLOAD_LOCK}"
if ! flock -n -x 9; then
  echo "fast_eval.sh: training or another evaluation is using the assigned GPU." >&2
  exit 75
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "fast_eval.sh: cannot verify that the assigned GPU is idle: nvidia-smi is absent." >&2
  exit 75
fi
if ! gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
  | awk '$1 ~ /^[0-9]+$/ { print $1 }'); then
  echo "fast_eval.sh: cannot verify that the assigned GPU is idle." >&2
  exit 75
fi
if [[ -n "${gpu_pids}" ]]; then
  echo "fast_eval.sh: GPU compute processes remain after acquiring the workload lock:" >&2
  printf '  pid %s\n' ${gpu_pids} >&2
  echo "fast_eval.sh: stop them and release GPU memory before evaluation." >&2
  exit 75
fi

exec python3 /opt/harness/fast_eval.py \
  --checkpoint "${CHECKPOINT}" \
  --data "${FAST_EVAL_DATA:-/assets/data/math500}" \
  --out "${OUT_DIR}/fast_eval-$(date +%s).json" \
  "$@"
