#!/usr/bin/env bash
# Score a checkpoint on the 512-pair proxy: RewardBench v1, one GPU.
#
# The same rows, weights and aggregation the hidden final uses, on 512 of its 2985
# pairs. A proxy point and a final point are the same unit -- which was not true on
# the reference protocol, where the proxy was 64 UltraFeedback pairs and the final was
# RewardBench.
#
# Read the stderr field. It is propagated through the metric's own weights rather
# than treated as a binomial over 512 rows, because the score is a weighted mean of
# 23 subset accuracies inside four sections. A gap inside stderr is not an
# improvement.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: fast_eval.sh <checkpoint-dir>" >&2
  exit 2
fi

CHECKPOINT=$1
shift

# /tmp is a 256 MiB tmpfs; the transformers and safetensors caches do not fit.
OUT_DIR=${OUT_DIR:-/out}
export TMPDIR=${TMPDIR:-${OUT_DIR}/tmp}
export HF_HOME=${HF_HOME:-${OUT_DIR}/tmp/hf}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-${OUT_DIR}/tmp/inductor}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${OUT_DIR}/tmp/triton}
export AI4AI_LOCK_ROOT=${AI4AI_LOCK_ROOT:-${OUT_DIR}}
mkdir -p "${TMPDIR}" "${HF_HOME}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

exec python3 /opt/harness/fast_eval.py \
  --checkpoint "${CHECKPOINT}" \
  --base-model "${BASE_MODEL:-/assets/models/base}" \
  --data "${FAST_EVAL_DATA:-/assets/data/rewardbench_proxy}" \
  --out "${OUT_DIR}/fast_eval-$(date +%s%N)-$$.json" \
  "$@"
