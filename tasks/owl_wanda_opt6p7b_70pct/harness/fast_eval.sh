#!/usr/bin/env bash
# Score a pruned checkpoint on WikiText2 raw validation, on one GPU.
#
# The same evaluator arithmetic as the final, on validation rather than test text.
# Re-run a promising pruning change at another calibration seed before treating a
# small change as robust.
#
# Read the sparsity fields. An artifact outside [0.699, 0.701] scores nothing in the
# final, whatever its perplexity here.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: fast_eval.sh <checkpoint-dir>" >&2
  echo "  e.g. fast_eval.sh /out/pruned" >&2
  exit 2
fi

CHECKPOINT=$1
shift

# /tmp is a 256 MiB tmpfs; transformers and torch caches do not fit in it.
OUT_DIR=${OUT_DIR:-/out}
export TMPDIR=${TMPDIR:-${OUT_DIR}/fast-eval-cache/tmp}
export HF_HOME=${HF_HOME:-${OUT_DIR}/fast-eval-cache/hf}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${OUT_DIR}/fast-eval-cache/hf/transformers}
export TORCH_HOME=${TORCH_HOME:-${OUT_DIR}/fast-eval-cache/torch}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${OUT_DIR}/fast-eval-cache/triton}
mkdir -p "${TMPDIR}" "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${TORCH_HOME}" "${TRITON_CACHE_DIR}"

exec python3 /opt/harness/fast_eval.py \
  --checkpoint "${CHECKPOINT}" \
  --data "${FAST_EVAL_DATA:-/assets/data/wikitext2/validation}" \
  --out "${OUT_DIR}/fast_eval-$(date +%s).json" \
  "$@"
