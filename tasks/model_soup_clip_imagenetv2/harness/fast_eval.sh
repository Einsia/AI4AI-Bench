#!/usr/bin/env bash
# Score a soup on the 2000 proxy rows: offsets 0-1 of each of 1000 classes.
#
# This is the final forward path over offsets 0-1 of every class. The metric is
# deterministic for a fixed checkpoint and row set. Use `--classes` with different
# seeds to test an ordering on separately sampled class subsets.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: fast_eval.sh <checkpoint> [--classes N] [--seed S]" >&2
  exit 2
fi

CHECKPOINT=$1
shift

OUT_DIR=${OUT_DIR:-/out}
# /tmp is a small tmpfs. torch's extension and inductor caches do not belong in
# it; OPD lost an evaluation run to exactly this, with an error that named an
# ImportError rather than a full filesystem.
export TMPDIR=${TMPDIR:-${OUT_DIR}/fast-eval-cache/tmp}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-${OUT_DIR}/fast-eval-cache/torch-extensions}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-${OUT_DIR}/fast-eval-cache/inductor}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${OUT_DIR}/fast-eval-cache/xdg}
mkdir -p "${TMPDIR}" "${TORCH_EXTENSIONS_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${XDG_CACHE_HOME}"

exec python3 /opt/harness/fast_eval.py \
  --checkpoint "${CHECKPOINT}" \
  --data "${PROXY_DATA:-/assets/data/imagenetv2_proxy}" \
  --clip-cache "${CLIP_CACHE:-/assets/models/clip}" \
  --out "${OUT_DIR}/fast_eval-$(date +%s).json" \
  "$@"
