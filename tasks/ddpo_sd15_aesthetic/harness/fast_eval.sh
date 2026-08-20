#!/usr/bin/env bash
# Score a LoRA or full-pipeline checkpoint on 64 generated images, on one GPU.
#
# Same scorer and diagnostics as the final, at a lower sample count.
# The default proxy-tier diagnostic reference is attached automatically. Override the
# sample count or seed and the script suppresses that reference; measure the changed
# tier's own base model before interpreting diagnostic deltas.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: fast_eval.sh <checkpoint-dir> [--samples N] [--seed S]" >&2
  echo "                   [--diagnostic-reference FILE]" >&2
  echo "       fast_eval.sh --base            # the untrained start" >&2
  exit 2
fi

OUT_DIR=${OUT_DIR:-/out}
HARNESS_DIR=${HARNESS_DIR:-/opt/harness}
# The training start at THIS tier -- 64 rows, seed 20269700. Only valid there.
PROXY_REFERENCE=${PROXY_REFERENCE:-${HARNESS_DIR}/proxy_reference.json}

# /tmp is a 256 MiB tmpfs; diffusers, transformers and Triton all cache under it.
export TMPDIR=${TMPDIR:-${OUT_DIR}/fast-eval-cache/tmp}
export HF_HOME=${HF_HOME:-${OUT_DIR}/fast-eval-cache/hf}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${OUT_DIR}/fast-eval-cache/xdg}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${OUT_DIR}/fast-eval-cache/triton}
mkdir -p "${TMPDIR}" "${HF_HOME}" "${XDG_CACHE_HOME}" "${TRITON_CACHE_DIR}"

if [[ "${1:-}" == "--base" ]]; then
  shift
  # No reference: --base IS the reference. Its output is directly usable as one --
    # mean_clip_alignment and mean_pairwise_clip_distance are top-level keys.
  exec python3 "${HARNESS_DIR}/fast_eval.py" \
    --base --out "${OUT_DIR}/fast_eval-base-$(date +%s).json" "$@"
fi

CHECKPOINT=$1
shift

# Attach the tier's reference unless the caller supplied one or moved off the tier.
# A reference is only meaningful at the (samples, seed) it was measured at, so an
# override has to suppress it rather than silently compare across tiers.
attach_reference=1
reason=""
for argument in "$@"; do
  case "${argument}" in
    --diagnostic-reference|--diagnostic-reference=*)
      attach_reference=0; reason="you passed --diagnostic-reference" ;;
    --samples|--samples=*)
      attach_reference=0; reason="--samples moves off the measured tier" ;;
    --seed|--seed=*)
      attach_reference=0; reason="--seed moves off the measured tier" ;;
  esac
done

# Built as one array that is never empty, because `"${empty[@]}"` is an unbound-variable
# error under `set -u` on bash 3.2, so keep one non-empty array.
ARGS=(--checkpoint "${CHECKPOINT}" --out "${OUT_DIR}/fast_eval-$(date +%s).json")

if [[ "${attach_reference}" == "1" ]]; then
  if [[ -f "${PROXY_REFERENCE}" ]]; then
    ARGS+=(--diagnostic-reference "${PROXY_REFERENCE}")
  else
    echo "fast_eval: ${PROXY_REFERENCE} is missing, so the output will carry no" >&2
    echo "fast_eval: diagnostic comparison. Measure this tier with --base and" >&2
    echo "fast_eval: pass the result back with --diagnostic-reference." >&2
  fi
elif [[ -n "${reason}" && "${reason}" != "you passed --diagnostic-reference" ]]; then
  echo "fast_eval: no diagnostic comparison -- ${reason}." >&2
  echo "fast_eval: measure it with '--base ${*}' and pass the result back with" >&2
  echo "fast_eval: --diagnostic-reference. Do NOT compare against the final tier's" >&2
  echo "fast_eval: reference: each tier needs its own base-model measurement." >&2
fi

exec python3 "${HARNESS_DIR}/fast_eval.py" "${ARGS[@]}" "$@"
