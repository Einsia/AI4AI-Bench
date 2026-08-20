#!/usr/bin/env bash
# Activation-aware unstructured pruning | OPT-6.7B | C4 calibration | single GPU.
#
# Defaults are environment variables and extra arguments are forwarded to the
# pruning implementation.
#
# ONE VALUE IS NOT REALLY YOURS. SPARSITY_RATIO is checked against the exported
# weights by the scoring harness, which you cannot reach, and an artifact outside
# [0.699, 0.701] scores nothing. That is not a rule about method: a less sparse model
# has lower perplexity for a reason that has nothing to do with pruning quality, so
# it is not the same measurement. Everything else about how you reach 70% is open.

set -euo pipefail

# ---- paths (read-only mounts) ----
DENSE_MODEL=${DENSE_MODEL:-/assets/models/opt-dense}
CALIBRATION_DATA=${CALIBRATION_DATA:-/assets/data/c4}
OUTPUT_DIR=${OUTPUT_DIR:-/out}
# One completed construction is legitimate; it is published as numeric candidate 1.
PRUNED_DIR=${PRUNED_DIR:-${OUTPUT_DIR}/work/pruned}
# The pinned OWL tree, editable. An edit here takes effect on the next run.
OWL_SOURCE=${OWL_SOURCE:-/workspace/owl}

# ---- algorithm ----
# wanda_owl is OWL's non-uniform layer allocation on top of Wanda's
# activation-aware score. wanda is the uniform in-family control.
PRUNE_METHOD=${PRUNE_METHOD:-wanda_owl}
SPARSITY_RATIO=${SPARSITY_RATIO:-0.7}
SPARSITY_TYPE=${SPARSITY_TYPE:-unstructured}
# OWL's two allocation hyperparameters. LAMBDA bounds how far a layer's sparsity may
# deviate from the global target; HYPER_M scales the outlier ratio that decides the
# direction. The shipped pair is (0.08, 5.0); both are editable.
LAMBDA=${LAMBDA:-0.08}
HYPER_M=${HYPER_M:-5.0}
# Upstream's bisection on a per-row alpha instead of a fixed per-row count. Off in
# the shipped recipe.
USE_VARIANT=${USE_VARIANT:-false}

# ---- calibration ----
# THIS WAS FROZEN AND IS NOW YOURS. The old task pinned 128 sequences at seed 0 for
# every run and enforced it with a checker. The mount holds the whole C4 train shard,
# so more sequences is a real option -- roughly linear in cost, since the pass is one
# forward per sequence per decoder layer.
#
CALIBRATION_SAMPLES=${CALIBRATION_SAMPLES:-128}
# The seed decides which sequences are drawn. Its effect on the score has never been
# measured on this task -- see instruction.md.
CALIBRATION_SEED=${CALIBRATION_SEED:-0}

# The host supplies a formal wall clock. Direct runs default to no internal limit.
MAX_WALL_TIME_SECONDS=${MAX_WALL_TIME_SECONDS:-0}
DEADLINE_RESERVE_SECONDS=${DEADLINE_RESERVE_SECONDS:-600}

SAVE_UNIT="${SAVE_UNIT:-candidate}"          # candidate
SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"  # 0 means unlimited
[[ "${SAVE_UNIT}" == "candidate" ]] || { echo "OWL supports SAVE_UNIT=candidate" >&2; exit 78; }
[[ "${SAVE_INTERVAL}" =~ ^[1-9][0-9]*$ ]] || { echo "SAVE_INTERVAL must be positive" >&2; exit 78; }
[[ "${SAVE_TOTAL_LIMIT}" =~ ^[0-9]+$ ]] || { echo "SAVE_TOTAL_LIMIT must be nonnegative" >&2; exit 78; }

# /tmp is a 256 MiB tmpfs. The C4 shard decompresses well past that, and
# transformers, torch and datasets all cache there by default.
export TMPDIR=${TMPDIR:-${OUTPUT_DIR}/prune-cache/tmp}
export HF_HOME=${HF_HOME:-${OUTPUT_DIR}/prune-cache/hf}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${OUTPUT_DIR}/prune-cache/hf/datasets}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${OUTPUT_DIR}/prune-cache/hf/transformers}
export TORCH_HOME=${TORCH_HOME:-${OUTPUT_DIR}/prune-cache/torch}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${OUTPUT_DIR}/prune-cache/triton}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${OUTPUT_DIR}/prune-cache/xdg}
mkdir -p "${TMPDIR}" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" \
         "${TORCH_HOME}" "${TRITON_CACHE_DIR}" "${XDG_CACHE_HOME}"

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export WANDB_MODE=${WANDB_MODE:-offline}

# The dense model and the calibration data are the two fixed inputs. /assets is a
# read-only mount, but that only stops them being edited -- nothing stopped this
# script from being pointed somewhere else, and weights written under /workspace
# would ride into the pruning container inside candidate.patch, which is generated
# with --binary. So refuse a source outside /assets.
#
# Set ALLOW_UNPINNED_INPUTS=1 to override, which the orchestrator never does.
for _path in "${DENSE_MODEL}" "${CALIBRATION_DATA}"; do
  case "${_path}" in
    /assets/*) ;;
    *)
      if [[ "${ALLOW_UNPINNED_INPUTS:-0}" != "1" ]]; then
        echo "run.sh: fixed inputs must live under /assets, got '${_path}'." >&2
        echo "run.sh: the dense model, the calibration data and the evaluator are" >&2
        echo "run.sh: the fixed inputs; everything else about the method is yours." >&2
        exit 78
      fi
      echo "run.sh: WARNING fixed input outside /assets: ${_path}" >&2
      ;;
  esac
done

mkdir -p "${OUTPUT_DIR}"

prune_command=(python3 /workspace/prune.py \
  --dense-model "${DENSE_MODEL}" \
  --calibration-data "${CALIBRATION_DATA}" \
  --owl-source "${OWL_SOURCE}" \
  --output "${OUTPUT_DIR}" \
  --pruned-dir "${PRUNED_DIR}" \
  --prune-method "${PRUNE_METHOD}" \
  --sparsity-ratio "${SPARSITY_RATIO}" \
  --sparsity-type "${SPARSITY_TYPE}" \
  --lambda "${LAMBDA}" \
  --hyper-m "${HYPER_M}" \
  --use-variant "${USE_VARIANT}" \
  --calibration-samples "${CALIBRATION_SAMPLES}" \
  --calibration-seed "${CALIBRATION_SEED}" \
  "$@")

if [[ "${MAX_WALL_TIME_SECONDS}" != "0" ]]; then
  prune_budget=$(( MAX_WALL_TIME_SECONDS - DEADLINE_RESERVE_SECONDS ))
  if (( prune_budget <= 0 )); then
    echo "run.sh: wall clock leaves no time before the export reserve" >&2
    exit 78
  fi
  prune_command=(timeout --signal=TERM --kill-after=60 "${prune_budget}" "${prune_command[@]}")
fi

"${prune_command[@]}"

python3 /opt/harness/save_checkpoint.py \
  --output "${OUTPUT_DIR}" \
  --progress 1 \
  --source "${PRUNED_DIR}" \
  --payload-name . \
  --retention "${SAVE_TOTAL_LIMIT}"
