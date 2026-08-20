#!/usr/bin/env bash
# Discrete graph diffusion | QM9 without hydrogens | Lightning.
#
# Absorbed from the reference protocol's baseline/recipe.toml. That file was a TOML recipe
# read by a wrapper that validated it against an allowlist -- seed, learning rate,
# depth, batch size, diffusion steps -- and refused anything else. The allowlist is
# gone in v1, so the recipe is gone with it: every value it held is a shell variable
# with a default, and the file that reads them is yours.
#
# Change a default, export a different value, or append Hydra overrides as
# positional arguments -- the last wins, same as upstream.

set -euo pipefail

# ---- paths ----
# The one fixed input. /assets is a read-only mount and the only data in the
# container; the retrain phase also exports this, which beats an edited default but
# not a rewritten Hydra line. What actually fixes the data is that there is nowhere
# else to load from and no network.
QM9_DATA=${QM9_DATA:-/assets/data/qm9_no_h}
OUTPUT_DIR=${OUTPUT_DIR:-/out}
CKPT_DIR=${CKPT_DIR:-${OUTPUT_DIR}/work/digress}

resolved_output=$(readlink -m "${OUTPUT_DIR}")
case "${resolved_output}" in
  /out|/out/*) ;;
  *)
    if [[ "${ALLOW_OUTPUT_OUTSIDE_OUT:-0}" != "1" ]]; then
      echo "run.sh: OUTPUT_DIR must be /out or below it, got '${OUTPUT_DIR}'." >&2
      echo "run.sh: formal receipts and checkpoints persist only below /out." >&2
      exit 78
    fi
    ;;
esac
OUTPUT_DIR=${resolved_output}
CKPT_DIR=$(readlink -m "${CKPT_DIR}")
case "${CKPT_DIR}" in
  /out|/out/*) ;;
  *)
    if [[ "${ALLOW_OUTPUT_OUTSIDE_OUT:-0}" != "1" ]]; then
      echo "run.sh: CKPT_DIR must stay below /out, got '${CKPT_DIR}'." >&2
      exit 78
    fi
    ;;
esac

# Training and fast evaluation must not share the GPU. Re-exec the whole run under
# the same lock fast_eval.sh uses, so checkpoint finalization is covered as well.
PHASE_LOCK=${AI4AI_GPU_PHASE_LOCK:-/out/.ai4ai-gpu-phase.lock}
if [[ "${AI4AI_GPU_PHASE_LOCK_HELD:-0}" != "1" ]]; then
  export AI4AI_GPU_PHASE_LOCK_HELD=1
  exec python3 /opt/harness/gpu_phase_lock.py \
    --lock "${PHASE_LOCK}" \
    --label "digress-train" \
    -- bash "$(readlink -f "${BASH_SOURCE[0]}")" "$@"
fi

# ---- algorithm ----
# The five the reference protocol let you touch, plus the three it declared and froze.
# There is no longer a difference between those two groups.
SEED=${SEED:-42}
NUM_LAYERS=${NUM_LAYERS:-9}
DIFFUSION_STEPS=${DIFFUSION_STEPS:-500}
LEARNING_RATE=${LEARNING_RATE:-2.0e-4}
BATCH_SIZE=${BATCH_SIZE:-512}
EMA_DECAY=${EMA_DECAY:-0.0}
WEIGHT_DECAY=${WEIGHT_DECAY:-1.0e-12}
CLIP_GRAD=${CLIP_GRAD:-1.0}

# ---- schedule ----
# MAX_EPOCHS deliberately exceeds the wall-clock horizon. MAX_WALL_TIME_SECONDS stops training
# -- it sends SIGTERM so Lightning finishes its epoch and writes a complete
# last.ckpt -- so too many epochs costs nothing while too few silently forfeits the
# budget. The resolved Hydra config and learning-rate dynamics remain part of every
# receipt; if a candidate adds a horizon-dependent schedule, account for the
# wall-clock stop.
MAX_EPOCHS=${MAX_EPOCHS:-1000}
# A positive value bounds a normal training epoch without enabling Lightning's
# fast_dev_run, so checkpoint callbacks remain active. The formal recipe leaves it at 0.
MAX_TRAIN_BATCHES=${MAX_TRAIN_BATCHES:-0}
# 0 disables the wall-clock stop. The retrain phase exports both of these; see
# container.py's exports_with_wall_clock.
MAX_WALL_TIME_SECONDS=${MAX_WALL_TIME_SECONDS:-0}
DEADLINE_RESERVE_SECONDS=${DEADLINE_RESERVE_SECONDS:-600}

# ---- checkpoint policy ----
# These are ordinary Recipe defaults: the Agent may change them or replace the saving code.
SAVE_UNIT="${SAVE_UNIT:-epoch}"              # step | epoch
SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"  # 0 means unlimited
case "${SAVE_UNIT}" in step|epoch) ;; *) echo "invalid SAVE_UNIT: ${SAVE_UNIT}" >&2; exit 78 ;; esac
[[ "${SAVE_INTERVAL}" =~ ^[1-9][0-9]*$ ]] || { echo "SAVE_INTERVAL must be positive" >&2; exit 78; }
[[ "${SAVE_TOTAL_LIMIT}" =~ ^[0-9]+$ ]] || { echo "SAVE_TOTAL_LIMIT must be nonnegative" >&2; exit 78; }

# Molecules sampled at the end of training, for the run's own log. Not the score:
# the score comes from the harness at a fixed count. The declared seed is inert in
# this pinned tree, so sampling is not deterministic. Keep it small: this is not
# what the task is measured on.
TRAIN_SAMPLES=${TRAIN_SAMPLES:-512}

RUN_NAME=${RUN_NAME:-digress_qm9_${SEED}}

# W&B starts local services and creates sockets even when offline.
export WANDB_MODE=${WANDB_MODE:-offline}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}

# /tmp is a tmpfs -- 8 GiB under the current ContainerSpec, and small enough to matter
# on any host that lowers it. RDKit, matplotlib, torch and hydra all cache below it, so
# send them to the output mount. HOME matters too: the container runs as the host's uid,
# which has no /etc/passwd entry, and matplotlib and wandb both want a home directory.
export TMPDIR=${OUTPUT_DIR}/tmp
# HOME is unconditional because the image already defines HOME=/tmp; a `:-` default
# would not redirect caches to persistent output storage.
export HOME=${OUTPUT_DIR}/tmp/home
export XDG_CACHE_HOME=${OUTPUT_DIR}/tmp/cache
export MPLCONFIGDIR=${OUTPUT_DIR}/tmp/matplotlib
export TORCH_HOME=${OUTPUT_DIR}/tmp/torch
mkdir -p "${TMPDIR}" "${HOME}" "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}" "${TORCH_HOME}"

# The dataset is the fixed input. /assets is read-only, which stops it being edited
# but not being bypassed: a preprocessed tensor written under /workspace would ride
# into the retrain container inside candidate.patch, which submit.sh generates with
# --binary. The runner's inspect_patch hook refuses a patch carrying data and the
# Agent cannot reach that one, but refuse it here too, where the error is legible.
#
# Set ALLOW_UNPINNED_DATA=1 to override, which the orchestrator never does.
case "${QM9_DATA}" in
  /assets/*) ;;
  *)
    if [[ "${ALLOW_UNPINNED_DATA:-0}" != "1" ]]; then
      echo "run.sh: QM9 data must live under /assets, got '${QM9_DATA}'." >&2
      echo "run.sh: the data source and the grader are the fixed inputs; how much of" >&2
      echo "run.sh: the data you use, and everything about the method, is yours." >&2
      exit 78
    fi
    echo "run.sh: WARNING data outside /assets: ${QM9_DATA}" >&2
    ;;
esac

exec python3 /workspace/train.py \
  --upstream /workspace/digress \
  --data "${QM9_DATA}" \
  --output "${OUTPUT_DIR}" \
  --checkpoint-dir "${CKPT_DIR}" \
  --run-name "${RUN_NAME}" \
  --seed "${SEED}" \
  --epochs "${MAX_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --learning-rate "${LEARNING_RATE}" \
  --num-layers "${NUM_LAYERS}" \
  --diffusion-steps "${DIFFUSION_STEPS}" \
  --ema-decay "${EMA_DECAY}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --clip-grad "${CLIP_GRAD}" \
  --samples "${TRAIN_SAMPLES}" \
  --max-train-batches "${MAX_TRAIN_BATCHES}" \
  --max-wall-seconds "${MAX_WALL_TIME_SECONDS}" \
  --reserve-seconds "${DEADLINE_RESERVE_SECONDS}" \
  --save-unit "${SAVE_UNIT}" \
  --save-interval "${SAVE_INTERVAL}" \
  --save-total-limit "${SAVE_TOTAL_LIMIT}" \
  "$@"
