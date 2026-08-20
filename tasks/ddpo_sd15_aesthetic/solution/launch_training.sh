#!/usr/bin/env bash
# DDPO LoRA on Stable Diffusion v1.5 | aesthetic reward | PPO clipping.
# Called by train.py after the public driver has established export and time semantics.

set -euo pipefail

# ---- paths (read-only mounts) ----
DDPO_MODEL=${DDPO_MODEL:-/assets/models/stable-diffusion-v1-5}
# The aesthetic reward is CLIP-L/14 plus a 5-layer MLP. The MLP weights ship inside
# the pinned upstream tree; this is the CLIP half.
DDPO_CLIP_PATH=${DDPO_CLIP_PATH:-/assets/models/clip}
OUTPUT_DIR=${OUTPUT_DIR:-/out}
LOG_DIR=${LOG_DIR:-${OUTPUT_DIR}/logs}
UPSTREAM=${UPSTREAM:-/workspace/ddpo-pytorch}
CONFIG=${CONFIG:-/workspace/ddpo_config.py}

# ---- algorithm ----
LEARNING_RATE=${LEARNING_RATE:-3.0e-4}
PPO_CLIP_RANGE=${PPO_CLIP_RANGE:-1.0e-4}
ADV_CLIP_MAX=${ADV_CLIP_MAX:-5.0}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-5.0}
SAMPLE_STEPS=${SAMPLE_STEPS:-50}
# Match the upstream normalization rule. With 32 samples per epoch, the buffer can
# satisfy min_count=16; disabling it changes the advantage distribution materially.
PER_PROMPT_STAT_TRACKING=${PER_PROMPT_STAT_TRACKING:-1}
PER_PROMPT_BUFFER_SIZE=${PER_PROMPT_BUFFER_SIZE:-32}
PER_PROMPT_MIN_COUNT=${PER_PROMPT_MIN_COUNT:-16}

# ---- shapes ----
# samples_per_epoch = SAMPLE_BATCH_SIZE x SAMPLES_PER_EPOCH. The selected recipe
# generates 32 trajectories per epoch and optimizes at effective batch 16.
SAMPLE_BATCH_SIZE=${SAMPLE_BATCH_SIZE:-8}
SAMPLES_PER_EPOCH=${SAMPLES_PER_EPOCH:-4}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-4}

# ---- schedule ----
NUM_EPOCHS=${NUM_EPOCHS:-13}
SAVE_UNIT=${SAVE_UNIT:-epoch}
SAVE_INTERVAL=${SAVE_INTERVAL:-1}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-3}
[[ "${SAVE_UNIT}" == epoch ]] || { echo "DDPO supports SAVE_UNIT=epoch" >&2; exit 2; }
SAVE_FREQ=${SAVE_FREQ:-${SAVE_INTERVAL}}
NUM_CHECKPOINT_LIMIT=${NUM_CHECKPOINT_LIMIT:-${SAVE_TOTAL_LIMIT}}
SEED=${SEED:-43}
MIXED_PRECISION=${MIXED_PRECISION:-fp16}
RUN_NAME=${RUN_NAME:-ddpo_sd15_aesthetic-${SEED}}

export WANDB_MODE=${WANDB_MODE:-disabled}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

# /tmp is a small tmpfs, so caches belong on the writable output mount.
export TMPDIR=${TMPDIR:-${OUTPUT_DIR}/retrain-cache/tmp}
export HF_HOME=${HF_HOME:-${OUTPUT_DIR}/retrain-cache/hf}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${OUTPUT_DIR}/retrain-cache/xdg}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${OUTPUT_DIR}/retrain-cache/triton}
mkdir -p "${TMPDIR}" "${HF_HOME}" "${XDG_CACHE_HOME}" "${TRITON_CACHE_DIR}" "${LOG_DIR}"

for _path in "${DDPO_MODEL}" "${DDPO_CLIP_PATH}"; do
  case "${_path}" in
    /assets/*) ;;
    *)
      if [[ "${ALLOW_UNPINNED_ASSETS:-0}" != "1" ]]; then
        echo "launch_training.sh: model and reward CLIP must live under /assets: ${_path}" >&2
        exit 78
      fi
      echo "launch_training.sh: WARNING asset outside /assets: ${_path}" >&2
      ;;
  esac
done

export DDPO_MODEL DDPO_CLIP_PATH LOG_DIR RUN_NAME SEED
export NUM_EPOCHS SAVE_UNIT SAVE_INTERVAL SAVE_TOTAL_LIMIT
export SAVE_FREQ NUM_CHECKPOINT_LIMIT MIXED_PRECISION
export SAMPLE_STEPS GUIDANCE_SCALE SAMPLE_BATCH_SIZE SAMPLES_PER_EPOCH
export TRAIN_BATCH_SIZE GRAD_ACCUM LEARNING_RATE ADV_CLIP_MAX PPO_CLIP_RANGE
export PER_PROMPT_STAT_TRACKING PER_PROMPT_BUFFER_SIZE PER_PROMPT_MIN_COUNT
export DDPO_AESTHETIC_CLIP_MODEL=${DDPO_CLIP_PATH}

repro_dir="${OUTPUT_DIR}/repro"
mkdir -p "${repro_dir}"
cp "$(readlink -f "${BASH_SOURCE[0]}")" "${repro_dir}/launch_training.sh"
{
  printf '%q ' "$(readlink -f "${BASH_SOURCE[0]}")" "$@"
  printf '\n'
} > "${repro_dir}/command.txt"
(env | grep -v -E '(^|_)(API_)?KEY=|TOKEN=|PASSWORD=|PASS=|SECRET=|CREDENTIAL=|COOKIE=' | sort \
  || true) > "${repro_dir}/env.txt"

cd "${UPSTREAM}"
export PYTHONPATH="${UPSTREAM}${PYTHONPATH:+:${PYTHONPATH}}"

set -x
exec python3 "${UPSTREAM}/tools/train.py" --config "${CONFIG}" "$@"
