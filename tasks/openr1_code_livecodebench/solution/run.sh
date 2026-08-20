#!/usr/bin/env bash
# Completion-only code SFT from the fixed Qwen2.5-Coder training start.
# Formal replay reads defaults from this file; probe-only environment overrides do
# not survive submission.

set -euo pipefail

# ---- paths (read-only mounts) ----
TRAINING_START=${TRAINING_START:-/assets/models/training_start}
TRAIN_DATA=${TRAIN_DATA:-/assets/data/codeforces_cots}
OUTPUT_DIR=${OUTPUT_DIR:-/out}
CKPT_DIR=${CKPT_DIR:-${OUTPUT_DIR}/checkpoints}

AI4AI_LOCK_ROOT=${AI4AI_LOCK_ROOT:-/out}
if [[ "${AI4AI_OUTPUT_LOCK_HELD:-0}" != "1" ]]; then
  export AI4AI_LOCK_ROOT
  exec python3 /workspace/runtime_guard.py "${AI4AI_LOCK_ROOT}" \
    bash "$(readlink -f "${BASH_SOURCE[0]}")" "$@"
fi

# ---- data ----
# The subset directory inside the fixed mount.
SOURCE_SUBSET=${SOURCE_SUBSET:-solutions_py_decontaminated}
# Changing the split seed changes both training and validation row identities.
SPLIT_SEED=${SPLIT_SEED:-20260727}
# The mounted snapshot contains exactly 8133 rows. The shipped split consumes all
# of them as 8005 training plus 128 validation rows; raising either count is invalid.
TRAIN_ROWS=${TRAIN_ROWS:-8005}
VALIDATION_ROWS=${VALIDATION_ROWS:-128}
SEED=${SEED:-42}

# ---- algorithm ----
# The pinned OpenR1 OlympicCoder SFT recipe uses 1e-5. Keep that optimizer scale
# while adapting its multi-GPU recipe to this fixed 1.5B, single-accelerator protocol.
LEARNING_RATE=${LEARNING_RATE:-1e-5}
# Sixty optimizer steps preserve a midpoint plus endpoint for paired evaluation.
MAX_STEPS=${MAX_STEPS:-60}
MAX_LENGTH=${MAX_LENGTH:-32768}
# The batch shape is the task's fixed single-device default and retains the 32K context.
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-3}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-6}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
GRADIENT_CLIP_NORM=${GRADIENT_CLIP_NORM:-0.2}
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-cosine_with_min_lr}
MIN_LR_RATE=${MIN_LR_RATE:-0.1}

# ---- checkpointing ----
# Preserve the midpoint and endpoint. train.py also exports the final step under
# checkpoints/checkpoint-60 for the task checkpoint contract.
EVAL_STEPS=${EVAL_STEPS:-30}
SAVE_UNIT=${SAVE_UNIT:-step}
SAVE_INTERVAL=${SAVE_INTERVAL:-30}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-3}
case "${SAVE_UNIT}" in step|epoch) ;; *) echo "SAVE_UNIT must be step or epoch" >&2; exit 2 ;; esac
[[ "${SAVE_INTERVAL}" =~ ^[1-9][0-9]*$ ]] || { echo "SAVE_INTERVAL must be positive" >&2; exit 2; }
[[ "${SAVE_TOTAL_LIMIT}" =~ ^[0-9]+$ ]] || { echo "SAVE_TOTAL_LIMIT must be non-negative" >&2; exit 2; }
SAVE_STEPS=${SAVE_STEPS:-${SAVE_INTERVAL}}
# Optional validation-NLL checkpoint selection. NLL is a diagnostic and must not be
# reported as LiveCodeBench pass@1.
LOAD_BEST_MODEL_AT_END=${LOAD_BEST_MODEL_AT_END:-0}

# ---- wall clock ----
# The retrain phase exports both of these; they are 0 and unused in a hand-run.
# Stopping one step early is what keeps the last checkpoint whole: HF Trainer
# writes model shards, optimizer state and RNG state as separate files, and a kill
# between them leaves a directory that exists and cannot be loaded.
MAX_WALL_TIME_SECONDS=${MAX_WALL_TIME_SECONDS:-0}
DEADLINE_RESERVE_SECONDS=${DEADLINE_RESERVE_SECONDS:-600}

# W&B and friends start local services and create unix sockets even when offline.
export WANDB_MODE=${WANDB_MODE:-disabled}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}

# /tmp is a 256 MiB tmpfs, and this is not a small detail here: `datasets` builds
# an Arrow cache of the corpus under HF_HOME, and the corpus is about 500 MB. With
# HOME=/tmp in the image, the default cache location fills the tmpfs and the run
# dies during dataset loading -- before a single step, with a disk error that
# reads like a permissions problem.
export TMPDIR=${TMPDIR:-${OUTPUT_DIR}/tmp}
export HF_HOME=${HF_HOME:-${OUTPUT_DIR}/tmp/hf}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${OUTPUT_DIR}/tmp/hf/datasets}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-${OUTPUT_DIR}/tmp/inductor}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${OUTPUT_DIR}/tmp/triton}
mkdir -p "${TMPDIR}" "${HF_HOME}" "${HF_DATASETS_CACHE}" \
  "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${CKPT_DIR}"

# The training data and the training start are the fixed inputs. /assets is a
# read-only mount, which stops them being edited but not bypassed: a parquet
# written under /workspace rides into the retrain container inside
# candidate.patch, which submit.sh generates with --binary. So refuse a source
# outside /assets.
#
# This is a guard-rail, not the boundary -- you own this file and can delete the
# check. What actually fixes the inputs is that the container has nowhere else to
# load from, and orchestrator/task.py:describe_patch_rejections refuses a patch
# carrying data or weights before the retrain phase claims a GPU.
for _path in "${TRAINING_START}" "${TRAIN_DATA}"; do
  case "${_path}" in
    /assets/*) ;;
    *)
      echo "run.sh: model and data must live under /assets, got '${_path}'." >&2
      echo "run.sh: the model, corpus and evaluator are fixed inputs." >&2
      exit 78
      ;;
  esac
done

repro_dir="${CKPT_DIR}/repro"
mkdir -p "${repro_dir}"
cp "$(readlink -f "${BASH_SOURCE[0]}")" "${repro_dir}/run.sh"
{
  printf '%q ' "$(readlink -f "${BASH_SOURCE[0]}")" "$@"
  printf '\n'
} > "${repro_dir}/command.txt"
(env | grep -v -E '(^|_)(API_)?KEY=|TOKEN=|PASSWORD=|PASS=|SECRET=|CREDENTIAL=|COOKIE=' | sort \
  || true) > "${repro_dir}/env.txt"

set -x
python3 /workspace/train.py \
  --training-start "${TRAINING_START}" \
  --data "${TRAIN_DATA}" \
  --source-subset "${SOURCE_SUBSET}" \
  --output "${OUTPUT_DIR}" \
  --checkpoint-dir "${CKPT_DIR}" \
  --seed "${SEED}" \
  --split-seed "${SPLIT_SEED}" \
  --train-rows "${TRAIN_ROWS}" \
  --validation-rows "${VALIDATION_ROWS}" \
  --learning-rate "${LEARNING_RATE}" \
  --max-steps "${MAX_STEPS}" \
  --max-length "${MAX_LENGTH}" \
  --per-device-batch-size "${PER_DEVICE_BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --warmup-ratio "${WARMUP_RATIO}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --gradient-clip-norm "${GRADIENT_CLIP_NORM}" \
  --lr-scheduler-type "${LR_SCHEDULER_TYPE}" \
  --min-lr-rate "${MIN_LR_RATE}" \
  --eval-steps "${EVAL_STEPS}" \
  --save-unit "${SAVE_UNIT}" \
  --save-interval "${SAVE_INTERVAL}" \
  --save-total-limit "${SAVE_TOTAL_LIMIT}" \
  --load-best-model-at-end "${LOAD_BEST_MODEL_AT_END}" \
  --max-wall-time-seconds "${MAX_WALL_TIME_SECONDS}" \
  --deadline-reserve-seconds "${DEADLINE_RESERVE_SECONDS}" \
  "$@"

# The final export is already an atomic standard checkpoint. Normalize retained
# Trainer saves as additional complete model candidates.
while IFS= read -r source; do
  step=${source##*-}
  [[ "${step}" =~ ^[0-9]+$ ]] || continue
  python3 /opt/harness/save_checkpoint.py --output "${OUTPUT_DIR}" \
    --progress "${step}" --source "${source}" --retention "${SAVE_TOTAL_LIMIT}"
done < <(find "${OUTPUT_DIR}/trainer" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V)
