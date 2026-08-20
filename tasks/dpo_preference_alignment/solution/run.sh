#!/usr/bin/env bash
# Shipped pairwise DPO optimization from the fixed merged Zephyr SFT start.
# Formal replay reads defaults from this file; probe-only environment overrides do
# not survive submission.

set -euo pipefail

# ---- paths (read-only mounts) ----
POLICY_START=${POLICY_START:-/assets/models/policy_start}
TRAIN_DATA=${TRAIN_DATA:-/assets/data/ultrafeedback}
OUTPUT_DIR=${OUTPUT_DIR:-/out}
CHECKPOINT_NAME=${CHECKPOINT_NAME:-checkpoint}

# One train or evaluator process may use a run root at a time. The inherited file
# lock survives both shell and Python exec and is released by the kernel on exit.
AI4AI_LOCK_ROOT=${AI4AI_LOCK_ROOT:-/out}
if [[ "${AI4AI_OUTPUT_LOCK_HELD:-0}" != "1" ]]; then
  export AI4AI_LOCK_ROOT
  exec python3 /workspace/runtime_guard.py "${AI4AI_LOCK_ROOT}" \
    bash "$(readlink -f "${BASH_SOURCE[0]}")" "$@"
fi

# ---- objective ----
# Pairwise sigmoid DPO on the policy/reference log-ratio. The reference is the same
# frozen start, reached by disabling the LoRA adapter rather than by loading a second
# copy of the model, keeping the QLoRA recipe memory-efficient.
DPO_BETA=${DPO_BETA:-0.01}
DPO_LOSS_TYPE=${DPO_LOSS_TYPE:-sigmoid}
# This is the shipped loss setting, not a restriction on candidate objectives.

# ---- data ----
# Use the complete pinned train split. train.py records pool and shard identities.
TRAIN_SAMPLES=${TRAIN_SAMPLES:-61135}
# The held-out rows the trainer scores itself on, from test_prefs. Not the task's
# metric -- the task is graded on IFEval -- but it is what SELECT_BEST below reads.
EVAL_SAMPLES=${EVAL_SAMPLES:-128}
# Changing this seed changes both row identities and ordering.
DATA_ORDER_SEED=${DATA_ORDER_SEED:-42}
MAX_LENGTH=${MAX_LENGTH:-1024}

# ---- shapes ----
# 4 x 4 = 16 pairs per optimizer step, so 32 sequences through a 4-bit 7B backbone
# with a second forward pass for the reference.
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}

# ---- optimizer and schedule ----
# The endpoint is explicit so formal replay does not depend on an external signal.
MAX_STEPS=${MAX_STEPS:-772}
LEARNING_RATE=${LEARNING_RATE:-5.0e-6}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
WARMUP_RATIO=${WARMUP_RATIO:-0.10}
GRADIENT_CLIP_NORM=${GRADIENT_CLIP_NORM:-1.0}
OPTIMIZER=${OPTIMIZER:-paged_adamw_32bit}
SCHEDULER=${SCHEDULER:-cosine}
SEED=${SEED:-42}
LOGGING_STEPS=${LOGGING_STEPS:-1}

# ---- what is trainable ----
# QLoRA: the backbone is loaded in 4-bit NF4 and frozen; the adapter configuration
# is part of the editable training implementation.
LORA_R=${LORA_R:-128}
LORA_ALPHA=${LORA_ALPHA:-${LORA_R}}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}

# ---- checkpointing ----
# The selected artifact is the fixed endpoint. UltraFeedback preference accuracy is
# a diagnostic and must not choose an IFEval checkpoint.
SELECT_BEST=${SELECT_BEST:-0}
EVAL_STEPS=${EVAL_STEPS:-32}
SAVE_UNIT=${SAVE_UNIT:-step}
SAVE_INTERVAL=${SAVE_INTERVAL:-386}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-3}
case "${SAVE_UNIT}" in step|epoch) ;; *) echo "SAVE_UNIT must be step or epoch" >&2; exit 2 ;; esac
[[ "${SAVE_INTERVAL}" =~ ^[1-9][0-9]*$ ]] || { echo "SAVE_INTERVAL must be positive" >&2; exit 2; }
[[ "${SAVE_TOTAL_LIMIT}" =~ ^[0-9]+$ ]] || { echo "SAVE_TOTAL_LIMIT must be non-negative" >&2; exit 2; }
SAVE_STEPS=${SAVE_STEPS:-${SAVE_INTERVAL}}

# ---- what gets exported ----
# `merged` reloads the pinned backbone in bf16 and writes a self-contained model.
# `adapter` writes only the LoRA delta; the evaluator assembles it on the same start.
EXPORT_MODE=${EXPORT_MODE:-adapter}
EXPORT_SHARD_SIZE=${EXPORT_SHARD_SIZE:-4GB}

# ---- wall clock ----
# 0 disables it. The retrain phase exports a real value; see declaration.py.
MAX_WALL_TIME_SECONDS=${MAX_WALL_TIME_SECONDS:-0}
DEADLINE_RESERVE_SECONDS=${DEADLINE_RESERVE_SECONDS:-1200}

# The start and the pairs are two of the three fixed inputs. /assets is a read-only
# mount, which stops them being edited but not being bypassed: a parquet written
# under /workspace would ride into the retrain container inside candidate.patch,
# which submit.sh generates with --binary. So refuse a source outside /assets. The
# orchestrator also screens the patch for data files, and that check is the one you
# cannot reach from here.
for _path in "${POLICY_START}" "${TRAIN_DATA}"; do
  case "${_path}" in
    /assets/*) ;;
    *)
      echo "run.sh: model and data must live under /assets, got '${_path}'." >&2
      echo "run.sh: the model source, the data source and the grader are fixed;" >&2
      echo "run.sh: everything else about the method is yours." >&2
      exit 78
      ;;
  esac
done

# Caches go on /out. /tmp is a tmpfs and docker adds noexec to every --tmpfs, so a
# JIT-compiled kernel written under /tmp cannot be loaded; /out is the writable bind
# every phase has. See orchestrator/container.py.
export TMPDIR=${TMPDIR:-${OUTPUT_DIR}/tmp}
export HF_HOME=${HF_HOME:-${OUTPUT_DIR}/tmp/hf}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${OUTPUT_DIR}/tmp/hf/datasets}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-${OUTPUT_DIR}/tmp/inductor}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${OUTPUT_DIR}/tmp/triton}
mkdir -p "${TMPDIR}" "${HF_DATASETS_CACHE}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

export POLICY_START TRAIN_DATA OUTPUT_DIR CHECKPOINT_NAME
export DPO_BETA DPO_LOSS_TYPE
export TRAIN_SAMPLES EVAL_SAMPLES DATA_ORDER_SEED MAX_LENGTH
export PER_DEVICE_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS
export MAX_STEPS LEARNING_RATE WEIGHT_DECAY WARMUP_RATIO GRADIENT_CLIP_NORM
export OPTIMIZER SCHEDULER SEED LOGGING_STEPS
export LORA_R LORA_ALPHA LORA_DROPOUT LORA_TARGET_MODULES ATTN_IMPLEMENTATION
export SELECT_BEST EVAL_STEPS SAVE_UNIT SAVE_INTERVAL SAVE_STEPS SAVE_TOTAL_LIMIT
export EXPORT_MODE EXPORT_SHARD_SIZE
export MAX_WALL_TIME_SECONDS DEADLINE_RESERVE_SECONDS

repro_dir="${OUTPUT_DIR}/repro"
mkdir -p "${repro_dir}"
cp "$(readlink -f "${BASH_SOURCE[0]}")" "${repro_dir}/run.sh"
(env | grep -v -E '(^|_)(API_)?KEY=|TOKEN=|PASSWORD=|PASS=|SECRET=|CREDENTIAL=|COOKIE=' | sort \
  || true) > "${repro_dir}/env.txt"

set -x
python3 /workspace/train.py "$@"
progress=$(python3 - "${OUTPUT_DIR}/train_summary.json" <<'PY'
import json, sys
print(int(json.load(open(sys.argv[1], encoding="utf-8"))["steps_completed"]))
PY
)
# The unconditional final export wins a same-progress collision with Trainer state.
python3 /opt/harness/save_checkpoint.py --output "${OUTPUT_DIR}" \
  --progress "${progress}" --source "${OUTPUT_DIR}/${CHECKPOINT_NAME}" \
  --retention "${SAVE_TOTAL_LIMIT}"
while IFS= read -r source; do
  step=${source##*-}
  [[ "${step}" =~ ^[0-9]+$ ]] || continue
  python3 /opt/harness/save_checkpoint.py --output "${OUTPUT_DIR}" \
    --progress "${step}" --source "${source}" --retention "${SAVE_TOTAL_LIMIT}"
done < <(find "${OUTPUT_DIR}/work" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V)
