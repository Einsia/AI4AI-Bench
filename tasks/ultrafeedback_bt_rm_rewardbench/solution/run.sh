#!/usr/bin/env bash
# Bradley-Terry reward model | Mistral-7B-Instruct-v0.2 + LoRA | scalar head.
# Formal replay reads these defaults; every persistent recipe change belongs in
# source rather than in a probe-only environment override.

set -euo pipefail

# ---- paths (read-only mounts) ----
BASE_MODEL=${BASE_MODEL:-/assets/models/base}
TRAIN_DATA=${TRAIN_DATA:-/assets/data/pairs.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-/out}
CHECKPOINT_NAME=${CHECKPOINT_NAME:-checkpoint}

# Acquire the output/cache lock before creating any receipt or cache. The file
# descriptor remains inherited by the training process until it exits.
if [[ ${AI4AI_OUTPUT_LOCK_HELD:-0} != 1 ]]; then
  exec python3 /workspace/runtime_guard.py "${OUTPUT_DIR}" bash "$0" "$@"
fi

# ---- objective ----
# The shipped recipe uses Bradley-Terry pairwise logistic on (chosen - rejected).
# The frozen evaluator accepts any loadable scalar reward-model artifact.
BT_TEMPERATURE=${BT_TEMPERATURE:-1.0}
BT_MARGIN=${BT_MARGIN:-0.0}
# Bradley-Terry only constrains the *difference* of the two rewards, so the
# absolute scale is free to drift. A positive value penalises
# (chosen + rejected)^2 and pins the scale.
REWARD_CENTERING_WEIGHT=${REWARD_CENTERING_WEIGHT:-0.0}

# ---- data ----
# The mounted file contains exactly 8192 pairs. A smaller deterministic prefix is
# permitted; a larger value fails rather than reading a different source.
TRAIN_PAIRS=${TRAIN_PAIRS:-8192}
MAX_LENGTH=${MAX_LENGTH:-4096}

# ---- shapes ----
# One pair per micro-batch with accumulation 64 gives the required effective
# batch while leaving memory headroom for 4,096-token examples.
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-64}

# ---- optimizer and schedule ----
# Use a fixed endpoint so formal replay does not depend on an external signal.
MAX_STEPS=${MAX_STEPS:-252}
LEARNING_RATE=${LEARNING_RATE:-5.0e-6}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.001}
WARMUP_STEPS=${WARMUP_STEPS:-4}
GRADIENT_CLIP_NORM=${GRADIENT_CLIP_NORM:-1.0}
OPTIMIZER=${OPTIMIZER:-paged_adamw_32bit}
SCHEDULER=${SCHEDULER:-cosine}
SEED=${SEED:-42}
LOGGING_STEPS=${LOGGING_STEPS:-10}

# ---- what is trainable ----
# The shipped recipe freezes the backbone and exports LoRA plus the scalar head.
# Other candidates may export a compatible full scalar reward model instead.
LORA_R=${LORA_R:-128}
LORA_ALPHA=${LORA_ALPHA:-$(( 2 * LORA_R ))}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
SAVE_UNIT=${SAVE_UNIT:-step}
SAVE_INTERVAL=${SAVE_INTERVAL:-126}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-3}
case "${SAVE_UNIT}" in step|epoch) ;; *) echo "SAVE_UNIT must be step or epoch" >&2; exit 2 ;; esac
[[ "${SAVE_INTERVAL}" =~ ^[1-9][0-9]*$ ]] || { echo "SAVE_INTERVAL must be positive" >&2; exit 2; }
[[ "${SAVE_TOTAL_LIMIT}" =~ ^[0-9]+$ ]] || { echo "SAVE_TOTAL_LIMIT must be non-negative" >&2; exit 2; }
SAVE_STEPS=${SAVE_STEPS:-${SAVE_INTERVAL}}

# ---- wall clock ----
# 0 disables it. The retrain phase exports a real value; see declaration.py.
MAX_WALL_TIME_SECONDS=${MAX_WALL_TIME_SECONDS:-0}
DEADLINE_RESERVE_SECONDS=${DEADLINE_RESERVE_SECONDS:-600}

# The base model and the pairs are two of the three fixed inputs. /assets is a
# read-only mount, which stops them being edited but not being bypassed: a JSONL
# written under /workspace would ride into the retrain container inside
# candidate.patch, which submit.sh generates with --binary. So refuse a source
# outside /assets. The orchestrator also screens the patch for data files, and
# that check is the one you cannot reach from here.
for _path in "${BASE_MODEL}" "${TRAIN_DATA}"; do
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

# /tmp is a 256 MiB tmpfs; HF datasets and torch caches will overrun it.
export TMPDIR=${TMPDIR:-${OUTPUT_DIR}/tmp}
export HF_HOME=${HF_HOME:-${OUTPUT_DIR}/tmp/hf}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${OUTPUT_DIR}/tmp/hf/datasets}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-${OUTPUT_DIR}/tmp/inductor}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${OUTPUT_DIR}/tmp/triton}
mkdir -p "${TMPDIR}" "${HF_DATASETS_CACHE}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

export BASE_MODEL TRAIN_DATA OUTPUT_DIR CHECKPOINT_NAME
export BT_TEMPERATURE BT_MARGIN REWARD_CENTERING_WEIGHT
export TRAIN_PAIRS MAX_LENGTH MICRO_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS
export MAX_STEPS LEARNING_RATE WEIGHT_DECAY WARMUP_STEPS GRADIENT_CLIP_NORM
export OPTIMIZER SCHEDULER SEED LOGGING_STEPS
export LORA_R LORA_ALPHA LORA_DROPOUT LORA_TARGET_MODULES ATTN_IMPLEMENTATION
export SAVE_UNIT SAVE_INTERVAL SAVE_STEPS SAVE_TOTAL_LIMIT
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
python3 /opt/harness/save_checkpoint.py --output "${OUTPUT_DIR}" \
  --progress "${progress}" --source "${OUTPUT_DIR}/${CHECKPOINT_NAME}" \
  --retention "${SAVE_TOTAL_LIMIT}"
while IFS= read -r source; do
  step=${source##*-}
  [[ "${step}" =~ ^[0-9]+$ ]] || continue
  python3 /opt/harness/save_checkpoint.py --output "${OUTPUT_DIR}" \
    --progress "${step}" --source "${source}" --retention "${SAVE_TOTAL_LIMIT}"
done < <(find "${OUTPUT_DIR}/trainer" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V)
