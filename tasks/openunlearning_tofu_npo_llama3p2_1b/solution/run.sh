#!/usr/bin/env bash
set -euo pipefail

export TRAINING_START=${TRAINING_START:-/assets/models/training_start}
export TRAIN_DATA=${TRAIN_DATA:-/assets/data/train}
export OUTPUT_DIR=${OUTPUT_DIR:-/out/npo-output}
FORMAL_OUTPUT_ROOT=${FORMAL_OUTPUT_ROOT:-${OUTPUT_DIR%/*}}
export LEARNING_RATE=${LEARNING_RATE:-1.5e-5}
export NPO_BETA=${NPO_BETA:-0.1}
export NPO_ALPHA=${NPO_ALPHA:-1.0}
export RETAIN_GAMMA=${RETAIN_GAMMA:-1.0}
export EPOCHS=${EPOCHS:-10}
export PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-8}
export GRAD_ACCUM=${GRAD_ACCUM:-4}
export SEED=${SEED:-0}
# -1 preserves the epoch-bounded shipped recipe. Positive values provide a bounded
# direct-run canary without changing formal defaults.
export MAX_STEPS=${MAX_STEPS:--1}
export SAVE_UNIT=${SAVE_UNIT:-epoch}
export SAVE_INTERVAL=${SAVE_INTERVAL:-5}
export SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-3}
case "${SAVE_UNIT}" in step|epoch) ;; *) echo "SAVE_UNIT must be step or epoch" >&2; exit 2 ;; esac
[[ "${SAVE_INTERVAL}" =~ ^[1-9][0-9]*$ ]] || { echo "SAVE_INTERVAL must be positive" >&2; exit 2; }
[[ "${SAVE_TOTAL_LIMIT}" =~ ^[0-9]+$ ]] || { echo "SAVE_TOTAL_LIMIT must be non-negative" >&2; exit 2; }
export MAX_WALL_TIME_SECONDS=${MAX_WALL_TIME_SECONDS:-0}
export DEADLINE_RESERVE_SECONDS=${DEADLINE_RESERVE_SECONDS:-900}

python3 /workspace/train.py

final_progress=$(python3 - "${OUTPUT_DIR}/training_metadata.json" "${SAVE_UNIT}" <<'PY'
import json, math, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
if sys.argv[2] == "epoch":
    print(int(p.get("completed_epoch") or p["recipe"]["epochs"]))
else:
    print(int(p.get("completed_steps") or p["recipe"].get("max_steps") or 0))
PY
)
python3 /opt/harness/save_checkpoint.py --output "${FORMAL_OUTPUT_ROOT}" \
  --progress "${final_progress}" --source "${FORMAL_OUTPUT_ROOT}/final-model" \
  --retention "${SAVE_TOTAL_LIMIT}"
while IFS= read -r source; do
  read -r step epoch < <(python3 - "${source}/trainer_state.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
print(int(p.get("global_step", 0)), int(float(p.get("epoch", 0))))
PY
  )
  if [[ "${SAVE_UNIT}" == step ]]; then
    progress=${step}
  else
    (( epoch > 0 && epoch % SAVE_INTERVAL == 0 )) || continue
    progress=${epoch}
  fi
  python3 /opt/harness/save_checkpoint.py --output "${FORMAL_OUTPUT_ROOT}" \
    --progress "${progress}" --source "${source}" --retention "${SAVE_TOTAL_LIMIT}"
done < <(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V)
