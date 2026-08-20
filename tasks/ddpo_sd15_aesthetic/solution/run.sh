#!/usr/bin/env bash
# Public DDPO entry point. Formal replay and direct runs use the same driver, which
# launches training and exports standard recipe-owned checkpoints.

set -euo pipefail

OUTPUT_DIR=${OUTPUT_DIR:-/out}
DDPO_PROFILE=${DDPO_PROFILE:-retrain}
SAVE_UNIT=${SAVE_UNIT:-epoch}
SAVE_INTERVAL=${SAVE_INTERVAL:-1}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-3}

[[ "${SAVE_UNIT}" == epoch ]] || { echo "DDPO supports SAVE_UNIT=epoch" >&2; exit 2; }
[[ "${SAVE_INTERVAL}" =~ ^[1-9][0-9]*$ ]] || { echo "SAVE_INTERVAL must be positive" >&2; exit 2; }
[[ "${SAVE_TOTAL_LIMIT}" =~ ^[0-9]+$ ]] || { echo "SAVE_TOTAL_LIMIT must be non-negative" >&2; exit 2; }
export SAVE_UNIT SAVE_INTERVAL SAVE_TOTAL_LIMIT
export SAVE_FREQ=${SAVE_FREQ:-${SAVE_INTERVAL}}
export NUM_CHECKPOINT_LIMIT=${NUM_CHECKPOINT_LIMIT:-${SAVE_TOTAL_LIMIT}}

python3 /workspace/train.py \
  --profile "${DDPO_PROFILE}" \
  --output "${OUTPUT_DIR}" \
  "$@"

# Upstream automatic checkpoint numbers are monotonic save progress. The final
# export is published first so a same-progress periodic directory cannot replace it.
latest=$({ find "${OUTPUT_DIR}/logs" -type d -name 'checkpoint_*' -printf '%f\n' \
  2>/dev/null || true; } | sed -n 's/^checkpoint_//p' | sort -n | tail -1)
if [[ -n "${latest}" && -d "${OUTPUT_DIR}/checkpoint" ]]; then
  python3 /opt/harness/save_checkpoint.py --output "${OUTPUT_DIR}" \
    --progress "${latest}" --source "${OUTPUT_DIR}/checkpoint" \
    --retention "${SAVE_TOTAL_LIMIT}"
fi
while IFS= read -r source; do
  progress=${source##*checkpoint_}
  [[ "${progress}" =~ ^[0-9]+$ ]] || continue
  python3 /opt/harness/save_checkpoint.py --output "${OUTPUT_DIR}" \
    --progress "${progress}" --source "${source}" \
    --retention "${SAVE_TOTAL_LIMIT}"
done < <(find "${OUTPUT_DIR}/logs" -type d -name 'checkpoint_*' 2>/dev/null | sort -V)
