#!/usr/bin/env bash
# Model soup | CLIP ViT-B/32 | 72 frozen ingredients | single GPU.
#
# Defaults are environment variables and extra arguments are forwarded to soup.py.
#
# Nothing here is a boundary. What fixes the ingredient set is that /assets is the
# only weights mount, the patch cannot carry weights, there is no network, and the
# exported soup is checked against the 72 ingredients at score time. See
# instruction.md.

set -euo pipefail

# ---- paths (read-only mounts) ----
INGREDIENTS=${INGREDIENTS:-/assets/models/ingredients}
CLIP_CACHE=${CLIP_CACHE:-/assets/models/clip}
PROXY_DATA=${PROXY_DATA:-/assets/data/imagenetv2_proxy}
OUTPUT_DIR=${OUTPUT_DIR:-/out}
WORK_DIR=${WORK_DIR:-${OUTPUT_DIR}/work/soup}

# Model Soup is one completed construction by default; it need not manufacture copies.
SAVE_UNIT="${SAVE_UNIT:-candidate}"          # candidate
SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"  # 0 means unlimited
[[ "${SAVE_UNIT}" == "candidate" ]] || { echo "Model Soup supports SAVE_UNIT=candidate" >&2; exit 78; }
[[ "${SAVE_INTERVAL}" =~ ^[1-9][0-9]*$ ]] || { echo "SAVE_INTERVAL must be positive" >&2; exit 78; }
[[ "${SAVE_TOTAL_LIMIT}" =~ ^[0-9]+$ ]] || { echo "SAVE_TOTAL_LIMIT must be nonnegative" >&2; exit 78; }

# ---- algorithm ----
# uniform is the baseline: the plain average of all 72. best_single takes the
# strongest ingredient on the proxy; strict_greedy adds ingredients in accuracy
# order and keeps each one only if it improves the proxy score.
#
SELECTION_RULE=${SELECTION_RULE:-uniform}
# Images per class used to score candidate merges from the 2000-row proxy.
# The proxy holds offsets 0-1, so 2 is also its maximum -- raising this needs more
# rows than the mount has, which is the constraint to notice before planning
# around it.
VALIDATION_PER_CLASS=${VALIDATION_PER_CLASS:-2}
# Ingredients strict_greedy is allowed to consider, in proxy-accuracy order.
MAX_INGREDIENTS=${MAX_INGREDIENTS:-72}
BATCH_SIZE=${BATCH_SIZE:-256}
SEED=${SEED:-42}

# ---- wall clock ----
# soup.py stops starting trials at the declared reserve and exports the best
# complete soup available.
MAX_WALL_TIME_SECONDS=${MAX_WALL_TIME_SECONDS:-0}
DEADLINE_RESERVE_SECONDS=${DEADLINE_RESERVE_SECONDS:-900}

# /tmp is a small tmpfs; keep the torch caches off it.
export TMPDIR=${TMPDIR:-${OUTPUT_DIR}/build-cache/tmp}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-${OUTPUT_DIR}/build-cache/torch-extensions}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-${OUTPUT_DIR}/build-cache/inductor}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${OUTPUT_DIR}/build-cache/xdg}
mkdir -p "${TMPDIR}" "${TORCH_EXTENSIONS_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${XDG_CACHE_HOME}"

python3 /workspace/soup.py \
  --ingredients "${INGREDIENTS}" \
  --clip-cache "${CLIP_CACHE}" \
  --data "${PROXY_DATA}" \
  --output "${WORK_DIR}" \
  --selection-rule "${SELECTION_RULE}" \
  --validation-per-class "${VALIDATION_PER_CLASS}" \
  --max-ingredients "${MAX_INGREDIENTS}" \
  --batch-size "${BATCH_SIZE}" \
  --seed "${SEED}" \
  --max-wall-seconds "${MAX_WALL_TIME_SECONDS}" \
  --reserve-seconds "${DEADLINE_RESERVE_SECONDS}" \
  "$@"

python3 /opt/harness/save_checkpoint.py \
  --output "${OUTPUT_DIR}" \
  --progress 1 \
  --source "${WORK_DIR}/model.pt" \
  --payload-name model.pt \
  --retention "${SAVE_TOTAL_LIMIT}"
