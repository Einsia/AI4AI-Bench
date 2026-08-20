#!/usr/bin/env bash
# Wait for a running retrain container to exit, then score its last checkpoint.
#
#   bash score_when_done.sh <container> <run-dir> [--run-config FILE]
#
# Task, assets, preferred GPU, image override and verification policy are recovered
# from the lifecycle's immutable run-config. Explicit overrides are accepted only as
# assertions and must match that receipt.
#
# trial.sh chains a trial it started itself. This recovery utility handles a retrain
# container whose original host-side supervisor exited before it could start scoring.
# The wall clock remains inside the container, but a replacement owner is needed for
# the transition to `score`.
#
# Run it detached, or it inherits the same failure it exists to fix:
#   setsid nohup bash score_when_done.sh ... >> watch.log 2>&1 &
set -uo pipefail

CONTAINER=${1:?usage: score_when_done.sh <container> <run-dir> [--run-config FILE]}
RUN=${2:?missing run directory}
shift 2

ASSETS="" GPU="" TASK="" IMAGE="" RUN_CONFIG=""
SOURCE_CHECK=${AI4AI_SOURCE_CHECK:-}
IMAGE_CHECK=${AI4AI_IMAGE_CHECK:-}
HARDWARE_CHECK=${AI4AI_HARDWARE_CHECK:-}
IMAGE_PULL_POLICY=${AI4AI_IMAGE_PULL_POLICY:-}
while [ $# -gt 0 ]; do
  case "$1" in
    --assets) ASSETS=$2; shift 2 ;;
    --gpu)    GPU=$2;    shift 2 ;;
    --task)   TASK=$2;   shift 2 ;;
    --image) IMAGE=$2; shift 2 ;;
    --run-config) RUN_CONFIG=$2; shift 2 ;;
    --source-check) SOURCE_CHECK=$2; shift 2 ;;
    --image-check) IMAGE_CHECK=$2; shift 2 ;;
    --hardware-check) HARDWARE_CHECK=$2; shift 2 ;;
    --image-pull-policy) IMAGE_PULL_POLICY=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python3}
command -v "$PY" >/dev/null || {
  echo "python interpreter is not available: $PY" >&2
  exit 2
}
export AI4AI_DOCKER=${AI4AI_DOCKER:-docker}
read -r -a DOCKER_CMD < <(printf '%s\n' "$AI4AI_DOCKER")
docker_cmd() { "${DOCKER_CMD[@]}" "$@"; }
NVIDIA_SMI=${NVIDIA_SMI:-nvidia-smi}

if [ -z "$RUN_CONFIG" ]; then
  if [ -f "$RUN/run-config.json" ]; then
    RUN_CONFIG="$RUN/run-config.json"
  elif [ -f "$RUN/../run-config.json" ]; then
    RUN_CONFIG="$RUN/../run-config.json"
  else
    echo "no immutable run-config.json for recovery; refusing to infer lifecycle policy" >&2
    exit 2
  fi
fi
[ -f "$RUN_CONFIG" ] || { echo "run config is not a file: $RUN_CONFIG" >&2; exit 2; }

mapfile -t RECORDED_CONFIG < <("$PY" -B - "$RUN_CONFIG" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    value = json.loads(path.read_text(encoding="utf-8"))
except (OSError, ValueError) as error:
    raise SystemExit(f"cannot read immutable run config {path}: {error}")
defaults = {"hardware_check": "warn", "image_pull_policy": "missing"}
keys = (
    "task", "assets", "gpu", "image", "source_check", "image_check",
    "hardware_check", "image_pull_policy",
)
for key in keys:
    item = value.get(key, defaults.get(key))
    if not isinstance(item, str) or not item or "\n" in item or "\r" in item:
        raise SystemExit(f"immutable run config has invalid {key!r}")
    print(item)
score_phase = value.get("score_phase", "score")
if score_phase not in {"score", "score-mock"}:
    raise SystemExit("immutable run config has invalid score_phase")
print(score_phase)
score_timeout = (value.get("effective_timeouts") or {}).get("score")
if score_timeout is None:
    print("declared")
elif isinstance(score_timeout, int) and not isinstance(score_timeout, bool) and score_timeout > 0:
    print(score_timeout)
else:
    raise SystemExit("immutable run config has invalid effective score timeout")
PY
)
[ ${#RECORDED_CONFIG[@]} -eq 10 ] || { echo "invalid immutable run config" >&2; exit 2; }

RECORDED_TASK=${RECORDED_CONFIG[0]}
RECORDED_ASSETS=${RECORDED_CONFIG[1]}
RECORDED_GPU=${RECORDED_CONFIG[2]}
RECORDED_IMAGE=${RECORDED_CONFIG[3]}
RECORDED_SOURCE_CHECK=${RECORDED_CONFIG[4]}
RECORDED_IMAGE_CHECK=${RECORDED_CONFIG[5]}
RECORDED_HARDWARE_CHECK=${RECORDED_CONFIG[6]}
RECORDED_IMAGE_PULL_POLICY=${RECORDED_CONFIG[7]}
SCORE_PHASE=${RECORDED_CONFIG[8]}
SCORE_TIMEOUT=${RECORDED_CONFIG[9]}

assert_same() {
  local name=$1 requested=$2 recorded=$3
  [ -z "$requested" ] || [ "$requested" = "$recorded" ] || {
    echo "$name conflicts with immutable run config: requested=$requested recorded=$recorded" >&2
    exit 2
  }
}
if [ -n "$TASK" ]; then TASK=$(realpath -m "$TASK"); fi
if [ -n "$ASSETS" ]; then ASSETS=$(realpath -m "$ASSETS"); fi
assert_same task "$TASK" "$RECORDED_TASK"
assert_same assets "$ASSETS" "$RECORDED_ASSETS"
assert_same gpu "$GPU" "$RECORDED_GPU"
assert_same image "$IMAGE" "$RECORDED_IMAGE"
assert_same source-check "$SOURCE_CHECK" "$RECORDED_SOURCE_CHECK"
assert_same image-check "$IMAGE_CHECK" "$RECORDED_IMAGE_CHECK"
assert_same hardware-check "$HARDWARE_CHECK" "$RECORDED_HARDWARE_CHECK"
assert_same image-pull-policy "$IMAGE_PULL_POLICY" "$RECORDED_IMAGE_PULL_POLICY"

TASK=$RECORDED_TASK
ASSETS=$RECORDED_ASSETS
GPU=$RECORDED_GPU
IMAGE=$RECORDED_IMAGE
SOURCE_CHECK=$RECORDED_SOURCE_CHECK
IMAGE_CHECK=$RECORDED_IMAGE_CHECK
HARDWARE_CHECK=$RECORDED_HARDWARE_CHECK
IMAGE_PULL_POLICY=$RECORDED_IMAGE_PULL_POLICY
RUNNER_POLICY_ARGS=(--image "$IMAGE" --source-check "$SOURCE_CHECK"
  --image-check "$IMAGE_CHECK" --hardware-check "$HARDWARE_CHECK"
  --image-pull-policy "$IMAGE_PULL_POLICY" --run-config "$RUN_CONFIG")

say() { echo "[$(date +%H:%M:%S)] $*"; }

# --- wait ---------------------------------------------------------------------------
# Poll rather than `docker wait`: this must survive the daemon being restarted under it,
# and a container that is already gone has to read as done rather than as an error.
#
# The subtlety is that "inspect did not say true" covers three different situations, and
# only two of them mean the run is over:
#
#   Running=false      the container finished          -> score it
#   inspect errors     the container is gone/removed   -> score what is on disk
#   inspect errors     a transient failure             -> KEEP WAITING
#
# A first version treated all three the same, which turns a daemon restart or a socket
# hiccup into a score taken mid-training. That is worse than a crash: the resolver would
# find the newest complete checkpoint, write summary.json, and that file then makes the
# run look already-scored when it really finishes, so the correct score never happens.
#
# So separate "answered" from "failed to answer", and only accept a failure as terminal
# after several consecutive ones -- a removed container fails every time, a blip does not.
say "watching $CONTAINER"
MISSES=0
MISS_LIMIT=${MISS_LIMIT:-5}
while :; do
  state=$(docker_cmd inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null | head -1)
  case "$state" in
    true)
      MISSES=0
      ;;
    false)
      say "$CONTAINER reports not running"
      break
      ;;
    *)
      MISSES=$(( MISSES + 1 ))
      say "docker inspect gave no answer ($MISSES/$MISS_LIMIT)"
      if [ "$MISSES" -ge "$MISS_LIMIT" ]; then
        say "$CONTAINER is gone after $MISSES consecutive failures"
        break
      fi
      ;;
  esac
  sleep 60
done
CODE=$(docker_cmd inspect -f '{{.State.ExitCode}}' "$CONTAINER" 2>/dev/null | head -1)
CODE=${CODE:-gone}
say "$CONTAINER stopped, exit=$CODE"

# A non-zero exit is not automatically fatal here. run.sh ends via the wall clock, and
# the question that decides whether scoring is meaningful is whether a complete
# checkpoint exists -- which the next step answers directly.

# --- fixed inputs --------------------------------------------------------------------
# Written by verify_fixed_inputs from what Hydra actually resolved. The three runs this
# script was written for started before that hook existed, so the marker will be absent
# for them and their configs were checked by hand instead -- all four read clean.
if [ -e "$RUN/out/FIXED_INPUTS_VIOLATED" ]; then
  say "retrain trained on something other than the fixed inputs, not scoring:"
  sed 's/^/     /' "$RUN/out/FIXED_INPUTS_VIOLATED"
  exit 1
fi

# --- discover candidates -------------------------------------------------------------
# Frozen validation needs a GPU and therefore runs after the device claim below.
if [ ! -e "$RUN/artifacts.json" ]; then
  "$PY" -B "$HERE/artifact.py" discover --task "$TASK" --out "$RUN/out" \
    --receipt "$RUN/artifacts.json" --format json >/dev/null || exit 1
fi
say "checkpoint inventory recorded in $RUN/artifacts.json"

# Avoid waiting forever for a GPU when there is nothing to validate or score.  A
# receipt with no pending validation is safe to finalize before device selection;
# this also covers a resumed run whose validations are already terminal.
PENDING_VALIDATIONS=$(
  "$PY" -B "$HERE/artifact.py" list --receipt "$RUN/artifacts.json" \
    --format validation-tsv
)
if [ -z "$PENDING_VALIDATIONS" ]; then
  CKPT=$("$PY" -B "$HERE/artifact.py" finalize --receipt "$RUN/artifacts.json" \
    --out "$RUN/out") || exit 1
  [ -n "$CKPT" ] || { say "no frozen-loadable standard checkpoint -- not scoring"; exit 1; }
fi

# --- pick the device at score time ---------------------------------------------------
# The requested GPU is a preference. If it is occupied, scan for an idle alternative;
# atomic claims prevent concurrent watchers from selecting the same device.
# Every watcher sharing a host must use the same claim store. Keep the old
# CLAIMS variable as a compatibility fallback, but give public deployments an
# AI4AI-scoped setting and honor the host's temporary-directory convention.
CLAIMS=${AI4AI_GPU_CLAIM_ROOT:-${CLAIMS:-${TMPDIR:-/tmp}/ai4ai-gpu-claims}}
mkdir -p "$CLAIMS" 2>/dev/null
CLAIM_FDS=()

free_pct() {  # device index -> "used total"; empty if nvidia-smi cannot say
  "$NVIDIA_SMI" --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
    2>/dev/null | awk -F', ' -v d="$1" '$1 == d { print $2, $3 }'
}

is_free() {
  local vals used total
  vals=$(free_pct "$1")
  [ -n "$vals" ] || return 1
  used=${vals% *}; total=${vals#* }
  [ "$total" -gt 0 ] 2>/dev/null || return 1
  [ "$(( used * 100 / total ))" -lt 15 ]
}

claim() {  # an open flock is atomic and the kernel releases it if this watcher dies
  local d=$1 fd
  exec {fd}>"$CLAIMS/gpu$d.lock" || return 1
  if flock -n "$fd"; then
    CLAIM_FDS+=("$fd")
    return 0
  fi
  exec {fd}>&-
  return 1
}

release_claims() {
  local fd
  for fd in "${CLAIM_FDS[@]}"; do
    flock -u "$fd" 2>/dev/null || true
    exec {fd}>&-
  done
}

pick_device() {
  if [ -n "$GPU" ] && is_free "$GPU" && claim "$GPU"; then
    say "scoring on requested device $GPU"
    return 0
  fi
  [ -n "$GPU" ] && say "requested device $GPU is occupied or claimed, looking for another"
  local d
  for d in $("$NVIDIA_SMI" --query-gpu=index --format=csv,noheader 2>/dev/null); do
    if is_free "$d" && claim "$d"; then
      GPU=$d
      say "scoring on device $GPU instead"
      return 0
    fi
  done
  return 1
}

if command -v "$NVIDIA_SMI" >/dev/null 2>&1; then
  command -v flock >/dev/null 2>&1 || { say "flock is required for atomic GPU claims"; exit 1; }
  until pick_device; do
    say "no device is 85% free; waiting 5 min. nvidia-smi says:"
    "$NVIDIA_SMI" --query-gpu=index,memory.used,memory.total --format=csv,noheader | sed 's/^/     /'
    sleep 300
  done
  trap release_claims EXIT
fi
GPU=${GPU:-0}

# --- frozen loadability and latest-three publication --------------------------------
while IFS=$'\t' read -r PROGRESS CHECKPOINT; do
  [ -n "$PROGRESS" ] || continue
  VALIDATION_ROOT="$RUN/validation/checkpoint-$PROGRESS"
  VALIDATION_JSON="$VALIDATION_ROOT/out/validation.json"
  mkdir -p "$VALIDATION_ROOT/out" "$VALIDATION_ROOT/logs"
  if [ ! -e "$VALIDATION_ROOT/.complete" ]; then
    [ -e "$VALIDATION_ROOT/.started" ] || date -u +"%Y-%m-%dT%H:%M:%SZ" > "$VALIDATION_ROOT/.started"
    say "validating checkpoint-$PROGRESS: $CHECKPOINT"
    if ! "$PY" -B "$HERE/runner.py" checkpoint-validate --task "$TASK" \
         --assets "$ASSETS" --checkpoint "$CHECKPOINT" --out "$VALIDATION_ROOT/out" \
         --logs "$VALIDATION_ROOT/logs" --gpu "$GPU" "${RUNNER_POLICY_ARGS[@]}" \
         2>&1 | tee -a "$VALIDATION_ROOT/validate.log"; then
      say "checkpoint-$PROGRESS validator infrastructure failure; left retryable"
      exit 1
    fi
    [ -s "$VALIDATION_JSON" ] || { say "checkpoint-$PROGRESS has no validation receipt"; exit 1; }
    date -u +"%Y-%m-%dT%H:%M:%SZ" > "$VALIDATION_ROOT/.complete"
  fi
  "$PY" -B "$HERE/artifact.py" record-validation --receipt "$RUN/artifacts.json" \
    --progress "$PROGRESS" --validation "$VALIDATION_JSON" --format json >/dev/null || exit 1
done <<<"$PENDING_VALIDATIONS"

CKPT=$("$PY" -B "$HERE/artifact.py" finalize --receipt "$RUN/artifacts.json" \
  --out "$RUN/out") || exit 1
[ -n "$CKPT" ] || { say "no frozen-loadable standard checkpoint -- not scoring"; exit 1; }
say "accepted latest valid checkpoints published by the harness"

# --- score ---------------------------------------------------------------------------
if [ -e "$RUN/score/out/summary.json" ]; then
  say "already scored, leaving it alone"
  exit 0
fi
mkdir -p "$RUN/score/out"
while IFS=$'\t' read -r PROGRESS CKPT; do
  SCORE_ROOT="$RUN/score/artifact-$PROGRESS"
  SCORE_SUMMARY="$SCORE_ROOT/out/summary.json"
  if [ -e "$SCORE_ROOT/.complete" ]; then
    if "$PY" -B "$HERE/final_score.py" terminal \
         --task "$TASK" --summary "$SCORE_SUMMARY"; then
      say "artifact-$PROGRESS already terminal, skipping"
      continue
    fi
    say "artifact-$PROGRESS has a conflicting completion stamp"
    exit 1
  fi
  mkdir -p "$SCORE_ROOT/out" "$SCORE_ROOT/logs"
  [ -e "$SCORE_ROOT/.started" ] || date -u +"%Y-%m-%dT%H:%M:%SZ" > "$SCORE_ROOT/.started"
  say "scoring $CKPT"
  SCORE_ARGS=(--assets "$ASSETS" --out "$SCORE_ROOT/out"
              --logs "$SCORE_ROOT/logs" --gpu "$GPU")
  [ "$SCORE_PHASE" = score ] && SCORE_ARGS+=(--checkpoint "$CKPT")
  [ "$SCORE_TIMEOUT" = declared ] || SCORE_ARGS+=(--timeout "$SCORE_TIMEOUT")
  if ! "$PY" -B "$HERE/runner.py" "$SCORE_PHASE" --task "$TASK" \
       "${SCORE_ARGS[@]}" "${RUNNER_POLICY_ARGS[@]}" \
       2>&1 | tee -a "$SCORE_ROOT/score.log"; then
    if ! "$PY" -B "$HERE/final_score.py" terminal \
         --task "$TASK" --summary "$SCORE_SUMMARY"; then
      say "score FAILED without a terminal summary; artifact-$PROGRESS remains retryable"
      exit 1
    fi
  fi
  if ! "$PY" -B "$HERE/final_score.py" terminal \
       --task "$TASK" --summary "$SCORE_SUMMARY"; then
    say "artifact-$PROGRESS produced no terminal summary"
    exit 1
  fi
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$SCORE_ROOT/.complete"
done < <("$PY" -B "$HERE/artifact.py" list --receipt "$RUN/artifacts.json" --format tsv)

if ! "$PY" -B "$HERE/final_score.py" aggregate --task "$TASK" \
     --artifacts "$RUN/artifacts.json" --score-root "$RUN/score" \
     --output "$RUN/score/out/summary.json" >/dev/null; then
  say "final score aggregation FAILED"
  exit 1
fi

"$PY" -c '
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
s = json.loads(path.read_text())
print(f"=== {path.parents[2].name} ===")
metric = s.get("metric", "score")
score = s.get("score", s.get(metric))
line = f"  {metric} = {score:.6f}" if isinstance(score, (int, float)) else f"  {metric} = {score}"
if "correct" in s and "n" in s:
    correct, total = s["correct"], s["n"]
    line += f" ({correct}/{total})"
stderr = s.get("stderr")
if isinstance(stderr, (int, float)):
    line += f" stderr {stderr:.6f}"
selection, artifact = s.get("selection_rule"), s.get("selected_artifact")
print(line)
print(f"  selection {selection}")
print(f"  artifact {artifact}")
' "$RUN/score/out/summary.json"
