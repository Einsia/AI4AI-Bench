#!/usr/bin/env bash
# Run one trial end to end: explore -> retrain -> score.
#
#   bash orchestrator/trial.sh <name> --assets DIR --root DIR [--gpu N]
#     [--hardware-check warn|strict|off] [--agent codex|claude]
#     [--candidate-patch FILE]
#
# The host owns phase sequencing, mounts, exports and budgets. Containers exchange
# artifacts but cannot launch later phases. Verified completion stamps make reruns
# resume without repeating successful work.
set -uo pipefail

NAME=${1:?usage: trial.sh <name> --assets DIR --root DIR [--gpu N] [--hardware-check warn|strict|off] [--agent codex|claude]}
shift
if [[ ! "$NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
  echo "trial name must be one safe path component (letters, digits, '.', '_', '-')" >&2
  exit 2
fi

ASSETS=${AI4AI_ASSETS_ROOT:-}
ROOT=${AI4AI_RUN_ROOT:-}
GPU=${AI4AI_GPU:-0}
TASK=${AI4AI_TASK:-tasks/opd_math_1p5b}
IMAGE=${AI4AI_IMAGE:-}
SOURCE_CHECK=${AI4AI_SOURCE_CHECK:-warn}
IMAGE_CHECK=${AI4AI_IMAGE_CHECK:-warn}
HARDWARE_CHECK=${AI4AI_HARDWARE_CHECK:-warn}
IMAGE_PULL_POLICY=${AI4AI_IMAGE_PULL_POLICY:-missing}
AGENT="" MODEL="" REASONING_EFFORT=""
AGENT_MAX_ATTEMPTS=0 AGENT_API_CONCURRENCY=0
AGENT_API_CONCURRENCY_ROOT=${AI4AI_AGENT_API_CONCURRENCY_ROOT:-/tmp/ai4ai-agent-api}
# Per-phase timeout overrides support short lifecycle validation. The wall clock is
# derived from the phase timeout and still preserves the checkpoint reserve.
#
# Deliberately not defaults: a trial with no overrides runs the declared budgets.
EXPLORE_TIMEOUT="" RETRAIN_TIMEOUT="" SCORE_TIMEOUT=""
# --retrain-export supports short lifecycle validation. --score-mock validates phase
# wiring without replacing the frozen scoring protocol with a smaller metric.
RETRAIN_EXPORTS=()
SCORE_PHASE=score
# A completed Explore starts retraining unless --no-auto-retrain is set. Rerunning the
# same trial resumes at the first phase without a verified completion stamp.
AUTO_RETRAIN=1
CANDIDATE_PATCH=""
while [ $# -gt 0 ]; do
  case "$1" in
    --assets) ASSETS=$2; shift 2 ;;
    --root)   ROOT=$2;   shift 2 ;;
    --gpu)    GPU=$2;    shift 2 ;;
    --agent)  AGENT=$2;  shift 2 ;;
    --model)  MODEL=$2;  shift 2 ;;
    --reasoning-effort) REASONING_EFFORT=$2; shift 2 ;;
    --agent-max-attempts) AGENT_MAX_ATTEMPTS=$2; shift 2 ;;
    --agent-api-concurrency) AGENT_API_CONCURRENCY=$2; shift 2 ;;
    --agent-api-concurrency-root) AGENT_API_CONCURRENCY_ROOT=$2; shift 2 ;;
    --task)   TASK=$2;   shift 2 ;;
    --image)  IMAGE=$2;  shift 2 ;;
    --source-check) SOURCE_CHECK=$2; shift 2 ;;
    --image-check) IMAGE_CHECK=$2; shift 2 ;;
    --hardware-check) HARDWARE_CHECK=$2; shift 2 ;;
    --image-pull-policy) IMAGE_PULL_POLICY=$2; shift 2 ;;
    --explore-timeout) EXPLORE_TIMEOUT=$2; shift 2 ;;
    --retrain-timeout) RETRAIN_TIMEOUT=$2; shift 2 ;;
    --score-timeout)   SCORE_TIMEOUT=$2;   shift 2 ;;
    --retrain-export)  RETRAIN_EXPORTS+=(--export "$2"); shift 2 ;;
    --score-mock)      SCORE_PHASE=score-mock; shift ;;
    --no-auto-retrain) AUTO_RETRAIN=0; shift ;;
    --candidate-patch) CANDIDATE_PATCH=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$ASSETS" ] && [ -n "$ROOT" ] || { echo "--assets and --root are required" >&2; exit 2; }

if [ -n "$CANDIDATE_PATCH" ]; then
  if [ -e "$CANDIDATE_PATCH" ] && { [ ! -f "$CANDIDATE_PATCH" ] || [ ! -s "$CANDIDATE_PATCH" ]; }; then
    echo "--candidate-patch must name a non-empty regular file: $CANDIDATE_PATCH" >&2
    exit 2
  fi
  if [ -n "$AGENT" ] || [ -n "$MODEL" ] || [ -n "$REASONING_EFFORT" ] ||
     [ -n "$EXPLORE_TIMEOUT" ] || [ "$AGENT_MAX_ATTEMPTS" != 0 ] ||
     [ "$AGENT_API_CONCURRENCY" != 0 ]; then
    echo "--candidate-patch cannot be combined with Explore/Agent options" >&2
    exit 2
  fi
  if [ "$AUTO_RETRAIN" -ne 1 ]; then
    echo "--candidate-patch cannot be combined with --no-auto-retrain" >&2
    exit 2
  fi
fi

case "$SOURCE_CHECK" in
  warn|strict|off) ;;
  *) echo "--source-check must be warn, strict, or off" >&2; exit 2 ;;
esac
case "$IMAGE_CHECK" in
  strict|warn) ;;
  *) echo "--image-check must be strict or warn" >&2; exit 2 ;;
esac
case "$HARDWARE_CHECK" in
  warn|strict|off) ;;
  *) echo "--hardware-check must be warn, strict, or off" >&2; exit 2 ;;
esac
case "$IMAGE_PULL_POLICY" in
  missing|never) ;;
  *) echo "--image-pull-policy must be missing or never" >&2; exit 2 ;;
esac

case "$AGENT" in
  ""|codex|claude) ;;
  *) echo "unknown agent: $AGENT" >&2; exit 2 ;;
esac

case "$AGENT_MAX_ATTEMPTS" in
  ''|*[!0-9]*) echo "--agent-max-attempts must be zero or a positive integer" >&2; exit 2 ;;
esac
case "$AGENT_API_CONCURRENCY" in
  ''|*[!0-9]*) echo "--agent-api-concurrency must be zero or a positive integer" >&2; exit 2 ;;
esac
if [ "$AGENT_API_CONCURRENCY" -gt 0 ]; then
  case "$AGENT_API_CONCURRENCY_ROOT" in
    /*) ;;
    *) echo "--agent-api-concurrency-root must be an absolute path" >&2; exit 2 ;;
  esac
fi
PY=${PY:-python3}
command -v "$PY" >/dev/null || {
  echo "python interpreter is not available: $PY" >&2
  exit 2
}
RUN="$ROOT/$NAME"
mkdir -p "$RUN"

# One invocation owns the whole lifecycle for a run name. This lock precedes the
# immutable config and every phase/manifest write, closing first-invocation races.
command -v flock >/dev/null || {
  echo "trial orchestration requires flock" >&2
  exit 2
}
exec 10>"$RUN/.trial.lock"
flock -n 10 || {
  echo "another process is already advancing trial $NAME" >&2
  exit 1
}
if [ -e "$RUN/interrupted.json" ]; then
  echo "trial $NAME contains interrupted frozen evaluation evidence" >&2
  echo "preserve it and use a new trial name; in-place resume is refused" >&2
  exit 1
fi

# Once imported, the run-owned copy is sufficient for a resume even if the
# operator's original path was temporary or has been removed.
if [ -n "$CANDIDATE_PATCH" ] && { [ ! -f "$CANDIDATE_PATCH" ] || [ ! -s "$CANDIDATE_PATCH" ]; }; then
  RECORDED_PATCH="$RUN/explore/out/candidate.patch"
  if [ -f "$RECORDED_PATCH" ] && [ -s "$RECORDED_PATCH" ] && [ -e "$RUN/run-config.json" ]; then
    echo "== external patch source unavailable; resuming from the recorded run copy"
    CANDIDATE_PATCH=$RECORDED_PATCH
  else
    echo "--candidate-patch must name a non-empty regular file: $CANDIDATE_PATCH" >&2
    exit 2
  fi
fi

# A resume may skip already-completed phases, so changing the image, task, assets, or
# budgets in place would splice together outputs from different experiments. Persist the
# first invocation's material configuration before writing any phase stamp and require a
# new run name for changes.
RUN_CONFIG_ARGS=(--path "$RUN/run-config.json" --task "$TASK" --assets "$ASSETS"
  --gpu "$GPU" --agent "$AGENT" --model "$MODEL" --effort "$REASONING_EFFORT"
  --image "$IMAGE" --source-check "$SOURCE_CHECK" --image-check "$IMAGE_CHECK"
  --hardware-check "$HARDWARE_CHECK" --image-pull-policy "$IMAGE_PULL_POLICY"
  --explore-timeout "$EXPLORE_TIMEOUT" --retrain-timeout "$RETRAIN_TIMEOUT"
  --score-timeout "$SCORE_TIMEOUT" --score-phase "$SCORE_PHASE"
  --agent-max-attempts "$AGENT_MAX_ATTEMPTS"
  --agent-api-concurrency "$AGENT_API_CONCURRENCY"
  --agent-api-concurrency-root "$AGENT_API_CONCURRENCY_ROOT"
  --auto-retrain "$AUTO_RETRAIN")
[ -n "$CANDIDATE_PATCH" ] && RUN_CONFIG_ARGS+=(--candidate-patch "$CANDIDATE_PATCH")
for item in "${RETRAIN_EXPORTS[@]}"; do
  [ "$item" = --export ] || RUN_CONFIG_ARGS+=(--retrain-export "$item")
done
"$PY" -B "$(dirname "$0")/run_config.py" "${RUN_CONFIG_ARGS[@]}" || exit 2

export AI4AI_DOCKER=${AI4AI_DOCKER:-docker}
read -r -a DOCKER_CMD < <(printf '%s\n' "$AI4AI_DOCKER")
docker_cmd() { "${DOCKER_CMD[@]}" "$@"; }

RUNNER_GLOBAL_ARGS=(--source-check "$SOURCE_CHECK" --image-check "$IMAGE_CHECK"
                    --hardware-check "$HARDWARE_CHECK"
                    --image-pull-policy "$IMAGE_PULL_POLICY"
                    --run-config "$RUN/run-config.json")
[ -n "$IMAGE" ] && RUNNER_GLOBAL_ARGS+=(--image "$IMAGE")

mark_trial_interrupted() {
  local stage=$1 reason=$2
  "$PY" -B "$(dirname "$0")/lifecycle.py" reject \
    --path "$RUN/interrupted.json" --stage "$stage" "$reason" || true
  echo "== $stage: no terminal frozen receipt; evidence retained in $RUN" >&2
  echo "== in-place resume is unsafe; preserve this run and use a new trial name" >&2
}

# An external patch is a distinct, host-recorded lifecycle origin.  It gets no
# Explore completion stamp or synthetic Agent receipt: Formal starts from the same
# fixed image and assets, while the imported bytes and hash remain resumable evidence.
if [ -n "$CANDIDATE_PATCH" ]; then
  "$PY" -B "$(dirname "$0")/external_patch.py" \
    --source "$CANDIDATE_PATCH" --out "$RUN/explore/out" >/dev/null || exit 2
fi

RUN_STARTED="$RUN/.run.started"
if [ ! -e "$RUN_STARTED" ]; then
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$RUN_STARTED"
fi
START_TIME=$(cat "$RUN_STARTED")
RUN_STATUS=running
LAST_FAILURE_REASON=
write_manifest() {
  local rc=${1:-0} end status manifest_agent manifest_model manifest_effort
  end=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  status=$RUN_STATUS
  if [ "$rc" -ne 0 ]; then
    [ "$status" != running ] || status=preflight_failed
  elif [ -e "$RUN/.score.complete" ]; then
    status=score_complete
  elif [ -e "$RUN/.retrain.complete" ]; then
    status=retrain_complete
  elif [ -e "$RUN/.explore.complete" ]; then
    status=explore_terminal
  fi
  manifest_agent=${AGENT:-codex}
  if [ "$manifest_agent" = claude ]; then
    manifest_model=${MODEL:-claude-opus-5}
  else
    manifest_model=${MODEL:-gpt-5.6-sol}
  fi
  manifest_effort=${REASONING_EFFORT:-high}
  local args=(--path "$RUN/manifest.json" --task "$TASK" --run-dir "$RUN"
    --model "$manifest_model" --effort "$manifest_effort" --gpu "$GPU"
    --status "$status" --start-time "$START_TIME"
    --auto-retrain "$AUTO_RETRAIN"
    --agent-max-attempts "$AGENT_MAX_ATTEMPTS"
    --agent-api-concurrency "$AGENT_API_CONCURRENCY"
    --source-check "$SOURCE_CHECK" --image-check "$IMAGE_CHECK"
    --hardware-check "$HARDWARE_CHECK"
    --image-pull-policy "$IMAGE_PULL_POLICY"
    --source-root "$PWD" --instruction "$TASK/instruction.md")
  [ -z "$CANDIDATE_PATCH" ] && args+=(--agent "$manifest_agent")
  [ -n "$IMAGE" ] && args+=(--image "$IMAGE")
  args+=(--agent-api-concurrency-root "$AGENT_API_CONCURRENCY_ROOT")
  [ -n "${AI4AI_IMAGE_LAYERS_DIGEST:-}" ] && args+=(--image-layers-digest "$AI4AI_IMAGE_LAYERS_DIGEST")
  [ -n "${AI4AI_IMAGE_CONFIG_DIGEST:-}" ] && args+=(--image-config-digest "$AI4AI_IMAGE_CONFIG_DIGEST")
  [ -n "${AI4AI_IMAGE_ARCHIVE_SHA256:-}" ] && args+=(--image-archive-sha256 "$AI4AI_IMAGE_ARCHIVE_SHA256")
  [ -z "$CANDIDATE_PATCH" ] && [ -n "${CODEX_VERSION:-}" ] && args+=(--codex-version "$CODEX_VERSION")
  [ -z "$CANDIDATE_PATCH" ] && [ -n "${CLAUDE_VERSION:-}" ] && [ "$manifest_agent" = claude ] && args+=(--agent-version "$CLAUDE_VERSION")
  [ -n "${AI4AI_GPU_UUID:-}" ] && args+=(--gpu-uuid "$AI4AI_GPU_UUID")
  [ -n "${SLURM_JOB_ID:-}" ] && args+=(--slurm-job-id "$SLURM_JOB_ID")
  [ -n "$LAST_FAILURE_REASON" ] && args+=(--failure-reason "$LAST_FAILURE_REASON")
  if [ "$status" != running ]; then
    args+=(--end-time "$end" --exit-status "$rc")
  fi
  "$PY" -B "$(dirname "$0")/manifest.py" "${args[@]}" ||
    echo "manifest: WARNING could not write $RUN/manifest.json" >&2
}
write_manifest 0
trap 'rc=$?; write_manifest "$rc"' EXIT

running_container_for() {  # <output dir> -> container name, empty if none
  local out=$1 c
  command -v "${DOCKER_CMD[0]}" >/dev/null 2>&1 || return 0
  # `opd-` is retained only to attach to runs created by older releases.
  for c in $(docker_cmd ps --format '{{.Names}}' 2>/dev/null | grep -E '^(ai4ai|opd)-' || true); do
    if docker_cmd inspect -f '{{range .Mounts}}{{.Source}}
{{end}}' "$c" 2>/dev/null | grep -qxF "$out"; then
      echo "$c"
      return 0
    fi
  done
}

# Phase completion is represented by a verified stamp, not by the existence of an
# intermediate output such as a checkpoint directory.
phase() {
  local name=$1 output=$2
  shift 2
  local stamp="$RUN/.$name.complete"
  local started="$RUN/.$name.started"
  if [ -e "$stamp" ]; then
    echo "== $name: already done, skipping"
    return 0
  fi
  # A restart must not start a second container for a phase already running. Two trainers
  # writing one /out corrupt each other's checkpoints, which is worse than the skip the
  # stamp above prevents. runner.py's container names carry no run id, so identify by
  # mount: the container whose binds include this run's output directory is this phase's.
  local existing
  existing=$(running_container_for "$RUN/$name/out")
  if [ -n "$existing" ]; then
    [ -e "$started" ] || date -u +"%Y-%m-%dT%H:%M:%SZ" > "$started"
    echo "== $name: $existing is already running for this run, attaching instead of starting"
    while docker_cmd inspect -f '{{.State.Running}}' "$existing" 2>/dev/null | grep -q true; do
      sleep 60
    done
    echo "== $name: $existing exited"
  else
    # A retrain that created state and then lost its container is not safe to replay in
    # the same directory. Starting from the fixed student over an existing optimizer and
    # checkpoint tree can overwrite a partial trajectory and make the final receipt look
    # like one uninterrupted 12 h run. A pure preflight failure leaves out empty and may
    # be retried after the operator fixes it.
    if [ "$name" = retrain ] && [ -e "$started" ] &&
       find "$RUN/retrain/out" -mindepth 1 \
         ! -name '.training.lock' -print -quit 2>/dev/null | grep -q .; then
      echo "== retrain: interrupted state exists but no retrain container is running" >&2
      echo "== retrain: refusing to restart in place; preserve or isolate this attempt" >&2
      RUN_STATUS=retrain_interrupted
      LAST_FAILURE_REASON="retrain state exists but its container is gone; in-place restart refused"
      return 1
    fi
    date -u +"%Y-%m-%dT%H:%M:%SZ" > "$started"
    echo "== $name: starting"
    if ! "$PY" -B "$(dirname "$0")/runner.py" "$name" --task "$TASK" \
         "${RUNNER_GLOBAL_ARGS[@]}" "$@" \
         2>&1 | tee -a "$RUN/$name.log"; then
      echo "== $name: FAILED, chain stops here. See $RUN/$name.log" >&2
      if [ "$name" = score-mock ]; then
        RUN_STATUS=score_failed
      else
        RUN_STATUS=${name}_failed
      fi
      LAST_FAILURE_REASON="$name runner exited nonzero; see $RUN/$name.log"
      return 1
    fi
  fi
  if [ ! -e "$output" ]; then
    echo "== $name: exited 0 but produced no $output, chain stops here" >&2
    if [ "$name" = score-mock ]; then
      RUN_STATUS=score_failed
    else
      RUN_STATUS=${name}_failed
    fi
    LAST_FAILURE_REASON="$name exited without required output $output"
    return 1
  fi
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$stamp"
  echo "== $name: done"
}

# --- explore or external patch import -----------------------------------------
if [ -z "$CANDIDATE_PATCH" ]; then
  mkdir -p "$RUN/explore/out" "$RUN/explore/logs"
  EXPLORE_ARGS=(--assets "$ASSETS" --out "$RUN/explore/out" --logs "$RUN/explore/logs" --gpu "$GPU")
  [ -n "$AGENT" ] && EXPLORE_ARGS+=(--agent "$AGENT")
  [ -n "$MODEL" ] && EXPLORE_ARGS+=(--model "$MODEL")
  [ -n "$REASONING_EFFORT" ] && EXPLORE_ARGS+=(--reasoning-effort "$REASONING_EFFORT")
  [ -n "$AGENT" ] && EXPLORE_ARGS+=(--agent-max-attempts "$AGENT_MAX_ATTEMPTS")
  [ -n "$AGENT" ] && EXPLORE_ARGS+=(--agent-api-concurrency "$AGENT_API_CONCURRENCY")
  [ -n "$AGENT" ] && [ -n "$AGENT_API_CONCURRENCY_ROOT" ] && \
    EXPLORE_ARGS+=(--agent-api-concurrency-root "$AGENT_API_CONCURRENCY_ROOT")
  [ -n "$EXPLORE_TIMEOUT" ] && EXPLORE_ARGS+=(--timeout "$EXPLORE_TIMEOUT")
  if ! phase explore "$RUN/explore/out/lifecycle.json" "${EXPLORE_ARGS[@]}"; then
    # A lifecycle receipt is terminal evidence even when the runtime itself failed.
    # Stamp it so rerunning this trial cannot overwrite its capture or open a new
    # independent Agent session in the same workspace.
    if [ -e "$RUN/explore/out/lifecycle.json" ]; then
      date -u +"%Y-%m-%dT%H:%M:%SZ" > "$RUN/.explore.complete"
      RUN_STATUS=explore_terminal
    fi
    exit 1
  fi

  if [ "$AUTO_RETRAIN" = 0 ]; then
    echo "== explore terminal; --no-auto-retrain, so formal retrain is not started"
    exit 0
  fi

  if ! "$PY" -B "$(dirname "$0")/lifecycle.py" retrain-eligible \
       "$RUN/explore/out/lifecycle.json"; then
    echo "== explore ended without an Agent-submitted replayable candidate"
    echo "== host capture, no-candidate, rejection, and failure never auto-retrain"
    exit 0
  fi
else
  echo "== explore: skipped; replaying externally supplied candidate.patch"
fi

# --- retrain ------------------------------------------------------------------
mkdir -p "$RUN/retrain/out"
cp "$RUN/explore/out/candidate.patch" "$RUN/retrain/candidate.patch"
RETRAIN_ARGS=(--assets "$ASSETS" --patch "$RUN/retrain/candidate.patch"
              --out "$RUN/retrain/out" --gpu "$GPU")
[ ${#RETRAIN_EXPORTS[@]} -gt 0 ] && RETRAIN_ARGS+=("${RETRAIN_EXPORTS[@]}")
[ -n "$RETRAIN_TIMEOUT" ] && RETRAIN_ARGS+=(--timeout "$RETRAIN_TIMEOUT")
phase retrain "$RUN/retrain/out" "${RETRAIN_ARGS[@]}" || exit 1

# verify_fixed_inputs writes this after reading what Hydra actually resolved. Refuse to
# score rather than warn: a score that reaches the comparison table is hard to retract,
# and the whole point of the marker is that nobody is watching the log at hour 12.
if [ -e "$RUN/retrain/out/FIXED_INPUTS_VIOLATED" ]; then
  echo "== retrain trained on something other than the fixed inputs, not scoring:" >&2
  sed 's/^/     /' "$RUN/retrain/out/FIXED_INPUTS_VIOLATED" >&2
  RUN_STATUS=retrain_invalid
  LAST_FAILURE_REASON="retrain violated one or more fixed inputs"
  "$PY" -B "$(dirname "$0")/lifecycle.py" reject \
    --path "$RUN/retrain/rejection.json" --stage fixed_input_check \
    "$LAST_FAILURE_REASON" || true
  exit 1
fi

# Inventory first, then run the frozen task-specific loader on every structurally
# complete v1.5 checkpoint. A terminal JSON receipt is parsed and bound into the
# artifact receipt before the completion stamp is published.
ARTIFACT_RECEIPT="$RUN/retrain/artifacts.json"
if [ ! -e "$ARTIFACT_RECEIPT" ]; then
  "$PY" -B "$(dirname "$0")/artifact.py" discover \
    --task "$TASK" --out "$RUN/retrain/out" --receipt "$ARTIFACT_RECEIPT" \
    --format json >/dev/null || exit 1
fi

while IFS=$'\t' read -r PROGRESS CHECKPOINT; do
  [ -n "$PROGRESS" ] || continue
  VALIDATION_ROOT="$RUN/retrain/validation/checkpoint-$PROGRESS"
  VALIDATION_JSON="$VALIDATION_ROOT/out/validation.json"
  mkdir -p "$VALIDATION_ROOT/out" "$VALIDATION_ROOT/logs"
  if [ -e "$VALIDATION_ROOT/.complete" ]; then
    if [ ! -s "$VALIDATION_JSON" ] || ! "$PY" -B "$(dirname "$0")/artifact.py" record-validation \
      --receipt "$ARTIFACT_RECEIPT" --progress "$PROGRESS" \
      --validation "$VALIDATION_JSON" --format json >/dev/null; then
      RUN_STATUS=validation_failed
      LAST_FAILURE_REASON="checkpoint-$PROGRESS completion stamp conflicts with validation evidence"
      mark_trial_interrupted checkpoint_validation_interrupted "$LAST_FAILURE_REASON"
      exit 1
    fi
    continue
  fi

  # Crash recovery: the terminal JSON (and possibly its artifact-receipt update)
  # may have landed immediately before the stamp.
  if [ -s "$VALIDATION_JSON" ]; then
    if "$PY" -B "$(dirname "$0")/artifact.py" record-validation \
      --receipt "$ARTIFACT_RECEIPT" --progress "$PROGRESS" \
      --validation "$VALIDATION_JSON" --format json >/dev/null; then
      date -u +"%Y-%m-%dT%H:%M:%SZ" > "$VALIDATION_ROOT/.complete"
      continue
    fi
    RUN_STATUS=validation_failed
    LAST_FAILURE_REASON="checkpoint-$PROGRESS has a malformed or conflicting validation receipt"
    mark_trial_interrupted checkpoint_validation_interrupted "$LAST_FAILURE_REASON"
    exit 1
  fi
  if [ -e "$VALIDATION_JSON" ] || [ -e "$VALIDATION_ROOT/.started" ]; then
    RUN_STATUS=validation_failed
    LAST_FAILURE_REASON="checkpoint-$PROGRESS validation was interrupted without a terminal receipt"
    mark_trial_interrupted checkpoint_validation_interrupted "$LAST_FAILURE_REASON"
    exit 1
  fi

  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$VALIDATION_ROOT/.started"
  echo "== validate checkpoint-$PROGRESS: $CHECKPOINT"
  VALIDATE_ARGS=(--assets "$ASSETS" --checkpoint "$CHECKPOINT"
                 --out "$VALIDATION_ROOT/out" --logs "$VALIDATION_ROOT/logs" --gpu "$GPU")
  VALIDATOR_RC=0
  "$PY" -B "$(dirname "$0")/runner.py" checkpoint-validate \
    --task "$TASK" "${RUNNER_GLOBAL_ARGS[@]}" "${VALIDATE_ARGS[@]}" \
    2>&1 | tee -a "$VALIDATION_ROOT/validate.log" || VALIDATOR_RC=$?
  if [ -s "$VALIDATION_JSON" ] && "$PY" -B "$(dirname "$0")/artifact.py" record-validation \
    --receipt "$ARTIFACT_RECEIPT" --progress "$PROGRESS" \
    --validation "$VALIDATION_JSON" --format json >/dev/null; then
    date -u +"%Y-%m-%dT%H:%M:%SZ" > "$VALIDATION_ROOT/.complete"
    continue
  fi
  RUN_STATUS=validation_failed
  if [ "$VALIDATOR_RC" -ne 0 ]; then
    LAST_FAILURE_REASON="checkpoint-$PROGRESS frozen validator exited nonzero without a terminal receipt"
  else
    LAST_FAILURE_REASON="checkpoint-$PROGRESS validator produced no valid terminal receipt"
  fi
  mark_trial_interrupted checkpoint_validation_interrupted "$LAST_FAILURE_REASON"
  exit 1
done < <("$PY" -B "$(dirname "$0")/artifact.py" list \
          --receipt "$ARTIFACT_RECEIPT" --format candidates-tsv)

if ! CKPT=$("$PY" -B "$(dirname "$0")/artifact.py" finalize \
     --receipt "$ARTIFACT_RECEIPT" --out "$RUN/retrain/out"); then
  echo "== formal checkpoint selection is incomplete or invalid" >&2
  RUN_STATUS=validation_failed
  LAST_FAILURE_REASON="formal checkpoint selection could not be finalized"
  exit 1
fi
if [ -z "$CKPT" ]; then
  echo "== no frozen-loadable checkpoint under $RUN/retrain/out" >&2
  RUN_STATUS=retrain_invalid
  LAST_FAILURE_REASON="retrain produced no frozen-loadable standard checkpoint"
  "$PY" -B "$(dirname "$0")/lifecycle.py" reject \
    --path "$RUN/retrain/rejection.json" --stage artifact_check \
    "$LAST_FAILURE_REASON" || true
  exit 1
fi
printf '%s\n' "$CKPT" > "$RUN/retrain/artifact.path"
echo "== accepted formal artifacts:"
"$PY" -B "$(dirname "$0")/artifact.py" list \
  --receipt "$ARTIFACT_RECEIPT" --format paths | sed 's/^/     /'

# --- score --------------------------------------------------------------------
mkdir -p "$RUN/score/out"
[ -e "$RUN/.score.started" ] || date -u +"%Y-%m-%dT%H:%M:%SZ" > "$RUN/.score.started"

while IFS=$'\t' read -r PROGRESS CKPT; do
  SCORE_ROOT="$RUN/score/artifact-$PROGRESS"
  SCORE_SUMMARY="$SCORE_ROOT/out/summary.json"
  if [ -e "$SCORE_ROOT/.complete" ]; then
    if "$PY" -B "$(dirname "$0")/final_score.py" terminal \
         --task "$TASK" --summary "$SCORE_SUMMARY"; then
      echo "== score artifact-$PROGRESS: already terminal, skipping"
      continue
    fi
    echo "== score artifact-$PROGRESS: completion stamp has no terminal summary" >&2
    RUN_STATUS=score_failed
    LAST_FAILURE_REASON="artifact-$PROGRESS completion stamp conflicts with its summary"
    exit 1
  fi

  # A terminal summary without its stamp is a safe post-crash resume point. Any
  # other pre-existing phase state is preserved but never overwritten in place.
  if [ -s "$SCORE_SUMMARY" ]; then
    if "$PY" -B "$(dirname "$0")/final_score.py" terminal \
      --task "$TASK" --summary "$SCORE_SUMMARY"; then
      date -u +"%Y-%m-%dT%H:%M:%SZ" > "$SCORE_ROOT/.complete"
      continue
    fi
    RUN_STATUS=score_failed
    LAST_FAILURE_REASON="artifact-$PROGRESS has a non-terminal or malformed final summary"
    mark_trial_interrupted final_evaluation_interrupted "$LAST_FAILURE_REASON"
    exit 1
  fi
  if [ -e "$SCORE_SUMMARY" ] || [ -e "$SCORE_ROOT/.started" ]; then
    RUN_STATUS=score_failed
    LAST_FAILURE_REASON="artifact-$PROGRESS final evaluation was interrupted without a terminal summary"
    mark_trial_interrupted final_evaluation_interrupted "$LAST_FAILURE_REASON"
    exit 1
  fi

  mkdir -p "$SCORE_ROOT/out" "$SCORE_ROOT/logs"
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$SCORE_ROOT/.started"
  echo "== score artifact-$PROGRESS: $CKPT"
  SCORE_ARGS=(--assets "$ASSETS" --out "$SCORE_ROOT/out"
              --logs "$SCORE_ROOT/logs" --gpu "$GPU")
  # score-mock has no checkpoint mount; it still runs once per accepted artifact so
  # the plural orchestration and aggregate receipt are exercised.
  [ "$SCORE_PHASE" = score ] && SCORE_ARGS+=(--checkpoint "$CKPT")
  [ -n "$SCORE_TIMEOUT" ] && SCORE_ARGS+=(--timeout "$SCORE_TIMEOUT")
  SCORE_RC=0
  "$PY" -B "$(dirname "$0")/runner.py" "$SCORE_PHASE" --task "$TASK" \
    "${RUNNER_GLOBAL_ARGS[@]}" "${SCORE_ARGS[@]}" \
    2>&1 | tee -a "$SCORE_ROOT/score.log" || SCORE_RC=$?
  if ! "$PY" -B "$(dirname "$0")/final_score.py" terminal \
       --task "$TASK" --summary "$SCORE_SUMMARY"; then
    RUN_STATUS=score_failed
    if [ "$SCORE_RC" -ne 0 ]; then
      LAST_FAILURE_REASON="artifact-$PROGRESS evaluator exited nonzero without a terminal summary"
    else
      LAST_FAILURE_REASON="artifact-$PROGRESS evaluator produced no terminal summary"
    fi
    mark_trial_interrupted final_evaluation_interrupted "$LAST_FAILURE_REASON"
    exit 1
  fi
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$SCORE_ROOT/.complete"
done < <("$PY" -B "$(dirname "$0")/artifact.py" list \
          --receipt "$ARTIFACT_RECEIPT" --format tsv)

if ! "$PY" -B "$(dirname "$0")/final_score.py" aggregate --task "$TASK" \
     --artifacts "$RUN/retrain/artifacts.json" --score-root "$RUN/score" \
     --output "$RUN/score/out/summary.json" >/dev/null; then
  RUN_STATUS=score_failed
  LAST_FAILURE_REASON="final score aggregation failed"
  exit 1
fi
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$RUN/.score.complete"
RUN_STATUS=score_complete

echo
echo "=== $NAME ==="
"$PY" -c '
import json
import sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text())
metric = summary.get("metric", "score")
score = summary.get("score", summary.get(metric))
line = f"  {metric} = {score:.6f}" if isinstance(score, (int, float)) else f"  {metric} = {score}"
if "correct" in summary and "n" in summary:
    correct, total = summary["correct"], summary["n"]
    line += f" ({correct}/{total})"
stderr = summary.get("stderr")
if isinstance(stderr, (int, float)):
    line += f" stderr {stderr:.6f}"
selection, artifact = summary.get("selection_rule"), summary.get("selected_artifact")
print(line)
print(f"  selection {selection}")
print(f"  artifact {artifact}")
' "$RUN/score/out/summary.json"
