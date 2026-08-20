#!/usr/bin/env bash
# Validate and score one to three existing checkpoints without training them.
set -uo pipefail

NAME=${1:?usage: evaluate.sh <name> --task DIR --assets DIR --root DIR --checkpoint PROGRESS=PATH [--checkpoint ...]}
shift
if [[ ! "$NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
  echo "evaluation name must be one safe path component (letters, digits, '.', '_', '-')" >&2
  exit 2
fi

ASSETS=${AI4AI_ASSETS_ROOT:-}
ROOT=${AI4AI_EVALUATION_ROOT:-}
GPU=${AI4AI_GPU:-0}
TASK=${AI4AI_TASK:-tasks/opd_math_1p5b}
IMAGE=${AI4AI_IMAGE:-}
SOURCE_CHECK=${AI4AI_SOURCE_CHECK:-warn}
IMAGE_CHECK=${AI4AI_IMAGE_CHECK:-warn}
HARDWARE_CHECK=${AI4AI_HARDWARE_CHECK:-warn}
IMAGE_PULL_POLICY=${AI4AI_IMAGE_PULL_POLICY:-missing}
SCORE_TIMEOUT=""
SCORE_PHASE=score
CHECKPOINTS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --assets) ASSETS=$2; shift 2 ;;
    --root) ROOT=$2; shift 2 ;;
    --gpu) GPU=$2; shift 2 ;;
    --task) TASK=$2; shift 2 ;;
    --image) IMAGE=$2; shift 2 ;;
    --source-check) SOURCE_CHECK=$2; shift 2 ;;
    --image-check) IMAGE_CHECK=$2; shift 2 ;;
    --hardware-check) HARDWARE_CHECK=$2; shift 2 ;;
    --image-pull-policy) IMAGE_PULL_POLICY=$2; shift 2 ;;
    --score-timeout) SCORE_TIMEOUT=$2; shift 2 ;;
    --score-mock) SCORE_PHASE=score-mock; shift ;;
    --checkpoint) CHECKPOINTS+=("$2"); shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[ -n "$ASSETS" ] && [ -n "$ROOT" ] || {
  echo "--assets and --root are required" >&2
  exit 2
}
if [ ${#CHECKPOINTS[@]} -lt 1 ] || [ ${#CHECKPOINTS[@]} -gt 3 ]; then
  echo "provide between one and three --checkpoint PROGRESS=PATH arguments" >&2
  exit 2
fi
case "$SOURCE_CHECK" in warn|strict|off) ;; *) echo "invalid --source-check" >&2; exit 2 ;; esac
case "$IMAGE_CHECK" in warn|strict) ;; *) echo "invalid --image-check" >&2; exit 2 ;; esac
case "$HARDWARE_CHECK" in warn|strict|off) ;; *) echo "invalid --hardware-check" >&2; exit 2 ;; esac
case "$IMAGE_PULL_POLICY" in missing|never) ;; *) echo "invalid --image-pull-policy" >&2; exit 2 ;; esac

PY=${PY:-python3}
command -v "$PY" >/dev/null || {
  echo "python interpreter is not available: $PY" >&2
  exit 2
}
EVAL="$ROOT/$NAME"
mkdir -p "$EVAL"

# Serialize before any receipt/config write. artifact.py imports the task declaration,
# so the trusted byte-level host contract check must also run before that import.
command -v flock >/dev/null || {
  echo "checkpoint-only evaluation requires flock" >&2
  exit 2
}
exec 9>"$EVAL/.evaluation.lock"
flock -n 9 || {
  echo "another process is already advancing evaluation $NAME" >&2
  exit 1
}
if [ -e "$EVAL/interrupted.json" ]; then
  echo "evaluation $NAME contains interrupted frozen-evaluator evidence" >&2
  echo "preserve it and use a new evaluation name; in-place resume is refused" >&2
  exit 1
fi
"$PY" -B "$(dirname "$0")/host_contract.py" \
  --task "$TASK" --mode "$SOURCE_CHECK" >/dev/null || exit 2

ARTIFACT_RECEIPT="$EVAL/artifacts.json"
ARTIFACT_OUT="$EVAL/artifacts/out"
EXTERNAL_ARGS=()
for checkpoint in "${CHECKPOINTS[@]}"; do
  EXTERNAL_ARGS+=(--checkpoint "$checkpoint")
done
"$PY" -B "$(dirname "$0")/artifact.py" external \
  --task "$TASK" --out "$ARTIFACT_OUT" --receipt "$ARTIFACT_RECEIPT" \
  "${EXTERNAL_ARGS[@]}" --format json >/dev/null || exit 2

CONFIG="$EVAL/evaluation-config.json"
"$PY" -B "$(dirname "$0")/evaluation_config.py" \
  --path "$CONFIG" --task "$TASK" --assets "$ASSETS" --artifacts "$ARTIFACT_RECEIPT" \
  --gpu "$GPU" --image "$IMAGE" --source-check "$SOURCE_CHECK" \
  --image-check "$IMAGE_CHECK" --hardware-check "$HARDWARE_CHECK" \
  --image-pull-policy "$IMAGE_PULL_POLICY" --score-phase "$SCORE_PHASE" \
  --score-timeout "$SCORE_TIMEOUT" || exit 2

RUNNER_GLOBAL_ARGS=(--source-check "$SOURCE_CHECK" --image-check "$IMAGE_CHECK"
  --hardware-check "$HARDWARE_CHECK" --image-pull-policy "$IMAGE_PULL_POLICY"
  --run-config "$CONFIG")
[ -n "$IMAGE" ] && RUNNER_GLOBAL_ARGS+=(--image "$IMAGE")

mark_evaluation_interrupted() {
  local stage=$1 reason=$2
  "$PY" -B "$(dirname "$0")/lifecycle.py" reject \
    --path "$EVAL/interrupted.json" --stage "$stage" "$reason" || true
  echo "== $stage: no terminal frozen receipt; evidence retained in $EVAL" >&2
  echo "== in-place resume is unsafe; preserve it and use a new evaluation name" >&2
}

[ -e "$EVAL/.evaluation.started" ] || date -u +"%Y-%m-%dT%H:%M:%SZ" > "$EVAL/.evaluation.started"

# Bind every candidate on every invocation, including terminal-invalid candidates
# that no longer appear in the pending-validation or accepted-score lists.
while IFS=$'\t' read -r PROGRESS _; do
  [ -n "$PROGRESS" ] || continue
  "$PY" -B "$(dirname "$0")/artifact.py" verify-external \
    --receipt "$ARTIFACT_RECEIPT" --progress "$PROGRESS" >/dev/null || exit 1
done < <("$PY" -B "$(dirname "$0")/artifact.py" list \
  --receipt "$ARTIFACT_RECEIPT" --format candidates-tsv)

# Validation is task-specific and terminal invalidity is evidence, not a zero score.
while IFS=$'\t' read -r PROGRESS CHECKPOINT; do
  [ -n "$PROGRESS" ] || continue
  VALIDATION_ROOT="$EVAL/validation/checkpoint-$PROGRESS"
  VALIDATION_JSON="$VALIDATION_ROOT/out/validation.json"
  mkdir -p "$VALIDATION_ROOT/out" "$VALIDATION_ROOT/logs"
  # Re-check even when consuming a terminal receipt on resume. This keeps an
  # invalid (and therefore unscored) checkpoint bound to the initialized bytes.
  "$PY" -B "$(dirname "$0")/artifact.py" verify-external \
    --receipt "$ARTIFACT_RECEIPT" --progress "$PROGRESS" >/dev/null || exit 1
  if [ -e "$VALIDATION_ROOT/.complete" ]; then
    if [ ! -s "$VALIDATION_JSON" ] || ! "$PY" -B "$(dirname "$0")/artifact.py" record-validation \
      --receipt "$ARTIFACT_RECEIPT" --progress "$PROGRESS" \
      --validation "$VALIDATION_JSON" --format json >/dev/null; then
      mark_evaluation_interrupted checkpoint_validation_interrupted \
        "checkpoint-$PROGRESS completion stamp conflicts with validation evidence"
      exit 1
    fi
    continue
  fi

  # Recover a crash after terminal JSON/receipt publication but before the stamp.
  if [ -s "$VALIDATION_JSON" ]; then
    if "$PY" -B "$(dirname "$0")/artifact.py" record-validation \
      --receipt "$ARTIFACT_RECEIPT" --progress "$PROGRESS" \
      --validation "$VALIDATION_JSON" --format json >/dev/null; then
      date -u +"%Y-%m-%dT%H:%M:%SZ" > "$VALIDATION_ROOT/.complete"
      continue
    fi
    mark_evaluation_interrupted checkpoint_validation_interrupted \
      "checkpoint-$PROGRESS has a malformed or conflicting validation receipt"
    exit 1
  fi
  if [ -e "$VALIDATION_JSON" ] || [ -e "$VALIDATION_ROOT/.started" ]; then
    mark_evaluation_interrupted checkpoint_validation_interrupted \
      "checkpoint-$PROGRESS validation was interrupted without a terminal receipt"
    exit 1
  fi

  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$VALIDATION_ROOT/.started"
  echo "== validate checkpoint-$PROGRESS: $CHECKPOINT"
  VALIDATOR_RC=0
  "$PY" -B "$(dirname "$0")/runner.py" checkpoint-validate \
    --task "$TASK" "${RUNNER_GLOBAL_ARGS[@]}" --assets "$ASSETS" \
    --checkpoint "$CHECKPOINT" --out "$VALIDATION_ROOT/out" \
    --logs "$VALIDATION_ROOT/logs" --gpu "$GPU" \
    2>&1 | tee -a "$VALIDATION_ROOT/validate.log" || VALIDATOR_RC=$?
  "$PY" -B "$(dirname "$0")/artifact.py" verify-external \
    --receipt "$ARTIFACT_RECEIPT" --progress "$PROGRESS" >/dev/null || exit 1
  if [ -s "$VALIDATION_JSON" ] && "$PY" -B "$(dirname "$0")/artifact.py" record-validation \
    --receipt "$ARTIFACT_RECEIPT" --progress "$PROGRESS" \
    --validation "$VALIDATION_JSON" --format json >/dev/null; then
    date -u +"%Y-%m-%dT%H:%M:%SZ" > "$VALIDATION_ROOT/.complete"
    continue
  fi
  if [ "$VALIDATOR_RC" -ne 0 ]; then
    VALIDATION_REASON="checkpoint-$PROGRESS validator exited nonzero without a terminal receipt"
  else
    VALIDATION_REASON="checkpoint-$PROGRESS validator produced no valid terminal receipt"
  fi
  mark_evaluation_interrupted checkpoint_validation_interrupted "$VALIDATION_REASON"
  exit 1
done < <("$PY" -B "$(dirname "$0")/artifact.py" list \
  --receipt "$ARTIFACT_RECEIPT" --format candidates-tsv)

"$PY" -B "$(dirname "$0")/artifact.py" finalize \
  --receipt "$ARTIFACT_RECEIPT" --out "$ARTIFACT_OUT" >/dev/null || exit 1

mkdir -p "$EVAL/score/out"
[ -e "$EVAL/.score.started" ] || date -u +"%Y-%m-%dT%H:%M:%SZ" > "$EVAL/.score.started"
while IFS=$'\t' read -r PROGRESS CHECKPOINT; do
  [ -n "$PROGRESS" ] || continue
  SCORE_ROOT="$EVAL/score/artifact-$PROGRESS"
  SCORE_SUMMARY="$SCORE_ROOT/out/summary.json"
  if [ -e "$SCORE_ROOT/.complete" ]; then
    if "$PY" -B "$(dirname "$0")/final_score.py" terminal \
      --task "$TASK" --summary "$SCORE_SUMMARY"; then
      echo "== score artifact-$PROGRESS: already terminal, skipping"
      continue
    fi
    echo "== score artifact-$PROGRESS: stamp conflicts with summary" >&2
    exit 1
  fi
  if [ -s "$SCORE_SUMMARY" ]; then
    if "$PY" -B "$(dirname "$0")/final_score.py" terminal \
      --task "$TASK" --summary "$SCORE_SUMMARY"; then
      date -u +"%Y-%m-%dT%H:%M:%SZ" > "$SCORE_ROOT/.complete"
      continue
    fi
    mark_evaluation_interrupted final_evaluation_interrupted \
      "artifact-$PROGRESS has a non-terminal or malformed final summary"
    exit 1
  fi
  if [ -e "$SCORE_SUMMARY" ] || [ -e "$SCORE_ROOT/.started" ]; then
    mark_evaluation_interrupted final_evaluation_interrupted \
      "artifact-$PROGRESS final evaluation was interrupted without a terminal summary"
    exit 1
  fi
  "$PY" -B "$(dirname "$0")/artifact.py" verify-external \
    --receipt "$ARTIFACT_RECEIPT" --progress "$PROGRESS" >/dev/null || exit 1
  mkdir -p "$SCORE_ROOT/out" "$SCORE_ROOT/logs"
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$SCORE_ROOT/.started"
  echo "== score artifact-$PROGRESS: $CHECKPOINT"
  SCORE_ARGS=(--assets "$ASSETS" --out "$SCORE_ROOT/out" --logs "$SCORE_ROOT/logs" --gpu "$GPU")
  [ "$SCORE_PHASE" = score ] && SCORE_ARGS+=(--checkpoint "$CHECKPOINT")
  [ -n "$SCORE_TIMEOUT" ] && SCORE_ARGS+=(--timeout "$SCORE_TIMEOUT")
  SCORE_RC=0
  "$PY" -B "$(dirname "$0")/runner.py" "$SCORE_PHASE" --task "$TASK" \
    "${RUNNER_GLOBAL_ARGS[@]}" "${SCORE_ARGS[@]}" \
    2>&1 | tee -a "$SCORE_ROOT/score.log" || SCORE_RC=$?
  "$PY" -B "$(dirname "$0")/artifact.py" verify-external \
    --receipt "$ARTIFACT_RECEIPT" --progress "$PROGRESS" >/dev/null || exit 1
  if ! "$PY" -B "$(dirname "$0")/final_score.py" terminal \
    --task "$TASK" --summary "$SCORE_SUMMARY"; then
    if [ "$SCORE_RC" -ne 0 ]; then
      SCORE_REASON="artifact-$PROGRESS evaluator exited nonzero without a terminal summary"
    else
      SCORE_REASON="artifact-$PROGRESS evaluator produced no terminal summary"
    fi
    mark_evaluation_interrupted final_evaluation_interrupted "$SCORE_REASON"
    exit 1
  fi
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$SCORE_ROOT/.complete"
done < <("$PY" -B "$(dirname "$0")/artifact.py" list \
  --receipt "$ARTIFACT_RECEIPT" --format tsv)

# Close the provenance interval after all containers have stopped, even when every
# checkpoint was terminal-invalid and therefore never entered the score loop.
while IFS=$'\t' read -r PROGRESS _; do
  [ -n "$PROGRESS" ] || continue
  "$PY" -B "$(dirname "$0")/artifact.py" verify-external \
    --receipt "$ARTIFACT_RECEIPT" --progress "$PROGRESS" >/dev/null || exit 1
done < <("$PY" -B "$(dirname "$0")/artifact.py" list \
  --receipt "$ARTIFACT_RECEIPT" --format candidates-tsv)

"$PY" -B "$(dirname "$0")/final_score.py" aggregate --task "$TASK" \
  --artifacts "$ARTIFACT_RECEIPT" --score-root "$EVAL/score" \
  --output "$EVAL/score/out/summary.json" --evaluation-config "$CONFIG" \
  >/dev/null || exit 1
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$EVAL/.score.complete"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$EVAL/.evaluation.complete"

echo
echo "=== $NAME ==="
"$PY" -B -c '
import json, sys
from pathlib import Path
s = json.loads(Path(sys.argv[1]).read_text())
c = json.loads(Path(sys.argv[2]).read_text())
print("  {} = {}".format(s.get("metric"), s.get("score")))
print("  status {}".format(s.get("status")))
print("  selection {}".format(s.get("selection_rule")))
print("  artifact {}".format(s.get("selected_artifact")))
verification = c.get("verification", {})
order = ("source_check", "image_check", "hardware_check", "score_phase",
         "score_timeout_overridden", "declared_score_timeout", "effective_score_timeout",
         "score_timeout_matches_declared")
print("  verification {}".format(" ".join(f"{k}={verification.get(k)}" for k in order)))
classification = str(c.get("result_classification", "non_official_local"))
classification = classification.replace("_", " ").replace("non official", "non-official")
print("  classification {}".format(classification))
print("  config {}".format(sys.argv[2]))
' "$EVAL/score/out/summary.json" "$CONFIG"
