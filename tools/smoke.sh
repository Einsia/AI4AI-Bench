#!/usr/bin/env bash
# Exercise the public image, Docker GPU path, host mounts and synthetic score phase.
set -euo pipefail

ROOT=""
GPU=0
while [ $# -gt 0 ]; do
  case "$1" in
    --root) ROOT=$2; shift 2 ;;
    --gpu) GPU=$2; shift 2 ;;
    -h|--help)
      echo "usage: smoke.sh --root DIR [--gpu N]"
      exit 0
      ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[ -n "$ROOT" ] || { echo "--root is required" >&2; exit 2; }
case "$GPU" in
  ''|*[!0-9]*) echo "--gpu must be a non-negative integer" >&2; exit 2 ;;
esac

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
PY=${PY:-python3}
command -v "$PY" >/dev/null || { echo "python interpreter is not available: $PY" >&2; exit 2; }

mkdir -p "$ROOT"
SMOKE=$(mktemp -d "$ROOT/smoke-XXXXXXXX")
mkdir -p "$SMOKE/assets" "$SMOKE/out" "$SMOKE/logs"

echo "AI4AI smoke output: $SMOKE"
"$PY" -B "$REPO/orchestrator/runner.py" score-mock \
  --task "$REPO/tasks/ddpo_sd15_aesthetic" \
  --assets "$SMOKE/assets" \
  --out "$SMOKE/out" \
  --logs "$SMOKE/logs" \
  --gpu "$GPU" \
  --source-check strict \
  --image-check strict \
  --hardware-check warn \
  --image-pull-policy missing

"$PY" -B - "$SMOKE/out/summary.json" "$SMOKE/preflight.json" <<'PY'
import json
import math
import sys
from pathlib import Path

summary_path, preflight_path = map(Path, sys.argv[1:])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
if (
    summary.get("schema_version") != 1
    or summary.get("status") != "passed"
    or summary.get("mock") is not True
    or summary.get("n") != 256
    or summary.get("metric") != "mean_aesthetic_score_final256"
    or summary.get("direction") != "maximize"
):
    raise SystemExit(f"mock evaluator returned an unexpected summary: {summary_path}")
if not isinstance(summary.get("score"), (int, float)) or not math.isfinite(summary["score"]):
    raise SystemExit(f"mock evaluator returned no finite score: {summary_path}")
if preflight.get("phase") != "score-mock" or preflight.get("capability_checks") != "passed":
    raise SystemExit(f"runtime capability receipt is incomplete: {preflight_path}")
print("AI4AI smoke passed: Docker, immutable image, GPU kernel, mounts, and mock score")
print(f"  summary:   {summary_path}")
print(f"  preflight: {preflight_path}")
PY
