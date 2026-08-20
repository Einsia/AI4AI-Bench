#!/usr/bin/env bash
set -euo pipefail

checkpoint=${1:?usage: fast_eval.sh CHECKPOINT [OUTPUT_JSON]}
output=${2:-/out/fast-eval.json}
exec python3 /opt/harness/fast_eval.py \
  --checkpoint "${checkpoint}" --assets /assets --output "${output}"
