#!/usr/bin/env bash
# Explicitly submit the current source candidate and end exploration.
set -euo pipefail

python3 /opt/harness/lifecycle.py submit --out "${OUT_DIR:-/out}"
echo "submit: ending the exploration container"
kill -TERM 1
