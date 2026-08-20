#!/usr/bin/env bash
# Explicitly end exploration without a candidate.
set -euo pipefail

if [[ $# -ne 1 || -z "$1" ]]; then
  echo 'usage: /opt/harness/no_candidate.sh "reason"' >&2
  exit 2
fi
python3 /opt/harness/lifecycle.py no-candidate --out "${OUT_DIR:-/out}" "$1"
echo "no-candidate: ending the exploration container"
kill -TERM 1
