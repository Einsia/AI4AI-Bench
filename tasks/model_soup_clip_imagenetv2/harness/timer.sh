#!/usr/bin/env bash
# Seconds left in the 4 h exploration phase.
#
# The host writes /logs/deadline.json with an absolute timestamp when it starts
# the container; /logs is read-only here. You can see the deadline, you cannot
# move it. When it arrives the container is killed from outside.

set -euo pipefail

DEADLINE_FILE=${DEADLINE_FILE:-/logs/deadline.json}

if [[ ! -f "${DEADLINE_FILE}" ]]; then
  echo "timer: no deadline at ${DEADLINE_FILE}" >&2
  exit 1
fi

remaining=$(python3 -c '
import json
import sys
import time
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(int(payload["deadline_unix"] - time.time()))
' "${DEADLINE_FILE}")

if [[ "${1:-}" == "--seconds" ]]; then
  echo "${remaining}"
  exit 0
fi

if (( remaining < 0 )); then
  printf 'past deadline by %dm %ds\n' $(( -remaining / 60 )) $(( -remaining % 60 ))
else
  printf '%dh %dm %ds remaining\n' \
    $(( remaining / 3600 )) $(( remaining % 3600 / 60 )) $(( remaining % 60 ))
fi
