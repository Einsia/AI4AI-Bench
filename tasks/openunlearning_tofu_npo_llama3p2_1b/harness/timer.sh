#!/usr/bin/env bash
set -euo pipefail

deadline_file=${DEADLINE_FILE:-/logs/deadline.json}
test -f "${deadline_file}" || {
  echo "timer: no deadline at ${deadline_file}" >&2
  exit 1
}
remaining=$(python3 -c '
import json, sys, time
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(int(payload["deadline_unix"] - time.time()))
' "${deadline_file}")
if [[ "${1:-}" == "--seconds" ]]; then
  echo "${remaining}"
elif (( remaining < 0 )); then
  printf 'past deadline by %dm %ds\n' $(( -remaining / 60 )) $(( -remaining % 60 ))
else
  printf '%dh %dm %ds remaining\n' \
    $(( remaining / 3600 )) $(( remaining % 3600 / 60 )) $(( remaining % 60 ))
fi
