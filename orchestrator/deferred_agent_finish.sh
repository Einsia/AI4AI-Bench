#!/usr/bin/env bash
# Claude must emit its terminal stream-json result before the container is removed.
# This read-only agent-phase mount replaces the image's immediate PID-1 kill wrappers;
# the host still force-removes the container as soon as the CLI exits.
set -euo pipefail

case "${0##*/}" in
  submit.sh)
    [[ $# -eq 0 ]] || { echo "usage: /opt/harness/submit.sh" >&2; exit 2; }
    python3 /opt/harness/lifecycle.py submit --out "${OUT_DIR:-/out}"
    echo "submit: receipt recorded; waiting for the agent result"
    ;;
  no_candidate.sh)
    if [[ $# -ne 1 || -z "$1" ]]; then
      echo 'usage: /opt/harness/no_candidate.sh "reason"' >&2
      exit 2
    fi
    python3 /opt/harness/lifecycle.py no-candidate --out "${OUT_DIR:-/out}" "$1"
    echo "no-candidate: receipt recorded; waiting for the agent result"
    ;;
  *)
    echo "unexpected deferred lifecycle entrypoint: $0" >&2
    exit 2
    ;;
esac
