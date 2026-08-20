#!/usr/bin/env python3
"""Run a formal recipe, forward signals to its process group, then publish.

A plain shell sequence (``trainer; publisher``) risks killing only the shell at
the outer deadline and leaving Ray, torchrun, or another training child alive.
This supervisor is exec'd by shipped recipes, gives the recipe its own process
group, and publishes artifacts only after a clean exit.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import publish_artifacts  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        separator = arguments.index("--run")
    except ValueError:
        print("run_and_publish: missing --run separator", file=sys.stderr)
        return 2
    publish_argv = arguments[:separator]
    command = arguments[separator + 1 :]
    if not command:
        print("run_and_publish: no recipe command after --run", file=sys.stderr)
        return 2

    child = subprocess.Popen(command, start_new_session=True)

    received_signal: int | None = None

    def forward(signum: int, _frame: object) -> None:
        nonlocal received_signal
        received_signal = signum
        try:
            os.killpg(child.pid, signum)
        except ProcessLookupError:
            pass

    previous = {
        signum: signal.signal(signum, forward)
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    }
    try:
        status = child.wait()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    if received_signal is not None:
        return 128 + received_signal
    if status < 0:
        return 128 - status
    if status != 0:
        return status
    return publish_artifacts.main(publish_argv)


if __name__ == "__main__":
    raise SystemExit(main())
