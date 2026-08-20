"""Run one GPU phase under a non-blocking advisory lock.

Training and fast evaluation share one visible GPU inside an exploration
container. Running both at once changes throughput, memory pressure and sometimes
the result. The shell entry points use this helper so a second phase fails quickly
instead of silently overlapping the first one.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

LOCKED_EXIT = 75


def acquire(path: Path, label: str) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o664)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.lseek(descriptor, 0, os.SEEK_SET)
        owner = os.read(descriptor, 4096).decode("utf-8", errors="replace").strip()
        os.close(descriptor)
        print(
            f"gpu_phase_lock: another train/eval phase owns {path}: "
            f"{owner or 'owner metadata unavailable'}",
            file=sys.stderr,
        )
        raise SystemExit(LOCKED_EXIT) from None

    metadata = {
        "label": label,
        "pid": os.getpid(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    encoded = (json.dumps(metadata, sort_keys=True) + "\n").encode()
    os.ftruncate(descriptor, 0)
    os.lseek(descriptor, 0, os.SEEK_SET)
    os.write(descriptor, encoded)
    os.fsync(descriptor)
    # Python file descriptors are non-inheritable by default. The lock must remain
    # held after exec replaces this helper with the real training/evaluation process.
    os.set_inheritable(descriptor, True)
    return descriptor


def smoke() -> None:
    with tempfile.TemporaryDirectory(prefix="gpu-phase-lock-") as directory:
        path = Path(directory) / "phase.lock"
        descriptor = acquire(path, "smoke")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("label") != "smoke":
            raise RuntimeError(f"lock metadata was not written: {payload}")
        contender = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--lock",
                str(path),
                "--label",
                "contender",
                "--",
                sys.executable,
                "-c",
                "raise SystemExit(0)",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if contender.returncode != LOCKED_EXIT or '"label": "smoke"' not in contender.stderr:
            raise RuntimeError(
                "a concurrent phase was not rejected with owner evidence: "
                f"status={contender.returncode} stderr={contender.stderr!r}"
            )
        os.close(descriptor)
    print(json.dumps({"gpu_phase_lock_smoke": "passed"}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--label", default="gpu-phase")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.smoke:
        smoke()
        return
    if args.lock is None:
        parser.error("--lock is required")
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        parser.error("a command is required after --")
    acquire(args.lock, args.label)
    os.execvpe(command[0], command, os.environ.copy())


if __name__ == "__main__":
    main()
