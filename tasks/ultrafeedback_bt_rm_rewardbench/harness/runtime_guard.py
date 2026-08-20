"""Runtime locking and CUDA telemetry shared by task entry points."""

from __future__ import annotations

import fcntl
import json
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

LOCK_NAME = ".ai4ai-operation.lock"


def lock_root() -> Path:
    return Path(os.environ.get("AI4AI_LOCK_ROOT", "/out")).resolve()


@contextmanager
def exclusive_output(operation: str, root: Path | None = None) -> Iterator[None]:
    """Fail if another train/eval process holds this run's output root."""

    target = (root or lock_root()).resolve()
    target.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target / LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another train/eval operation holds {target / LOCK_NAME}; "
                "use one operation per run output root"
            ) from error
        record = {
            "operation": operation,
            "pid": os.getpid(),
            "started_unix": time.time(),
        }
        os.ftruncate(descriptor, 0)
        os.write(descriptor, (json.dumps(record, sort_keys=True) + "\n").encode())
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def cuda_telemetry(torch: Any, *, require_single: bool = True) -> dict[str, Any]:
    """Return device facts or fail before a GPU result can be emitted without them."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    count = int(torch.cuda.device_count())
    if count < 1:
        raise RuntimeError("CUDA reported no visible devices")
    if require_single and count != 1:
        raise RuntimeError(f"expected exactly one visible GPU, found {count}")
    devices = []
    for index in range(count):
        properties = torch.cuda.get_device_properties(index)
        capability = tuple(int(value) for value in torch.cuda.get_device_capability(index))
        total_memory = int(properties.total_memory)
        name = str(properties.name).strip()
        if not name or len(capability) != 2 or total_memory <= 0:
            raise RuntimeError(f"incomplete CUDA telemetry for device {index}")
        devices.append(
            {
                "index": index,
                "name": name,
                "compute_capability": list(capability),
                "total_memory_bytes": total_memory,
            }
        )
    return {
        "visible_device_count": count,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "devices": devices,
    }


def peak_memory_bytes(torch: Any) -> int:
    peak = int(torch.cuda.max_memory_allocated())
    if peak <= 0:
        raise RuntimeError("CUDA peak-memory telemetry is empty")
    return peak


def reexec_locked() -> None:
    """CLI wrapper used by run.sh so locking precedes cache and receipt writes."""

    if len(sys.argv) < 3:
        raise SystemExit("usage: runtime_guard.py <lock-root> <command> [args ...]")
    root = Path(sys.argv[1]).resolve()
    command = sys.argv[2:]
    root.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(root / LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise SystemExit(
            f"another train/eval operation holds {root / LOCK_NAME}; "
            "use a unique run output and wait for the active operation"
        ) from error
    record = {"operation": "train", "pid": os.getpid(), "started_unix": time.time()}
    os.ftruncate(descriptor, 0)
    os.write(descriptor, (json.dumps(record, sort_keys=True) + "\n").encode())
    os.set_inheritable(descriptor, True)
    environment = dict(os.environ)
    environment["AI4AI_OUTPUT_LOCK_HELD"] = "1"
    environment["AI4AI_LOCK_ROOT"] = str(root)
    os.execvpe(command[0], command, environment)


if __name__ == "__main__":
    reexec_locked()
