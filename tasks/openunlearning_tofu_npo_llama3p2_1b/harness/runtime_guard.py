"""Runtime guards shared by the frozen NPO evaluators."""

from __future__ import annotations

import fcntl
import json
import math
import re
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

WEIGHT_PATTERNS = ("*.safetensors", "*.bin")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model")


def checkpoint_lock_path(checkpoint: Path) -> Path:
    checkpoint = checkpoint.resolve()
    return checkpoint.parent / f".{checkpoint.name}.train.lock"


@contextmanager
def checkpoint_read_lock(checkpoint: Path) -> Iterator[Path]:
    """Refuse to evaluate a checkpoint while its trainer holds the write lock."""

    lock_path = checkpoint_lock_path(checkpoint)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"checkpoint is still being trained; refusing concurrent evaluation: {checkpoint}"
            ) from exc
        try:
            yield lock_path
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def parse_gpu_sample(stdout: str) -> dict[str, Any]:
    rows = [line.strip() for line in stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"expected telemetry for exactly one visible GPU, got {len(rows)}")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 3:
        raise RuntimeError(f"malformed nvidia-smi telemetry row: {rows[0]!r}")
    uuid, memory_text, utilization_text = fields
    try:
        memory_mib = float(memory_text)
        utilization_percent = float(utilization_text)
    except ValueError as exc:
        raise RuntimeError(f"non-numeric nvidia-smi telemetry row: {rows[0]!r}") from exc
    if (
        not uuid
        or not math.isfinite(memory_mib)
        or not math.isfinite(utilization_percent)
        or memory_mib < 0
        or not 0 <= utilization_percent <= 100
    ):
        raise RuntimeError(f"invalid nvidia-smi telemetry row: {rows[0]!r}")
    return {
        "gpu_uuid": uuid,
        "memory_used_mib": memory_mib,
        "utilization_percent": utilization_percent,
    }


def sample_gpu() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "GPU telemetry unavailable: nvidia-smi executable not found"
        ) from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic"
        raise RuntimeError(
            f"GPU telemetry unavailable: nvidia-smi exited {result.returncode}: {detail}"
        )
    return parse_gpu_sample(result.stdout)


class GpuTelemetry:
    def __init__(self) -> None:
        self.gpu_uuid: str | None = None
        self.samples = 0
        self.peak_memory_mib = 0.0
        self.peak_utilization_percent = 0.0

    def observe(self) -> dict[str, Any]:
        sample = sample_gpu()
        uuid = str(sample["gpu_uuid"])
        if self.gpu_uuid is not None and uuid != self.gpu_uuid:
            raise RuntimeError(f"visible GPU changed during evaluation: {self.gpu_uuid} -> {uuid}")
        self.gpu_uuid = uuid
        self.samples += 1
        self.peak_memory_mib = max(self.peak_memory_mib, float(sample["memory_used_mib"]))
        self.peak_utilization_percent = max(
            self.peak_utilization_percent,
            float(sample["utilization_percent"]),
        )
        return sample

    def summary(self) -> dict[str, Any]:
        if self.samples <= 0 or self.gpu_uuid is None:
            raise RuntimeError("evaluation produced no valid GPU telemetry samples")
        return {
            "gpu_uuid": self.gpu_uuid,
            "sample_count": self.samples,
            "peak_memory_used_mib": self.peak_memory_mib,
            "peak_utilization_percent": self.peak_utilization_percent,
            "source": "nvidia-smi inside the single-GPU evaluation container",
        }


def _has_weights(path: Path) -> bool:
    return any(any(path.glob(pattern)) for pattern in WEIGHT_PATTERNS)


def is_loadable_checkpoint(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "config.json").is_file()
        and (path / "tokenizer_config.json").is_file()
        and any((path / name).is_file() for name in TOKENIZER_FILES)
        and _has_weights(path)
    )


def checkpoint_step(path: Path) -> int | None:
    state = path / "trainer_state.json"
    if state.is_file():
        try:
            value = json.loads(state.read_text(encoding="utf-8"))["global_step"]
            step = int(value)
            if step >= 0:
                return step
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else None


def resolve_checkpoint(root: Path) -> Path:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"checkpoint root is not a directory: {root}")

    directories = [root, *(path for path in root.rglob("*") if path.is_dir())]
    loadable = [path for path in directories if is_loadable_checkpoint(path)]
    if not loadable:
        raise ValueError(f"no self-contained loadable checkpoint below {root}")

    with_step = [(checkpoint_step(path), path) for path in loadable]
    numbered = [(step, path) for step, path in with_step if step is not None]
    if numbered:
        highest = max(step for step, _ in numbered)
        winners = [path for step, path in numbered if step == highest]
    else:
        winners = loadable
    if len(winners) != 1:
        raise ValueError(f"ambiguous highest loadable checkpoint below {root}: {winners}")
    return winners[0]
