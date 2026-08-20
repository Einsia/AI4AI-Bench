"""DDPO driver: launch the inner trainer, then export the last complete adapter.

launch_training.sh holds every recipe knob. This file adds two things:

1. **Export.** Upstream saves through `accelerate.save_state()` with automatic
   checkpoint naming, so adapters land at
   `${LOG_DIR}/<run_name>_<timestamp>/checkpoints/checkpoint_<n>` -- a path with a
   timestamp in it and a counter that is Accelerate's save index, not the epoch
   number. The score phase mounts one directory. This copies the highest-numbered
   adapter to `/out/checkpoint/pytorch_lora_weights.bin`.

2. **Schedule presets.** `--profile proxy|retrain` and `--one-step`, so the same tree
   serves a short probe and the full run.

The public `bash run.sh` invokes this driver, so direct and formal runs both produce
the artifact consumed by the score phase.

`training_metadata.json` is written for provenance and **nothing gates on it**. The old
branch's evaluator read five fields out of it and refused a checkpoint whose JSON did
not name the right algorithm family, upstream revision and so on. The candidate writes
that file, so it only ever proved the candidate agreed with itself; the two checks that
survive into v1 are computed from the images the adapter produces. See
harness/final_eval.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

TASK_ID = "ddpo_sd15_aesthetic"
ALGORITHM_FAMILY = "ddpo_lora_ppo"
WORKSPACE = Path(__file__).resolve().parent
INNER_LAUNCHER = WORKSPACE / "launch_training.sh"
LORA_WEIGHT_NAME = "pytorch_lora_weights.bin"

# epochs, and whether to save only at the end.
#
# `epochs = None` means "leave run.sh's own default alone", which is what the retrain
# profile does. Writing a number here would make it the third place the same quantity
# lives -- run.sh's default, this table, and the budget arithmetic -- and on the
# reference task those three drifted and the wrong one was in force.
PROFILES = {
    "proxy": 4,
    "retrain": None,
}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_is_loadable(path: Path) -> bool:
    """Return whether a LoRA file is a non-empty state dict that Torch can load."""

    try:
        import torch

        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # Older pinned Torch releases do not expose weights_only.
            state = torch.load(path, map_location="cpu")
    except Exception:  # A partially written save is evidence to skip, not to select.
        return False
    return isinstance(state, dict) and bool(state)


def latest_adapter(log_dir: Path) -> Path:
    """The highest-numbered Accelerate checkpoint holding a LoRA adapter.

    Sorted by the trailing integer, not by name: checkpoint_10 sorts before
    checkpoint_9 lexicographically, which is harmless at 4 epochs and wrong at 24.

    The counter is Accelerate's save index under `automatic_checkpoint_naming`, so it
    counts saves rather than epochs -- with save_freq 1 they coincide, and with
    anything else they do not. The reference protocol recorded exactly this confusion as
    invalid evidence: "checkpoint_0 is Accelerate automatic checkpoint numbering for
    the epoch-31 save, not the untrained initialization".
    """

    candidates: list[tuple[int, Path]] = []
    for path in log_dir.glob("*/checkpoints/checkpoint_*"):
        suffix = path.name.removeprefix("checkpoint_")
        if suffix.isdigit() and (path / LORA_WEIGHT_NAME).is_file():
            candidates.append((int(suffix), path))
    if not candidates:
        raise FileNotFoundError(
            f"no checkpoint_*/{LORA_WEIGHT_NAME} under {log_dir}. Upstream only saves "
            "when `epoch % save_freq == 0`, so check SAVE_FREQ against NUM_EPOCHS."
        )
    for _, path in sorted(candidates, reverse=True):
        if adapter_is_loadable(path / LORA_WEIGHT_NAME):
            return path
    raise RuntimeError(f"no loadable {LORA_WEIGHT_NAME} under {log_dir}")


def nonnegative_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def stop_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def train(*, profile: str, output: Path, one_step: bool, extra: list[str]) -> dict[str, Any]:
    epochs = PROFILES[profile]
    if one_step:
        epochs = 1

    output.mkdir(parents=True, exist_ok=True)
    log_dir = output / "logs"
    environment = os.environ.copy()
    environment.update({"OUTPUT_DIR": str(output), "LOG_DIR": str(log_dir)})
    # Omitted rather than set, so run.sh's own default applies. Setting it would
    # silently override whatever the candidate wrote there.
    if epochs is not None:
        environment["NUM_EPOCHS"] = str(epochs)
        environment["SAVE_FREQ"] = "1"

    max_wall_seconds = nonnegative_env("MAX_WALL_TIME_SECONDS", 0)
    reserve_seconds = nonnegative_env("DEADLINE_RESERVE_SECONDS", 300)
    if max_wall_seconds and max_wall_seconds <= reserve_seconds:
        raise ValueError(
            "MAX_WALL_TIME_SECONDS must exceed DEADLINE_RESERVE_SECONDS"
        )
    train_timeout = max_wall_seconds - reserve_seconds if max_wall_seconds else None

    started = time.monotonic()
    wall_clock_stop = False
    with (output / "train.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            ["bash", str(INNER_LAUNCHER), *extra],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=train_timeout)
        except subprocess.TimeoutExpired:
            wall_clock_stop = True
            stop_process_group(process)
            returncode = process.returncode if process.returncode is not None else 124
    elapsed = time.monotonic() - started
    if returncode != 0 and not wall_clock_stop:
        raise RuntimeError(
            f"training failed with exit {returncode}; see {output / 'train.log'}"
        )

    source = latest_adapter(log_dir)
    checkpoint = output / "checkpoint"
    checkpoint.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / LORA_WEIGHT_NAME, checkpoint / LORA_WEIGHT_NAME)

    adapter_sha256 = file_sha256(checkpoint / LORA_WEIGHT_NAME)
    metadata = {
        "schema_version": 2,
        "task_id": TASK_ID,
        "algorithm_family": ALGORITHM_FAMILY,
        "profile": profile,
        "selected_upstream_checkpoint": source.name,
        "selected_from": str(source),
        "adapter_sha256": adapter_sha256,
        "num_epochs": environment.get("NUM_EPOCHS", "run.sh default"),
        "learning_rate": environment.get("LEARNING_RATE", "run.sh default"),
        "ppo_clip_range": environment.get("PPO_CLIP_RANGE", "run.sh default"),
        "seed": environment.get("SEED", "run.sh default"),
        "max_wall_time_seconds": max_wall_seconds,
        "deadline_reserve_seconds": reserve_seconds,
        "wall_clock_stop": wall_clock_stop,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "note": "provenance only; nothing downstream gates on these fields",
    }
    atomic_json(checkpoint / "training_metadata.json", metadata)
    summary = {
        "schema_version": 2,
        "task_id": TASK_ID,
        "profile": profile,
        "checkpoint": str(checkpoint),
        "adapter_sha256": adapter_sha256,
        "wall_seconds": elapsed,
        "returncode": returncode,
        "wall_clock_stop": wall_clock_stop,
    }
    atomic_json(output / "retrain_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="proxy")
    parser.add_argument("--output", type=Path, default=Path("/out"))
    parser.add_argument(
        "--one-step", action="store_true", help="train exactly one epoch and export it"
    )
    args, extra = parser.parse_known_args()
    if extra[:1] == ["--"]:
        extra = extra[1:]

    print(
        json.dumps(
            train(
                profile=args.profile,
                output=args.output.resolve(),
                one_step=args.one_step,
                extra=extra,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
