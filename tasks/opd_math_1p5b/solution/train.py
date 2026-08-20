"""Driver around run.sh: pick a schedule, launch, then make the checkpoint loadable.

run.sh holds every Hydra override. This file adds three conveniences:

1. **Schedule presets.** `--profile proxy|retrain` and `--one-step` map to step
   counts and wall-clock limits, so the same tree serves a 40-step probe and a
   12 h run.
2. **Tolerating exit 134.** VERL/Ray reliably aborts on a corrupt mutex *after*
   writing a complete checkpoint. Treating that as a failure would throw away
   every finished run, so it is accepted -- but only once the checkpoint is
   verified present and indexed.
3. **Topping up the exported checkpoint.** VERL writes bos/eos/pad ids and the
   tokenizer itself, so this is defensive rather than required: a 40-step run
   through `bash run.sh` alone produced a checkpoint that fast_eval loaded and
   scored with no help from here. What it still adds is a `pad_token_id` in
   generation_config, which VERL leaves null, and merges.txt / vocab.json when a
   future model needs them.

None of this is contract. `bash run.sh` on its own trains and produces a
scorable checkpoint; delete or replace this file if it is in your way.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

TASK_ID = "opd_math_1p5b"
ALGORITHM_FAMILY = "sampled_token_on_policy_distillation"
WORKSPACE = Path(__file__).resolve().parent
RUN_SH = WORKSPACE / "run.sh"
WEIGHT_PATTERNS = ("*.safetensors", "pytorch_model*.bin")
TOKENIZER_FILES = (
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
# VERL/Ray aborts with SIGABRT on teardown after a successful save. Observed
# every run; accepted only when the checkpoint is already complete.
POST_CHECKPOINT_ABORT_RETURNCODE = 134

# steps, wall-clock budget for run.sh, reserve before the wall.
#
# `steps = None` means "leave run.sh's own default alone", which is what the
# retrain phase does. Writing a number here would make it the third place the same
# quantity lives -- run.sh's default, this table, and the budget arithmetic -- and
# the previous version of this table said 1105 against run.sh's 1200. When two
# copies of one number disagree, the wrong one is the one in force.
PROFILES = {
    "proxy": (40, 3600, 300),
    "retrain": (None, 43200, 1200),
}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def weight_sha256(checkpoint: Path) -> str:
    files = sorted({path for pattern in WEIGHT_PATTERNS for path in checkpoint.glob(pattern)})
    if not files:
        raise RuntimeError(f"checkpoint has no model weights: {checkpoint}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def latest_checkpoint(root: Path) -> tuple[int, Path]:
    """Return the highest global_step_N under root, and its exported HF directory."""

    candidates: list[tuple[int, Path]] = []
    for path in root.glob("global_step_*"):
        suffix = path.name.removeprefix("global_step_")
        if suffix.isdigit():
            candidates.append((int(suffix), path))
    if not candidates:
        raise FileNotFoundError(f"no global_step_* checkpoint under {root}")
    step, directory = max(candidates, key=lambda item: item[0])
    for nested in ("actor/huggingface", "huggingface"):
        exported = directory / nested
        if exported.is_dir() and any(
            exported.glob(pattern) for pattern in WEIGHT_PATTERNS
        ):
            return step, exported
    raise FileNotFoundError(f"checkpoint {directory} has no exported HF weights")


def verify_complete(checkpoint: Path) -> None:
    """Confirm an exported checkpoint is whole before trusting an aborted run."""

    config = checkpoint / "config.json"
    if not config.is_file() or config.stat().st_size == 0:
        raise ValueError(f"checkpoint has no usable config.json: {checkpoint}")
    index = checkpoint / "model.safetensors.index.json"
    if index.is_file():
        payload = json.loads(index.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"checkpoint index has no weight_map: {checkpoint}")
        expected = set(weight_map.values())
        actual = {path.name for path in checkpoint.glob("*.safetensors")}
        if expected != actual:
            missing = sorted(expected - actual)
            raise ValueError(f"checkpoint is missing shards {missing}: {checkpoint}")
    elif not any(checkpoint.glob(pattern) for pattern in WEIGHT_PATTERNS):
        raise ValueError(f"checkpoint has no weights: {checkpoint}")


def copy_model_metadata(student: Path, checkpoint: Path) -> None:
    """Top up the exported checkpoint's tokenizer and token ids.

    Measured on a 40-step run: VERL's own export already carries bos/eos/pad in
    config.json, bos/eos in generation_config.json, and tokenizer.json plus
    tokenizer_config.json -- enough for vLLM to load it. So this is belt and
    braces, not a fix. It fills the null `pad_token_id` in generation_config and
    copies merges.txt / vocab.json when a model needs them separately.
    """

    if not student.is_dir():
        raise FileNotFoundError(f"student model directory is missing: {student}")
    for name in TOKENIZER_FILES:
        source = student / name
        if source.is_file():
            shutil.copy2(source, checkpoint / name)

    source_config = json.loads((student / "config.json").read_text(encoding="utf-8"))
    target_path = checkpoint / "config.json"
    target_config = json.loads(target_path.read_text(encoding="utf-8"))
    generation_path = student / "generation_config.json"
    generation = (
        json.loads(generation_path.read_text(encoding="utf-8"))
        if generation_path.is_file()
        else {}
    )

    for key in ("max_position_embeddings", "rope_theta", "rope_scaling"):
        if key in source_config:
            target_config[key] = source_config[key]
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = source_config.get(key, generation.get(key))
        if value is None and key == "pad_token_id":
            value = source_config.get("eos_token_id", generation.get("eos_token_id"))
        if value is not None:
            target_config[key] = value
            generation.setdefault(key, value)
    if generation.get("eos_token_id") is None:
        raise ValueError("cannot determine eos_token_id for the exported checkpoint")

    atomic_json(target_path, target_config)
    atomic_json(checkpoint / "generation_config.json", generation)


def write_checkpoint_metadata(
    checkpoint: Path, *, profile: str, step: int, environment: dict[str, str]
) -> dict[str, Any]:
    """A short provenance record. Nothing downstream gates on its fields."""

    metadata = {
        "schema_version": 2,
        "task_id": TASK_ID,
        "algorithm_family": ALGORITHM_FAMILY,
        "profile": profile,
        "completed_step": step,
        "loss_mode": environment.get("DISTILLATION_LOSS_MODE", "k1"),
        "total_training_steps": environment.get("TOTAL_TRAINING_STEPS"),
        "actor_lr": environment.get("ACTOR_LR"),
        "train_batch_size": environment.get("TRAIN_BATCH_SIZE"),
        "rollout_n": environment.get("ROLLOUT_N"),
        "weight_sha256": weight_sha256(checkpoint),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    atomic_json(checkpoint / "training_metadata.json", metadata)
    return metadata


def train(
    *,
    profile: str,
    output: Path,
    student: Path,
    one_step: bool,
    extra: list[str],
) -> dict[str, Any]:
    steps, wall_seconds, reserve = PROFILES[profile]
    if one_step:
        steps, wall_seconds, reserve = 1, 3600, 60

    output.mkdir(parents=True, exist_ok=True)
    checkpoint_root = output / "checkpoints"
    environment = os.environ.copy()
    environment.update(
        {
            "OUTPUT_DIR": str(output),
            "CKPT_DIR": str(checkpoint_root),
            # Save at the end for a probe; periodically for a long run, so a crash
            # late in 12 h does not cost everything.
            "SAVE_FREQ": str(steps if steps and (profile == "proxy" or one_step) else 20),
            "MAX_WALL_TIME_SECONDS": str(wall_seconds),
            "DEADLINE_RESERVE_SECONDS": str(reserve),
            "EXPERIMENT_NAME": f"{TASK_ID}-{profile}-{int(time.time())}",
        }
    )
    # Omitted rather than set to "None", so run.sh's own default applies. Setting
    # it would silently override whatever the candidate wrote there.
    if steps is not None:
        environment["TOTAL_TRAINING_STEPS"] = str(steps)

    started = time.monotonic()
    with (output / "train.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(
            ["bash", str(RUN_SH), *extra],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.monotonic() - started

    step, checkpoint = latest_checkpoint(checkpoint_root)
    if process.returncode == POST_CHECKPOINT_ABORT_RETURNCODE:
        # Accept the known teardown abort, but only against a whole checkpoint.
        verify_complete(checkpoint)
        print(
            f"train: exit {POST_CHECKPOINT_ABORT_RETURNCODE} after a complete save at step {step}"
            " -- accepting the known Ray teardown abort",
            file=sys.stderr,
        )
    elif process.returncode != 0:
        raise RuntimeError(
            f"training failed with exit {process.returncode}; see {output / 'train.log'}"
        )
    else:
        verify_complete(checkpoint)

    copy_model_metadata(student, checkpoint)
    metadata = write_checkpoint_metadata(
        checkpoint, profile=profile, step=step, environment=environment
    )
    summary = {
        "schema_version": 2,
        "task_id": TASK_ID,
        "profile": profile,
        "completed_step": step,
        "checkpoint": str(checkpoint),
        "wall_seconds": elapsed,
        "seconds_per_step": elapsed / step if step else None,
        "returncode": process.returncode,
        "weight_sha256": metadata["weight_sha256"],
    }
    atomic_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="proxy")
    parser.add_argument("--output", type=Path, default=Path("/out"))
    parser.add_argument("--student", type=Path, default=Path("/assets/models/student"))
    parser.add_argument(
        "--one-step", action="store_true", help="train exactly one step and checkpoint it"
    )
    parser.add_argument(
        "extra", nargs="*", help="Hydra overrides passed straight through to run.sh"
    )
    args = parser.parse_args()

    print(
        json.dumps(
            train(
                profile=args.profile,
                output=args.output.resolve(),
                student=args.student,
                one_step=args.one_step,
                extra=list(args.extra),
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
