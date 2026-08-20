"""Turn RAGEN's sharded FSDP checkpoints into weights something can load.

run.sh calls this at the end of training. It is not a convenience: RAGEN saves
sharded FSDP state under `global_step_N/actor/`, and nothing -- not fast_eval, not
the hidden final, not vLLM -- can load that. `verl.model_merger` produces the HF
directory, and only merged weights score. A run that trains for 12 h and skips this
step hands back nothing.

Three things it does beyond the merge:

1. **Tops up the tokenizer and token ids** from the frozen policy. The merger writes
   weights and a config; the tokenizer files come from the source model. Copied
   rather than assumed present, and `pad_token_id` is filled from `eos_token_id`
   when the source leaves it null, which vLLM needs.
2. **Parses the training dynamics** out of the log into metrics.jsonl. The reference protocol
   listed six required dynamics in `[reporting]` -- success, pass_at_k, reward,
   episode_length, rollout_count, gpu_memory_peak_bytes -- and a run with no
   parseable dynamics was treated as a failed run. That check is kept because it
   catches a real thing: a trainer that starts, logs nothing and exits 0.
3. **Writes a short provenance record** beside each checkpoint. Nothing downstream
   gates on its fields, and that is deliberate: the old evaluator asserted 13 of
   them, all written by the candidate. The scoring path now computes what it needs
   from the artifact and the image instead -- see harness/final_eval.py.

None of this is contract. Replace it if it is in your way; what the score phase needs
is a directory of merged HF weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

TASK_ID = "ragen_sokoban_grpo"
ALGORITHM_FAMILY = "multi_turn_on_policy_environment_rl"
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
METRIC_RE = re.compile(r"([A-Za-z0-9_./-]+):\s*(-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)")
STEP_RE = re.compile(r"\bstep:(\d+)\b")

def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def weight_sha256(checkpoint: Path) -> str:
    files = sorted(
        {path for pattern in WEIGHT_PATTERNS for path in checkpoint.rglob(pattern)}
    )
    if not files:
        raise RuntimeError(f"checkpoint has no model weights: {checkpoint}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(checkpoint).as_posix().encode())
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def checkpoint_loadability_error(checkpoint: Path) -> str | None:
    """Return why a merged export cannot be loaded as a local HF checkpoint."""

    try:
        config_path = checkpoint / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or not config.get("model_type"):
            raise ValueError("config.json has no model_type")

        from transformers import AutoConfig, AutoTokenizer

        AutoConfig.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
        AutoTokenizer.from_pretrained(
            checkpoint, local_files_only=True, trust_remote_code=False, use_fast=True
        )
        files = sorted(
            {path for pattern in WEIGHT_PATTERNS for path in checkpoint.glob(pattern)}
        )
        if not files:
            raise ValueError("no direct model weight files")

        for index_path in checkpoint.glob("*.index.json"):
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index.get("weight_map") if isinstance(index, dict) else None
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(f"{index_path.name} has no weight_map")
            missing = sorted(
                name for name in set(weight_map.values()) if not (checkpoint / name).is_file()
            )
            if missing:
                raise ValueError(f"{index_path.name} references missing shards {missing[:3]}")

        safetensors = [path for path in files if path.suffix == ".safetensors"]
        if safetensors:
            from safetensors import safe_open

            for path in safetensors:
                with safe_open(path, framework="pt", device="cpu") as handle:
                    if not list(handle.keys()):
                        raise ValueError(f"{path.name} contains no tensors")
        for path in (path for path in files if path.suffix == ".bin"):
            import torch

            state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
            if not isinstance(state, dict) or not state:
                raise ValueError(f"{path.name} contains no state dict")
        return None
    except Exception as error:  # noqa: BLE001
        return f"{type(error).__name__}: {error}"


def saved_steps(work: Path, keep: int) -> list[tuple[int, Path]]:
    """The highest `keep` saved steps, sorted by step number rather than by name.

    Lexicographically global_step_80 sorts after global_step_100, so a name sort
    picks an early checkpoint as the latest. Harmless at 100 updates; wrong as soon
    as somebody raises the count past 100, which the budget arithmetic in run.sh
    invites.
    """

    root = work / "verl_checkpoints"
    found: list[tuple[int, Path]] = []
    for path in root.glob("global_step_*"):
        suffix = path.name.removeprefix("global_step_")
        if suffix.isdigit() and (path / "actor").is_dir():
            found.append((int(suffix), path / "actor"))
    if not found:
        raise RuntimeError(
            f"no global_step_*/actor under {root}. Training saved nothing, so there "
            "is nothing to merge and nothing to score."
        )
    return sorted(found)[-keep:] if keep > 0 else sorted(found)


def merge(actor: Path, target: Path, ragen: Path) -> None:
    """Sharded FSDP state -> a loadable HF directory."""

    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "verl.model_merger",
            "merge",
            "--backend",
            "fsdp",
            "--local_dir",
            str(actor),
            "--target_dir",
            str(target),
        ],
        cwd=str(ragen),
        check=True,
    )
    if not any(any(target.glob(pattern)) for pattern in WEIGHT_PATTERNS):
        raise RuntimeError(f"the merge produced no weights in {target}")


def copy_model_metadata(policy: Path, checkpoint: Path) -> None:
    """Top up the merged checkpoint's tokenizer and token ids from the frozen policy."""

    if not policy.is_dir():
        raise FileNotFoundError(f"policy directory is missing: {policy}")
    for name in TOKENIZER_FILES:
        source = policy / name
        if source.is_file():
            shutil.copy2(source, checkpoint / name)

    source_config = json.loads((policy / "config.json").read_text(encoding="utf-8"))
    target_path = checkpoint / "config.json"
    target_config = (
        json.loads(target_path.read_text(encoding="utf-8"))
        if target_path.is_file()
        else dict(source_config)
    )
    generation_path = policy / "generation_config.json"
    generation = (
        json.loads(generation_path.read_text(encoding="utf-8"))
        if generation_path.is_file()
        else {}
    )
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = source_config.get(key, generation.get(key))
        if value is None and key == "pad_token_id":
            value = source_config.get("eos_token_id", generation.get("eos_token_id"))
        if value is not None:
            target_config[key] = value
            generation.setdefault(key, value)
    if generation.get("eos_token_id") is None:
        raise ValueError("cannot determine eos_token_id for the merged checkpoint")
    atomic_json(target_path, target_config)
    atomic_json(checkpoint / "generation_config.json", generation)


def parse_dynamics(log_path: Path) -> list[dict[str, Any]]:
    """Pull `step:N ... key: value` lines out of the training log."""

    if not log_path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        step = STEP_RE.search(line)
        metrics = {key: float(value) for key, value in METRIC_RE.findall(line)}
        if step is None or not metrics:
            continue
        records.append({"step": int(step.group(1)), "metrics": metrics})
    return records


def write_metadata(checkpoint: Path, *, step: int, dynamics_records: int) -> dict[str, Any]:
    """Provenance beside the weights. Nothing gates on it -- see the module docstring."""

    metadata = {
        "schema_version": 2,
        "task_id": TASK_ID,
        "algorithm_family": ALGORITHM_FAMILY,
        "completed_step": step,
        "training_records": dynamics_records,
        "weight_sha256": weight_sha256(checkpoint),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    atomic_json(checkpoint / "training_metadata.json", metadata)
    return metadata


def finalize(
    *,
    work: Path,
    checkpoints: Path,
    policy: Path,
    ragen: Path,
    keep: int,
    train_log: Path,
    train_status: int,
) -> dict[str, Any]:
    dynamics = parse_dynamics(train_log)
    if not dynamics:
        raise RuntimeError(
            f"no parseable training dynamics in {train_log}. A trainer that starts, "
            "logs nothing and exits 0 produces a checkpoint that cannot be reasoned "
            "about, so this is treated as a failed run."
        )
    exported: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    # Scan every save from highest to lowest and stop after `keep` successful
    # exports. A truncated highest actor save must not hide a complete lower one.
    for step, actor in reversed(saved_steps(work, 0)):
        if keep > 0 and len(exported) >= keep:
            break
        target = checkpoints / f"checkpoint-{step}"
        incomplete = checkpoints / f".checkpoint-{step}.incomplete"
        try:
            if target.exists():
                error = checkpoint_loadability_error(target)
                if error is not None:
                    raise RuntimeError(f"existing export is not loadable: {error}")
            else:
                if incomplete.exists():
                    raise RuntimeError(
                        f"an earlier incomplete export is preserved at {incomplete}; "
                        "use a fresh OUTPUT_DIR rather than mixing attempts"
                    )
                merge(actor, incomplete, ragen)
                copy_model_metadata(policy, incomplete)
                error = checkpoint_loadability_error(incomplete)
                if error is not None:
                    raise RuntimeError(f"merged export is not loadable: {error}")
                write_metadata(incomplete, step=step, dynamics_records=len(dynamics))
                incomplete.replace(target)
            metadata = write_metadata(target, step=step, dynamics_records=len(dynamics))
        except Exception as error:  # noqa: BLE001
            failures.append(
                {
                    "step": step,
                    "actor": str(actor),
                    "incomplete": str(incomplete) if incomplete.exists() else None,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        exported.append(
            {
                "step": step,
                "checkpoint": str(target),
                "weight_sha256": metadata["weight_sha256"],
            }
        )
    checkpoints.mkdir(parents=True, exist_ok=True)
    (checkpoints / "metrics.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in dynamics),
        encoding="utf-8",
    )
    summary = {
        "schema_version": 2,
        "task_id": TASK_ID,
        "algorithm_family": ALGORITHM_FAMILY,
        "completed_steps": dynamics[-1]["step"],
        "training_records": len(dynamics),
        "checkpoints": exported,
        "failed_exports": failures,
        # 124 is the wall-clock stop, which is an expected end to a run that filled
        # its budget rather than a failure. Recorded so a reader can tell the two
        # apart without reading the log.
        "train_status": train_status,
        "stopped_at_wall_clock": train_status == 124,
        "offline": True,
    }
    atomic_json(checkpoints / "summary.json", summary)
    if failures:
        atomic_json(checkpoints / "finalize_failures.json", failures)
    if not exported:
        raise RuntimeError(
            "no loadable merged checkpoint was produced; see "
            f"{checkpoints / 'finalize_failures.json'}"
        )
    return summary


def smoke() -> None:
    """Check the parts that do not need a GPU, a checkpoint or the RAGEN tree."""

    log = Path("/tmp/ragen-finalize-smoke.log")
    log.write_text(
        "step:1 critic/score/mean: 0.125 actor/entropy: 0.75\n"
        "not a metric line\n"
        "step:20 critic/score/mean: 0.1875 actor/grad_norm: 0.5\n",
        encoding="utf-8",
    )
    records = parse_dynamics(log)
    assert [record["step"] for record in records] == [1, 20], records
    assert records[-1]["metrics"]["critic/score/mean"] == 0.1875, records

    # Step order, not name order: 100 must come after 80.
    root = Path("/tmp/ragen-finalize-smoke/verl_checkpoints")
    for step in (60, 80, 100):
        (root / f"global_step_{step}/actor").mkdir(parents=True, exist_ok=True)
    steps = [step for step, _ in saved_steps(root.parent, 3)]
    assert steps == [60, 80, 100], steps
    assert [step for step, _ in saved_steps(root.parent, 1)] == [100]
    print(json.dumps({"finalize_smoke": "passed", "steps": steps}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("/out/work"))
    parser.add_argument("--checkpoints", type=Path, default=Path("/out/checkpoints"))
    parser.add_argument("--policy", type=Path, default=Path("/assets/models/policy"))
    parser.add_argument("--ragen", type=Path, default=Path("/workspace/ragen"))
    parser.add_argument("--keep", type=int, default=3)
    parser.add_argument("--train-log", type=Path, default=None)
    parser.add_argument("--train-status", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        smoke()
        return
    train_log = args.train_log or (args.work / "train.log")
    print(
        json.dumps(
            finalize(
                work=args.work.resolve(),
                checkpoints=args.checkpoints.resolve(),
                policy=args.policy.resolve(),
                ragen=args.ragen.resolve(),
                keep=args.keep,
                train_log=train_log.resolve(),
                train_status=args.train_status,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
