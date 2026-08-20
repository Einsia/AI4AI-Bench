"""Score a merged checkpoint on 512 held-out Sokoban boards.

The host supplies the hidden environment seed, rollouts execute from the frozen RAGEN tree,
and artifact properties are computed rather than read from candidate metadata. The generator
creates 640 environments and scores the first 512 distinct boards, one trajectory each.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fast_eval import atomic_json, resolve_checkpoint  # noqa: E402
from grade import (  # noqa: E402
    ENGINE_SEED,
    EVAL_TEMPERATURE,
    RAGEN_COMMIT,
    VERL_COMMIT,
    frozen_environment_report,
    rollout,
    rollout_environment,
    score_bank,
    summarize,
    synthetic_bank,
)

TASK_ID = "ragen_sokoban_grpo"
METRIC = "held_out_512_board_solve_rate"
GENERATED_ENVIRONMENTS = 640
SCORED_BOARDS = 512
REWARD_PATH = Path("/logs/verifier/reward.txt")

# Keys that make a checkpoint directory executable. `auto_map` names modules inside
# the checkpoint that transformers imports when trust_remote_code is on, and the
# other two are the flags that turn it on from inside the artifact.
EXECUTABLE_CONFIG_KEYS = ("auto_map", "custom_code", "trust_remote_code", "auto_class")
# Suffixes that are code rather than weights or metadata.
CODE_SUFFIXES = frozenset({".py", ".pyc", ".pyo", ".so", ".pyd", ".sh", ".pth"})
# What a checkpoint legitimately contains. Anything else is reported, not refused --
# the refusal list above is the narrow one.
EXPECTED_CHECKPOINT_SUFFIXES = frozenset(
    {".safetensors", ".bin", ".json", ".txt", ".model", ".jinja", ".md", ""}
)


def check_checkpoint_carries_no_code(checkpoint: Path, model: Path | None = None) -> dict[str, Any]:
    """Refuse a checkpoint that can execute code inside the scoring process.

    This is the artifact-side half of "environment transitions and solve reward
    remain frozen", and it is the half no mount can hold.

    The scoring container runs the image's own harness with the candidate patch
    deliberately not applied, so the candidate has no source in this process -- with
    one exception. The checkpoint is mounted, and a checkpoint is not inert:
    transformers will import a module named in `config.json`'s `auto_map` when
    trust_remote_code is enabled, and that import runs arbitrary code in the same
    interpreter that is about to roll trajectories out. From there,
    `ragen.env...SokobanEnv.step` is an attribute like any other, and every board
    reports solved.

    So this refuses the capability rather than trying to detect its effect: no
    executable file anywhere under the checkpoint, and no config key that would load
    one. Computed from the artifact's own bytes and filenames; nothing here reads
    metadata the candidate wrote about itself.

    A violating artifact, to test it: take a real retrain checkpoint, drop in

        # modeling_patched.py
        import ragen.env as env
        env.CoordSokoban.step = lambda self, *a, **k: (self.render(), 1.0, True, {})
        from transformers import Qwen2ForCausalLM as PatchedQwen

    and add `"auto_map": {"AutoModelForCausalLM": "modeling_patched.PatchedQwen"}` to
    its config.json. Without this check the rollout loads the policy, the import runs,
    and the run reports a solve rate near 1.0 with a legitimate-looking weight hash.
    With it, the trial is invalidated before the first board is generated.
    """

    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint is not a directory: {checkpoint}")

    # Scan from the mount root rather than from the resolved weights directory.
    # resolve_checkpoint descends into `checkpoint-N/` or `actor/huggingface/`, so
    # auditing only what it returned would leave a `.py` one level up unexamined --
    # and one level up is still inside the mount, still on the same filesystem, and
    # still importable by anything that adds it to sys.path.
    code_files: list[str] = []
    unexpected: list[str] = []
    symlinks: list[str] = []
    for path in sorted(checkpoint.rglob("*")):
        if path.is_symlink():
            symlinks.append(path.relative_to(checkpoint).as_posix())
            continue
        if not path.is_file():
            continue
        relative = path.relative_to(checkpoint).as_posix()
        suffix = path.suffix.lower()
        if suffix in CODE_SUFFIXES:
            code_files.append(relative)
        elif suffix not in EXPECTED_CHECKPOINT_SUFFIXES:
            unexpected.append(relative)

    # Config keys are read wherever a config lives, because transformers resolves an
    # auto_map relative to the directory holding the config it loaded.
    executable_keys: dict[str, Any] = {}
    roots = [checkpoint] if model is None else [checkpoint, model]
    for root in roots:
        for name in ("config.json", "generation_config.json", "tokenizer_config.json"):
            path = root / name
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path} is not valid JSON: {error}") from error
            if not isinstance(payload, dict):
                continue
            for key in EXECUTABLE_CONFIG_KEYS:
                if key in payload:
                    executable_keys[f"{path.name}:{key}"] = payload[key]

    if code_files or executable_keys or symlinks:
        raise ValueError(
            "the checkpoint carries executable code, so it could redefine the "
            "environment or the reward inside the scoring process:\n"
            f"  code files       {code_files or 'none'}\n"
            f"  config keys      {sorted(executable_keys) or 'none'}\n"
            f"  symlinks         {symlinks or 'none'}\n"
            "A checkpoint is weights, a tokenizer and config. The environment "
            "transitions and the solve reward are the image's, and this trial is "
            "invalid."
        )
    return {
        "code_files": 0,
        "executable_config_keys": 0,
        "symlinks": 0,
        "unexpected_suffixes": unexpected,
    }


def weight_sha256(checkpoint: Path) -> str:
    """Hash the weight files, in sorted relative-path order.

    Recorded so a score can be tied to the exact bytes that produced it. Not a check:
    there is nothing to compare it against, because the retrain phase's checkpoint is
    whatever training produced.
    """

    import hashlib

    files = sorted(
        {
            path
            for pattern in ("*.safetensors", "pytorch_model*.bin")
            for path in checkpoint.rglob(pattern)
        }
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


def write_reward(score: float, reward_path: Path) -> None:
    """Harbor's verifier contract: the scalar lands in /logs/verifier."""

    reward_path.parent.mkdir(parents=True, exist_ok=True)
    reward_path.write_text(f"{score:.10f}\n", encoding="utf-8")


def evaluate(
    checkpoint: Path,
    output: Path,
    environment_seed: int,
    engine_seed: int,
    generated: int,
    scored: int,
    reward_path: Path,
) -> dict[str, Any]:
    model = resolve_checkpoint(checkpoint)
    output.mkdir(parents=True, exist_ok=True)

    # Both artifact-side checks run before a device is claimed, so a violation costs
    # nothing and the message is about the artifact rather than about a crash 3 h in.
    artifact = check_checkpoint_carries_no_code(checkpoint, model)
    weights_hash = weight_sha256(model)
    environment = rollout_environment(output)
    frozen = frozen_environment_report(environment)

    atomic_json(
        output / "resolved_config.json",
        {
            "schema_version": 1,
            "task_id": TASK_ID,
            "checkpoint": str(checkpoint),
            "model": str(model),
            "checkpoint_weight_sha256": weights_hash,
            "checkpoint_code_audit": artifact,
            "frozen_environment": frozen,
            "environment_seed": environment_seed,
            "engine_seed": engine_seed,
            "request_seed_scheme": "engine_seed + env_id",
            "generated_environments": generated,
            "scored_unique_boards": scored,
            "trajectories_per_board": 1,
            "temperature": EVAL_TEMPERATURE,
            "ragen_commit": RAGEN_COMMIT,
            "verl_commit": VERL_COMMIT,
            "input_contract": "checkpoint_only",
            "image_digest": os.environ.get("IMAGE_DIGEST"),
        },
    )

    started = time.monotonic()
    rows, reported, elapsed = rollout(
        checkpoint=model,
        boards=generated,
        environment_seed=environment_seed,
        engine_seed=engine_seed,
        output=output / "rollout",
        environment=environment,
    )
    bank = score_bank(
        rows,
        expected_rows=generated,
        scored_boards=scored,
        environment_seed=environment_seed,
        engine_seed=engine_seed,
        # Deliberately not cross-checked: RAGEN's own aggregate is over all 640
        # generated trajectories and the score is over the 512 distinct prefix, so
        # the two are means of different sets. Recorded instead.
        reported=None,
        wall_seconds=elapsed,
    )
    aggregate = summarize([bank])
    summary = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "passed",
        "benchmark": "RAGEN unseen Sokoban",
        "metric": METRIC,
        "direction": "maximize",
        "metrics": {METRIC: aggregate["score"]},
        **aggregate,
        "generated_environments": bank["generated"],
        "environment_reported_mean_over_generated": reported,
        "checkpoint_weight_sha256": weights_hash,
        "checkpoint_code_audit": artifact,
        "frozen_environment": frozen,
        "input_contract": "checkpoint_only",
        "wall_seconds": time.monotonic() - started,
        "offline": True,
    }
    atomic_json(output / "summary.json", summary)
    (output / "trajectories.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    write_reward(aggregate["score"], reward_path)
    return summary


def mock(output: Path, reward_path: Path, environment_seed: int) -> dict[str, Any]:
    """Synthetic rows through the real aggregation, for a host with no GPU.

    The frozen-environment and checkpoint-code checks are skipped, because neither
    the tree nor a checkpoint exists on such a host. Exercises the output shape a
    consumer parses and the over-generation prefix, which is the part with an
    off-by-one in it.
    """

    output.mkdir(parents=True, exist_ok=True)
    rows = synthetic_bank(
        environment_seed=environment_seed, boards=SCORED_BOARDS, generated=GENERATED_ENVIRONMENTS
    )
    bank = score_bank(
        rows,
        expected_rows=GENERATED_ENVIRONMENTS,
        scored_boards=SCORED_BOARDS,
        environment_seed=environment_seed,
        engine_seed=ENGINE_SEED,
    )
    aggregate = summarize([bank])
    summary = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "passed",
        "metric": METRIC,
        "direction": "maximize",
        "metrics": {METRIC: aggregate["score"]},
        **aggregate,
        "generated_environments": bank["generated"],
        "input_contract": "checkpoint_only",
        "mock": True,
    }
    atomic_json(output / "summary.json", summary)
    write_reward(aggregate["score"], reward_path)
    return summary


def smoke() -> None:
    rows = synthetic_bank(
        environment_seed=7, boards=SCORED_BOARDS, generated=GENERATED_ENVIRONMENTS
    )
    bank = score_bank(
        rows,
        expected_rows=GENERATED_ENVIRONMENTS,
        scored_boards=SCORED_BOARDS,
        environment_seed=7,
        engine_seed=ENGINE_SEED,
    )
    aggregate = summarize([bank])
    if aggregate["boards"] != SCORED_BOARDS:
        raise RuntimeError(f"expected {SCORED_BOARDS} scored boards, got {aggregate['boards']}")
    if bank["generated"] != GENERATED_ENVIRONMENTS:
        raise RuntimeError(f"expected {GENERATED_ENVIRONMENTS} generated rows")
    # 512 independent draws: the error bar the baselines have to be read against.
    if not 0.0 < aggregate["stderr"] < 0.05:
        raise RuntimeError(f"implausible stderr at 512 boards: {aggregate['stderr']}")
    print(
        json.dumps(
            {"final_eval_smoke": "passed", "boards": aggregate["boards"]},
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--environment-seed",
        type=int,
        default=None,
        help="the seed the 640 boards are generated from. Required for a real run "
        "and deliberately without a default: it is supplied by the score phase in "
        "declaration.py so that it is absent from the image the Agent explores in.",
    )
    parser.add_argument("--engine-seed", type=int, default=ENGINE_SEED)
    parser.add_argument("--generated", type=int, default=GENERATED_ENVIRONMENTS)
    parser.add_argument("--scored", type=int, default=SCORED_BOARDS)
    parser.add_argument("--reward-path", type=Path, default=REWARD_PATH)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        smoke()
        return
    if args.output is None:
        parser.error("--output is required")
    if args.mock:
        seed = args.environment_seed if args.environment_seed is not None else 0
        print(
            json.dumps(
                mock(args.output.resolve(), args.reward_path, seed),
                sort_keys=True,
            )
        )
        return
    if args.checkpoint is None:
        parser.error("--checkpoint is required outside --mock/--smoke")
    if args.environment_seed is None:
        parser.error(
            "--environment-seed is required. The final's boards are generated from "
            "it, and it has no default here on purpose -- see declaration.py."
        )
    if args.scored > args.generated:
        parser.error("--scored cannot exceed --generated")
    print(
        json.dumps(
            evaluate(
                args.checkpoint.resolve(),
                args.output.resolve(),
                args.environment_seed,
                args.engine_seed,
                args.generated,
                args.scored,
                args.reward_path,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"RAGEN final failed: {exc}", file=sys.stderr)
        raise
