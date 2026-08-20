"""256 Sokoban boards on one GPU, for use inside the 4 h phase.

Same environment, same reward, same rollout code and same sampling as the hidden
final -- only the boards differ, and they differ because they have to: see the
overlap note in task.toml. It is not a surrogate metric. It plays the game.

**Four banks, because one was not enough.** The reference protocol's first proxy was a
single 64-board bank at seed 4242, and the comment in its evaluator records the
result: "essentially zero rank correlation" with the 1,024-board selector it was
screening for. Four disjoint banks at seeds 4242-4245 were the smallest screen whose
ordering held, reaching Spearman 0.8144 across eight legal recipes. That calibration
was measured against a selector tier that v1 does not have, so it is history rather
than a current number -- but the failure it fixed is real and the fix is kept.

**More boards, not more trajectories per board.** One trajectory per board, and the
reason is the same one OPD measured on samples per question: a per-board outcome is
close to deterministic, so a second trajectory on a solved board re-measures
something already known while a new board adds information. Boards are also free --
they are generated, not stored.

**Read the stderr, and know what it is not.** Each board is one Bernoulli draw, so
the error on the mean is sqrt(p(1-p)/256), about 0.021 at the baseline's rate. The
old evaluator instead reported a spread across repeats of the *same* seed -- 0.0074,
about 6x too small, and it read like an error bar. Nothing here repeats a seed.

**Cost is not recorded.** No proxy evaluation was ever timed on the reference protocol, on
any device. Its two declared ceilings disagree with each other (1800 s in
proxy.toml, 3600 s in task.toml's [proxy]). Time the first call you make and size
your loop from that, not from either number.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from grade import (  # noqa: E402
    ENGINE_SEED,
    EVAL_TEMPERATURE,
    RAGEN_COMMIT,
    frozen_environment_report,
    rollout,
    rollout_environment,
    score_bank,
    summarize,
    synthetic_bank,
)

TASK_ID = "ragen_sokoban_grpo"
METRIC = "public_four_bank_solve_rate"
# The four proxy banks. Disjoint, public, and deliberately in the image: these are
# the boards the Agent is meant to tune against.
PROXY_ENVIRONMENT_SEEDS = (4242, 4243, 4244, 4245)
BOARDS_PER_BANK = 64

WEIGHT_PATTERNS = ("*.safetensors", "pytorch_model*.bin")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def has_weights(directory: Path) -> bool:
    return any(any(directory.glob(pattern)) for pattern in WEIGHT_PATTERNS)


def checkpoint_weight_sha256(checkpoint: Path) -> str:
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


def checkpoint_loadability_error(directory: Path) -> str | None:
    """Return why a merged HF checkpoint is not structurally loadable."""

    try:
        config_path = directory / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or not config.get("model_type"):
            raise ValueError("config.json has no model_type")

        from transformers import AutoConfig

        AutoConfig.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
        files = sorted(
            {path for pattern in WEIGHT_PATTERNS for path in directory.glob(pattern)}
        )
        if not files:
            raise ValueError("no direct model weight files")

        for index_path in directory.glob("*.index.json"):
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index.get("weight_map") if isinstance(index, dict) else None
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(f"{index_path.name} has no weight_map")
            missing = sorted(
                name for name in set(weight_map.values()) if not (directory / name).is_file()
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


def resolve_checkpoint(checkpoint: Path, loadability_check=checkpoint_loadability_error) -> Path:
    """Find the merged HF weights, given any of the paths someone would plausibly type.

    RAGEN writes sharded FSDP state under `global_step_N/actor/` and run.sh merges it
    into `/out/checkpoints/checkpoint-N`. All three are plausible arguments, and the
    checkpoints root takes the highest step that passes a real config/index/weight
    loadability check. A truncated highest save must not hide a complete lower one.
    """

    candidates: list[Path] = []

    def add(directory: Path) -> None:
        if directory.is_dir() and directory not in candidates:
            candidates.append(directory)

    add(checkpoint)
    for nested in ("huggingface", "hf_model", "actor/huggingface"):
        add(checkpoint / nested)

    steps: list[tuple[int, Path]] = []
    for pattern, prefix in (("checkpoint-*", "checkpoint-"), ("global_step_*", "global_step_")):
        for path in checkpoint.glob(pattern):
            suffix = path.name.removeprefix(prefix)
            if suffix.isdigit():
                steps.append((int(suffix), path))
    for _, directory in sorted(steps, reverse=True):
        add(directory)
        for nested in ("huggingface", "hf_model", "actor/huggingface"):
            add(directory / nested)

    rejected: list[str] = []
    for directory in candidates:
        if not has_weights(directory):
            continue
        error = loadability_check(directory)
        if error is None:
            return directory
        rejected.append(f"{directory}: {error}")

    raise FileNotFoundError(
        f"no loadable model weights under {checkpoint}. Looked for *.safetensors here, in "
        "huggingface/ hf_model/ actor/huggingface/, and in the highest checkpoint-N "
        "or global_step_N below. An unmerged FSDP checkpoint has no loadable weights "
        "-- run.sh merges them at the end of training. Rejected candidates: "
        + ("; ".join(rejected) if rejected else "none had direct weights")
    )


def evaluate(
    checkpoint: Path,
    out: Path,
    seeds: tuple[int, ...],
    boards: int,
    engine_seed: int,
) -> dict[str, Any]:
    model = resolve_checkpoint(checkpoint)
    work = out.parent / f"{out.stem}-work"
    work.mkdir(parents=True, exist_ok=True)
    environment = rollout_environment(work)
    frozen = frozen_environment_report(environment)

    started = time.monotonic()
    banks = []
    for index, environment_seed in enumerate(seeds):
        rows, reported, elapsed = rollout(
            checkpoint=model,
            boards=boards,
            environment_seed=environment_seed,
            engine_seed=engine_seed,
            output=work / f"bank-{index}",
            environment=environment,
        )
        banks.append(
            score_bank(
                rows,
                expected_rows=boards,
                scored_boards=boards,
                environment_seed=environment_seed,
                engine_seed=engine_seed,
                reported=reported,
                wall_seconds=elapsed,
            )
        )
    summary = summarize(banks)
    payload = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "metric": METRIC,
        "direction": "maximize",
        "checkpoint": str(checkpoint),
        "model": str(model),
        "checkpoint_weight_sha256": checkpoint_weight_sha256(model),
        "environment_seeds": list(seeds),
        "engine_seed": engine_seed,
        "boards_per_bank": boards,
        "trajectories_per_board": 1,
        "temperature": EVAL_TEMPERATURE,
        "proxy_final_overlap_fraction": 0.0,
        "ragen_commit": RAGEN_COMMIT,
        "frozen_environment": frozen,
        "seconds": time.monotonic() - started,
        **summary,
        "banks": banks,
    }
    atomic_json(out, payload)
    return payload


def mock(out: Path, seeds: tuple[int, ...], boards: int) -> dict[str, Any]:
    banks = [
        score_bank(
            synthetic_bank(environment_seed=seed, boards=boards),
            expected_rows=boards,
            scored_boards=boards,
            environment_seed=seed,
            engine_seed=ENGINE_SEED,
        )
        for seed in seeds
    ]
    payload = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "metric": METRIC,
        "direction": "maximize",
        "mock": True,
        "environment_seeds": list(seeds),
        "boards_per_bank": boards,
        "proxy_final_overlap_fraction": 0.0,
        "seconds": 0.0,
        **summarize(banks),
        "banks": banks,
    }
    atomic_json(out, payload)
    return payload


def smoke() -> None:
    out = Path("/tmp/ragen-fast-eval-smoke.json")
    payload = mock(out, PROXY_ENVIRONMENT_SEEDS, BOARDS_PER_BANK)
    expected = len(PROXY_ENVIRONMENT_SEEDS) * BOARDS_PER_BANK
    if payload["boards"] != expected:
        raise RuntimeError(f"expected {expected} boards, got {payload['boards']}")
    if payload["bank_count"] != 4:
        raise RuntimeError(f"expected four banks, got {payload['bank_count']}")
    if payload["stderr"] <= 0.0:
        raise RuntimeError(f"stderr must be positive on varied rows: {payload['stderr']}")
    with tempfile.TemporaryDirectory(prefix="ragen-resolve-") as directory:
        root = Path(directory)
        for step in (80, 100):
            candidate = root / f"checkpoint-{step}"
            candidate.mkdir()
            (candidate / "pytorch_model.bin").write_bytes(b"probe")
        selected = resolve_checkpoint(
            root,
            loadability_check=lambda path: "truncated" if path.name == "checkpoint-100" else None,
        )
        if selected.name != "checkpoint-80":
            raise RuntimeError(f"did not skip the broken highest checkpoint: {selected}")
    print(json.dumps({"fast_eval_smoke": "passed", "boards": payload["boards"]}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--environment-seeds",
        type=int,
        nargs="*",
        default=list(PROXY_ENVIRONMENT_SEEDS),
        help="the board banks to score. Default is the four calibrated ones. Adding "
        "seeds adds boards and lowers the error bar; repeating one measures engine "
        "nondeterminism and is refused.",
    )
    parser.add_argument("--boards", type=int, default=BOARDS_PER_BANK)
    parser.add_argument(
        "--engine-seed",
        type=int,
        default=ENGINE_SEED,
        help="rides into vLLM as sampling_seed, and per request as sampling_seed + "
        "env_id. Change it to re-score the same boards under different sampling.",
    )
    parser.add_argument("--mock", action="store_true", help="synthetic rows, no GPU")
    parser.add_argument("--smoke", action="store_true", help="self-check and exit")
    args = parser.parse_args()

    if args.smoke:
        smoke()
        return
    if args.out is None:
        parser.error("--out is required")
    seeds = tuple(args.environment_seeds)
    if len(set(seeds)) != len(seeds):
        parser.error(
            "a repeated environment seed scores the same boards twice, which measures "
            "engine nondeterminism rather than adding resolution"
        )
    if args.mock:
        print(json.dumps(mock(args.out.resolve(), seeds, args.boards), sort_keys=True))
        return
    if args.checkpoint is None:
        parser.error("--checkpoint is required outside --mock/--smoke")
    payload = evaluate(
        args.checkpoint.resolve(), args.out.resolve(), seeds, args.boards, args.engine_seed
    )
    print(
        json.dumps(
            {
                "score": payload["score"],
                "stderr": payload["stderr"],
                "boards": payload["boards"],
                "bank_scores": payload["bank_scores"],
                "bank_spread": payload["bank_spread"],
                "seconds": payload["seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"fast_eval failed: {exc}", file=sys.stderr)
        raise
