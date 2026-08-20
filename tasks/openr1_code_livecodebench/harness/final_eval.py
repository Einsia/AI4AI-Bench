"""Hidden final: LiveCodeBench v6, the whole 175-row release slice in hash order.

The deployment asset is materialized from the pinned trusted lock. This evaluator
accepts only a loadable checkpoint and that v6 mount, records hashes and exact row
IDs, and never falls back to the public v4/v5 exploration rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from grade import (  # noqa: E402
    LCB_COMMIT,
    MAX_NEW_TOKENS,
    atomic_json,
    build_prompts,
    execute,
    generate,
    load_evaluator,
    load_rows,
    summarize,
)
from runtime_guard import cuda_telemetry, exclusive_output, peak_memory_bytes  # noqa: E402

TASK_ID = "openr1_code_livecodebench"
METRIC = "livecodebench_v6_pass_at_1_full175"
# v6 only. Not test4/test5: a different release, disjoint from every public row by
# construction rather than by a slice boundary.
FINAL_FILES = ("test6.jsonl",)
# The fixed protocol is the whole v6-only release file, in hash order.
FINAL_ROWS = 175
FINAL_OFFSET = 0
REWARD_PATH = Path("/logs/verifier/reward")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def weight_sha256(checkpoint: Path) -> str:
    """Hash the weights on disk. Computed from the artifact, never read from it."""

    files = sorted(
        {
            path
            for pattern in ("*.safetensors", "pytorch_model*.bin")
            for path in checkpoint.glob(pattern)
        }
    )
    if not files:
        raise RuntimeError(f"checkpoint has no model weights: {checkpoint}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def has_weights(directory: Path) -> bool:
    return any(directory.glob("*.safetensors")) or any(directory.glob("pytorch_model*.bin"))


def resolve_checkpoint(checkpoint: Path) -> Path:
    """Same resolution as fast_eval, so a path that scores there scores here.

    Deliberately narrower in one way: no search for the highest numbered
    subdirectory. The operator names the checkpoint being scored for the final,
    rather than a root this file picks from. Accepting a `global_step_N` or
    `checkpoint-N` directory still matters, because that is what the retrain phase
    reports.
    """

    if has_weights(checkpoint):
        return checkpoint
    for name in ("huggingface", "hf_model"):
        nested = checkpoint / name
        if nested.is_dir() and has_weights(nested):
            return nested
    raise FileNotFoundError(
        f"no model weights under {checkpoint}; expected an HF directory, or a "
        "global_step_N / checkpoint-N directory holding one"
    )


def require_final_rows(data: Path) -> None:
    """Refuse to score anything if the v6 rows are absent. No fallback.

    The v4/v5 files live one directory over and would load cleanly, which is
    precisely why this check is explicit: a silent substitution makes the final a
    second proxy and nothing downstream can tell.
    """

    missing = [name for name in FINAL_FILES if not (data / name).is_file()]
    if not missing:
        return
    present = sorted(path.name for path in data.glob("*")) if data.is_dir() else []
    raise FileNotFoundError(
        f"the hidden final needs {list(FINAL_FILES)} under {data}, and {missing} "
        f"{'is' if len(missing) == 1 else 'are'} not there. Found: {present or 'nothing'}.\n"
        "Materialize assets/data/livecodebench_final from the pinned trusted "
        "asset lock before scoring.\n"
        "There is deliberately no fallback to the v4/v5 rows. Those are the "
        "exploration phase's rows -- scoring the final on them would report a "
        "number that looks valid and measures nothing."
    )


def write_reward(score: float, reward_path: Path) -> None:
    """Harbor's verifier contract: the scalar lands in /logs/verifier/reward."""

    reward_path.parent.mkdir(parents=True, exist_ok=True)
    reward_path.write_text(f"{score:.10f}\n", encoding="utf-8")


def evaluate(
    checkpoint: Path,
    assets: Path,
    output: Path,
    reward_path: Path,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> dict[str, Any]:
    data = assets / "data/livecodebench_final"
    require_final_rows(data)
    model = resolve_checkpoint(checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    occupied = [
        path
        for path in (output / "resolved_config.json", output / "graded_generations.jsonl")
        if path.exists()
    ]
    prior_summary = output / "summary.json"
    if prior_summary.is_file():
        try:
            prior_payload = json.loads(prior_summary.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"cannot classify existing receipt {prior_summary}: {error}"
            ) from error
        if prior_payload.get("metric") == METRIC:
            occupied.append(prior_summary)
    if occupied:
        raise FileExistsError(f"refusing to overwrite final-evaluation receipts: {occupied}")

    import torch

    gpu = cuda_telemetry(torch, require_single=True)
    torch.cuda.reset_peak_memory_stats()

    evaluator = load_evaluator()
    rows = load_rows(data, FINAL_FILES, offset=FINAL_OFFSET, count=FINAL_ROWS)
    problems, prompts = build_prompts(evaluator, rows)
    weights_hash = weight_sha256(model)

    atomic_json(
        output / "resolved_config.json",
        {
            "schema_version": 1,
            "task_id": TASK_ID,
            "checkpoint": str(checkpoint),
            "model": str(model),
            "checkpoint_weight_sha256": weights_hash,
            "livecodebench_commit": LCB_COMMIT,
            "release_slice": "v6_only",
            "release_files": list(FINAL_FILES),
            "release_file_sha256": {
                name: file_sha256(data / name) for name in FINAL_FILES
            },
            "row_order": "sha256_question_id",
            "row_offset": FINAL_OFFSET,
            "row_count": len(rows),
            "question_ids": [row["question_id"] for row in rows],
            "do_sample": False,
            "n": 1,
            "max_new_tokens": max_new_tokens,
            "image_digest": os.environ.get("IMAGE_DIGEST"),
            "gpu": gpu,
        },
    )

    started = time.monotonic()
    generations, token_counts = generate(model, prompts, max_new_tokens)
    records, reported = execute(evaluator, problems, generations)
    for record, generated_tokens in zip(records, token_counts, strict=True):
        record["generated_tokens"] = generated_tokens
        record["length_clipped"] = generated_tokens >= max_new_tokens
    aggregate = summarize(records, reported)
    (output / "graded_generations.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=True) + "\n" for record in records),
        encoding="utf-8",
    )

    summary = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "passed",
        "benchmark": "LiveCodeBench v6 pass@1",
        "metric": METRIC,
        "direction": "maximize",
        "metrics": {METRIC: aggregate["score"]},
        **aggregate,
        "release_slice": "v6_only",
        "wall_seconds": time.monotonic() - started,
        "gpu": gpu,
        "peak_memory_bytes": peak_memory_bytes(torch),
        "offline": True,
        "checkpoint_weight_sha256": weights_hash,
    }
    atomic_json(output / "summary.json", summary)
    write_reward(aggregate["score"], reward_path)
    return summary


def mock(output: Path, reward_path: Path) -> dict[str, Any]:
    records = [
        {
            "question_id": f"mock-v6-{index}",
            "example_id": index,
            "final_score": index % 10 == 0,
            "extracted": index % 25 != 24,
            "generated_tokens": MAX_NEW_TOKENS if index == 0 else 32,
            "length_clipped": index == 0,
        }
        for index in range(FINAL_ROWS)
    ]
    output.mkdir(parents=True, exist_ok=True)
    aggregate = summarize(records)
    if aggregate["n"] != FINAL_ROWS:
        raise RuntimeError(f"mock produced {aggregate['n']} rows, expected {FINAL_ROWS}")
    (output / "graded_generations.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "passed",
        "metric": METRIC,
        "direction": "maximize",
        "metrics": {METRIC: aggregate["score"]},
        **aggregate,
        "release_slice": "v6_only",
        "mock": True,
    }
    atomic_json(output / "summary.json", summary)
    write_reward(aggregate["score"], reward_path)
    return summary


def smoke() -> None:
    records = [
        {"question_id": f"q{index}", "example_id": index, "final_score": True, "extracted": True}
        for index in range(FINAL_ROWS)
    ]
    summary = summarize(records)
    if summary["score"] != 1.0 or summary["n"] != FINAL_ROWS:
        raise RuntimeError(f"unexpected final smoke result: {summary}")
    # The refusal is part of the contract, so the smoke test exercises it: an
    # empty directory must raise rather than resolve to anything.
    import tempfile

    with tempfile.TemporaryDirectory() as empty:
        try:
            require_final_rows(Path(empty))
        except FileNotFoundError:
            pass
        else:  # pragma: no cover
            raise RuntimeError("a missing test6.jsonl was accepted")
    print(json.dumps({"final_eval_smoke": "passed", "n": summary["n"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--assets", type=Path, default=Path("/assets"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reward-path", type=Path, default=REWARD_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        smoke()
        return
    if args.output is None:
        parser.error("--output is required")
    if args.mock:
        print(json.dumps(mock(args.output.resolve(), args.reward_path), sort_keys=True))
        return
    if args.checkpoint is None:
        parser.error("--checkpoint is required outside --mock/--smoke")
    resolved_output = args.output.resolve()
    with exclusive_output("openr1_final_eval", resolved_output):
        result = evaluate(
            args.checkpoint.resolve(),
            args.assets.resolve(),
            resolved_output,
            args.reward_path,
            args.max_new_tokens,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"OpenR1 final failed: {exc}", file=sys.stderr)
        raise
