"""LiveCodeBench pass@1 on the public v4/v5 exploration rows.

The canonical health tier is 64 rows at offset 0. The canonical confirmation
tier is the disjoint 204 rows at offset 64. Both use the final evaluator's prompt,
greedy generation, extraction, and official test execution; the hidden final uses
v6 instead. Custom row/count/token-cap overrides are diagnostics and are labelled
as such in the receipt.
"""

from __future__ import annotations

import argparse
import json
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
METRIC = "livecodebench_public_pass_at_1"
# The two public release files. Both the proxy and the confirmation slice are
# positions in the hash order of their union.
PUBLIC_FILES = ("test4.jsonl", "test5.jsonl")
# The frozen proxy: the first 64 rows. Do not change either number -- they define
# which rows every recorded proxy score was measured on.
PROXY_OFFSET = 0
PROXY_ROWS = 64
# The disjoint confirmation slice: rows 64..267 of the same order. Called
# public204 in the old evidence, where it was a separate selection tier; v1 has
# two tiers, so it survives here as the wider setting of one tool rather than as
# a stage of its own.
CONFIRM_OFFSET = 64
CONFIRM_ROWS = 204


def protocol_tier(offset: int, rows: int, max_new_tokens: int) -> str:
    if max_new_tokens != MAX_NEW_TOKENS:
        return "diagnostic_custom_protocol"
    if (offset, rows) == (PROXY_OFFSET, PROXY_ROWS):
        return "health_64"
    if (offset, rows) == (CONFIRM_OFFSET, CONFIRM_ROWS):
        return "confirmation_204"
    return "diagnostic_custom_protocol"


def has_weights(directory: Path) -> bool:
    return any(directory.glob("*.safetensors")) or any(directory.glob("pytorch_model*.bin"))


def resolve_checkpoint(checkpoint: Path) -> Path:
    """Find the weights, given any of the paths someone would plausibly type.

    An HF directory, a `global_step_N` export, or `/out/checkpoints` itself --
    the last takes the highest step, because that is what "checkpoints land in
    /out/checkpoints" invites you to type. HF Trainer's own `checkpoint-N` names
    are accepted too, since a candidate that bypasses run.sh's export step will
    have those and nothing else.
    """

    if has_weights(checkpoint):
        return checkpoint
    for name in ("huggingface", "hf_model"):
        nested = checkpoint / name
        if nested.is_dir() and has_weights(nested):
            return nested

    numbered: list[tuple[int, Path]] = []
    for prefix in ("global_step_", "checkpoint-"):
        for path in checkpoint.glob(f"{prefix}*"):
            suffix = path.name.removeprefix(prefix)
            if suffix.isdigit():
                numbered.append((int(suffix), path))
    for _, directory in sorted(numbered, reverse=True):
        if has_weights(directory):
            return directory
        for name in ("huggingface", "hf_model"):
            nested = directory / name
            if nested.is_dir() and has_weights(nested):
                return nested

    raise FileNotFoundError(
        f"no model weights under {checkpoint}. Looked for *.safetensors here, in "
        "huggingface/ and hf_model/, and in the highest global_step_* or "
        "checkpoint-* below."
    )


def evaluate(
    checkpoint: Path,
    data: Path,
    out: Path,
    *,
    offset: int,
    rows: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    rows_path = out.with_name(out.stem + "-rows.jsonl")
    occupied = [path for path in (out, rows_path) if path.exists()]
    if occupied:
        raise FileExistsError(f"refusing to overwrite evaluation receipts: {occupied}")
    model = resolve_checkpoint(checkpoint)
    evaluator = load_evaluator()
    selected = load_rows(data, PUBLIC_FILES, offset=offset, count=rows)
    problems, prompts = build_prompts(evaluator, selected)

    import torch

    gpu = cuda_telemetry(torch, require_single=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    generations, token_counts = generate(model, prompts, max_new_tokens)
    records, reported = execute(evaluator, problems, generations)
    for record, generated_tokens in zip(records, token_counts, strict=True):
        record["generated_tokens"] = generated_tokens
        record["length_clipped"] = generated_tokens >= max_new_tokens
    summary = summarize(records, reported)
    elapsed = time.monotonic() - started
    tier = protocol_tier(offset, len(selected), max_new_tokens)

    payload = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "metric": METRIC,
        "metrics": {METRIC: summary["score"]},
        "direction": "maximize",
        "checkpoint": str(checkpoint),
        "model": str(model),
        "release_slice": "v4_v5",
        "protocol_tier": tier,
        "canonical_protocol": tier != "diagnostic_custom_protocol",
        "row_offset": offset,
        "row_count": len(selected),
        "question_ids": [row["question_id"] for row in selected],
        "livecodebench_commit": LCB_COMMIT,
        "do_sample": False,
        "generations_per_prompt": 1,
        "max_new_tokens": max_new_tokens,
        "seconds": elapsed,
        "gpu": gpu,
        "peak_memory_bytes": peak_memory_bytes(torch),
        **summary,
    }
    atomic_json(out, payload)
    rows_path.write_text(
        "".join(json.dumps(record, ensure_ascii=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return payload


def mock(out: Path, rows: int) -> dict[str, Any]:
    records = [
        {
            "question_id": f"mock-{index}",
            "example_id": index,
            "final_score": index % 16 == 0,
            "extracted": index % 32 != 31,
            "generated_tokens": MAX_NEW_TOKENS if index == 0 else 32,
            "length_clipped": index == 0,
        }
        for index in range(rows)
    ]
    payload = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "metric": METRIC,
        "direction": "maximize",
        "mock": True,
        "seconds": 0.0,
        **summarize(records),
    }
    payload["metrics"] = {METRIC: payload["score"]}
    atomic_json(out, payload)
    return payload


def smoke() -> None:
    payload = mock(Path("/tmp/fast_eval-smoke.json"), PROXY_ROWS)
    if payload["n"] != PROXY_ROWS:
        raise RuntimeError(f"unexpected row count: {payload}")
    if payload["stderr"] <= 0.0:
        raise RuntimeError(f"stderr must be positive on varied data: {payload}")
    print(json.dumps({"fast_eval_smoke": "passed", "n": payload["n"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data", type=Path, default=Path("/assets/data/livecodebench_public"))
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--confirm",
        action="store_true",
        help=f"score the disjoint {CONFIRM_ROWS}-row confirmation slice at offset "
        f"{CONFIRM_OFFSET} instead of the {PROXY_ROWS}-row health slice",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="override the row count. 0 means every row from the offset on.",
    )
    parser.add_argument("--offset", type=int, default=None, help="override the row offset")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--mock", action="store_true", help="grade synthetic rows, no GPU")
    parser.add_argument("--smoke", action="store_true", help="self-check and exit")
    args = parser.parse_args()

    if args.smoke:
        smoke()
        return
    if args.out is None:
        parser.error("--out is required")
    if args.mock:
        rows = args.rows if args.rows else PROXY_ROWS
        print(json.dumps(mock(args.out.resolve(), rows), sort_keys=True))
        return
    if args.checkpoint is None:
        parser.error("--checkpoint is required outside --mock/--smoke")

    offset = CONFIRM_OFFSET if args.confirm else PROXY_OFFSET
    rows = CONFIRM_ROWS if args.confirm else PROXY_ROWS
    if args.offset is not None:
        offset = args.offset
    if args.rows is not None:
        rows = args.rows

    # Resolved before load_evaluator(), which chdirs into the pinned tree so the
    # upstream prompt module finds its few-shot examples. A relative --out would
    # otherwise land somewhere read-only and unrelated.
    resolved_out = args.out.resolve()
    with exclusive_output("openr1_fast_eval", resolved_out.parent):
        payload = evaluate(
            args.checkpoint.resolve(),
            args.data.resolve(),
            resolved_out,
            offset=offset,
            rows=rows,
            max_new_tokens=args.max_new_tokens,
        )
    print(
        json.dumps(
            {
                "score": payload["score"],
                "stderr": payload["stderr"],
                "n": payload["n"],
                "correct": payload["correct"],
                "extracted_rate": payload["extracted_rate"],
                "row_offset": payload["row_offset"],
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
