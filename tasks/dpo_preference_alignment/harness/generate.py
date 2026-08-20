"""Greedy IFEval generation, one row at a time.

Fast and final evaluation share this implementation. Each prompt is generated
without batch neighbours under the fixed chat template, greedy decoding and token
cap. Row records retain generated-token counts so clipping is observed rather than
inferred. Shard outputs are keyed and restored to source order.
"""

from __future__ import annotations

import concurrent.futures
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import grade  # noqa: E402


def generate_rows(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    progress_every: int = 0,
    label: str = "",
) -> list[dict[str, Any]]:
    """Generate and score every row, returning one scored record each.

    `fresh metric state per row` from the protocol string is honoured by
    `grade.score_completion`, which builds a new scorer for every row. Only the
    registry patch is installed once, because it is idempotent.
    """

    import torch

    scored: list[dict[str, Any]] = []
    started = time.monotonic()
    with torch.inference_mode():
        for position, row in enumerate(rows):
            input_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)
            output_ids = model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                do_sample=grade.DO_SAMPLE,
                max_new_tokens=grade.MAX_NEW_TOKENS,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            new_tokens = int(output_ids.shape[1] - input_ids.shape[1])
            completion = tokenizer.decode(
                output_ids[0, input_ids.shape[1] :], skip_special_tokens=True
            )
            scored.append(
                {
                    "key": grade.canonical_key(row),
                    "scores": grade.score_completion(row, completion),
                    "generated_tokens": new_tokens,
                }
            )
            if progress_every and (position + 1) % progress_every == 0:
                elapsed = time.monotonic() - started
                print(
                    f"  {label}generated {position + 1} of {len(rows)} rows, "
                    f"{elapsed / (position + 1):.1f} s/row",
                    flush=True,
                )
    return scored


def shard_worker(job: tuple[str, str | None, list[dict[str, Any]], str, int, str]) -> None:
    """One shard, one device, one model copy. Runs in a spawned process.

    CUDA_VISIBLE_DEVICES is set before torch is imported, which is the only order
    that works: torch caches the visible device list at import.
    """

    checkpoint, reference, rows, device, index, shard_dir = job
    os.environ["CUDA_VISIBLE_DEVICES"] = device

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import checkpoint as checkpoint_module

    model, tokenizer = checkpoint_module.load_model(
        Path(checkpoint), Path(reference) if reference else None
    )
    scored = generate_rows(model, tokenizer, rows, progress_every=16, label=f"shard {index}: ")
    Path(shard_dir, f"shard-{index:02d}.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in scored),
        encoding="utf-8",
    )


def merge_shards(shard_dir: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Read the shard files back and restore source order.

    Keyed rather than concatenated. A shard written twice while another is missing
    leaves the row count right and the content wrong, and `grade.summarize` refuses
    a duplicate key -- but only if the merge did not quietly drop one first.
    """

    by_key: dict[str, dict[str, Any]] = {}
    for path in sorted(shard_dir.glob("shard-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            key = str(record["key"])
            if key in by_key:
                raise ValueError(f"row {key} appears in more than one shard file")
            by_key[key] = record
    ordered: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in rows:
        key = grade.canonical_key(row)
        if key in by_key:
            ordered.append(by_key[key])
        else:
            missing.append(key)
    if missing:
        raise ValueError(
            f"{len(missing)} row(s) were never generated, e.g. {missing[:3]}. A shard "
            "process died without writing its file."
        )
    return ordered


def run(
    checkpoint: Path,
    reference: Path | None,
    rows: list[dict[str, Any]],
    output: Path,
    *,
    gpus: int = 1,
) -> list[dict[str, Any]]:
    """Generate every row, over `gpus` devices, and return them in source order."""

    import torch

    available = torch.cuda.device_count()
    if available == 0:
        raise RuntimeError("IFEval generation needs at least one GPU")
    # One device by default, the same as the exploration and retrain phases. Not
    # "all visible": a host with eight free devices should not silently change what
    # the task costs.
    count = max(1, min(gpus, available))
    buckets = [bucket for bucket in grade.shard(rows, count) if bucket]

    if len(buckets) == 1:
        import checkpoint as checkpoint_module

        model, tokenizer = checkpoint_module.load_model(checkpoint, reference)
        return generate_rows(model, tokenizer, buckets[0], progress_every=16)

    shard_dir = output / "generation_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (str(checkpoint), str(reference) if reference else None, bucket, str(index), index,
         str(shard_dir))
        for index, bucket in enumerate(buckets)
    ]
    context = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=len(jobs), mp_context=context
    ) as executor:
        for future in concurrent.futures.as_completed(
            [executor.submit(shard_worker, job) for job in jobs]
        ):
            future.result()
    return merge_shards(shard_dir, rows)


def smoke() -> None:
    """Check the shard/merge round trip, which is the only logic here that does not
    need a model."""

    import tempfile

    rows = grade.synthetic_source(50)
    with tempfile.TemporaryDirectory() as name:
        shard_dir = Path(name)
        buckets = grade.shard(rows, 3)
        for index, bucket in enumerate(buckets):
            Path(shard_dir, f"shard-{index:02d}.jsonl").write_text(
                "".join(
                    json.dumps(
                        {
                            "key": grade.canonical_key(row),
                            "scores": grade.synthetic_scores(position),
                            "generated_tokens": 32,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                    for position, row in enumerate(bucket)
                ),
                encoding="utf-8",
            )
        merged = merge_shards(shard_dir, rows)
        if [record["key"] for record in merged] != [grade.canonical_key(row) for row in rows]:
            raise RuntimeError("merge did not restore source order")

        Path(shard_dir, "shard-00.jsonl").unlink()
        try:
            merge_shards(shard_dir, rows)
        except ValueError:
            pass
        else:
            raise RuntimeError("a missing shard must be refused")
    print("generate.py smoke passed")


if __name__ == "__main__":
    smoke()
