"""Strict IFEval on the fixed 128-row proxy subset of the 413-row final.

The proxy and final use the same row scorer, greedy chat-template generation and
aggregation. The payload records actual wall time, generated-token counts and the
exact number reaching the fixed cap. Its ``stderr`` is descriptive binomial
uncertainty for one absolute score; it is not a paired interval or a model-seed
variance estimate. Partial-row plumbing runs receive a different metric name.

"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import checkpoint as checkpoint_module  # noqa: E402
import generate  # noqa: E402
import grade  # noqa: E402
from runtime_guard import cuda_telemetry, exclusive_output, peak_memory_bytes  # noqa: E402

TASK_ID = "dpo_preference_alignment"
METRIC = "ifeval_strict_accuracy_public128"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_proxy(data: Path) -> list[dict[str, Any]]:
    """Read the 128-row proxy asset and re-derive that it is the right 128 rows.

    The membership check is not a formality. The asset is produced on the host by
    `environment/build_proxy_asset.py` from the full 541, and `final_eval.py`
    recomputes membership independently from those 541. If the file and the selector
    disagree, score(P) here and score(P) there would name different row sets.

    The digest also distinguishes this projection from any same-sized adjacent
    slice in the deterministic ordering.
    """

    path = data if data.is_file() else data / "proxy.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"no proxy rows at {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != grade.PROXY_ROWS:
        raise ValueError(f"proxy holds {len(rows)} rows, expected {grade.PROXY_ROWS}")
    required = {"key", "prompt", "instruction_id_list", "kwargs"}
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError(f"proxy row {row.get('key')} is missing {sorted(missing)}")

    manifest_path = path.with_name("proxy-manifest.json")
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("proxy_rows") != grade.PROXY_ROWS:
            raise ValueError("proxy manifest row count disagrees with the harness")
        expected = manifest.get("selection_digest")
        actual = grade.keys_digest(grade.canonical_key(row) for row in rows)
        if expected and expected != actual:
            raise ValueError(
                "the proxy asset is not the row set the harness selects. Rebuild it with "
                "environment/build_proxy_asset.py against google/IFEval@966cd895."
            )
    return rows


def evaluate(
    checkpoint: Path,
    reference: Path,
    data: Path,
    out: Path,
    *,
    rows: int = 0,
    gpus: int = 1,
) -> dict[str, Any]:
    if out.exists() or out.with_name(out.stem + "-rows.jsonl").exists():
        raise FileExistsError(f"fast-eval output already exists for {out}")
    if gpus != 1:
        raise ValueError("this task protocol uses exactly one visible GPU per evaluation")
    pairs = load_proxy(data)
    # A truncated run is for checking the plumbing, and it is renamed so it cannot be
    # mistaken for a proxy score: fewer rows is a different row set, and the whole
    # point of this tier is that the row set is fixed.
    partial = 0 < rows < len(pairs)
    if partial:
        pairs = pairs[:rows]
    report = checkpoint_module.describe(checkpoint, reference)

    import torch

    gpu = cuda_telemetry(torch)
    torch.cuda.reset_peak_memory_stats()

    started = time.monotonic()
    scored = generate.run(checkpoint, reference, pairs, out.parent, gpus=gpus)
    summary = grade.summarize(scored)
    metric = f"{METRIC}_partial{len(pairs)}" if partial else METRIC

    payload = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "metric": metric,
        "direction": "maximize",
        "checkpoint": str(checkpoint),
        "artifact": report,
        "proxy_rows": len(pairs),
        "final_rows": grade.FINAL_ROWS,
        "overlap_fraction": len(pairs) / grade.FINAL_ROWS,
        "partial": partial,
        "protocol": grade.PROTOCOL,
        "max_new_tokens": grade.MAX_NEW_TOKENS,
        "tokenization_protocol": grade.TOKENIZATION_PROTOCOL,
        "seconds": time.monotonic() - started,
        "stderr_kind": "binomial_descriptive_not_seed_or_paired_uncertainty",
        "gpu": gpu,
        "gpu_memory_peak_bytes": peak_memory_bytes(torch),
        **summary,
    }
    atomic_json(out, payload)
    out.with_name(out.stem + "-rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in scored),
        encoding="utf-8",
    )
    return payload


def mock(out: Path) -> dict[str, Any]:
    """Score synthetic rows over the real split structure, no GPU and no model."""

    split = grade.split_source(grade.synthetic_source())
    scored = [
        {
            "key": grade.canonical_key(row),
            "scores": grade.synthetic_scores(index),
            "generated_tokens": 96,
        }
        for index, row in enumerate(split["proxy"])
    ]
    payload = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "metric": METRIC,
        "direction": "maximize",
        "mock": True,
        "proxy_rows": len(scored),
        "final_rows": grade.FINAL_ROWS,
        "overlap_fraction": len(scored) / grade.FINAL_ROWS,
        "seconds": 0.0,
        **grade.summarize(scored),
    }
    atomic_json(out, payload)
    return payload


def smoke() -> None:
    payload = mock(Path("/tmp/dpo-fast_eval-smoke.json"))
    if payload["n"] != grade.PROXY_ROWS:
        raise RuntimeError(f"unexpected row count: {payload['n']}")
    if not 0.0 < payload["score"] < 1.0:
        raise RuntimeError(f"score out of range: {payload['score']}")
    if payload["stderr"] <= 0.0:
        raise RuntimeError("stderr must be positive on varied data")
    if abs(payload["overlap_fraction"] - grade.PROXY_ROWS / grade.FINAL_ROWS) > 1e-12:
        raise RuntimeError(f"unexpected overlap: {payload['overlap_fraction']}")
    if payload["overlap_fraction"] > 0.5:
        raise RuntimeError("proxy/final overlap is over the 50% ceiling")
    if grade.METRIC not in payload["metrics"]:
        raise RuntimeError("the headline metric must be reported")
    print(json.dumps({"fast_eval_smoke": "passed", "n": payload["n"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reference", type=Path, default=Path("/assets/models/policy_start"))
    parser.add_argument("--data", type=Path, default=Path("/assets/data/ifeval_proxy"))
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--rows",
        type=int,
        default=0,
        help="score only the first N of the 128. For checking the plumbing: the result "
        "is renamed ifeval_proxy_128_partialN, because fewer rows is a different row "
        "set and this tier's whole value is that the row set is fixed.",
    )
    parser.add_argument("--gpus", type=int, default=1, help="devices; only changes the cost")
    parser.add_argument("--mock", action="store_true", help="synthetic rows, no GPU")
    parser.add_argument("--smoke", action="store_true", help="self-check and exit")
    args = parser.parse_args()

    if args.smoke:
        smoke()
        return
    if args.out is None:
        parser.error("--out is required")
    if args.mock:
        print(json.dumps(mock(args.out.resolve()), sort_keys=True))
        return
    if args.checkpoint is None:
        parser.error("--checkpoint is required outside --mock/--smoke")
    with exclusive_output("dpo_fast_eval"):
        payload = evaluate(
            args.checkpoint.resolve(),
            args.reference.resolve(),
            args.data.resolve(),
            args.out.resolve(),
            rows=args.rows,
            gpus=args.gpus,
        )
    print(
        json.dumps(
            {
                "metric": payload["metric"],
                "score": payload["score"],
                "stderr": payload["stderr"],
                "n": payload["n"],
                "correct": payload["correct"],
                "length_clipped": payload["length_clipped"],
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
