"""RewardBench v1 fast evaluation on the fixed 512-row stratified proxy.

The proxy uses the final evaluator's tokenization and aggregation. It is a visible
subset of the final; held-out behavior is reported only by final evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import artifact  # noqa: E402
import grade  # noqa: E402
from runtime_guard import cuda_telemetry, exclusive_output, peak_memory_bytes  # noqa: E402

TASK_ID = "ultrafeedback_bt_rm_rewardbench"
METRIC = "rewardbench_proxy_512"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_proxy(data: Path) -> list[dict[str, Any]]:
    """Read the 512-row proxy asset and re-derive that it is the right 512 rows.

    The membership check is not a formality. The proxy file is produced on the host
    by `environment/build_proxy_asset.py` from the full RewardBench, and
    `final_eval.py` recomputes membership independently from the 2985 rows. If the
    file and the selector disagree, `score(P)` here and `score(P)` there would
    describe different row sets while carrying the same name.
    """

    path = data if data.is_file() else data / "proxy.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"no proxy pairs at {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != grade.PROXY_ROWS:
        raise ValueError(f"proxy holds {len(rows)} rows, expected {grade.PROXY_ROWS}")
    for row in rows:
        missing = {"id", "subset", "prompt", "chosen", "rejected"} - set(row)
        if missing:
            raise ValueError(f"proxy row {row.get('id')} is missing {sorted(missing)}")

    manifest_path = path.with_name("proxy-manifest.json")
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("proxy_rows") != grade.PROXY_ROWS:
            raise ValueError("proxy manifest row count disagrees with the harness")
        expected = manifest.get("selection_digest")
        actual = grade.selection_digest(row["id"] for row in rows)
        if expected and expected != actual:
            raise ValueError(
                "the proxy asset is not the row set the harness selects. Rebuild it with "
                "environment/build_proxy_asset.py against the pinned RewardBench revision."
            )
    return rows


def evaluate(checkpoint: Path, base_model: Path, data: Path, out: Path) -> dict[str, Any]:
    rows_path = out.with_name(out.stem + "-rows.jsonl")
    occupied = [path for path in (out, rows_path) if path.exists()]
    if occupied:
        raise FileExistsError(f"refusing to overwrite evaluation receipts: {occupied}")
    report = artifact.check(checkpoint, base_model)
    pairs = load_proxy(data)
    import torch

    gpu = cuda_telemetry(torch, require_single=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model, tokenizer = artifact.load_model(checkpoint, base_model)
    loaded = time.monotonic()
    rows = grade.score_pairs(model, tokenizer, pairs, progress_every=8)
    summary = grade.summarize(rows)

    payload = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "metric": METRIC,
        "direction": "maximize",
        "checkpoint": str(checkpoint),
        "proxy_rows": len(pairs),
        "final_rows": grade.FINAL_ROWS,
        "overlap_fraction": len(pairs) / grade.FINAL_ROWS,
        "max_length": grade.EVAL_MAX_LENGTH,
        "eval_batch_pairs": grade.EVAL_BATCH_PAIRS,
        "tokenization": "apply_chat_template_tokenize_true_no_special_token_readdition",
        "artifact": report,
        "model_load_seconds": loaded - started,
        "seconds": time.monotonic() - started,
        "gpu": gpu,
        "gpu_memory_peak_bytes": peak_memory_bytes(torch),
        **summary,
    }
    atomic_json(out, payload)
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return payload


def mock(out: Path) -> dict[str, Any]:
    """Score synthetic rows over the real subset structure, no GPU and no model."""

    counts = grade.expected_row_counts()
    allocation = grade.stratified_allocation(grade.PROXY_ROWS, counts)
    rows = [
        {
            "id": f"{subset}-{index}",
            "subset": subset,
            "chosen_reward": 1.0,
            "rejected_reward": 0.0,
            "reward_margin": 1.0,
            "correct": float((index + len(subset)) % 3 != 0),
        }
        for subset, taken in allocation.items()
        for index in range(taken)
    ]
    payload = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "metric": METRIC,
        "direction": "maximize",
        "mock": True,
        "proxy_rows": len(rows),
        "seconds": 0.0,
        **grade.summarize(rows),
    }
    atomic_json(out, payload)
    return payload


def smoke() -> None:
    payload = mock(Path("/tmp/btrm-fast_eval-smoke.json"))
    if payload["n"] != grade.PROXY_ROWS:
        raise RuntimeError(f"unexpected row count: {payload['n']}")
    if not 0.0 < payload["score"] < 100.0:
        raise RuntimeError(f"score out of range: {payload['score']}")
    if payload["stderr"] <= 0.0:
        raise RuntimeError("stderr must be positive on varied data")
    if set(payload["section_scores"]) != set(grade.SECTIONS):
        raise RuntimeError("the four sections must all be reported")
    print(json.dumps({"fast_eval_smoke": "passed", "n": payload["n"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--base-model", type=Path, default=Path("/assets/models/base"))
    parser.add_argument("--data", type=Path, default=Path("/assets/data/rewardbench_proxy"))
    parser.add_argument("--out", type=Path)
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
    resolved_out = args.out.resolve()
    with exclusive_output("btrm_fast_eval", resolved_out.parent):
        payload = evaluate(
            args.checkpoint.resolve(),
            args.base_model.resolve(),
            args.data.resolve(),
            resolved_out,
        )
    print(
        json.dumps(
            {
                "score": payload["score"],
                "stderr": payload["stderr"],
                "n": payload["n"],
                "section_scores": payload["section_scores"],
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
