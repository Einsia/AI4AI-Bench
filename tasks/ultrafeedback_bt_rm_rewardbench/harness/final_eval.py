"""Hidden RewardBench v1 final on all 2985 pinned pairs.

The headline is the official four-section score. The receipt also reports the
visible 512-row proxy, its 2473-row complement, artifact validity, decontamination,
fixed tokenization, truncation counts, and B300-readable CUDA telemetry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import artifact  # noqa: E402
import grade  # noqa: E402
from runtime_guard import cuda_telemetry, exclusive_output, peak_memory_bytes  # noqa: E402

TASK_ID = "ultrafeedback_bt_rm_rewardbench"
METRIC = "rewardbench_v1_score"
BASE_MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
REWARDBENCH_SOURCE = "allenai/reward-bench"
REWARDBENCH_REVISION = "168d848cdbbea9764fae4a544dc9ca1e6cca4931"
TRAIN_DATA_REVISION = "3949bf5f8c17c394422ccfab0c31ea9c20bdeb85"
UPSTREAM_REVISION = "fc39179f5a9a4dd75047a4c0311672905b9d9a04"
# `reward`, not `reward.txt`: orchestrator/runner.py:report_reward reads
# `<logs>/verifier/reward`. OPD writes reward.txt, so its reward line never prints.
REWARD_PATH = Path("/logs/verifier/reward")


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


def write_reward(score: float, reward_path: Path) -> None:
    """Harbor's verifier contract: the scalar lands in /logs/verifier."""

    reward_path.parent.mkdir(parents=True, exist_ok=True)
    reward_path.write_text(f"{score:.10f}\n", encoding="utf-8")


def load_rewardbench(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read the pinned filtered split, refusing content drift.

    The row count, the subset inventory and every per-subset count are checked
    against the metric definition in `grade.py`. That is stronger than the old
    `len(dataset) != 2985`, which passes on a dataset whose subsets have been
    reshuffled -- and the reference protocol had exactly that kind of error in its
    weight table.

    A per-file sha256 is recorded if `rewardbench-files.json` sits beside the data,
    and verified. The reference protocol recorded only a four-file tree-manifest hash
    for this asset, not the parquet's own digest, so there is no pinned per-file
    value to compare against yet: this reports the digest it saw so the first real
    run can pin it. OPD pins its question files this way and it caught nothing only
    because nothing drifted.
    """

    import pyarrow.parquet as pq

    files = sorted((root / "data").glob("filtered-*.parquet")) or sorted(
        root.glob("filtered-*.parquet")
    )
    if not files:
        raise FileNotFoundError(f"no filtered RewardBench parquet under {root}")
    table = pq.read_table([str(path) for path in files])
    required = {"id", "subset", "prompt", "chosen", "rejected"}
    if not required <= set(table.column_names):
        raise ValueError(f"unexpected RewardBench schema: {sorted(table.column_names)}")
    rows = [
        {
            "id": str(record["id"]),
            "subset": str(record["subset"]),
            "prompt": str(record["prompt"]),
            "chosen": str(record["chosen"]),
            "rejected": str(record["rejected"]),
        }
        for record in table.select(sorted(required)).to_pylist()
    ]
    if len(rows) != grade.FINAL_ROWS:
        raise ValueError(f"RewardBench has {len(rows)} rows, expected {grade.FINAL_ROWS}")

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["subset"]] = counts.get(row["subset"], 0) + 1
    grade.require_weights_match_rows(counts)

    digests = {path.name: file_sha256(path) for path in files}
    expected_path = root / "rewardbench-files.json"
    if expected_path.is_file():
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        mismatched = {
            name: {"expected": expected[name], "actual": digest}
            for name, digest in digests.items()
            if name in expected and expected[name] != digest
        }
        if mismatched:
            raise ValueError(f"pinned RewardBench file hash mismatch: {mismatched}")
    return rows, {"files": digests, "subset_rows": dict(sorted(counts.items()))}


def canonical_text(value: str) -> str:
    """The projection's canonicalization, reproduced so overlap can be recomputed.

    NFKC, whitespace collapse, casefold -- lifted verbatim from
    `judge/tools/project_public_assets.py`, which is what produced the training
    pool. If it drifts from that function the check below stops meaning anything,
    so it is short on purpose.
    """

    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def assistant_text(conversation: Any) -> str:
    if isinstance(conversation, str):
        return canonical_text(conversation)
    for message in reversed(conversation):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return canonical_text(str(message.get("content", "")))
    raise ValueError("conversation has no assistant response")


def user_text(conversation: Any) -> str:
    for message in conversation:
        if isinstance(message, dict) and message.get("role") == "user":
            return canonical_text(str(message.get("content", "")))
    raise ValueError("conversation has no user turn")


def verify_disjoint(pairs_path: Path, final_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute that the training pool does not overlap RewardBench.

    This was the reference protocol's largest unverifiable claim. The pool was
    produced by `judge/tools/project_public_assets.py`, which drops any
    UltraFeedback row whose canonical prompt matches a RewardBench prompt or whose
    chosen/rejected response matches any RewardBench response -- but the only
    evidence shipped in the repository was `decontamination_status: "passed"` in a
    manifest, plus a `hidden_signature_digest` that stayed judge-side by design.
    The repository asserted decontamination and could not check it, because
    checking needs RewardBench and RewardBench was never mounted anywhere.

    Under the new split RewardBench *is* mounted here, so the check is now
    possible, and the training pool is mounted read-only beside it for exactly this
    purpose. It leaks nothing: the Agent already has the whole pool during
    exploration.

    An overlap is fatal. If a training pair carries a RewardBench prompt or
    response, the final is partly measuring memorisation and the number is not a
    generalisation result.
    """

    if not pairs_path.is_file():
        raise FileNotFoundError(
            f"cannot verify the zero-overlap gate because {pairs_path} is not mounted"
        )
    prompts = {canonical_text(row["prompt"]) for row in final_rows}
    responses = {canonical_text(row["chosen"]) for row in final_rows}
    responses |= {canonical_text(row["rejected"]) for row in final_rows}

    checked = 0
    prompt_hits: list[str] = []
    response_hits: list[str] = []
    with pairs_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            checked += 1
            if user_text(row["chosen"]) in prompts:
                prompt_hits.append(str(row.get("pair_sha256", checked)))
            if (
                assistant_text(row["chosen"]) in responses
                or assistant_text(row["rejected"]) in responses
            ):
                response_hits.append(str(row.get("pair_sha256", checked)))
    result = {
        "status": "passed" if not (prompt_hits or response_hits) else "failed",
        "training_pairs_checked": checked,
        "rewardbench_prompts": len(prompts),
        "rewardbench_responses": len(responses),
        "prompt_overlap": len(prompt_hits),
        "response_overlap": len(response_hits),
        "canonicalization": "NFKC, whitespace collapse, casefold",
    }
    if prompt_hits or response_hits:
        raise ValueError(
            f"the training pool overlaps RewardBench: {len(prompt_hits)} prompt and "
            f"{len(response_hits)} response collisions over {checked} pairs, e.g. "
            f"{(prompt_hits + response_hits)[:3]}. The final would be measuring "
            f"memorisation rather than generalisation."
        )
    return result


def split_report(rows: list[dict[str, Any]], proxy_ids: dict[str, str]) -> dict[str, Any]:
    """score(F), score(P), score(F\\P), and the difference with its error bar."""

    inside = [row for row in rows if row["id"] in proxy_ids]
    outside = [row for row in rows if row["id"] not in proxy_ids]
    if len(inside) != grade.PROXY_ROWS:
        raise ValueError(
            f"{len(inside)} of the scored rows are in the proxy, expected "
            f"{grade.PROXY_ROWS}. The proxy selector and the final's row set disagree."
        )
    if not outside:
        raise ValueError("the proxy covers the whole final; there is nothing held out")

    whole = grade.summarize(rows)
    proxy = grade.summarize(inside)
    held_out = grade.summarize(outside)
    gap = proxy["score"] - held_out["score"]
    return {
        "score_final": whole["score"],
        "score_proxy": proxy["score"],
        "score_held_out": held_out["score"],
        # The two row sets are disjoint, so the variances add.
        "overfitting_gap": gap,
        "overfitting_gap_stderr": math.sqrt(proxy["stderr"] ** 2 + held_out["stderr"] ** 2),
        "rows_final": whole["n"],
        "rows_proxy": proxy["n"],
        "rows_held_out": held_out["n"],
        "overlap_fraction": proxy["n"] / whole["n"],
        "final": whole,
        "proxy": proxy,
        "held_out": held_out,
    }


def evaluate(
    checkpoint: Path,
    assets: Path,
    output: Path,
    reward_path: Path,
) -> dict[str, Any]:
    base_model = assets / "models/base"
    # The artifact check runs before anything is loaded onto a device, so a
    # violation costs a config read rather than a model load and 2985 forward
    # passes.
    report = artifact.check(checkpoint, base_model)
    final_rows, provenance = load_rewardbench(assets / "data/rewardbench")
    decontamination = verify_disjoint(assets / "data/pairs.jsonl", final_rows)
    proxy_ids = grade.select_proxy_ids(final_rows)
    output.mkdir(parents=True, exist_ok=True)
    occupied = [
        path
        for path in (output / "resolved_config.json", output / "metrics.jsonl")
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

    atomic_json(
        output / "resolved_config.json",
        {
            "schema_version": 1,
            "task_id": TASK_ID,
            "checkpoint": str(checkpoint),
            "artifact": report,
            "base_model_source": "mistralai/Mistral-7B-Instruct-v0.2",
            "base_model_revision": BASE_MODEL_REVISION,
            "rewardbench_source": REWARDBENCH_SOURCE,
            "rewardbench_revision": REWARDBENCH_REVISION,
            "rewardbench_files": provenance["files"],
            "rewardbench_subset_rows": provenance["subset_rows"],
            "train_data_revision": TRAIN_DATA_REVISION,
            "upstream_revision": UPSTREAM_REVISION,
            "max_length": grade.EVAL_MAX_LENGTH,
            "eval_batch_pairs": grade.EVAL_BATCH_PAIRS,
            "proxy_rows": grade.PROXY_ROWS,
            "section_aggregation": "example-weighted subset accuracy within section",
            "overall_aggregation": "unweighted mean of four section scores",
            "example_counts": grade.EXAMPLE_COUNTS,
            "upweighted_subsets": grade.UPWEIGHTED,
            "decontamination": decontamination,
            "input_contract": "checkpoint_only",
            "image_digest": os.environ.get("IMAGE_DIGEST"),
            "gpu": gpu,
            "offline": True,
        },
    )

    started = time.monotonic()
    model, tokenizer = artifact.load_model(checkpoint, base_model)
    loaded = time.monotonic()
    print(f"final: model loaded in {loaded - started:.1f} s", flush=True)
    scored = grade.score_pairs(model, tokenizer, final_rows, progress_every=32)
    split = split_report(scored, proxy_ids)

    (output / "metrics.jsonl").write_text(
        "".join(
            json.dumps({**row, "in_proxy": row["id"] in proxy_ids}, sort_keys=True) + "\n"
            for row in scored
        ),
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "passed",
        "benchmark": "RewardBench v1 filtered split",
        "metric": METRIC,
        "direction": "maximize",
        # `score` and `metrics[METRIC]` are score(F). The reward is the benchmark's
        # own definition over its own rows; the split below is a diagnostic.
        "score": split["score_final"],
        "metrics": {
            METRIC: split["score_final"],
            "rewardbench_v1_score_proxy_rows": split["score_proxy"],
            "rewardbench_v1_score_held_out": split["score_held_out"],
            "overfitting_gap": split["overfitting_gap"],
        },
        "stderr": split["final"]["stderr"],
        "stderr_kind": split["final"]["stderr_kind"],
        "n": split["rows_final"],
        "correct": split["final"]["correct"],
        "section_scores": split["final"]["section_scores"],
        "subset_accuracy": split["final"]["subset_accuracy"],
        "overall_pair_accuracy": split["final"]["overall_pair_accuracy"],
        "mean_reward_margin": split["final"]["mean_reward_margin"],
        "ties": split["final"]["ties"],
        "split": split,
        "artifact": report,
        "decontamination": decontamination,
        "model_load_seconds": loaded - started,
        "wall_seconds": time.monotonic() - started,
        "gpu": gpu,
        "gpu_memory_peak_bytes": peak_memory_bytes(torch),
        "offline": True,
    }
    atomic_json(output / "summary.json", summary)
    write_reward(split["score_final"], reward_path)
    return summary


def synthetic_rows() -> list[dict[str, Any]]:
    """Full-size scored rows over the real subset structure, no model."""

    return [
        {
            "id": f"{subset}-{index}",
            "subset": subset,
            "chosen_reward": 1.0,
            "rejected_reward": 0.0,
            "reward_margin": 1.0,
            "correct": float((index * 7 + len(subset)) % 3 != 0),
        }
        for subset, count in grade.expected_row_counts().items()
        for index in range(count)
    ]


def mock(output: Path, reward_path: Path) -> dict[str, Any]:
    """Exercise the whole output shape without a GPU or the question set."""

    rows = synthetic_rows()
    proxy_ids = grade.select_proxy_ids(rows)
    split = split_report(rows, proxy_ids)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.jsonl").write_text(
        "".join(
            json.dumps({**row, "in_proxy": row["id"] in proxy_ids}, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "passed",
        "metric": METRIC,
        "direction": "maximize",
        "score": split["score_final"],
        "metrics": {
            METRIC: split["score_final"],
            "rewardbench_v1_score_proxy_rows": split["score_proxy"],
            "rewardbench_v1_score_held_out": split["score_held_out"],
            "overfitting_gap": split["overfitting_gap"],
        },
        "stderr": split["final"]["stderr"],
        "n": split["rows_final"],
        "correct": split["final"]["correct"],
        "section_scores": split["final"]["section_scores"],
        "split": split,
        "mock": True,
    }
    atomic_json(output / "summary.json", summary)
    write_reward(split["score_final"], reward_path)
    return summary


def smoke() -> None:
    rows = synthetic_rows()
    if len(rows) != grade.FINAL_ROWS:
        raise RuntimeError(f"synthetic set has {len(rows)} rows")
    split = split_report(rows, grade.select_proxy_ids(rows))
    if split["rows_proxy"] + split["rows_held_out"] != grade.FINAL_ROWS:
        raise RuntimeError("the split does not partition the final")
    if abs(split["overlap_fraction"] - grade.PROXY_ROWS / grade.FINAL_ROWS) > 1e-12:
        raise RuntimeError(f"unexpected overlap: {split['overlap_fraction']}")
    if split["overlap_fraction"] > 0.5:
        raise RuntimeError("proxy/final overlap is over the 50% ceiling")
    for key in ("score_final", "score_proxy", "score_held_out"):
        if not 0.0 <= split[key] <= 100.0:
            raise RuntimeError(f"{key} out of range: {split[key]}")
    if split["overfitting_gap_stderr"] <= 0.0:
        raise RuntimeError("the gap must carry an error bar")

    perfect = [{**row, "correct": 1.0} for row in rows]
    if abs(split_report(perfect, grade.select_proxy_ids(rows))["score_final"] - 100.0) > 1e-9:
        raise RuntimeError("an all-correct set must score 100")
    print(
        json.dumps(
            {
                "final_eval_smoke": "passed",
                "n": split["rows_final"],
                "proxy_rows": split["rows_proxy"],
                "held_out_rows": split["rows_held_out"],
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--assets", type=Path, default=Path("/assets"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reward-path", type=Path, default=REWARD_PATH)
    parser.add_argument("--mock", action="store_true", help="synthetic rows, no GPU")
    parser.add_argument("--smoke", action="store_true", help="self-check and exit")
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
    with exclusive_output("btrm_final_eval", resolved_output):
        result = evaluate(
            args.checkpoint.resolve(),
            args.assets.resolve(),
            resolved_output,
            args.reward_path,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"BT-RM final failed: {exc}", file=sys.stderr)
        raise
