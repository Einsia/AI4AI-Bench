"""Hidden final: strict IFEval on all 413 sealed rows, checkpoint only.

The evaluator re-derives the fixed tiers from the pinned 541-row source, resolves
the submitted merged model or adapter, and loads it against the exact merged policy
start. Candidate metadata is not trusted as a gate. Fixed model/data mounts, patch
screening and the method boundary in instruction.md define the task contract.

## score(F), score(P), score(F\\P)

The proxy is the first 128 of these 413 rows, so 31.0% of the final moves with
whatever the candidate tuned against. Rather than pretend otherwise:

    score(F)      all 413 rows -- the headline, and the reward
    score(P)      the 128 proxy rows, the ones exploration could see
    score(F\\P)    the remaining 285, which no exploration container mounts

`score(P) - score(F\\P)` is the overfitting measurement. It is a diagnostic, not a
penalty: nothing is subtracted from the reward, because the reward is the
benchmark's own definition over its own rows.

The reported standard errors describe row-level binomial uncertainty only. They do
not include training-seed or replay variance and are not paired confidence
intervals. The full row payload is retained for matched comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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
METRIC = grade.METRIC
IFEVAL_SOURCE = "google/IFEval"
IFEVAL_REVISION = "966cd89545d6b6acfd7638bc708b98261ca58e84"
POLICY_START_SOURCE = "mistralai/Mistral-7B-v0.1 + alignment-handbook/zephyr-7b-sft-qlora"
# The base revision and the SFT adapter revision, joined -- the frozen start is a
# merge of the two, so neither alone identifies it.
POLICY_START_REVISION = (
    "27d67f1b5f57dc0953326b2601d68371d40ea8da+156bec577ff12a65236cfc90860dcc61e96c6fd6"
)
TRAIN_DATA_REVISION = "3949bf5f8c17c394422ccfab0c31ea9c20bdeb85"
UPSTREAM_REVISION = "1de1fc996972aa76b7d40c64c07b66dec8b6976a"
# `reward`, not `reward.txt`: orchestrator/runner.py:report_reward reads
# `<logs>/verifier/reward`, so OPD's `reward.txt` writes a file the hook never finds
# and its reward line is silently absent from every score run.
REWARD_PATH = Path("/logs/verifier/reward")
SOURCE_FILENAME = "ifeval_input_data.jsonl"


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


def load_source(root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Read all 541 IFEval rows and cut the tiers, refusing content drift.

    The whole source is mounted here rather than a pre-sliced 413, so the split is
    recomputed in the container that scores it. That is what makes the row arithmetic
    checkable: 541 = 128 retired + 413 final, and 413 = 128 proxy + 285 held out, all
    from one file and one ordering.

    `grade.require_legacy_ordering` is the anchor. It reproduces the
    `public_keys_sha256` recorded in the reference protocol's `assets.lock.yaml`, which
    is the only committed value that ties today's rows to the runs the recorded
    baselines came from.

    The file's own sha256 is reported and, if `ifeval-files.json` sits beside the
    data, verified. The reference protocol recorded a `source_sha256` only inside a
    host-side projection manifest that was never committed, so there is no pinned
    value to compare against yet: this reports what it saw so a first real run can
    pin it.
    """

    path = root / SOURCE_FILENAME
    if not path.is_file():
        candidates = sorted(root.glob("*.jsonl"))
        if len(candidates) != 1:
            raise FileNotFoundError(f"no {SOURCE_FILENAME} under {root}")
        path = candidates[0]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    split = grade.split_source(rows)
    ordering_digest = grade.require_legacy_ordering(split["legacy_public"])

    digest = file_sha256(path)
    expected_path = root / "ifeval-files.json"
    if expected_path.is_file():
        expected = json.loads(expected_path.read_text(encoding="utf-8")).get(path.name)
        if expected and expected != digest:
            raise ValueError(
                f"pinned IFEval file hash mismatch for {path.name}: expected {expected}, "
                f"actual {digest}"
            )
    provenance = {
        "file": path.name,
        "file_sha256": digest,
        "source_rows": len(rows),
        "legacy_public_rows": len(split["legacy_public"]),
        "legacy_public_keys_sha256": ordering_digest,
        "final_rows": len(split["final"]),
        "proxy_rows": len(split["proxy"]),
        "held_out_rows": len(split["held_out"]),
    }
    return split, provenance


def split_report(
    scored: list[dict[str, Any]], proxy_keys: set[str]
) -> dict[str, Any]:
    """score(F), score(P), score(F\\P), and the difference with its error bar."""

    inside = [row for row in scored if str(row["key"]) in proxy_keys]
    outside = [row for row in scored if str(row["key"]) not in proxy_keys]
    if len(inside) != grade.PROXY_ROWS:
        raise ValueError(
            f"{len(inside)} of the scored rows are in the proxy, expected "
            f"{grade.PROXY_ROWS}. The proxy selector and the final's row set disagree."
        )
    if len(outside) != grade.HELD_OUT_ROWS:
        raise ValueError(
            f"{len(outside)} rows are outside the proxy, expected {grade.HELD_OUT_ROWS}"
        )

    whole = grade.summarize(scored)
    proxy = grade.summarize(inside)
    held_out = grade.summarize(outside)
    gap = proxy["score"] - held_out["score"]
    return {
        "score_final": whole["score"],
        "score_proxy": proxy["score"],
        "score_held_out": held_out["score"],
        "overfitting_gap": gap,
        # The two row sets are disjoint, so the variances add.
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
    *,
    gpus: int = 1,
) -> dict[str, Any]:
    if gpus != 1:
        raise ValueError("this task protocol uses exactly one visible GPU per evaluation")
    reference = assets / "models/policy_start"
    # Shape resolution runs before anything is loaded onto a device, so an unusable
    # submission costs a directory listing rather than a model load and 413 decodes.
    report = checkpoint_module.describe(checkpoint, reference)
    split, provenance = load_source(assets / "data/ifeval_final")
    final_rows = split["final"]
    proxy_keys = {grade.canonical_key(row) for row in split["proxy"]}
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

    gpu = cuda_telemetry(torch)
    torch.cuda.reset_peak_memory_stats()

    atomic_json(
        output / "resolved_config.json",
        {
            "schema_version": 1,
            "task_id": TASK_ID,
            "checkpoint": str(checkpoint),
            "artifact": report,
            "policy_start_source": POLICY_START_SOURCE,
            "policy_start_revision": POLICY_START_REVISION,
            "ifeval_source": IFEVAL_SOURCE,
            "ifeval_revision": IFEVAL_REVISION,
            "ifeval_provenance": provenance,
            "train_data_revision": TRAIN_DATA_REVISION,
            "upstream_revision": UPSTREAM_REVISION,
            "partition": "sha256_key_order: final = after_first128, proxy = first128_of_final",
            "protocol": grade.PROTOCOL,
            "max_new_tokens": grade.MAX_NEW_TOKENS,
            "do_sample": grade.DO_SAMPLE,
            "chat_template": True,
            "fresh_metric_instance_per_row": True,
            "tokenization_protocol": grade.TOKENIZATION_PROTOCOL,
            "gpus": gpus,
            "input_contract": "checkpoint_only",
            "image_digest": os.environ.get("IMAGE_DIGEST"),
            "offline": True,
        },
    )

    started = time.monotonic()
    scored = generate.run(checkpoint, reference, final_rows, output, gpus=gpus)
    split_result = split_report(scored, proxy_keys)

    (output / "metrics.jsonl").write_text(
        "".join(
            json.dumps({**row, "in_proxy": str(row["key"]) in proxy_keys}, sort_keys=True) + "\n"
            for row in scored
        ),
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "passed",
        "benchmark": "Google IFEval, sealed 413-row projection",
        "metric": METRIC,
        "direction": "maximize",
        # `score` and `metrics[METRIC]` are score(F). The reward is the benchmark's
        # own definition over its own rows; the split below is a diagnostic.
        "score": split_result["score_final"],
        "metrics": {
            **split_result["final"]["metrics"],
            "ifeval_prompt_level_strict_accuracy_proxy_rows": split_result["score_proxy"],
            "ifeval_prompt_level_strict_accuracy_held_out": split_result["score_held_out"],
            "overfitting_gap": split_result["overfitting_gap"],
        },
        "stderr": split_result["final"]["stderr"],
        "n": split_result["rows_final"],
        "correct": split_result["final"]["correct"],
        "instructions": split_result["final"]["instructions"],
        "mean_generated_tokens": split_result["final"]["mean_generated_tokens"],
        "length_clipped": split_result["final"]["length_clipped"],
        "split": split_result,
        "artifact": report,
        "ifeval_provenance": provenance,
        "protocol": grade.PROTOCOL,
        "wall_seconds": time.monotonic() - started,
        "stderr_kind": "binomial_descriptive_not_seed_or_paired_uncertainty",
        "gpu": gpu,
        "gpu_memory_peak_bytes": peak_memory_bytes(torch),
        "offline": True,
    }
    atomic_json(output / "summary.json", summary)
    write_reward(split_result["score_final"], reward_path)
    return summary


def synthetic_split() -> tuple[list[dict[str, Any]], set[str]]:
    """Full-size scored rows over the real split structure, no model."""

    split = grade.split_source(grade.synthetic_source())
    proxy_keys = {grade.canonical_key(row) for row in split["proxy"]}
    scored = [
        {
            "key": grade.canonical_key(row),
            "scores": grade.synthetic_scores(index),
            "generated_tokens": 128 + index % 7,
        }
        for index, row in enumerate(split["final"])
    ]
    return scored, proxy_keys


def mock(output: Path, reward_path: Path) -> dict[str, Any]:
    """Exercise the whole output shape without a GPU or the question set."""

    scored, proxy_keys = synthetic_split()
    split_result = split_report(scored, proxy_keys)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.jsonl").write_text(
        "".join(
            json.dumps({**row, "in_proxy": str(row["key"]) in proxy_keys}, sort_keys=True) + "\n"
            for row in scored
        ),
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "passed",
        "metric": METRIC,
        "direction": "maximize",
        "score": split_result["score_final"],
        "metrics": {
            **split_result["final"]["metrics"],
            "ifeval_prompt_level_strict_accuracy_proxy_rows": split_result["score_proxy"],
            "ifeval_prompt_level_strict_accuracy_held_out": split_result["score_held_out"],
            "overfitting_gap": split_result["overfitting_gap"],
        },
        "stderr": split_result["final"]["stderr"],
        "n": split_result["rows_final"],
        "correct": split_result["final"]["correct"],
        "split": split_result,
        "mock": True,
    }
    atomic_json(output / "summary.json", summary)
    write_reward(split_result["score_final"], reward_path)
    return summary


def smoke() -> None:
    scored, proxy_keys = synthetic_split()
    if len(scored) != grade.FINAL_ROWS:
        raise RuntimeError(f"synthetic final has {len(scored)} rows")
    split_result = split_report(scored, proxy_keys)
    if split_result["rows_proxy"] + split_result["rows_held_out"] != grade.FINAL_ROWS:
        raise RuntimeError("the split does not partition the final")
    if abs(split_result["overlap_fraction"] - grade.PROXY_ROWS / grade.FINAL_ROWS) > 1e-12:
        raise RuntimeError(f"unexpected overlap: {split_result['overlap_fraction']}")
    if split_result["overlap_fraction"] > 0.5:
        raise RuntimeError("proxy/final overlap is over the 50% ceiling")
    for key in ("score_final", "score_proxy", "score_held_out"):
        if not 0.0 <= split_result[key] <= 1.0:
            raise RuntimeError(f"{key} out of range: {split_result[key]}")
    if split_result["overfitting_gap_stderr"] <= 0.0:
        raise RuntimeError("the gap must carry an error bar")

    perfect = [{**row, "scores": grade.synthetic_scores(1)} for row in scored]
    if abs(split_report(perfect, proxy_keys)["score_final"] - 1.0) > 1e-12:
        raise RuntimeError("an all-correct set must score 1.0")
    print(
        json.dumps(
            {
                "final_eval_smoke": "passed",
                "n": split_result["rows_final"],
                "proxy_rows": split_result["rows_proxy"],
                "held_out_rows": split_result["rows_held_out"],
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--assets", type=Path, default=Path("/assets"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reward-path", type=Path, default=REWARD_PATH)
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="devices to shard the 413 rows over. Default 1, matching the training "
        "phases. More is only faster -- each row is generated alone, so no row's "
        "output depends on the shard layout.",
    )
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
    with exclusive_output("dpo_final_eval"):
        print(
            json.dumps(
                evaluate(
                    args.checkpoint.resolve(),
                    args.assets.resolve(),
                    args.output.resolve(),
                    args.reward_path,
                    gpus=args.gpus,
                ),
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"DPO final failed: {exc}", file=sys.stderr)
        raise
