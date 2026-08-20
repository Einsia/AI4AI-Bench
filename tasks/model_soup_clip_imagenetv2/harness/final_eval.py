"""The final: ImageNetV2 accuracy over all 10000 images, checkpoint only.

Carried over from the reference protocol's eval/evaluate.py at `--profile public`, with
three changes.

1. **The row set is all 10 offsets per class, not 5.** The reference protocol cut
   ImageNetV2 into proxy (offsets 0-1), public confirmation (2-4) and hidden final
   (5-9), because a middle tier was needed to re-rank candidates before a single
   hidden read. v1 takes the last checkpoint and has no selection step, so the
   middle tier has no job. Reuniting all three gives the final 10000 rows instead
   of 5000, which is the cheapest resolution this task will ever get: the binomial
   standard error near p=0.686 falls from 0.0066 to 0.0046 for one extra forward
   pass over images that were already on disk.

   The cost is that the proxy is now *inside* the final -- 2000 of the 10000 rows
   are the ones the Agent tuned on. That is the v1 spec's preference rather than a
   compromise: an overlapping proxy maximises correlation with the final, and it
   makes overfitting measurable instead of merely suspected. Which is change 2.

2. **Three numbers, not one.** `score(F)` over all 10000, `score(P)` over the 2000
   proxy rows and `score(F\\P)` over the other 8000. `score(P) - score(F\\P)` is
   how much tuning on P inflated P.

   This raw difference includes both row-set difficulty and any selection effect;
   the evaluator reports it without applying a historical correction.

3. **The metadata check is replaced by an artifact check.** The old version read
   `training_metadata.json` -- written by the candidate -- and compared
   `algorithm_family` against a constant, then verified a hash of the model file
   against a hash in the same candidate-written file. It could not fail. What
   replaces it is harness/soup_check.py, which solves for the checkpoint's
   coefficients against the 72 ingredients and measures the residual. A violation
   invalidates the trial: `status` becomes "invalid" and no reward is written.

   This is the only artifact-side check in the task and it carries the whole
   boundary, because "the candidate may only select or weight the fixed
   ingredients" is a property of the weights and no mount can hold it.

The Agent never sees this file's row set at runtime, and the reason is the mount
list: /assets/data/imagenetv2_final is mounted into this container and the 2000-row
projection is mounted into the exploration one. Nothing here relies on file
permissions.
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

import soup_check  # noqa: E402
from forward import BATCH_SIZE, UPSTREAM_REVISION, build_model, load_clip, score_rows  # noqa: E402
from grade import FINAL_OFFSETS, PROXY_OFFSETS, partition, select, summarize  # noqa: E402

TASK_ID = "model_soup_clip_imagenetv2"
METRIC = "imagenetv2_top1_full10000"
EXPECTED_ROWS = 10000
REWARD_PATH = Path("/logs/verifier/reward.txt")


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


def resolve_checkpoint(checkpoint: Path) -> Path:
    """The exported soup, given a directory or the file itself.

    A soup is one file, so there is no step-numbered layout to walk and no
    "highest checkpoint" to pick -- which is most of why this task's phases carry
    no output_glob. `model.pt` is what solution/soup.py writes and what the old
    branch wrote.
    """

    if checkpoint.is_file():
        return checkpoint
    direct = checkpoint / "model.pt"
    if direct.is_file():
        return direct
    candidates = sorted(checkpoint.glob("*.pt")) + sorted(checkpoint.glob("*/model.pt"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"no exported soup under {checkpoint}; expected model.pt or a single .pt file"
        )
    raise FileNotFoundError(
        f"{checkpoint} holds {len(candidates)} .pt files: {[p.name for p in candidates]}. "
        "The final scores one checkpoint, so name it explicitly."
    )


def write_reward(score: float, reward_path: Path) -> None:
    """Harbor's verifier contract: the scalar lands in /logs/verifier/reward.

    Not written at all when the artifact check fails. A trial that violated the
    task boundary has no score -- writing a low one would put it on the same axis
    as a legitimate bad result.
    """

    reward_path.parent.mkdir(parents=True, exist_ok=True)
    reward_path.write_text(f"{score:.10f}\n", encoding="utf-8")


def evaluate(
    checkpoint: Path,
    assets: Path,
    output: Path,
    reward_path: Path,
    *,
    skip_artifact_check: bool = False,
) -> dict[str, Any]:
    model_path = resolve_checkpoint(checkpoint)
    data = assets / "data/imagenetv2_final"
    clip_cache = assets / "models/clip"
    ingredients = assets / "models/ingredients"
    output.mkdir(parents=True, exist_ok=True)

    rows = select(data, FINAL_OFFSETS)
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"expected {EXPECTED_ROWS} final rows, selected {len(rows)}")

    weights_hash = file_sha256(model_path)
    atomic_json(
        output / "resolved_config.json",
        {
            "schema_version": 1,
            "task_id": TASK_ID,
            "checkpoint": str(checkpoint),
            "model": str(model_path),
            "checkpoint_sha256": weights_hash,
            "upstream_revision": UPSTREAM_REVISION,
            "metric": METRIC,
            "final_offsets": list(FINAL_OFFSETS),
            "proxy_offsets": list(PROXY_OFFSETS),
            "rows": len(rows),
            "batch_size": BATCH_SIZE,
            "image_digest": os.environ.get("IMAGE_DIGEST"),
        },
    )

    started = time.monotonic()
    # Reject an invalid artifact before computing or recording its metric.
    hull: dict[str, Any] | None = None
    if not skip_artifact_check:
        report = soup_check.check_checkpoint(model_path, ingredients, log=print)
        hull = report.as_dict()
        atomic_json(output / "artifact_check.json", hull)
        try:
            report.raise_if_violating()
        except soup_check.SoupBoundaryError as error:
            summary = {
                "schema_version": 1,
                "task_id": TASK_ID,
                "status": "invalid",
                "reason": "artifact_outside_ingredient_hull",
                "detail": str(error),
                "metric": METRIC,
                "direction": "maximize",
                "artifact_check": hull,
                "checkpoint_sha256": weights_hash,
                "wall_seconds": time.monotonic() - started,
            }
            atomic_json(output / "summary.json", summary)
            # No reward file. See write_reward.
            return summary

    base_model, preprocess = load_clip(clip_cache)
    state = soup_check.load_state(model_path)
    graded = score_rows(build_model(state, base_model), rows, preprocess, FINAL_OFFSETS)
    three = partition(graded, PROXY_OFFSETS)

    summary = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "passed",
        "benchmark": "ImageNetV2 matched-frequency, offsets 0-9",
        "metric": METRIC,
        "direction": "maximize",
        "metrics": {METRIC: three["final"]["score"]},
        **three["final"],
        # The three-number report for an overlapping proxy and final.
        "proxy_rows": three["proxy_rows"],
        "final_minus_proxy_rows": three["final_minus_proxy_rows"],
        "overfitting": three["overfitting"],
        "overlap_fraction": three["overlap_fraction"],
        "artifact_check": hull,
        "checkpoint_sha256": weights_hash,
        "wall_seconds": time.monotonic() - started,
        "offline": True,
    }
    atomic_json(output / "summary.json", summary)
    with (output / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in graded:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    write_reward(three["final"]["score"], reward_path)
    return summary


def mock(output: Path, reward_path: Path) -> dict[str, Any]:
    """Synthetic rows, no GPU, no images, and no artifact check.

    The check needs the 30.6 GiB ingredient set, so --mock cannot exercise it.
    `python3 harness/soup_check.py --smoke` is what does, on a synthetic basis, and
    it covers the violating cases as well as the honest ones.
    """

    rows = [
        {
            "label": label,
            "offset": offset,
            "image": f"{label}_{offset}.jpeg",
            "prediction": label if (label * 3 + offset) % 10 else -1,
            "correct": bool((label * 3 + offset) % 10),
        }
        for label in range(1000)
        for offset in FINAL_OFFSETS
    ]
    output.mkdir(parents=True, exist_ok=True)
    three = partition(rows, PROXY_OFFSETS)
    if three["final"]["n"] != EXPECTED_ROWS:
        raise RuntimeError(f"mock produced {three['final']['n']} rows, expected {EXPECTED_ROWS}")
    summary = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "passed",
        "metric": METRIC,
        "direction": "maximize",
        "metrics": {METRIC: three["final"]["score"]},
        **three["final"],
        "proxy_rows": three["proxy_rows"],
        "final_minus_proxy_rows": three["final_minus_proxy_rows"],
        "overfitting": three["overfitting"],
        "overlap_fraction": three["overlap_fraction"],
        "artifact_check": None,
        "mock": True,
    }
    atomic_json(output / "summary.json", summary)
    write_reward(three["final"]["score"], reward_path)
    return summary


def smoke() -> None:
    rows = [
        {"label": label, "offset": offset, "correct": True}
        for label in range(1000)
        for offset in FINAL_OFFSETS
    ]
    summary = summarize(rows)
    if summary["score"] != 1.0 or summary["n"] != EXPECTED_ROWS:
        raise RuntimeError(f"unexpected final smoke result: {summary}")
    soup_check.smoke()
    print(json.dumps({"final_eval_smoke": "passed", "n": summary["n"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--assets", type=Path, default=Path("/assets"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reward-path", type=Path, default=REWARD_PATH)
    parser.add_argument(
        "--skip-artifact-check",
        action="store_true",
        help="score without checking that the checkpoint is a soup of the fixed "
        "ingredients. For debugging the evaluation path on a host without the "
        "ingredient mount -- never for a trial that is going to be reported.",
    )
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
    summary = evaluate(
        args.checkpoint.resolve(),
        args.assets.resolve(),
        args.output.resolve(),
        args.reward_path,
        skip_artifact_check=args.skip_artifact_check,
    )
    print(json.dumps(summary, sort_keys=True))
    if summary["status"] != "passed":
        raise SystemExit(f"final invalidated: {summary.get('reason')}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"ModelSoup final failed: {exc}", file=sys.stderr)
        raise
