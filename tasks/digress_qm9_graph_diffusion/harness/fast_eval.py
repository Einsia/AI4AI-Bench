"""Sample and score 2,000 molecules with the pinned DiGress evaluator.

This is an independent unseeded draw at a smaller count than formal evaluation because
the pinned trainer does not consume its seed setting. Inspect validation NLL as well as
the generative score, and do not treat small differences as ordered without repeats.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from final_eval import (  # noqa: E402
    FINAL_SEED,
    PROXY_SAMPLES,
    atomic_json,
    file_sha256,
    find_smiles_dump,
    inspect_artifact,
    load_train_smiles,
    prepare_upstream,
    read_smiles_dump,
    resolve_checkpoint,
    run_upstream,
)
from grade import (  # noqa: E402
    REQUIRED_METRICS,
    composite,
    molecule_metrics,
    observed_sample_count,
    parse_metrics,
)

TASK_ID = "digress_qm9_graph_diffusion"
# The metric an Agent watches. Direction max -- and NOT the metric that ranks the
# task, which is NLL, minimised. Both are reported below; see task.toml [metadata].
METRIC = "validity_uniqueness_novelty"
DEFAULT_SAMPLES = PROXY_SAMPLES
DEFAULT_SEED = FINAL_SEED


def evaluate(
    checkpoint: Path,
    data: Path,
    out: Path,
    samples: int,
    seed: int,
    smiles_file: Path | None = None,
) -> dict[str, Any]:
    model = resolve_checkpoint(checkpoint)
    if not data.is_dir():
        raise FileNotFoundError(data)
    work = out.parent / (out.stem + "-work")
    work.mkdir(parents=True, exist_ok=True)

    # Record the same structural diagnostic as final scoring. The upstream load below
    # determines whether the checkpoint is compatible.
    structure = inspect_artifact(model)

    upstream = prepare_upstream(work)
    runtime_checkpoint = work / "work/checkpoint/last.ckpt"
    runtime_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copy2(str(model), str(runtime_checkpoint))

    started = time.monotonic()
    # Every invocation owns its work and caches. Reusing a warm cache would make an
    # evaluation's resource receipt depend on whichever run happened before it.
    log_text = run_upstream(
        upstream, runtime_checkpoint, data, work, samples, seed, "fast",
        cache_root=work,
    )
    metrics = parse_metrics(log_text)
    missing = [name for name in REQUIRED_METRICS if metrics.get(name) is None]
    if missing:
        raise RuntimeError(
            f"DiGress emitted incomplete molecule metrics, missing {missing}: {metrics}"
        )
    observed = observed_sample_count(log_text)
    if observed != samples:
        raise RuntimeError(
            f"sample-count mismatch: declared {samples}, upstream generated {observed}"
        )

    # Recompute from the dump when there is one as an auxiliary diagnostic. The
    # upstream stdout remains authoritative; this path never uses a consistency
    # comparison to reject or change the upstream score.
    # Same exclusion as the final, and for the same reason: `upstream` is a copy of the
    # pinned repo under `work`, and it ships its own generated_samples/*.txt. Without this
    # the recomputed cross-check reads the paper's MOSES dump -- which the final treats as
    # fatal and this treats as a stderr warning, so here it degraded the `recomputed` field
    # silently while the headline numbers still came from upstream's stdout.
    dump = smiles_file or find_smiles_dump(
        [runtime_checkpoint.parent, work, work / "hydra"],
        exclude=[upstream],
        preferred=[runtime_checkpoint.parent / "final_smiles.txt"],
    )
    recomputed: dict[str, float | None] | None = None
    if dump is not None:
        molecules = read_smiles_dump(dump)[:samples]
        if len(molecules) == samples:
            recomputed = molecule_metrics(molecules, load_train_smiles(data))

    elapsed = time.monotonic() - started
    payload = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "metric": METRIC,
        "direction": "max",
        "score": composite(metrics),
        # The metric that actually ranks the task, reported beside the one above.
        "ranking_metric": "nll",
        "ranking_direction": "min",
        "nll_split": "validation",
        "nll": metrics["nll"],
        "metrics": dict(metrics),
        "samples": samples,
        "observed_samples": observed,
        "declared_seed": seed,
        "seed_effective": False,
        "checkpoint": str(checkpoint),
        "model": str(model),
        "checkpoint_sha256": file_sha256(model),
        "artifact_description": structure,
        "recomputed": recomputed,
        "upstream_agreement": "not_applicable",
        "seconds": elapsed,
        "proxy_is_prefix_of_final": False,
        "final_samples": 10000,
        "overlap_fraction": None,
        "sampling_relation_to_final": "independent unseeded draw",
    }
    atomic_json(out, payload)
    return payload


def mock(out: Path, samples: int, seed: int) -> dict[str, Any]:
    def identity(value: str) -> str | None:
        return value or None

    molecules = [f"C{index % 900}" for index in range(samples)]
    metrics = molecule_metrics(molecules, ["C0"], canonicalize=identity)
    payload = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "metric": METRIC,
        "direction": "max",
        "mock": True,
        "score": composite(metrics),
        "ranking_metric": "nll",
        "ranking_direction": "min",
        "nll_split": "validation",
        "nll": None,
        "metrics": metrics,
        "samples": samples,
        "declared_seed": seed,
        "seed_effective": False,
        "seconds": 0.0,
        "proxy_is_prefix_of_final": False,
        "overlap_fraction": None,
        "sampling_relation_to_final": "independent unseeded draw",
    }
    atomic_json(out, payload)
    return payload


def smoke() -> None:
    payload = mock(Path("/tmp/digress-fast-eval-smoke.json"), DEFAULT_SAMPLES, DEFAULT_SEED)
    if payload["metrics"]["n"] != float(DEFAULT_SAMPLES):
        raise RuntimeError(f"unexpected sample count: {payload}")
    if not 0.0 < payload["score"] <= 1.0:
        raise RuntimeError(f"the composite must be a fraction: {payload}")
    if payload["proxy_is_prefix_of_final"] or payload["seed_effective"]:
        raise RuntimeError(f"the pinned tree's seed is inert: {payload}")
    print(json.dumps({"fast_eval_smoke": "passed", "n": payload["samples"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data", type=Path, default=Path("/assets/data/qm9_no_h"))
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help="molecules to draw. The default is the public fast-eval count; every "
        "invocation is an independent draw.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="declared protocol label retained for compatibility. The pinned DiGress "
        "tree does not read train.seed, so changing it does not control sampling.",
    )
    parser.add_argument("--smiles-file", type=Path, default=None)
    parser.add_argument("--mock", action="store_true", help="synthetic molecules, no GPU")
    parser.add_argument("--smoke", action="store_true", help="self-check and exit")
    args = parser.parse_args()

    if args.smoke:
        smoke()
        return
    if args.out is None:
        parser.error("--out is required")
    if args.mock:
        print(json.dumps(mock(args.out, args.samples, args.seed), sort_keys=True))
        return
    if args.checkpoint is None:
        parser.error("--checkpoint is required outside --mock/--smoke")
    payload = evaluate(
        args.checkpoint.resolve(),
        args.data.resolve(),
        args.out.resolve(),
        args.samples,
        args.seed,
        args.smiles_file,
    )
    print(
        json.dumps(
            {
                "score": payload["score"],
                "nll": payload["nll"],
                "validity": payload["metrics"]["validity"],
                "uniqueness": payload["metrics"]["uniqueness"],
                "novelty": payload["metrics"]["novelty"],
                "n": payload["samples"],
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
