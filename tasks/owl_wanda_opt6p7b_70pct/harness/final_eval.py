"""Hidden final: WikiText2 raw *test* perplexity, checkpoint only.

Carried over from the reference protocol's eval/evaluate.py. Four changes:

1. **The split is the test split.** The reference protocol's two profiles, `proxy` and
   `public`, both read
   `wikitext-2-raw-v1/validation-00000-of-00001.parquet` with the same 128
   calibration samples and differed only in an error message -- they were one
   measurement reported twice. v1 has two tiers and they are different text:
   fast_eval reads validation, this reads test. Overlap 0%.

2. **`load_metadata` is gone.** It read task_id, sparsity_type, sparsity_ratio and
   calibration_samples out of `training_metadata.json` -- a file the pruning script
   writes. A candidate that wanted to claim 70% only had to write 0.7 into it. The
   check that survives is `decoder_sparsity`, which counts zeros in the tensors about
   to be scored. The old recipe-side copy of the same check
   (environment/check_candidate.py, `algorithm["sparsity_ratio"] == 0.7` read from the
   candidate's recipe.toml) is also gone: v1's boundary is the mount list plus this.

3. **The sparsity gate invalidates the trial** rather than raising an anonymous
   RuntimeError. Same window, [0.699, 0.701], and it now writes a summary saying
   which side it failed on before exiting non-zero, so a violation is diagnosable
   without the container.

4. Scoring goes through harness/grade.py, the same entry point fast_eval uses.

The Agent never sees this file's input at runtime, and the reason is the mount list
rather than any file permission: /assets/data/wikitext2/test is mounted into this
container and not into the exploration one. Nothing here relies on the asset tree
being unreadable to the account running the benchmark.

WHAT IS NOT HERE, AND WHY. The spec asks a final whose proxy overlaps it to report
score(F), score(P) and score(F\\P), because score(P) - score(F\\P) is the overfitting
measurement. WikiText2's validation and test splits are disjoint by construction, so
there are no proxy blocks inside the final and that difference is not defined. The
fields are still emitted, as nulls with a stated reason, so a consumer parsing the
shape sees a deliberate absence rather than a missing key.
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


def _redirect_caches(root: str = "/out/final-eval-cache") -> None:
    """Move library caches off /tmp before anything imports torch.

    /tmp is a 256 MiB tmpfs -- ContainerSpec.tmpfs_tmp_size, applied to every phase.
    transformers and torch both write there by default, and this container's root
    filesystem is read-only, so an unredirected cache fails on a path that has
    nothing to do with the failure.
    """

    for name, relative in (
        ("TMPDIR", "tmp"),
        ("HF_HOME", "tmp/hf"),
        ("TRANSFORMERS_CACHE", "tmp/hf/transformers"),
        ("TORCH_HOME", "tmp/torch"),
        ("TRITON_CACHE_DIR", "tmp/triton"),
        ("XDG_CACHE_HOME", "tmp/cache"),
    ):
        path = Path(os.environ.get(name, f"{root}/{relative}"))
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            # A read-only or absent /out means --mock or --smoke, which never reaches
            # a model loader.
            continue
        os.environ[name] = str(path)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


_redirect_caches()

sys.path.insert(0, str(Path(__file__).resolve().parent))

from grade import (  # noqa: E402
    SEQUENCE_LENGTH,
    SPARSITY_WINDOW,
    block_nlls,
    decoder_sparsity,
    load_model,
    load_text,
    perplexity,
    sparsity_in_window,
    summarize,
)

TASK_ID = "owl_wanda_opt6p7b_70pct"
METRIC = "wikitext2_test_perplexity"
DIRECTION = "minimize"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKITEXT_SOURCE = "Salesforce/wikitext"
# The WikiText2 raw test split is 140 complete 2048-token blocks under the pinned
# OPT tokenizer. Check it at load so a changed tokenizer or concatenation cannot
# silently change the metric.
EXPECTED_BLOCKS = 140
REWARD_PATH = Path("/logs/verifier/reward.txt")


class SparsityViolation(RuntimeError):
    """The artifact is not at the sparsity the measurement assumes."""


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def weight_sha256(checkpoint: Path) -> str:
    files = sorted(
        {
            path
            for pattern in ("*.safetensors", "pytorch_model*.bin")
            for path in checkpoint.glob(pattern)
        }
    )
    if not files:
        raise FileNotFoundError(f"checkpoint has no model weights: {checkpoint}")
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
    """Find the weights, given either the artifact directory or its parent.

    Shorter than OPD's equivalent because there is no step sequence to search: the
    retrain phase writes exactly one model. `pruned/` is where solution/run.sh puts
    it, so accepting a mount of the parent costs one line and saves an operator
    guessing.
    """

    if has_weights(checkpoint):
        return checkpoint
    for candidate in ("pruned", "checkpoint"):
        nested = checkpoint / candidate
        if nested.is_dir() and has_weights(nested):
            return nested
    raise FileNotFoundError(
        f"no model weights under {checkpoint}. Looked for *.safetensors and "
        "pytorch_model*.bin here and in pruned/ and checkpoint/."
    )


def write_reward(score: float, reward_path: Path) -> None:
    """Write higher-is-better utility while summary.json retains raw perplexity."""

    reward_path.parent.mkdir(parents=True, exist_ok=True)
    reward_path.write_text(f"{-score:.10f}\n", encoding="utf-8")


def evaluate(
    checkpoint: Path,
    assets: Path,
    output: Path,
    reward_path: Path,
) -> dict[str, Any]:
    model_path = resolve_checkpoint(checkpoint)
    test_parquet = assets / "data/wikitext2/test"
    if not test_parquet.exists():
        raise FileNotFoundError(
            f"{test_parquet} is absent. The WikiText2 test split is the final's text "
            "and must be staged from the pinned asset manifest before scoring."
        )
    output.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    weights_hash = weight_sha256(model_path)
    model, tokenizer = load_model(model_path)

    # The gate comes before generation, so a violating artifact costs a model load
    # rather than a full scoring pass.
    actual_sparsity, layer_sparsities = decoder_sparsity(model)
    resolved = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "checkpoint": str(checkpoint),
        "model": str(model_path),
        "checkpoint_weight_sha256": weights_hash,
        "eval_source": WIKITEXT_SOURCE,
        "eval_revision": WIKITEXT_REVISION,
        "eval_split": "test",
        "sequence_length": SEQUENCE_LENGTH,
        "sparsity_window": list(SPARSITY_WINDOW),
        "actual_global_sparsity": actual_sparsity,
        "image_digest": os.environ.get("IMAGE_DIGEST"),
    }
    atomic_json(output / "resolved_config.json", resolved)

    if not sparsity_in_window(actual_sparsity):
        low, high = SPARSITY_WINDOW
        side = "below" if actual_sparsity < low else "above"
        detail = (
            f"decoder sparsity {actual_sparsity:.6f} is {side} the required "
            f"[{low}, {high}]"
        )
        atomic_json(
            output / "summary.json",
            {
                "schema_version": 1,
                "task_id": TASK_ID,
                "status": "invalid",
                "reason": detail,
                "metric": METRIC,
                "direction": DIRECTION,
                "metrics": {},
                "actual_global_sparsity": actual_sparsity,
                "layer_sparsity_min": min(layer_sparsities),
                "layer_sparsity_max": max(layer_sparsities),
                "sparsity_window": list(SPARSITY_WINDOW),
                "checkpoint_weight_sha256": weights_hash,
            },
        )
        # No reward is written. An invalid trial has no score, and writing one -- even
        # a deliberately terrible one -- would put it on the same axis as real
        # results and let a 50%-sparse model rank above a bad 70% one.
        raise SparsityViolation(
            f"{detail}. 70% is not a rule about method, it is what makes the "
            "perplexity comparable: a less sparse model scores better and is not the "
            "same measurement. The trial is invalid."
        )

    text = load_text(test_parquet)
    nlls = block_nlls(model, tokenizer, text)
    if len(nlls) != EXPECTED_BLOCKS:
        raise ValueError(
            f"WikiText2 test gave {len(nlls)} blocks, expected {EXPECTED_BLOCKS}. The "
            "block count is a function of the tokeniser and the concatenation, so a "
            "mismatch means the text or the tokeniser changed and the score is not "
            "comparable."
        )

    aggregate = summarize(nlls, actual_sparsity, layer_sparsities)
    summary = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "passed",
        "benchmark": "WikiText2 raw test perplexity",
        "metric": METRIC,
        "direction": DIRECTION,
        "metrics": {METRIC: aggregate["score"]},
        "reward": -aggregate["score"],
        **aggregate,
        # Proxy and final are disjoint splits, so there is no proxy-inside-final
        # subset to report and no overfitting delta to compute. Stated rather than
        # omitted -- see the module docstring.
        "proxy_overlap_blocks": 0,
        "score_proxy_subset": None,
        "score_final_minus_proxy": None,
        "overfitting_delta": None,
        "overlap_note": "validation and test are disjoint splits; overlap is 0%",
        "wall_seconds": time.monotonic() - started,
        "offline": True,
        "checkpoint_weight_sha256": weights_hash,
    }
    (output / "block_nll.jsonl").write_text(
        "".join(
            json.dumps({"block": index, "negative_log_likelihood": value}, sort_keys=True) + "\n"
            for index, value in enumerate(nlls)
        ),
        encoding="utf-8",
    )
    atomic_json(output / "summary.json", summary)
    write_reward(aggregate["score"], reward_path)
    return summary


def mock(output: Path, reward_path: Path) -> dict[str, Any]:
    """Synthetic per-block NLLs at exactly 70% sparsity, no GPU and no test split.

    Exercises the gate, the block-count check, the summary shape and the reward file.
    """

    nlls = [(3.9 + 0.001 * index) * SEQUENCE_LENGTH for index in range(EXPECTED_BLOCKS)]
    layer_sparsities = [0.70 + 0.001 * ((index % 5) - 2) for index in range(32)]
    output.mkdir(parents=True, exist_ok=True)
    aggregate = summarize(nlls, 0.7000, layer_sparsities)
    if aggregate["blocks"] != EXPECTED_BLOCKS:
        raise RuntimeError(f"mock produced {aggregate['blocks']} blocks")
    summary = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "passed",
        "metric": METRIC,
        "direction": DIRECTION,
        "metrics": {METRIC: aggregate["score"]},
        "reward": -aggregate["score"],
        **aggregate,
        "proxy_overlap_blocks": 0,
        "score_proxy_subset": None,
        "score_final_minus_proxy": None,
        "overfitting_delta": None,
        "mock": True,
    }
    atomic_json(output / "summary.json", summary)
    write_reward(aggregate["score"], reward_path)
    return summary


def smoke() -> None:
    nlls = [4.0 * SEQUENCE_LENGTH] * EXPECTED_BLOCKS
    summary = summarize(nlls, 0.7, [0.7] * 32)
    if summary["blocks"] != EXPECTED_BLOCKS:
        raise RuntimeError(f"unexpected block count: {summary}")
    if abs(summary["score"] - perplexity(nlls)) > 1e-9:
        raise RuntimeError("summarize and perplexity disagree")
    # The gate must reject the two cases it exists for: a half-pruned model and a
    # fully dense one.
    for bad in (0.5, 0.0, 0.6989, 0.7011):
        if sparsity_in_window(bad):
            raise RuntimeError(f"sparsity gate accepted {bad}")
    print(json.dumps({"final_eval_smoke": "passed", "blocks": summary["blocks"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--assets", type=Path, default=Path("/assets"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reward-path", type=Path, default=REWARD_PATH)
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
    )
    # runner.py has no report_reward hook on this phase -- that hook prints an
    # accuracy shape -- so the one human-readable line is printed here.
    print(
        f"final: {METRIC} = {summary['score']:.6f} (minimise) over "
        f"{summary['blocks']} blocks at sparsity "
        f"{summary['actual_global_sparsity']:.6f}"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"OWL final failed: {exc}", file=sys.stderr)
        raise
