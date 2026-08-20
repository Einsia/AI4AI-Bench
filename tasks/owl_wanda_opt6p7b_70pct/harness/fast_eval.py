"""WikiText2 raw validation perplexity for the 4 h exploration phase.

This uses the final evaluator's tokenisation, 2,048-token blocks, fp16 forward path
and artifact sparsity calculation on a disjoint validation split. Compare candidate
deltas within this tier, and repeat a promising pruning change with another
calibration seed before treating a small change as robust.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def _redirect_caches(root: str | None = None) -> None:
    """Move library caches off the 256 MiB /tmp tmpfs. Same reason as run.sh."""

    base = root or str(Path(os.environ.get("OUT_DIR", "/out")) / "fast-eval-cache")
    for name, relative in (
        ("TMPDIR", "tmp"),
        ("HF_HOME", "tmp/hf"),
        ("TRANSFORMERS_CACHE", "tmp/hf/transformers"),
        ("TORCH_HOME", "tmp/torch"),
        ("TRITON_CACHE_DIR", "tmp/triton"),
        ("XDG_CACHE_HOME", "tmp/cache"),
    ):
        path = Path(os.environ.get(name, f"{base}/{relative}"))
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
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
METRIC = "wikitext2_validation_perplexity"
DIRECTION = "minimize"
DEFAULT_DATA = Path("/assets/data/wikitext2/validation")
def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def has_weights(directory: Path) -> bool:
    return any(directory.glob("*.safetensors")) or any(directory.glob("pytorch_model*.bin"))


def resolve_checkpoint(checkpoint: Path) -> Path:
    """Same resolution as the final, so a path that scores there scores here."""

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


def evaluate(checkpoint: Path, data: Path, out: Path) -> dict[str, Any]:
    """Score a pruned checkpoint and report its sparsity alongside.

    THE SPARSITY GATE IS NOT FATAL HERE, and that is a deliberate split of
    responsibilities from the final, which does invalidate. During exploration a
    refusal costs you the number you needed to understand what went wrong, so this
    prints the perplexity and flags the sparsity loudly. The final is where it counts:
    an artifact outside [0.699, 0.701] scores nothing at all there, so treat
    `sparsity_in_window: false` here as a failed run whatever the perplexity says.
    """

    model_path = resolve_checkpoint(checkpoint)
    started = time.monotonic()
    model, tokenizer = load_model(model_path)
    actual_sparsity, layer_sparsities = decoder_sparsity(model)
    text = load_text(data)
    nlls = block_nlls(model, tokenizer, text)
    aggregate = summarize(nlls, actual_sparsity, layer_sparsities)
    payload = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "metric": METRIC,
        "direction": DIRECTION,
        "eval_split": "validation",
        "checkpoint": str(checkpoint),
        "model": str(model_path),
        **aggregate,
        "seconds": time.monotonic() - started,
    }
    atomic_json(out, payload)
    rows_path = out.with_name(out.stem + "-blocks.jsonl")
    rows_path.write_text(
        "".join(
            json.dumps({"block": index, "negative_log_likelihood": value}, sort_keys=True) + "\n"
            for index, value in enumerate(nlls)
        ),
        encoding="utf-8",
    )
    if not sparsity_in_window(actual_sparsity):
        low, high = SPARSITY_WINDOW
        print(
            f"\n*** WARNING: decoder sparsity {actual_sparsity:.6f} is outside "
            f"[{low}, {high}]. The final INVALIDATES this artifact and it will score "
            f"nothing. The perplexity below is not comparable with anything.\n",
            file=sys.stderr,
        )
    return payload


def mock(out: Path) -> dict[str, Any]:
    """Synthetic per-block NLLs, no GPU. Checks the output shape only.

    The block count here is arbitrary; the real evaluator derives it from the
    validation parquet and tokenizer.
    """

    blocks = 119
    nlls = [(4.05 + 0.002 * index) * SEQUENCE_LENGTH for index in range(blocks)]
    aggregate = summarize(nlls, 0.7, [0.7 + 0.001 * ((i % 5) - 2) for i in range(32)])
    payload = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "metric": METRIC,
        "direction": DIRECTION,
        "eval_split": "validation",
        "mock": True,
        **aggregate,
        "seconds": 0.0,
    }
    atomic_json(out, payload)
    return payload


def smoke() -> None:
    payload = mock(Path("/tmp/owl-fast_eval-smoke.json"))
    if payload["blocks"] != 119:
        raise RuntimeError(f"unexpected block count: {payload}")
    if payload["block_nll_per_token_sd"] <= 0.0:
        raise RuntimeError(f"spread must be positive on varied blocks: {payload}")
    if abs(payload["score"] - perplexity([(4.05 + 0.002 * i) * SEQUENCE_LENGTH
                                          for i in range(119)])) > 1e-9:
        raise RuntimeError("mock score does not match perplexity()")
    print(json.dumps({"fast_eval_smoke": "passed", "blocks": payload["blocks"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--mock", action="store_true", help="synthetic blocks, no GPU")
    parser.add_argument("--smoke", action="store_true", help="self-check and exit")
    args = parser.parse_args()

    if args.smoke:
        smoke()
        return
    if args.out is None:
        parser.error("--out is required")
    if args.mock:
        print(json.dumps(mock(args.out), sort_keys=True))
        return
    if args.checkpoint is None:
        parser.error("--checkpoint is required outside --mock/--smoke")
    payload = evaluate(args.checkpoint.resolve(), args.data.resolve(), args.out.resolve())
    print(
        json.dumps(
            {
                "score": payload["score"],
                "blocks": payload["blocks"],
                "actual_global_sparsity": payload["actual_global_sparsity"],
                "sparsity_in_window": payload["sparsity_in_window"],
                "layer_sparsity_min": payload["layer_sparsity_min"],
                "layer_sparsity_max": payload["layer_sparsity_max"],
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
