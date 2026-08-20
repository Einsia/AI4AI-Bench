"""Score 64 generated images with the final evaluator's model arithmetic.

The default proxy uses the same inference steps, guidance scale, aesthetic scorer
and two auxiliary metrics as the final. Only its sample count and generation seed differ.
The proxy is a noisy exploration instrument, so preserve per-run rows and repeat a
promising direction before treating a small change as an ordering.
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
    GUIDANCE_SCALE,
    INFERENCE_STEPS,
    atomic_json,
    draw_prompts,
    generate,
    import_scorer,
    score,
    summarize,
    synthetic_rows,
    write_rows,
)

TASK_ID = "ddpo_sd15_aesthetic"
METRIC = "mean_aesthetic_score_public64"

# The proxy tier. Its own seed, disjoint from the final's -- see the module docstring
# of final_eval.py for why the two tiers do not overlap. Unlike the final's seed, this
# one is deliberately in plain sight: the proxy is the Agent's instrument.
DEFAULT_SAMPLES = 64
DEFAULT_SEED = 20269700

def evaluate(
    checkpoint: Path | None,
    assets: Path,
    out: Path,
    samples: int,
    seed: int,
    reference: dict[str, Any] | None,
) -> dict[str, Any]:
    model = assets / "models/stable-diffusion-v1-5"
    clip = assets / "models/clip"
    for required in (model / "model_index.json", clip / "config.json"):
        if not required.is_file():
            raise FileNotFoundError(required)

    artifact: Path | None = None
    if checkpoint is not None:
        from grade import resolve_generation_artifact

        _, artifact = resolve_generation_artifact(checkpoint)

    scorer_class, simple_animals = import_scorer(clip)
    prompts = draw_prompts(simple_animals, samples, seed)
    started = time.monotonic()
    images, rows = generate(model, artifact, prompts, seed, out.parent / f"{out.stem}-images")
    metrics = score(scorer_class, images, prompts, rows)
    elapsed = time.monotonic() - started

    payload: dict[str, Any] = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "metric": METRIC,
        "direction": "maximize",
        "checkpoint": str(checkpoint) if checkpoint else None,
        "artifact": str(artifact) if artifact else "pinned base model",
        "samples": samples,
        "seed": seed,
        "distinct_prompts": len(set(prompts)),
        "inference_steps": INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "seconds": elapsed,
        **metrics,
    }
    if reference is not None:
        payload["diagnostics"] = {
            "status": "not_applicable",
            "reason": "the original DDPO benchmark has no alignment/diversity guard",
        }
    atomic_json(out, payload)
    write_rows(out.with_name(out.stem + "-rows.jsonl"), rows)
    return payload


def mock(out: Path, samples: int, seed: int) -> dict[str, Any]:
    rows, pairwise = synthetic_rows(samples, seed)
    payload = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "metric": METRIC,
        "direction": "maximize",
        "mock": True,
        "samples": samples,
        "seed": seed,
        "seconds": 0.0,
        **summarize(rows, pairwise),
    }
    atomic_json(out, payload)
    return payload


def smoke() -> None:
    payload = mock(Path("/tmp/fast_eval-smoke.json"), DEFAULT_SAMPLES, DEFAULT_SEED)
    if payload["n"] != DEFAULT_SAMPLES:
        raise RuntimeError(f"unexpected row count: {payload}")
    if not payload["aesthetic_stderr"] > 0.0:
        raise RuntimeError(f"stderr must be positive on varied rows: {payload}")
    print(json.dumps({"fast_eval_smoke": "passed", "n": payload["n"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--base",
        action="store_true",
        help="score the pinned base model -- the training start",
    )
    parser.add_argument("--assets", type=Path, default=Path("/assets"))
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help="rows to draw. 64 by default; see the module docstring before changing it",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--diagnostic-reference",
        type=Path,
        default=None,
        help="JSON holding same-tier training-start alignment and pairwise-distance "
        "diagnostics. Optional and advisory.",
    )
    parser.add_argument("--mock", action="store_true", help="synthetic rows, no GPU")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        smoke()
        return
    if args.out is None:
        parser.error("--out is required")
    if args.mock:
        print(json.dumps(mock(args.out, args.samples, args.seed), sort_keys=True))
        return
    if args.checkpoint is None and not args.base:
        parser.error("pass --checkpoint, or --base to score the untrained start")
    if args.checkpoint is not None and args.base:
        parser.error("--base scores the training start, so it cannot be combined with --checkpoint")

    reference = None
    if args.diagnostic_reference is not None:
        reference = json.loads(args.diagnostic_reference.read_text(encoding="utf-8"))

    payload = evaluate(
        args.checkpoint.resolve() if args.checkpoint else None,
        args.assets.resolve(),
        args.out.resolve(),
        args.samples,
        args.seed,
        reference,
    )
    print(
        json.dumps(
            {
                "mean_aesthetic_score": payload["mean_aesthetic_score"],
                "aesthetic_stderr": payload["aesthetic_stderr"],
                "mean_clip_alignment": payload["mean_clip_alignment"],
                "mean_pairwise_clip_distance": payload["mean_pairwise_clip_distance"],
                "n": payload["n"],
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
