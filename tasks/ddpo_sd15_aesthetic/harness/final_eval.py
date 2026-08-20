"""Hidden final: 256 generated images and mean aesthetic score.

Carried over from the trusted judge-side evaluator (judge/tasks/ddpo_sd15_aesthetic
on the reference protocol). Four changes:

1. **The metadata checks are gone.** The old evaluator read `training_metadata.json`
   out of the checkpoint and refused it unless five fields matched -- task_id,
   algorithm_family, and three upstream revisions -- then checked a `checkpoint_sha256`
   field against the weight file's real hash. The candidate writes that file. It is a
   check that the candidate agreed with itself, and it would pass for any artifact
   produced by any method as long as the JSON said the right words. Dropped. The
   scorer accepts either a loadable LoRA export or a complete diffusers pipeline.

2. **The benchmark has no alignment/diversity guard.** CLIP text alignment and mean
   pairwise CLIP distance are reported as auxiliary metrics beside the aesthetic score.
   They do not affect validity or the reward.

3. **The protocol comes from a mount, not from this file.** samples, generation seed
   and the training-start reference live in /assets/data/final_reference.json, which
   is mounted into the score phase and not into exploration. Spelling the seed here
   would publish it: /opt/harness is in the image, so the 4 h container can read
   every line of this file. With a generated dataset the seed *is* the held-out set --
   45 prompts and a latent per row -- so an Agent that knows it can tune against the
   exact 256 images the result is reported on. This is the same mechanism OPD uses for
   its AIME questions: a mount present in score and absent from explore.

4. Scoring goes through harness/grade.py, the same entry point fast_eval uses.

There is no test set to hold out in the usual sense. The "dataset" is (prompt, latent
seed) pairs drawn from a finite 45-word prompt list, so the tiers differ by sample
count and seed rather than by rows -- and the proxy and the final are two independent
draws from one distribution, not two slices of one pool.
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

from grade import (  # noqa: E402
    CLIP_REVISION,
    GUIDANCE_SCALE,
    INFERENCE_STEPS,
    LORA_WEIGHT_NAME,
    MODEL_REVISION,
    UPSTREAM_REVISION,
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
METRIC = "mean_aesthetic_score_final256"

REWARD_PATH = Path("/logs/verifier/reward")
DEFAULT_REFERENCE = Path("/assets/data/final_reference.json")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_reference(path: Path) -> dict[str, Any]:
    """Read the final's sample-count and seed protocol from the mounted asset.

    Legacy ``training_start`` metrics may still be present in the JSON for provenance,
    but they are not part of the benchmark score and are intentionally ignored.
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"the final's protocol is missing at {path}. It is a mounted asset, not a "
            "constant in this file, so that the exploration container cannot read the "
            "generation seed. Mount asset:data/final_reference.json, or pass --mock."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("samples", "generation_seed"):
        if key not in payload:
            raise ValueError(f"{path} has no {key!r}")
    samples = int(payload["samples"])
    if samples < 2:
        raise ValueError(f"{path}: samples must be at least 2, got {samples}")
    return {
        "samples": samples,
        "generation_seed": int(payload["generation_seed"]),
        "reference_sha256": file_sha256(path),
    }


def resolve_checkpoint(checkpoint: Path) -> tuple[str, Path]:
    """Find a LoRA export or complete diffusers pipeline."""

    from grade import resolve_generation_artifact

    return resolve_generation_artifact(checkpoint)


def artifact_sha256(path: Path) -> str:
    """Hash either one weight file or a complete checkpoint directory."""

    if path.is_file():
        return file_sha256(path)
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise FileNotFoundError(f"checkpoint directory is empty: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(file_sha256(item)))
    return digest.hexdigest()


def write_reward(score_value: float, reward_path: Path) -> None:
    """Harbor's verifier contract: the scalar lands in /logs/verifier/reward."""

    reward_path.parent.mkdir(parents=True, exist_ok=True)
    reward_path.write_text(f"{score_value:.10f}\n", encoding="utf-8")


def build_summary(
    metrics: dict[str, Any],
    diagnostics: dict[str, Any],
    *,
    protocol: dict[str, Any],
    extra: dict[str, Any],
) -> dict[str, Any]:
    """Assemble summary.json.

    `correct` carries no information and is not a count of anything. It is here
    because orchestrator/runner.py's report_reward hook formats the line
    "(correct/n)", which assumes a count-correct metric; this task's metric is a
    continuous mean, so there is nothing to count. Setting it to n keeps the
    host-side hook from raising KeyError after a completed run. orchestrator/ is
    off-limits in this port -- see the report's "orchestrator changes needed".
    """

    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "benchmark": "DDPO simple-animals held-out generation stream",
        "metric": METRIC,
        "direction": "maximize",
        "status": "passed",
        "metrics": {key: metrics[key] for key in
                    ("mean_aesthetic_score", "mean_clip_alignment",
                     "mean_pairwise_clip_distance")},
        "score": metrics["mean_aesthetic_score"],
        "stderr": metrics["aesthetic_stderr"],
        "n": metrics["n"],
        "correct": metrics["n"],
        "aesthetic_sd": metrics["aesthetic_sd"],
        "diagnostics": diagnostics,
        "protocol": protocol,
        **extra,
    }


def evaluate(
    checkpoint: Path,
    assets: Path,
    output: Path,
    reference_path: Path,
    reward_path: Path,
) -> dict[str, Any]:
    artifact_kind, artifact = resolve_checkpoint(checkpoint)
    model = assets / "models/stable-diffusion-v1-5"
    clip = assets / "models/clip"
    for required in (model / "model_index.json", clip / "config.json"):
        if not required.is_file():
            raise FileNotFoundError(required)
    reference = load_reference(reference_path)
    output.mkdir(parents=True, exist_ok=True)

    samples = reference["samples"]
    seed = reference["generation_seed"]
    scorer_class, simple_animals = import_scorer(clip)
    prompts = draw_prompts(simple_animals, samples, seed)
    artifact_hash = artifact_sha256(
        artifact / LORA_WEIGHT_NAME if artifact_kind == "lora" else artifact
    )

    protocol = {
        "samples": samples,
        "inference_steps": INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "distinct_prompts": len(set(prompts)),
        # The seed itself is deliberately not echoed here: resolved_config.json lands
        # in /out, and /out is the Agent's own output mount in the explore phase.
        "generation_seed_sha256": hashlib.sha256(str(seed).encode()).hexdigest(),
        "reference_sha256": reference["reference_sha256"],
    }
    atomic_json(
        output / "resolved_config.json",
        {
            "schema_version": 1,
            "task_id": TASK_ID,
            "checkpoint": str(checkpoint),
            "artifact_kind": artifact_kind,
            "artifact": str(artifact),
            "artifact_sha256": artifact_hash,
            "upstream_revision": UPSTREAM_REVISION,
            "model_revision": MODEL_REVISION,
            "clip_revision": CLIP_REVISION,
            "image_digest": os.environ.get("IMAGE_DIGEST"),
            **protocol,
        },
    )

    started = time.monotonic()
    images, rows = generate(model, artifact, prompts, seed, output / "images")
    metrics = score(scorer_class, images, prompts, rows)
    diagnostics = {
        "status": "not_applicable",
        "reason": "the original DDPO benchmark has no alignment/diversity guard",
    }

    summary = build_summary(
        metrics,
        diagnostics,
        protocol=protocol,
        extra={
            "artifact_kind": artifact_kind,
            "artifact_sha256": artifact_hash,
            "wall_seconds": time.monotonic() - started,
            "offline": True,
            "mock": False,
        },
    )
    atomic_json(output / "summary.json", summary)
    write_rows(output / "rows.jsonl", rows)
    write_reward(metrics["mean_aesthetic_score"], reward_path)
    return summary


def mock(output: Path, reward_path: Path, *, collapsed: bool) -> dict[str, Any]:
    """Synthetic rows, no GPU and no mounted protocol. Exercises diagnostics.

    `--mock-collapsed` produces the diversity-collapse case without training: every row
    the same high aesthetic score, alignment down, and pairwise distance near zero.
    """

    samples, seed = 256, 20269702
    rows, pairwise = synthetic_rows(samples, seed, collapsed=collapsed)
    metrics = summarize(rows, pairwise)
    diagnostics = {
        "status": "not_applicable",
        "reason": "the original DDPO benchmark has no alignment/diversity guard",
    }
    output.mkdir(parents=True, exist_ok=True)
    summary = build_summary(
        metrics,
        diagnostics,
        protocol={
            "samples": samples,
            "mock": True,
        },
        extra={"mock": True, "mock_collapsed": collapsed},
    )
    atomic_json(output / "summary.json", summary)
    write_rows(output / "rows.jsonl", rows)
    write_reward(metrics["mean_aesthetic_score"], reward_path)
    return summary


def smoke() -> None:
    rows, pairwise = synthetic_rows(256, 20269702)
    metrics = summarize(rows, pairwise)
    if metrics["n"] != 256:
        raise RuntimeError(f"unexpected row count: {metrics}")
    print(json.dumps({"final_eval_smoke": "passed", "n": metrics["n"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--assets", type=Path, default=Path("/assets"))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE,
        help="the final's protocol and training-start reference. A mounted asset, so "
        "the generation seed is not readable from the exploration container.",
    )
    parser.add_argument("--reward-path", type=Path, default=REWARD_PATH)
    parser.add_argument("--mock", action="store_true", help="synthetic rows, no GPU")
    parser.add_argument(
        "--mock-collapsed",
        action="store_true",
        help="synthetic diversity-collapse case; retained as an auxiliary-metric fixture",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        smoke()
        return
    if args.output is None:
        parser.error("--output is required")
    if args.mock or args.mock_collapsed:
        summary = mock(
            args.output.resolve(), args.reward_path, collapsed=args.mock_collapsed
        )
    else:
        if args.checkpoint is None:
            parser.error("--checkpoint is required outside --mock/--smoke")
        summary = evaluate(
            args.checkpoint.resolve(),
            args.assets.resolve(),
            args.output.resolve(),
            args.reference,
            args.reward_path,
        )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"DDPO final failed: {exc}", file=sys.stderr)
        raise
