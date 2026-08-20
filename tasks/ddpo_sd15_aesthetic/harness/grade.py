"""The one measurement path, shared by fast_eval and the hidden final.

Both tiers generate images from the candidate artifact and score them with the
*image's* copy of the official DDPO aesthetic scorer, baked
in at /opt/harness/ddpo-pytorch. Nothing here reads /workspace. That is the whole
reason "the reward does not change" can be checked by reading one file: a candidate
that rewrites its own aesthetic_scorer.py to return 10.0 changes what it sees while
training and changes nothing about what it is scored on.

Three numbers come out, not one:

    mean_aesthetic_score          the metric
    mean_clip_alignment           diagnostic: prompt/image agreement
    mean_pairwise_clip_distance   diagnostic: diversity across generated images

The two diagnostics make prompt drift and collapse visible, but they are not gates
and are not folded into the headline aesthetic score. Both are computed from the
pixels the checkpoint actually produced.

The arithmetic below is the fixed evaluator contract. Changing the pairwise term,
generation dtype or scorer would change the metric rather than improve a candidate.
"""

from __future__ import annotations

import json
import math
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Any

# The harness's own copy of the pinned upstream, not /workspace/ddpo-pytorch.
HARNESS_DDPO = Path("/opt/harness/ddpo-pytorch")
UPSTREAM_REVISION = "1958463f020112c9a7bc85768d296daacc2e1b4b"
MODEL_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
CLIP_REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"

# Generation protocol. Part of the metric, not a performance knob.
INFERENCE_STEPS = 20
GUIDANCE_SCALE = 5.0
LORA_WEIGHT_NAME = "pytorch_lora_weights.bin"


WORKSPACE = Path("/workspace")


def import_scorer(clip_path: Path) -> tuple[Any, Any]:
    """Import AestheticScorer and simple_animals from the harness's pinned tree.

    More care than an `sys.path.insert(0, ...)` because `ddpo_pytorch` has no
    __init__.py -- it is a **namespace package**, and that changes what "first on the
    path wins" means. Measured, not assumed: with the candidate's tree and the
    harness's tree both on sys.path, `ddpo_pytorch.__path__` holds *both* directories
    and `resources.files("ddpo_pytorch.assets")` returns a MultiplexedPath spanning
    both. Same-named files do resolve to whichever tree comes first, so the reward
    MLP would still be the harness's -- but a file present only in the candidate's
    copy is visible through the harness's package, and `__file__` is None, so the
    obvious `assert module.__file__.startswith(...)` guard cannot even be written.

    So the workspace is removed from sys.path before the import rather than merely
    out-ranked, and the assertion is on `__path__` being exactly one directory. That
    is the difference between "the harness's scorer usually wins" and "the
    candidate's tree is not reachable from here at all".

    This matters in the 4 h container, where /workspace/ddpo-pytorch is the
    candidate's edited copy and is on PYTHONPATH so training picks up its edits. The
    score phase runs with no patch applied, so there is nothing to shadow there --
    but fast_eval has to measure the same thing the final does, or the Agent spends
    four hours tuning against its own arithmetic.
    """

    if not HARNESS_DDPO.is_dir():
        raise FileNotFoundError(f"pinned DDPO tree missing from the image: {HARNESS_DDPO}")
    if "ddpo_pytorch" in sys.modules:
        raise RuntimeError("ddpo_pytorch was imported before the scoring path could pin it")

    workspace = str(WORKSPACE)
    # `''` and `'.'` mean the current directory and neither starts with /workspace, so a
    # startswith filter alone can expose both source trees. Remove them explicitly to
    # keep the reward implementation pinned under every Python invocation form.
    relative = {"", "."}
    sys.path[:] = [
        entry
        for entry in sys.path
        if entry not in relative and not entry.startswith(workspace)
    ]
    sys.path.insert(0, str(HARNESS_DDPO))
    # Prevent future child processes from inheriting the candidate workspace tree.
    os.environ.pop("PYTHONPATH", None)
    # The patched scorer reads this, defaulting to the mount; set it explicitly so a
    # non-default --clip is honoured.
    os.environ["DDPO_CLIP_PATH"] = str(clip_path)

    import ddpo_pytorch

    resolved = [str(entry) for entry in ddpo_pytorch.__path__]
    expected = [str(HARNESS_DDPO / "ddpo_pytorch")]
    if resolved != expected:
        raise RuntimeError(
            f"ddpo_pytorch resolves to {resolved}, expected exactly {expected}. The "
            "scoring path must use the image's scorer, never the candidate's."
        )

    from ddpo_pytorch.aesthetic_scorer import AestheticScorer
    from ddpo_pytorch.prompts import simple_animals

    scorer_file = Path(sys.modules["ddpo_pytorch.aesthetic_scorer"].__file__ or "")
    if HARNESS_DDPO not in scorer_file.parents:
        raise RuntimeError(f"aesthetic_scorer loaded from {scorer_file}, outside {HARNESS_DDPO}")
    return AestheticScorer, simple_animals


def draw_prompts(simple_animals: Any, samples: int, seed: int) -> list[str]:
    """The (prompt, latent) stream for a tier, reproduced exactly.

    `simple_animals()` is `random.choice` over a 45-line list in the pinned tree, so
    it draws from the *global* random state. Seeding `random` and then calling it
    `samples` times is what fixes which prompts a tier uses -- this is the old
    evaluator's construction.

    A tier is therefore identified by (samples, seed) and nothing else. There is no
    row file to hold out: the "dataset" is generated, and two tiers at different
    seeds are two independent draws from one distribution.
    """

    random.seed(seed)
    return [simple_animals()[0] for _ in range(samples)]


def resolve_generation_artifact(checkpoint: Path) -> tuple[str, Path]:
    """Resolve either a diffusers LoRA export or a complete diffusers pipeline."""

    candidates = (checkpoint, checkpoint / "checkpoint")
    for candidate in candidates:
        if (candidate / LORA_WEIGHT_NAME).is_file():
            return "lora", candidate
        if (candidate / "model_index.json").is_file():
            return "pipeline", candidate
    raise FileNotFoundError(
        f"no {LORA_WEIGHT_NAME} or model_index.json under {checkpoint}; export a "
        "diffusers LoRA directory or complete Stable Diffusion pipeline"
    )


def generate(model: Path, checkpoint: Path | None, prompts: list[str], seed: int, images_dir: Path):
    """Generate one image per prompt, latent seed = seed + row index.

    `checkpoint=None` evaluates the pinned base model. A candidate checkpoint may
    be either a LoRA directory applied to that base or a complete diffusers pipeline.
    """

    import numpy as np
    import torch
    from diffusers import DDIMScheduler, StableDiffusionPipeline

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    artifact_kind = "base"
    artifact = model
    if checkpoint is not None:
        artifact_kind, artifact = resolve_generation_artifact(checkpoint)
    pipeline_source = artifact if artifact_kind == "pipeline" else model
    pipeline = StableDiffusionPipeline.from_pretrained(
        str(pipeline_source), torch_dtype=torch.float16, local_files_only=True
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    # Off for the same reason upstream turns it off: a refused generation would
    # return a black image and score as if the policy had produced one.
    pipeline.safety_checker = None
    if artifact_kind == "lora":
        pipeline.unet.load_attn_procs(str(artifact))
    pipeline = pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=True)

    images_dir.mkdir(parents=True, exist_ok=True)
    images = []
    rows: list[dict[str, Any]] = []
    for index, prompt in enumerate(prompts):
        latent_seed = seed + index
        image = pipeline(
            prompt,
            num_inference_steps=INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=torch.Generator(device="cuda").manual_seed(latent_seed),
        ).images[0]
        image.save(images_dir / f"{index:03d}.png")
        images.append(image)
        rows.append({"index": index, "prompt": prompt, "seed": latent_seed})

    del pipeline
    torch.cuda.empty_cache()
    return images, rows


def score(scorer_class: Any, images: list[Any], prompts: list[str], rows: list[dict[str, Any]]):
    """Aesthetic score, CLIP text alignment, and mean pairwise CLIP distance.

    One CLIP forward pass serves all three: the aesthetic MLP consumes the image
    embedding, alignment is the cosine of image against its own prompt, and pairwise
    distance is 1 - the mean off-diagonal cosine of images against each other.

    Dtypes and the float32 cast are fixed parts of the evaluator.
    """

    import torch

    scorer = scorer_class(dtype=torch.float32).cuda()
    with torch.no_grad():
        aesthetic = scorer(images).float()
        inputs = scorer.processor(text=prompts, images=images, return_tensors="pt", padding=True)
        inputs = {key: value.cuda() for key, value in inputs.items()}
        image_features = scorer.clip.get_image_features(pixel_values=inputs["pixel_values"])
        text_features = scorer.clip.get_text_features(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        alignment = (image_features * text_features).sum(dim=-1)
        similarities = image_features @ image_features.T
        mask = ~torch.eye(len(images), dtype=torch.bool, device=similarities.device)
        diversity = 1.0 - similarities[mask].mean()

    for index, row in enumerate(rows):
        row["aesthetic_score"] = float(aesthetic[index])
        row["clip_alignment"] = float(alignment[index])
    return summarize(rows, float(diversity))


def summarize(rows: list[dict[str, Any]], pairwise_distance: float) -> dict[str, Any]:
    """Aggregate per-row scores, with a standard error on the primary metric.

    `aesthetic_stderr` is the plain sem across rows, which is the right shape here
    because each row is an independent (prompt, latent) draw -- unlike a pass@k
    metric there is nothing to cluster by. It measures sampling error *within* one
    run. It does not replace repetition across independently trained checkpoints.

    No error bar is reported for pairwise distance on purpose: it is a U-statistic
    over all n(n-1) pairs, so the row-wise sem does not apply to it and a naive one
    would be wrong in a flattering direction.
    """

    if not rows:
        raise ValueError("no rows to summarize")
    aesthetic = [float(row["aesthetic_score"]) for row in rows]
    alignment = [float(row["clip_alignment"]) for row in rows]
    count = len(rows)
    mean_aesthetic = statistics.fmean(aesthetic)
    if not math.isfinite(mean_aesthetic):
        raise RuntimeError("mean aesthetic score is non-finite")
    stderr = statistics.stdev(aesthetic) / math.sqrt(count) if count > 1 else float("nan")
    return {
        "mean_aesthetic_score": mean_aesthetic,
        "mean_clip_alignment": statistics.fmean(alignment),
        "mean_pairwise_clip_distance": pairwise_distance,
        "aesthetic_stderr": stderr,
        "aesthetic_sd": statistics.stdev(aesthetic) if count > 1 else float("nan"),
        "n": count,
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def synthetic_rows(samples: int, seed: int, *, collapsed: bool = False) -> tuple[list, float]:
    """Rows for --mock and --smoke, with no GPU and no model.

    `collapsed=True` is a collapse diagnostic in miniature: every row gets a high
    aesthetic score, alignment drops, and pairwise distance goes to almost zero.
    """

    generator = random.Random(seed)
    rows = []
    for index in range(samples):
        if collapsed:
            aesthetic, alignment = 6.5, 0.15
        else:
            aesthetic = 5.4 + generator.uniform(-0.4, 0.4)
            alignment = 0.235 + generator.uniform(-0.02, 0.02)
        rows.append(
            {
                "index": index,
                "prompt": "cat",
                "seed": seed + index,
                "aesthetic_score": aesthetic,
                "clip_alignment": alignment,
            }
        )
    return rows, 0.01 if collapsed else 0.36


def smoke() -> None:
    rows, pairwise = synthetic_rows(256, 20269702)
    summary = summarize(rows, pairwise)
    assert summary["n"] == 256, summary
    assert summary["aesthetic_stderr"] > 0.0, summary
    print("grade.py smoke passed")


if __name__ == "__main__":
    smoke()
