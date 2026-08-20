"""Build one soup from the 72 frozen CLIP ingredients. Yours to rewrite.

Ported from the reference protocol's baseline/method/train.py, with the parts that were
enforcement rather than method removed:

  the recipe loader        43 lines validating schema_version, task_id and six
                           algorithm fields against an allowlist, including a
                           check that weight_temperature was exactly 1.0. v1 has
                           no allowlist; configuration is run.sh's defaults.
  training_metadata.json   written by this file and then verified by the old
                           evaluator, which is a signature on a claim about
                           itself. The final now solves the exported weights
                           against the ingredient basis instead. This file still
                           writes a summary, because it is useful, but nothing
                           downstream trusts it.
  the mock path            the orchestrator has score-mock for plumbing checks.

Two things were kept deliberately.

**The tensor cache.** Selection scores the same frozen validation images for every
ingredient and every candidate merge. The transformed tensors are materialized once
and reused. This changes data staging only: row identities, transforms, model states
and metrics are identical to the uncached protocol.

**Streaming accumulation.** Upstream averages with
`sum(state[key] * scale for state in states)` over a list of all 72 loaded state
dicts, which is resident in 30.6 GiB. This loads one at a time into a running sum,
which needs about 2 GiB. The arithmetic is the same up to summation order. The
declared 64 GiB host ceiling covers the upstream form too, so writing it back is
not a mistake -- but it is 15x the memory for the same answer.

## What the search may and may not do

May: any selection rule, any weights, any number of ingredients, any objective for
choosing them, any framework. Look at the proxy rows as often as you like.

May not: produce weights that are not a combination of the 72. That includes
fine-tuning the soup, and it is checked at score time from the weights themselves
rather than from anything this file writes -- see harness/soup_check.py. The
weights must also sum to 1: a soup scaled by 1.0004 is not on the ingredients'
affine hull and is rejected with that diagnosis.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# The evaluator's own module construction, from the read-only harness. Not copied
# in: `get_model_from_sd` defines what a state dict means, so the search and the
# final have to agree about it or a soup that scores well here scores differently
# there.
sys.path.insert(0, "/opt/harness")

from forward import build_model, load_clip  # noqa: E402
from grade import class_offsets  # noqa: E402

EXPECTED_INGREDIENTS = 72
RULES = ("uniform", "best_single", "strict_greedy")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


class Deadline:
    """When to stop starting new trials.

    `max_wall_seconds <= 0` means no deadline, which is what an interactive run in
    the 4 h phase gets. The 12 h phase gets one from the host and the reserve is
    subtracted from it, so the last trial has room to finish and the export has
    room to complete. A truncated .pt is worse than a smaller search.
    """

    def __init__(self, max_wall_seconds: float, reserve_seconds: float) -> None:
        self.started = time.monotonic()
        # None means no deadline; a number means one, and it may be zero. Those two
        # were one value until a test asked for a reserve larger than the budget and
        # got an unlimited search: `limit <= 0` had been standing in for both "not
        # set" and "no time left", which are opposite instructions.
        self.limit: float | None = (
            None if max_wall_seconds <= 0 else max(max_wall_seconds - reserve_seconds, 0.0)
        )

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    @property
    def remaining(self) -> float:
        return float("inf") if self.limit is None else self.limit - self.elapsed

    def would_overrun(self, cost: float) -> bool:
        """Is there room for one more trial costing `cost` seconds?"""

        return self.remaining < cost


def ingredient_paths(root: Path) -> list[Path]:
    """The ingredients, ordered numerically rather than lexicographically.

    `model_9` sorts after `model_71` by name, which would scramble the recorded
    ingredient list against the weights beside it.
    """

    paths = sorted(root.glob("model_*.pt"), key=lambda item: int(item.stem.split("_")[-1]))
    if len(paths) != EXPECTED_INGREDIENTS:
        raise ValueError(
            f"expected {EXPECTED_INGREDIENTS} ingredients under {root}, found {len(paths)}"
        )
    return paths


def load_state(path: Path) -> dict[str, Any]:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def cache_validation(data: Path, per_class: int, preprocess: Any) -> Any:
    """Decode and transform the validation images once, then reuse them.

    See the module docstring. This changes repeated input preparation, not which rows
    are scored; the offsets and deterministic transform are unchanged.
    """

    import torch
    from PIL import Image
    from torch.utils.data import TensorDataset

    by_class = class_offsets(data)
    images: list[Any] = []
    labels: list[int] = []
    for label in sorted(by_class):
        for path in by_class[label][:per_class]:
            with Image.open(path) as handle:
                images.append(preprocess(handle.convert("RGB")))
            labels.append(label)
    if not images:
        raise ValueError(f"no validation images under {data}")
    return TensorDataset(torch.stack(images), torch.tensor(labels, dtype=torch.long))


def scorer(dataset: Any, base_model: Any, batch_size: int) -> Any:
    """A function from state dict to proxy accuracy."""

    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=True)
    use_cuda = torch.cuda.is_available()

    def score(state: dict[str, Any]) -> float:
        model = build_model(state, base_model)
        correct = total = 0
        with torch.no_grad():
            for images, labels in loader:
                if use_cuda:
                    images = images.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)
                correct += int((model(images).argmax(dim=1) == labels).sum())
                total += labels.numel()
        del model
        if use_cuda:
            torch.cuda.empty_cache()
        return correct / total

    return score


def weighted_average(paths: list[Path], weights: dict[int, float]) -> dict[str, Any]:
    """Stream the ingredients into one weighted sum.

    The weights are normalized here rather than trusted, because a soup whose
    weights do not sum to 1 is not on the ingredients' affine hull and the final
    rejects it with exactly that diagnosis. Normalizing is cheap; discovering it
    12 hours later is not.
    """

    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError(f"weights must sum to something positive, got {total}")
    accumulator: dict[str, Any] | None = None
    # Non-float tensors are copied from the first ingredient rather than averaged.
    # Multiplying an integer buffer by 1/72 promotes it to float, which changes the
    # checkpoint's dtype signature -- and the final compares dtypes, because a soup
    # of float32 ingredients is float32. Upstream's average multiplied every key
    # and got away with it only because this state dict happens to be all float.
    fixed: dict[str, Any] = {}
    for position, weight in sorted(weights.items()):
        if weight == 0.0:
            continue
        state = load_state(paths[position])
        scale = weight / total
        if accumulator is None:
            fixed = {
                key: value.clone()
                for key, value in state.items()
                if not value.is_floating_point()
            }
            accumulator = {
                key: value * scale for key, value in state.items() if value.is_floating_point()
            }
        else:
            for key in accumulator:
                accumulator[key] = accumulator[key] + state[key] * scale
        del state
    if accumulator is None:
        raise ValueError("no ingredient carried a non-zero weight")
    return {**accumulator, **fixed}


def rank_ingredients(
    paths: list[Path], score: Any, deadline: Deadline, log: Any
) -> list[tuple[int, float]]:
    """Score each ingredient alone, best first. 72 forward passes.

    The tie-break is the ingredient index, so a tie resolves the same way twice.
    """

    ranked: list[tuple[int, float]] = []
    per_trial = 0.0
    for position, path in enumerate(paths):
        if deadline.would_overrun(per_trial):
            log(f"soup: deadline reached after {len(ranked)} of {len(paths)} ingredients")
            break
        started = time.monotonic()
        accuracy = score(load_state(path))
        per_trial = max(per_trial, time.monotonic() - started)
        ranked.append((position, accuracy))
        log(f"soup: {path.stem} proxy {accuracy:.4f}")
    if not ranked:
        # Only reachable when the reserve is as large as the whole budget, which is a
        # misconfiguration rather than a search that ran long. Fail loudly: falling
        # back to the uniform soup here would export the baseline under a
        # candidate's name, which reads as a result rather than as a problem.
        raise RuntimeError(
            f"no ingredient could be scored: {deadline.remaining:.0f}s remaining of a "
            f"{deadline.limit}s limit. Check MAX_WALL_TIME_SECONDS against "
            "DEADLINE_RESERVE_SECONDS."
        )
    return sorted(ranked, key=lambda row: (-row[1], row[0]))


def build(
    paths: list[Path],
    rule: str,
    score: Any,
    max_ingredients: int,
    deadline: Deadline,
    log: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the soup and what was learned building it."""

    if rule == "uniform":
        # No scoring at all. This is the fixed uniform control.
        weights = {position: 1.0 for position in range(len(paths))}
        state = weighted_average(paths, weights)
        return state, {"rule": rule, "selected": [path.stem for path in paths], "weights": weights}

    ranked = rank_ingredients(paths, score, deadline, log)
    best_position, best_accuracy = ranked[0]
    if rule == "best_single":
        return load_state(paths[best_position]), {
            "rule": rule,
            "selected": [paths[best_position].stem],
            "best_single_proxy": best_accuracy,
            "ranking": [(paths[p].stem, a) for p, a in ranked],
        }
    if rule != "strict_greedy":
        raise ValueError(f"unknown selection rule {rule!r}; shipped rules are {RULES}")

    # Greedy with strict improvement: walk the ranking, and keep an ingredient only
    # if adding it to the running average raises the proxy score. This is upstream's
    # algorithm: add an ingredient only when it improves the proxy score.
    chosen = [best_position]
    current = load_state(paths[best_position])
    current_accuracy = best_accuracy
    per_trial = 0.0
    for position, _ in ranked[1:max_ingredients]:
        if deadline.would_overrun(per_trial):
            log(f"soup: deadline reached with {len(chosen)} ingredient(s) chosen")
            break
        started = time.monotonic()
        count = len(chosen)
        candidate = load_state(paths[position])
        trial = {
            key: value * (count / (count + 1.0)) + candidate[key] * (1.0 / (count + 1.0))
            if value.is_floating_point()
            else value
            for key, value in current.items()
        }
        del candidate
        accuracy = score(trial)
        per_trial = max(per_trial, time.monotonic() - started)
        keep = accuracy > current_accuracy
        log(
            f"soup: + {paths[position].stem} -> {accuracy:.4f} "
            f"({'keep' if keep else 'drop'}, best {current_accuracy:.4f})"
        )
        if keep:
            current, current_accuracy = trial, accuracy
            # The count above is len(chosen), so this has to grow or every
            # subsequent merge would weight the running soup as if it held one
            # ingredient -- the running average would drift towards the last
            # candidate instead of staying an average.
            chosen.append(position)
        else:
            del trial
    return current, {
        "rule": rule,
        "selected": [paths[position].stem for position in chosen],
        "best_single_proxy": best_accuracy,
        "greedy_proxy": current_accuracy,
        "ranking": [(paths[p].stem, a) for p, a in ranked],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ingredients", type=Path, default=Path("/assets/models/ingredients"))
    parser.add_argument("--clip-cache", type=Path, default=Path("/assets/models/clip"))
    parser.add_argument("--data", type=Path, default=Path("/assets/data/imagenetv2_proxy"))
    parser.add_argument("--output", type=Path, default=Path("/out/soup"))
    parser.add_argument("--selection-rule", default="uniform")
    parser.add_argument("--validation-per-class", type=int, default=2)
    parser.add_argument("--max-ingredients", type=int, default=EXPECTED_INGREDIENTS)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-wall-seconds",
        type=float,
        default=0.0,
        help="0 means no deadline. The 12 h phase passes one; see run.sh.",
    )
    parser.add_argument("--reserve-seconds", type=float, default=900.0)
    args = parser.parse_args()

    import torch

    torch.manual_seed(args.seed)
    deadline = Deadline(args.max_wall_seconds, args.reserve_seconds)
    paths = ingredient_paths(args.ingredients)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        print(f"[{deadline.elapsed:7.1f}s] {message}", flush=True)

    log(f"soup: {len(paths)} ingredients, rule {args.selection_rule}")
    base_model, preprocess = load_clip(args.clip_cache)
    dataset = cache_validation(args.data, args.validation_per_class, preprocess)
    log(f"soup: cached {len(dataset)} validation images")
    score = scorer(dataset, base_model, args.batch_size)

    state, trace = build(
        paths, args.selection_rule, score, args.max_ingredients, deadline, log
    )
    checkpoint = output / "model.pt"
    torch.save(state, checkpoint)
    summary = {
        "schema_version": 1,
        "task_id": "model_soup_clip_imagenetv2",
        "ingredients": len(paths),
        "validation_examples": len(dataset),
        "validation_per_class": args.validation_per_class,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "wall_seconds": deadline.elapsed,
        "deadline_seconds": args.max_wall_seconds,
        "checkpoint": str(checkpoint),
        **trace,
    }
    # Written because it is useful to read, not because anything trusts it. The old
    # branch's evaluator read a file like this one and compared its
    # `algorithm_family` field against a constant, which is a check a candidate
    # passes by typing. The final measures the exported weights instead.
    atomic_json(output / "retrain_summary.json", summary)
    log(f"soup: exported {checkpoint}")
    print(json.dumps({key: value for key, value in summary.items() if key != "ranking"},
                     sort_keys=True))


if __name__ == "__main__":
    main()
