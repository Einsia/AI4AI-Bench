"""ImageNetV2 accuracy on the 2,000 proxy rows used during exploration.

This is the final evaluator's forward path over offsets 0-1 of each class; the final
uses offsets 0-9. The metric is deterministic for a fixed checkpoint and row set.
Use `--classes` and different seeds to test an ordering on separately sampled class
subsets, and do not report a selected proxy score as the 10,000-row final result.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import soup_check  # noqa: E402
from forward import BATCH_SIZE, build_model, load_clip, score_rows  # noqa: E402
from grade import PROXY_OFFSETS, class_offsets, select, summarize  # noqa: E402

TASK_ID = "model_soup_clip_imagenetv2"
METRIC = "imagenetv2_top1_proxy2000"
EXPECTED_ROWS = 2000
# 0 means every class, which is the default. A subset is for splitting the row set
# in half to test whether a ranking survives, not for saving time -- 2000 images is
# already the cheapest thing in this task.
DEFAULT_CLASSES = 0
DEFAULT_SEED = 42


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def resolve_checkpoint(checkpoint: Path) -> Path:
    """The soup, given the file or a directory holding one."""

    if checkpoint.is_file():
        return checkpoint
    direct = checkpoint / "model.pt"
    if direct.is_file():
        return direct
    candidates = sorted(checkpoint.glob("*.pt")) + sorted(checkpoint.glob("*/model.pt"))
    if not candidates:
        raise FileNotFoundError(f"no .pt under {checkpoint}")
    # Newest, so `fast_eval.sh /out` scores what was just written. The final does
    # not do this -- there, the operator names the checkpoint.
    return max(candidates, key=lambda path: path.stat().st_mtime)


def restrict(
    rows: list[tuple[Path, int]], data: Path, classes: int, seed: int
) -> list[tuple[Path, int]]:
    """Keep `classes` classes, chosen by seed. 0 keeps all of them.

    The subset is a function of the seed alone, not of the checkpoint, so two calls
    at one seed are comparable and two calls at different seeds are a genuine
    split-half test of whether a ranking holds.
    """

    if classes <= 0:
        return rows
    labels = sorted(class_offsets(data))
    if classes > len(labels):
        raise ValueError(f"asked for {classes} classes, the tree has {len(labels)}")
    order = list(labels)
    random.Random(seed).shuffle(order)
    keep = set(order[:classes])
    return [(path, label) for path, label in rows if label in keep]


def evaluate(
    checkpoint: Path,
    data: Path,
    clip_cache: Path,
    out: Path,
    classes: int,
    seed: int,
) -> dict[str, Any]:
    model_path = resolve_checkpoint(checkpoint)
    rows = restrict(select(data, PROXY_OFFSETS), data, classes, seed)
    if classes <= 0 and len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"expected {EXPECTED_ROWS} proxy rows, selected {len(rows)}")
    base_model, preprocess = load_clip(clip_cache)
    state = soup_check.load_state(model_path)
    started = time.monotonic()
    graded = score_rows(build_model(state, base_model), rows, preprocess, PROXY_OFFSETS)
    elapsed = time.monotonic() - started
    payload = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "metric": METRIC,
        "direction": "maximize",
        "checkpoint": str(model_path),
        "offsets": list(PROXY_OFFSETS),
        "classes_requested": classes,
        "seed": seed,
        "batch_size": BATCH_SIZE,
        "seconds": elapsed,
        **summarize(graded),
    }
    atomic_json(out, payload)
    out.with_name(out.stem + "-rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in graded), encoding="utf-8"
    )
    return payload


def mock(out: Path, classes: int, seed: int) -> dict[str, Any]:
    count = classes if classes > 0 else 1000
    rows = [
        {
            "label": label,
            "offset": offset,
            "correct": bool((label * 5 + offset) % 8),
        }
        for label in range(count)
        for offset in PROXY_OFFSETS
    ]
    payload = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "metric": METRIC,
        "direction": "maximize",
        "mock": True,
        "seed": seed,
        "seconds": 0.0,
        **summarize(rows),
    }
    atomic_json(out, payload)
    return payload


def smoke() -> None:
    payload = mock(Path("/tmp/fast_eval-smoke.json"), DEFAULT_CLASSES, DEFAULT_SEED)
    if payload["n"] != EXPECTED_ROWS:
        raise RuntimeError(f"unexpected row count: {payload}")
    if payload["stderr"] <= 0.0:
        raise RuntimeError(f"stderr must be positive on varied data: {payload}")
    print(json.dumps({"fast_eval_smoke": "passed", "n": payload["n"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data", type=Path, default=Path("/assets/data/imagenetv2_proxy"))
    parser.add_argument("--clip-cache", type=Path, default=Path("/assets/models/clip"))
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--classes",
        type=int,
        default=DEFAULT_CLASSES,
        help="0 (default) means all 1000. A subset is for a split-half check -- two "
        "different seeds give two disjoint-ish halves -- not for saving time.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="chooses classes only")
    parser.add_argument("--mock", action="store_true", help="synthetic rows, no GPU")
    parser.add_argument("--smoke", action="store_true", help="self-check and exit")
    args = parser.parse_args()

    if args.smoke:
        smoke()
        return
    if args.out is None:
        parser.error("--out is required")
    if args.mock:
        print(json.dumps(mock(args.out, args.classes, args.seed), sort_keys=True))
        return
    if args.checkpoint is None:
        parser.error("--checkpoint is required outside --mock/--smoke")
    payload = evaluate(
        args.checkpoint.resolve(),
        args.data.resolve(),
        args.clip_cache.resolve(),
        args.out.resolve(),
        args.classes,
        args.seed,
    )
    print(
        json.dumps(
            {
                "score": payload["score"],
                "stderr": payload["stderr"],
                "n": payload["n"],
                "classes": payload["classes"],
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
