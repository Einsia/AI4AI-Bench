"""The one scoring path, shared by fast_eval and the final.

Two things live here: how an ImageNetV2 image tree becomes an ordered row set,
and how graded rows become a score with an honest error bar. Both stages call
them, so a proxy score and a final score are the same arithmetic over different
rows -- which is what makes `score(P)` computed inside the final comparable with
the proxy number the Agent saw.

Nothing here loads a model or touches CUDA. Keep it that way: this file is the
reason "the evaluator does not change" can be checked by reading one thing.

## The offset rule

ImageNetV2 matched-frequency is 1000 numeric class directories of 10 images each.
`class_offsets` orders the files inside each directory by name and calls position
i the offset-i image of that class. Every split in this task is a set of offsets:

    proxy   offsets 0-1     2000 images
    final   offsets 0-9     10000 images

The rule is deliberately the dullest one available -- sorted position within the
class directory -- because it has to be reproduced in two places that never see
each other. The proxy asset is materialized on the host by taking offsets 0-1;
final_eval.py re-derives the same offsets from the full tree to identify which of
its 10000 rows are the proxy rows. A manifest shipped between them would be one
more thing to keep in sync, and the failure would be silent: the three-number
report would still print, against the wrong partition.

## The error bar

`summarize` reports two standard errors and they differ by about 40% here, so
which one you quote matters.

`stderr` clusters by class: score each of the 1000 classes, then take the
standard error across classes. That is the right one, because the 10 images of a
class are not independent -- CLIP either has the concept or it does not, and
ImageNetV2's per-class accuracy is strongly bimodal.

`stderr_naive_binomial` treats all N images as independent draws. It is kept only
to show what the clustering costs, in the same spirit as OPD's grade.py. Do not
compare two scores against it.

Note what is absent: any notion of a sampling seed. This metric is a deterministic
argmax over fixed images, so re-running an unchanged checkpoint returns the
identical number. OPD's noise floor -- three runs at three seeds on one
checkpoint -- has no analogue. The only thing that moves this score is which rows
are in the set, which is exactly what the P versus F\\P report measures.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

EXPECTED_CLASSES = 1000
EXPECTED_PER_CLASS = 10
PROXY_OFFSETS = (0, 1)
FINAL_OFFSETS = tuple(range(EXPECTED_PER_CLASS))
# Anything torchvision's ImageFolder would take, minus the formats ImageNetV2
# does not ship. Filtering by suffix rather than by `is_file()` keeps a stray
# .DS_Store or a checksum file from becoming an image with a garbage label.
IMAGE_SUFFIXES = frozenset({".jpeg", ".jpg", ".png"})


def class_offsets(root: Path) -> dict[int, list[Path]]:
    """Map each class label to its images, ordered by name.

    The directory name is the label, and it is the *index* CLIP's classifier head
    predicts rather than a WordNet id -- ImageNetV2's matched-frequency release
    numbers its directories 0..999 in the same order the head is trained on, and
    the upstream evaluator relies on that by comparing `argmax` against
    `int(path.parent.name)` directly.
    """

    if not root.is_dir():
        raise FileNotFoundError(f"no ImageNetV2 tree at {root}")
    by_class: dict[int, list[Path]] = defaultdict(list)
    for path in sorted(root.glob("*/*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            label = int(path.parent.name)
        except ValueError as error:
            raise ValueError(
                f"class directory {path.parent.name!r} is not numeric; this tree is "
                "not the matched-frequency-format-val layout"
            ) from error
        by_class[label].append(path)
    if not by_class:
        raise ValueError(f"no images under {root}")
    return {label: sorted(paths) for label, paths in sorted(by_class.items())}


def select(root: Path, offsets: tuple[int, ...]) -> list[tuple[Path, int]]:
    """The (path, label) rows for the given offsets, class-major then offset.

    Refuses a tree that is not 1000 x len(offsets) rather than scoring whatever it
    finds. The reference protocol checked this too, and it is worth keeping for a reason
    that is easy to miss: a half-extracted asset produces a *higher* accuracy on
    the classes that survived, so a silent partial tree looks like an improvement.
    """

    by_class = class_offsets(root)
    if len(by_class) != EXPECTED_CLASSES:
        raise ValueError(f"expected {EXPECTED_CLASSES} classes under {root}, found {len(by_class)}")
    needed = max(offsets) + 1
    short = {label: len(paths) for label, paths in by_class.items() if len(paths) < needed}
    if short:
        sample = sorted(short.items())[:3]
        raise ValueError(
            f"{len(short)} class(es) hold fewer than {needed} images, e.g. {sample}. "
            f"Offsets {offsets} cannot be selected from this tree."
        )
    return [(by_class[label][offset], label) for label in sorted(by_class) for offset in offsets]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate graded rows, clustering the standard error by class.

    A row is `{"label": int, "prediction": int, "correct": bool}`. See the module
    docstring for why `stderr` and `stderr_naive_binomial` are both reported and
    why only the first is comparable.
    """

    if not rows:
        raise ValueError("no rows to summarize")
    by_class: dict[int, list[bool]] = defaultdict(list)
    for row in rows:
        by_class[int(row["label"])].append(bool(row["correct"]))

    per_class = [sum(values) / len(values) for values in by_class.values()]
    classes = len(per_class)
    total = len(rows)
    correct = sum(bool(row["correct"]) for row in rows)
    # The unweighted mean over classes equals correct/total only when every class
    # contributes the same number of rows, which every split in this task does.
    # Score the images rather than the classes so the number does not silently
    # change meaning if that ever stops being true.
    score = correct / total
    if classes > 1:
        stderr = statistics.stdev(per_class) / math.sqrt(classes)
    else:
        stderr = float("nan")
    naive = math.sqrt(max(score * (1.0 - score), 0.0) / total)
    if not math.isfinite(score):
        raise RuntimeError("score is non-finite")
    return {
        "score": score,
        "stderr": stderr,
        "stderr_naive_binomial": naive,
        "classes": classes,
        "images_per_class": total // classes,
        "n": total,
        "correct": correct,
        "step_size": 1.0 / total,
        "class_accuracy_perfect": sum(1 for value in per_class if value == 1.0),
        "class_accuracy_zero": sum(1 for value in per_class if value == 0.0),
    }


def offset_of(row: dict[str, Any]) -> int:
    if "offset" not in row:
        raise KeyError("row has no offset; the three-number report cannot be partitioned")
    return int(row["offset"])


def partition(rows: list[dict[str, Any]], proxy: tuple[int, ...]) -> dict[str, Any]:
    """The three numbers the v1 spec asks for when proxy and final overlap.

    `score(F)` over everything, `score(P)` over the proxy offsets, `score(F\\P)`
    over the rest, and the difference. `overfitting` is the raw
    `score(P) - score(F\\P)`; it is not the selection premium on its own because
    offsets 0-1 need not have the same difficulty as offsets 2-9.
    """

    proxy_set = set(proxy)
    inside = [row for row in rows if offset_of(row) in proxy_set]
    outside = [row for row in rows if offset_of(row) not in proxy_set]
    if not inside or not outside:
        raise ValueError(
            f"partition needs rows on both sides: {len(inside)} in the proxy "
            f"offsets {sorted(proxy_set)}, {len(outside)} outside"
        )
    whole = summarize(rows)
    within = summarize(inside)
    without = summarize(outside)
    return {
        "final": whole,
        "proxy_rows": within,
        "final_minus_proxy_rows": without,
        "overfitting": within["score"] - without["score"],
        "overlap_fraction": within["n"] / whole["n"],
    }


def smoke() -> None:
    """Synthetic rows with a known answer, so the arithmetic is checked without data."""

    rows = [
        {
            "label": label,
            "offset": offset,
            "prediction": label if (label + offset) % 4 else -1,
            "correct": bool((label + offset) % 4),
        }
        for label in range(EXPECTED_CLASSES)
        for offset in FINAL_OFFSETS
    ]
    whole = summarize(rows)
    assert whole["n"] == 10000, whole
    assert whole["classes"] == EXPECTED_CLASSES, whole
    assert whole["images_per_class"] == EXPECTED_PER_CLASS, whole
    assert abs(whole["score"] - 0.75) < 1e-12, whole
    assert whole["step_size"] == 1.0 / 10000, whole

    three = partition(rows, PROXY_OFFSETS)
    assert three["proxy_rows"]["n"] == 2000, three
    assert three["final_minus_proxy_rows"]["n"] == 8000, three
    assert abs(three["overlap_fraction"] - 0.2) < 1e-12, three
    # (label + offset) % 4 makes every class 7/10 or 8/10 correct and puts the
    # same pattern in offsets 0-1 as in 4-5 and 8-9, so P and F\P agree exactly.
    # A three-number report that cannot produce a zero here is measuring its own
    # bookkeeping rather than the checkpoint.
    assert abs(three["overfitting"]) < 1e-12, three

    # Every class identical -> the clustered stderr is 0 while the naive one is
    # not. The clearest possible demonstration that the two are different claims.
    flat = [
        {"label": label, "offset": offset, "correct": offset < 5}
        for label in range(EXPECTED_CLASSES)
        for offset in FINAL_OFFSETS
    ]
    summary = summarize(flat)
    assert summary["stderr"] == 0.0, summary
    assert summary["stderr_naive_binomial"] > 0.0, summary
    print("grade.py smoke passed")


if __name__ == "__main__":
    smoke()
