"""Is this checkpoint actually a soup of the 72 frozen ingredients?

This is the artifact-side check, and it is the whole boundary of the task. Every
other constraint here is held by a mount: the ingredients are read-only, the
images are read-only, there is no network. The one invariant no mount can hold is
a property of the *produced weights* -- "the candidate may only select or weight
the fixed ingredients". A container that can read 72 checkpoints and write a file
can write a fine-tuned checkpoint, and nothing about the mount list distinguishes
that file from a soup.

## Affine-hull test

The 72 ingredients span an affine subspace of dimension at most 71. Any
selection, any weighting, any greedy merge -- everything the task permits -- lands
inside it. Fine-tuning does not: a gradient step moves along a direction that has
no reason to lie in the span of 72 fixed checkpoints, and in a parameter space of
this size a direction that is not deliberately inside a 71-dimensional subspace is
very nearly orthogonal to it. So the question "is this a soup" becomes "is this
point in the affine hull of the ingredients", and that is a least-squares problem
with a residual.

    pass 1   stream the 72 ingredients. Accumulate the exact mean in float64,
             sketch each one onto a fixed pseudorandom coordinate subset, and
             accumulate a uniform soup in the ingredients' own dtype -- a
             known-honest artifact, built by the same arithmetic a candidate uses.
    solve    least squares on the sketch, in differences from ingredient 0, so
             the coefficients are affine by construction.
    pass 2   stream again and evaluate the residual on every coordinate, not on
             the sketch.

The sketch chooses the coefficients; the full pass measures the truth. That
ordering is what makes coordinate subsampling safe. A candidate whose deviation
lives in coordinates the sketch never looked at gets coefficients fitted as if it
were honest -- and then pass 2 reports the deviation at full size. The sketch
cannot hide anything, because nothing is concluded from it.

## The tolerance, and why it is not a number I picked

An honest soup does not have residual zero: it is 72 rounded multiply-adds in
float32. So the check needs to know what honest float noise looks like on this
data, in this dtype, and that quantity is measured in the same run -- the uniform
soup accumulated in pass 1, against the float64 mean of the same ingredients, is
exactly the rounding error of an honest average and nothing else.

Both residuals are then expressed relative to the norm of the checkpoint they
came from. That choice is load-bearing rather than cosmetic: the ingredients are
all fine-tunes of one initialization, so they sit close together relative to their
own length, and a residual expressed relative to the *spread* of the ingredient
set would carry that unknown ratio into the threshold. Relative to ||y|| the
ratio cancels and the candidate and the reference are the same units.

    accept when   residual/||y||  <=  max(FLOOR, MULTIPLE x reference residual/||ref||)

FLOOR is 1e-6, which is about sqrt(72) x float32 epsilon: the noise a
72-term float32 average carries even if the in-run reference comes out
anomalously small. MULTIPLE is 10, for a candidate that accumulates in a
different order or through a different intermediate dtype.

Both are loose, and they can afford to be, because the threshold does not sit
between two touching distributions -- it sits in an empty region. From `--smoke`,
which runs the real code path over a synthetic basis with the same close-together
geometry as the real ingredients:

    honest, float32-rounded 8-term soup     5.1e-08
    FLOOR                                   1.0e-06     19x above honest
    smallest violation tested (1e-4 noise)  1.0e-04    100x above the floor

So the check resolves any deviation larger than about one part per million of the
weight norm. A fine-tuning run that moves the metric moves weights by percents,
four to five orders of magnitude above that. There is no fine-tune that is both
consequential and small enough to hide here, which is why the numbers above are
not a tuned trade between false positives and false negatives.

These are synthetic. What they establish is that the arithmetic separates the two
cases by ~2000x on data with the right geometry; the real separation has not been
measured, because that needs the 30.6 GiB asset and a machine to stream it on.

`residual/spread` is reported too, because it is the interpretable one: it says
how far off the hull the checkpoint is in units of how far apart the ingredients
are. It is not what the acceptance test reads.

## Test cases

`--smoke` does this synthetically and is the test that runs in CI. On real
assets, the cheapest true violation is:

    1. build any legal soup, e.g. the uniform average of all 72;
    2. wrap it with `get_model_from_sd` and fine-tune it on ImageNet-scale data --
       or on nothing at all: a hundred steps of SGD at lr=1e-4 against a random
       label permutation is enough, since the check reads the weights and not the
       score;
    3. write the resulting state dict, with a `training_metadata.json` claiming
       `algorithm_family = "model_soup_weight_averaging"`.

The synthetic smoke also checks a rescaled ingredient, added Gaussian noise and a
replaced classifier head. These cases exercise affine scaling, dense deviation and
sparse deviation respectively.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Coordinates kept in the sketch. The system being solved has 71 columns, so this
# is about 3700x oversampled -- the sketch is not where the accuracy comes from
# (pass 2 is), it only has to pin the coefficients. 262144 float64 per ingredient
# is 151 MiB for all 72 held at once.
SKETCH_COORDINATES = 262144
# Fixed, so two runs of the check on one checkpoint agree exactly.
SKETCH_SEED = 20260803
# See the module docstring. FLOOR is sqrt(72) x float32 epsilon rounded up;
# MULTIPLE is slack for an honest candidate whose accumulation differs from ours.
RESIDUAL_FLOOR = 1e-6
RESIDUAL_MULTIPLE = 10.0
class SoupBoundaryError(RuntimeError):
    """The checkpoint is not a combination of the fixed ingredients."""


# --------------------------------------------------------------------------- #
# linear algebra
#
# Written against two vector types on purpose: numpy arrays in a real run, plain
# lists in --smoke. Everything below reduces to `dot`, and the only object that
# survives it is a 71x71 Gram matrix -- small enough that the solve itself is
# pure Python in both cases. Keeping it stdlib-only is what lets the smoke test,
# and therefore the violating-artifact test, run on a machine with no torch and
# no GPU. That test is the reason to trust the check at all, so it should not
# need the hardware the check runs on.
# --------------------------------------------------------------------------- #


def dot(left: Any, right: Any) -> float:
    """Inner product in float64, for a torch tensor, a numpy array or a list."""

    if isinstance(left, list):
        return math.fsum(a * b for a, b in zip(left, right, strict=True))
    return float(left @ right)


def subtract(left: Any, right: Any) -> Any:
    if isinstance(left, list):
        return [a - b for a, b in zip(left, right, strict=True)]
    return left - right


def take(vector: Any, index: Any) -> Any:
    """Gather the sketch coordinates. `index` is whatever `vector` indexes with."""

    if isinstance(vector, list):
        return [vector[position] for position in index]
    return vector[index]


def norm(vector: Any) -> float:
    if isinstance(vector, list):
        return math.sqrt(math.fsum(value * value for value in vector))
    return float((vector @ vector) ** 0.5)


def cholesky_solve(gram: list[list[float]], rhs: list[float], ridge: float) -> list[float]:
    """Solve a symmetric positive-definite system, with a ridge if it is not.

    The Gram matrix is built from the 71 differences x_i - x_0, which are affinely
    independent for any 72 distinct checkpoints, so it is positive definite in
    every case that is not degenerate. `ridge` is a relative Tikhonov term applied
    to the diagonal; it exists so a degenerate ingredient set -- two byte-identical
    checkpoints in the release, say -- produces a slightly-off set of coefficients
    and a reported residual instead of a crash. A coefficient set that is off does
    not weaken the check: pass 2 measures the residual those coefficients actually
    achieve, so a bad solve can only make a checkpoint look less like a soup, never
    more.
    """

    size = len(gram)
    scale = max((gram[i][i] for i in range(size)), default=1.0) or 1.0
    matrix = [[gram[i][j] + (ridge * scale if i == j else 0.0) for j in range(size)]
              for i in range(size)]
    lower = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1):
            total = matrix[i][j] - math.fsum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if total <= 0.0:
                    raise SoupBoundaryError(
                        "the ingredient set is affinely degenerate: the Gram matrix of "
                        f"differences is not positive definite at column {i} even with a "
                        f"{ridge:g} ridge. Two ingredients may be identical."
                    )
                lower[i][i] = math.sqrt(total)
            else:
                lower[i][j] = total / lower[j][j]
    forward = [0.0] * size
    for i in range(size):
        forward[i] = (rhs[i] - math.fsum(lower[i][k] * forward[k] for k in range(i))) / lower[i][i]
    result = [0.0] * size
    for i in reversed(range(size)):
        tail = math.fsum(lower[k][i] * result[k] for k in range(i + 1, size))
        result[i] = (forward[i] - tail) / lower[i][i]
    return result


def affine_coefficients(sketch: list[Any], target: Any, ridge: float = 1e-12) -> list[float]:
    """Coefficients of the best affine combination of `sketch` matching `target`.

    Solved in differences from the first ingredient, so `sum(weights) == 1` holds
    by construction rather than by assertion. That matters, because with 72
    near-collinear checkpoints the weights are not identifiable: many coefficient
    vectors reproduce the same point to within float noise, and a check that
    tested `abs(sum(w) - 1) < tol` on a solved vector would be testing an artifact
    of the solver. The affine hull is well defined even when the coordinates on it
    are not, so the hull is what gets tested and the weights are only reported.
    """

    anchor = sketch[0]
    columns = [subtract(item, anchor) for item in sketch[1:]]
    goal = subtract(target, anchor)
    gram = [[0.0] * len(columns) for _ in columns]
    for i, left in enumerate(columns):
        for j in range(i, len(columns)):
            value = dot(left, columns[j])
            gram[i][j] = gram[j][i] = value
    rhs = [dot(column, goal) for column in columns]
    tail = cholesky_solve(gram, rhs, ridge)
    return [1.0 - math.fsum(tail), *tail]


def implied_scale(sketch: list[Any], target: Any, ridge: float = 1e-10) -> float:
    """Sum of the unconstrained coefficients: 1.0 for a soup, S for S x a soup.

    Diagnostic only, and computed on the sketch, so it never decides anything. It
    makes coefficient normalization errors visible. A checkpoint that is 1.0004 times a
    legal soup is not on the affine hull -- it is a rescaled network rather than an
    average of the ingredients -- and the residual alone would send someone
    looking for a bug in their averaging instead of at their normalization. With
    this, the error message can name the number.

    The individual coefficients here are not trustworthy, because the ingredients
    are near-collinear and the unconstrained Gram matrix is correspondingly
    ill-conditioned. Their sum is the component along the direction all the
    ingredients share, which is the well-determined part and the only part used.
    """

    gram = [[0.0] * len(sketch) for _ in sketch]
    for i, left in enumerate(sketch):
        for j in range(i, len(sketch)):
            gram[i][j] = gram[j][i] = dot(left, sketch[j])
    rhs = [dot(column, target) for column in sketch]
    return math.fsum(cholesky_solve(gram, rhs, ridge))


# --------------------------------------------------------------------------- #
# the verdict
# --------------------------------------------------------------------------- #


@dataclass
class HullReport:
    """Everything the check measured, and whether it accepts.

    Kept as data rather than as a bool so the summary can carry the margin. A
    check that prints "passed" tells an operator nothing about how close it was,
    and this one is expected to pass by five orders of magnitude -- if that margin
    ever narrows, the number is the warning.
    """

    residual: float
    checkpoint_norm: float
    reference_residual: float
    reference_norm: float
    ingredient_spread: float
    distance_from_mean: float
    weights: list[float] = field(default_factory=list)
    sketch_coordinates: int = 0
    dimension: int = 0
    # Sum of the unconstrained coefficients, from the sketch. 1.0 for a soup.
    scale_estimate: float = 1.0

    @property
    def residual_over_norm(self) -> float:
        return self.residual / self.checkpoint_norm if self.checkpoint_norm else float("inf")

    @property
    def reference_residual_over_norm(self) -> float:
        return self.reference_residual / self.reference_norm if self.reference_norm else 0.0

    @property
    def residual_over_spread(self) -> float:
        return self.residual / self.ingredient_spread if self.ingredient_spread else float("inf")

    @property
    def extrapolation_ratio(self) -> float:
        spread = self.ingredient_spread
        return self.distance_from_mean / spread if spread else float("inf")

    @property
    def tolerance(self) -> float:
        return max(RESIDUAL_FLOOR, RESIDUAL_MULTIPLE * self.reference_residual_over_norm)

    @property
    def in_hull(self) -> bool:
        return self.residual_over_norm <= self.tolerance

    def as_dict(self) -> dict[str, Any]:
        return {
            "in_affine_hull": self.in_hull,
            "residual_over_norm": self.residual_over_norm,
            "tolerance": self.tolerance,
            "margin": self.tolerance / self.residual_over_norm
            if self.residual_over_norm
            else float("inf"),
            "reference_residual_over_norm": self.reference_residual_over_norm,
            "residual_over_spread": self.residual_over_spread,
            "extrapolation_ratio": self.extrapolation_ratio,
            "residual": self.residual,
            "checkpoint_norm": self.checkpoint_norm,
            "ingredient_spread": self.ingredient_spread,
            "distance_from_mean": self.distance_from_mean,
            "weight_sum": math.fsum(self.weights) if self.weights else None,
            "weight_min": min(self.weights) if self.weights else None,
            "weight_max": max(self.weights) if self.weights else None,
            # Reported, never enforced. With near-collinear ingredients the
            # coefficients are not identifiable, so "some solution has a negative
            # weight" is a fact about the solver and not about the candidate. The
            # artifact boundary is affine-hull membership, not coefficient sign.
            "weights_are_convex_as_solved": bool(self.weights) and min(self.weights) >= 0.0,
            "weights": self.weights,
            "sketch_coordinates": self.sketch_coordinates,
            "dimension": self.dimension,
            "scale_estimate": self.scale_estimate,
        }

    @property
    def looks_rescaled(self) -> bool:
        """Is the residual explained by weights that do not sum to 1?

        `(S - 1) x ||checkpoint||` is how far a soup scaled by S sits from the
        affine hull. If that accounts for most of the measured residual, the
        checkpoint is a legal soup that was not normalized, and saying so is more
        use than reporting a distance.

        Only meaningful once the hull test has failed. On a checkpoint that passes,
        the residual is float dust and so is `S - 1`, and their ratio is noise
        divided by noise -- which read as "rescaled" for a soup that was nothing of
        the kind until this was gated on `in_hull`.
        """

        if self.in_hull:
            return False
        explained = abs(self.scale_estimate - 1.0) * self.checkpoint_norm
        return self.residual > 0.0 and explained > 0.5 * self.residual

    def raise_if_violating(self) -> None:
        if not self.in_hull:
            detail = ""
            if self.looks_rescaled:
                detail = (
                    f"\nThe coefficients sum to about {self.scale_estimate:.6f} rather than "
                    "1.0, and that alone accounts for most of the distance: this looks like "
                    "a legal soup whose weights were never normalized. Divide the weights by "
                    "their sum and export again."
                )
            raise SoupBoundaryError(
                "the exported checkpoint is not a combination of the 72 fixed "
                "ingredients.\n"
                f"  residual / ||checkpoint||   {self.residual_over_norm:.3e}\n"
                f"  tolerance                   {self.tolerance:.3e}\n"
                f"  honest float32 reference    {self.reference_residual_over_norm:.3e}\n"
                f"  residual / ingredient spread {self.residual_over_spread:.3e}\n"
                "The distance from the ingredients' affine hull is far larger than the "
                "rounding error of an average of them, which is what training on top of "
                "a soup looks like. Selecting and weighting the ingredients cannot "
                "produce this checkpoint." + detail
            )


# --------------------------------------------------------------------------- #
# reading checkpoints
# --------------------------------------------------------------------------- #


def load_state(path: Path) -> dict[str, Any]:
    """Load a state dict, unwrapping the containers people save them inside.

    The ingredients are plain state dicts -- `torch.load` then
    `utils.get_model_from_sd` is what upstream does -- so a candidate that saves a
    module, an optimizer bundle or a Lightning-style checkpoint gets unwrapped
    here rather than rejected, since the packaging is not the boundary.
    """

    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("state_dict", "model_state_dict", "model"):
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            obj = obj[key]
            break
    if hasattr(obj, "state_dict") and not isinstance(obj, dict):
        obj = obj.state_dict()
    if not isinstance(obj, dict):
        raise SoupBoundaryError(f"{path} does not hold a state dict, it holds {type(obj)!r}")
    import torch as _torch

    bad = [key for key, value in obj.items() if not _torch.is_tensor(value)]
    if bad:
        raise SoupBoundaryError(f"{path}: {len(bad)} entries are not tensors, e.g. {bad[:3]}")
    return obj


def schema_of(state: dict[str, Any]) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    """Keys, shapes and dtypes, ordered by key. The identity of the architecture."""

    return tuple(
        (key, tuple(state[key].shape), str(state[key].dtype)) for key in sorted(state)
    )


def require_same_schema(candidate: dict[str, Any], reference: dict[str, Any]) -> None:
    """Refuse a checkpoint whose tensors are not the ingredients' tensors.

    This is the cheap half of the check and it runs first, because it catches the
    coarse violations -- a different CLIP variant, an added adapter, a head resized
    to a different label set -- with no arithmetic at all, and because the residual
    below is only meaningful between two vectors of the same length.

    dtype is part of the comparison rather than coerced. A soup of float32
    ingredients is float32; exporting float16 would halve the precision of the
    thing being measured and, more to the point, would make the in-run float noise
    reference an answer to a different question than the one being asked.
    """

    want, got = schema_of(reference), schema_of(candidate)
    if want == got:
        return
    want_keys = {item[0] for item in want}
    got_keys = {item[0] for item in got}
    missing = sorted(want_keys - got_keys)
    extra = sorted(got_keys - want_keys)
    reference_by_key = {key: (shape, dtype) for key, shape, dtype in want}
    changed = [
        f"{key}: ingredients have {reference_by_key[key]}, checkpoint has {(shape, dtype)}"
        for key, shape, dtype in got
        if key in reference_by_key and reference_by_key[key] != (shape, dtype)
    ]
    detail = []
    if missing:
        detail.append(f"  missing {len(missing)} tensor(s): {missing[:5]}")
    if extra:
        detail.append(f"  {len(extra)} tensor(s) the ingredients do not have: {extra[:5]}")
    if changed:
        detail.append("  " + "\n  ".join(changed[:5]))
    raise SoupBoundaryError(
        "the exported checkpoint does not have the ingredients' tensor layout, so it is "
        "not a combination of them:\n" + "\n".join(detail)
    )


def flatten(state: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """One float64 vector, in sorted key order. Deterministic across calls.

    float64 at the concatenation rather than after it, for two reasons: a state
    dict whose tensors are not all one float dtype cannot be concatenated at all,
    and the residual being measured is smaller than float32 can represent next to
    the weights themselves.
    """

    import torch

    return torch.cat([state[key].reshape(-1).double() for key in keys])


def float_keys(state: dict[str, Any]) -> tuple[str, ...]:
    """The tensors that participate in an average, sorted.

    Integer buffers -- position ids, token type ids, anything registered as a
    counter -- are excluded from the residual because averaging them is not
    defined, and they are covered instead by require_same_schema, which demands
    they be bit-identical to the ingredients'.
    """

    return tuple(key for key in sorted(state) if state[key].is_floating_point())


def require_identical_integer_buffers(
    candidate: dict[str, Any], reference: dict[str, Any]
) -> list[str]:
    """Non-float tensors must equal the ingredients' exactly, not approximately."""

    import torch

    mismatched = [
        key
        for key in sorted(reference)
        if not reference[key].is_floating_point()
        and not torch.equal(candidate[key], reference[key])
    ]
    if mismatched:
        raise SoupBoundaryError(
            f"{len(mismatched)} non-float tensor(s) differ from the ingredients, e.g. "
            f"{mismatched[:3]}. These are buffers rather than weights; an average of the "
            "ingredients leaves them untouched."
        )
    return mismatched


EXPECTED_INGREDIENTS = 72


def ingredient_paths(root: Path) -> list[Path]:
    """The 72 ingredients, ordered numerically. Refuses a partial mount.

    Sorted by the trailing integer rather than by name, for the reason OPD's
    checkpoint collection was: `model_9` sorts after `model_71` lexicographically.
    Here it would not change the residual -- the hull does not care about column
    order -- but it would scramble the reported weight vector against the
    ingredient names, which is the part a human reads.
    """

    paths = sorted(root.glob("model_*.pt"), key=lambda item: int(item.stem.split("_")[-1]))
    if len(paths) != EXPECTED_INGREDIENTS:
        raise SoupBoundaryError(
            f"expected {EXPECTED_INGREDIENTS} ingredients under {root}, found {len(paths)}. "
            "The check solves against the whole fixed set; a partial mount would make a "
            "legal soup look like a violation."
        )
    return paths


def sketch_indices(dimension: int, count: int) -> list[int]:
    """A fixed pseudorandom coordinate subset, sorted.

    Sorted for locality when gathering, and fixed so two runs agree. Nothing is
    concluded from these coordinates -- they choose the coefficients and pass 2
    measures the residual on all of them -- so the exact subset is not part of the
    task's definition, which is why a stdlib RNG is enough.
    """

    if count >= dimension:
        return list(range(dimension))
    return sorted(random.Random(SKETCH_SEED).sample(range(dimension), count))


def measure(
    load: Any,
    count: int,
    target: Any,
    index: Any,
    *,
    round_reference: Any,
    zeros: Any,
    log: Any = None,
) -> HullReport:
    """The two passes, over anything that can be loaded, summed and dotted.

    `load(i)` returns ingredient i as a float64 vector; `zeros()` a fresh
    accumulator; `round_reference(v)` rounds a vector to whatever precision an
    honest candidate would accumulate in. Written this way so that --smoke runs
    exactly this function over a synthetic basis of Python lists -- the violating
    artifact is tested against the code that scores the real one, not against a
    reimplementation of it that could agree with the check while both are wrong.
    """

    def note(message: str) -> None:
        if log is not None:
            log(message)

    scale = 1.0 / count
    mean = zeros()
    reference = zeros()
    rows = []
    for position in range(count):
        vector = load(position)
        mean = add(mean, vector, 1.0)
        # Scale-then-sum, which is exactly upstream's
        # `sum(state[key] * scale for state in states)`. The point of building this
        # is not the soup -- it is the rounding error, which is the tolerance.
        reference = add(reference, round_reference(scale_vector(vector, scale)), 1.0)
        rows.append(take(vector, index))
    mean = scale_vector(mean, scale)
    reference = round_reference(reference)
    note(f"soup_check: pass 1 over {count} ingredients done, solving for the coefficients")

    sketched_target = take(target, index)
    weights = affine_coefficients(rows, sketched_target)
    scale = implied_scale(rows, sketched_target)
    del rows

    estimate = zeros()
    spread = 0.0
    for position in range(count):
        vector = load(position)
        estimate = add(estimate, vector, weights[position])
        spread += norm(subtract(vector, mean))
    note("soup_check: pass 2 done")

    return HullReport(
        residual=norm(subtract(target, estimate)),
        checkpoint_norm=norm(target),
        reference_residual=norm(subtract(reference, mean)),
        reference_norm=norm(reference),
        ingredient_spread=spread / count,
        distance_from_mean=norm(subtract(target, mean)),
        weights=weights,
        sketch_coordinates=len(index),
        dimension=len(target),
        scale_estimate=scale,
    )


def add(accumulator: Any, vector: Any, weight: float) -> Any:
    if isinstance(accumulator, list):
        return [a + weight * b for a, b in zip(accumulator, vector, strict=True)]
    return accumulator + vector * weight


def scale_vector(vector: Any, factor: float) -> Any:
    if isinstance(vector, list):
        return [value * factor for value in vector]
    return vector * factor


def check_checkpoint(
    checkpoint: Path,
    ingredients: Path,
    *,
    coordinates: int = SKETCH_COORDINATES,
    log: Any = None,
) -> HullReport:
    """Measure how far the exported checkpoint sits from the ingredients' hull.

    Two streaming passes over the ingredient set, about 30.6 GiB each. Nothing is
    held that scales with 72: the largest live objects are three vectors the length
    of the model and the 72-row sketch.
    """

    import torch

    def note(message: str) -> None:
        if log is not None:
            log(message)

    paths = ingredient_paths(ingredients)
    first = load_state(paths[0])
    candidate = load_state(checkpoint)
    require_same_schema(candidate, first)
    keys = float_keys(first)
    if not keys:
        raise SoupBoundaryError(f"{paths[0]} holds no floating-point tensors")

    dtypes = {str(first[key].dtype) for key in keys}
    # The dtype an honest candidate would accumulate in. When the ingredients
    # share one, use it, so the in-run float-noise reference is produced by the
    # same arithmetic a candidate's average is. Mixed dtypes cannot be
    # concatenated, so those fall back to float32 and the reference becomes an
    # under-estimate -- which only tightens the check.
    accumulate_dtype = getattr(torch, dtypes.pop().split(".")[-1]) if len(dtypes) == 1 else None
    if accumulate_dtype is None:
        accumulate_dtype = torch.float32

    target = flatten(candidate, keys)
    dimension = int(target.numel())
    index = torch.tensor(sketch_indices(dimension, coordinates), dtype=torch.long)
    note(f"soup_check: {dimension} float parameters, sketching {index.numel()} of them")

    # Every ingredient is read twice, once per pass, rather than held. 72 x 456 MiB
    # does fit in the declared 64 GiB, but only just, and the second read is disk
    # rather than memory pressure.
    varying_buffers: list[str] = []

    def load(position: int) -> Any:
        state = first if position == 0 else load_state(paths[position])
        if position:
            require_same_schema(state, first)
            varying_buffers.extend(
                key
                for key in sorted(state)
                if not state[key].is_floating_point()
                and not torch.equal(state[key], first[key])
                and key not in varying_buffers
            )
        return flatten(state, keys)

    report = measure(
        load,
        len(paths),
        target,
        index,
        # An honest candidate averages in the ingredients' dtype, so the float-noise
        # reference is built by rounding through it. float64 in, float64 out, with
        # one trip through float32 in between -- which is where the rounding is.
        round_reference=lambda vector: vector.to(accumulate_dtype).double(),
        zeros=lambda: torch.zeros(dimension, dtype=torch.float64),
        log=log,
    )

    # The non-float buffers are only required to match when all 72 ingredients
    # agree on them; if the release itself varies, "equal to ingredient 0" would
    # reject a legal soup of the others.
    if not varying_buffers:
        require_identical_integer_buffers(candidate, first)
    return report


# --------------------------------------------------------------------------- #
# --smoke: the violating artifacts, synthetically
# --------------------------------------------------------------------------- #


def float32(value: float) -> float:
    """Round a Python float through float32, using nothing but the stdlib."""

    import struct

    return struct.unpack("<f", struct.pack("<f", value))[0]


def synthetic_basis(dimension: int, count: int, spread: float) -> list[list[float]]:
    """`count` points that look like fine-tunes of one initialization.

    The geometry matters more than the numbers. Real ingredients are all
    fine-tunes of one CLIP, so they sit close together relative to their own
    length -- `spread` is that ratio, and it is set to 2% here. A synthetic basis of
    independent random vectors would make every test easier to pass than the real
    case, which is the wrong direction for a test of a boundary.
    """

    rng = random.Random(11)
    base = [rng.gauss(0.0, 1.0) for _ in range(dimension)]
    scale = spread * math.sqrt(math.fsum(value * value for value in base) / dimension)
    return [
        [value + rng.gauss(0.0, scale) for value in base] for _ in range(count)
    ]


def combine(basis: list[list[float]], weights: list[float], *, rounding: Any = None) -> list[float]:
    """Weighted sum of the basis, optionally rounding the way float32 would.

    `rounding=float32` is what an honest candidate's average actually produces, and
    the residual it leaves is the reason this check has a tolerance at all.
    """

    total = [0.0] * len(basis[0])
    for weight, vector in zip(weights, basis, strict=True):
        for position, value in enumerate(vector):
            total[position] += weight * value
            if rounding is not None:
                total[position] = rounding(total[position])
    return total


def smoke() -> None:
    dimension, count = 2048, 8
    basis = synthetic_basis(dimension, count, spread=0.02)
    index = sketch_indices(dimension, 256)
    outside_sketch = sorted(set(range(dimension)) - set(index))
    uniform = [1.0 / count] * count

    def run(target: list[float]) -> HullReport:
        return measure(
            lambda position: basis[position],
            count,
            target,
            index,
            round_reference=lambda vector: [float32(value) for value in vector],
            zeros=lambda: [0.0] * dimension,
        )

    rng = random.Random(4242)
    scale = math.sqrt(math.fsum(value * value for value in basis[0]))
    reports: dict[str, HullReport] = {}

    # 1. an exact float64 combination. The residual is float64 noise, not zero.
    weights = [0.4, 0.3, 0.1, 0.1, 0.05, 0.05, 0.0, 0.0]
    assert abs(math.fsum(weights) - 1.0) < 1e-12, weights
    reports["exact_convex"] = run(combine(basis, weights))
    # 2. the same soup as a candidate would actually produce it: float32 rounding
    #    on every partial sum. This is the case the tolerance exists for.
    reports["float32_convex"] = run(combine(basis, weights, rounding=float32))
    # 3. one ingredient, unchanged. best_single is a legal answer and has to pass.
    reports["single_ingredient"] = run(list(basis[3]))
    # 4. mass moved between two ingredients, and one weight taken negative. Still
    #    an affine combination, so it passes -- the check is about the hull, not
    #    about the weights being tidy or even non-negative.
    moved = [0.6, -0.1, 0.1, 0.1, 0.05, 0.05, 0.1, 0.1]
    assert abs(math.fsum(moved) - 1.0) < 1e-12, moved
    reports["reweighted"] = run(combine(basis, moved))
    # 5. a legal soup, scaled by 1.0004. Not on the affine hull: the weights no
    #    longer sum to 1, so this is a rescaled network rather than an average. It
    #    has to fail, and the report has to say why -- this is the only way an
    #    honest candidate trips the check, and "residual too large" would send them
    #    looking for a bug in their averaging.
    reports["unnormalized"] = run([1.0004 * value for value in combine(basis, weights)])
    # 6. a soup plus dense noise at 1e-4 of the weight norm. This is the shape of
    #    "fine-tuned a little on top of a soup".
    honest = combine(basis, uniform)
    noise = [rng.gauss(0.0, 1.0) for _ in range(dimension)]
    noise_scale = 1e-4 * scale / math.sqrt(math.fsum(v * v for v in noise))
    reports["noised_1e-4"] = run([a + noise_scale * b for a, b in zip(honest, noise, strict=True)])
    # 7. the same magnitude of deviation, but confined to coordinates the sketch
    #    never looks at -- the classifier head alone, in the real case. This is the
    #    test that the sketch is not the last word: it fits coefficients as if the
    #    checkpoint were honest, and pass 2 reports the deviation anyway.
    sparse = list(honest)
    step = 1e-4 * scale / math.sqrt(len(outside_sketch))
    for position in outside_sketch:
        sparse[position] += step
    reports["sparse_outside_sketch"] = run(sparse)
    # 8. a full-size fine-tune, 1e-3 of the norm.
    big = [a + 10.0 * noise_scale * b for a, b in zip(honest, noise, strict=True)]
    reports["noised_1e-3"] = run(big)
    # 9. inside the hull, far outside the ingredient set. This remains a legal
    #    affine combination; distance from the ingredient cloud is diagnostic only.
    reports["extrapolated"] = run(combine(basis, [40.0, -39.0] + [0.0] * (count - 2)))

    passing = (
        "exact_convex",
        "float32_convex",
        "single_ingredient",
        "reweighted",
        "extrapolated",
    )
    failing = ("unnormalized", "noised_1e-4", "sparse_outside_sketch", "noised_1e-3")
    for name in passing:
        report = reports[name]
        assert report.in_hull, (name, report.as_dict())
        report.raise_if_violating()
    for name in failing:
        report = reports[name]
        assert not report.in_hull, (name, report.as_dict())
    assert reports["extrapolated"].in_hull, reports["extrapolated"].as_dict()
    assert reports["extrapolated"].extrapolation_ratio > 10.0
    reports["extrapolated"].raise_if_violating()
    for name in failing:
        try:
            reports[name].raise_if_violating()
        except SoupBoundaryError:
            pass
        else:  # pragma: no cover - the assertions above make this unreachable
            raise AssertionError(f"{name} did not raise")

    # The claim the tolerance rests on: honest and violating are orders of
    # magnitude apart, so the threshold sits in an empty region rather than
    # between two touching distributions.
    honest_worst = max(reports[name].residual_over_norm for name in passing)
    violating_best = min(reports[name].residual_over_norm for name in failing)
    assert violating_best > 100 * honest_worst, (honest_worst, violating_best)
    # And the un-normalized soup is diagnosed as one rather than as a fine-tune.
    assert reports["unnormalized"].looks_rescaled, reports["unnormalized"].as_dict()
    assert abs(reports["unnormalized"].scale_estimate - 1.0004) < 1e-5, (
        reports["unnormalized"].scale_estimate
    )
    for name in (*passing, "noised_1e-3"):
        assert not reports[name].looks_rescaled, (name, reports[name].as_dict())
    # And the sparse case is not detected any less well than the dense one, which
    # is the whole argument for measuring on every coordinate.
    assert reports["sparse_outside_sketch"].residual_over_norm > 20 * honest_worst

    print(
        json.dumps(
            {
                "soup_check_smoke": "passed",
                "honest_worst_residual_over_norm": honest_worst,
                "violating_best_residual_over_norm": violating_best,
                "separation": violating_best / honest_worst,
                "tolerance": reports["float32_convex"].tolerance,
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="the exported soup, a .pt file")
    parser.add_argument("--ingredients", type=Path, default=Path("/assets/models/ingredients"))
    parser.add_argument("--coordinates", type=int, default=SKETCH_COORDINATES)
    parser.add_argument("--smoke", action="store_true", help="self-check on synthetic weights")
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return
    if args.checkpoint is None:
        parser.error("--checkpoint is required outside --smoke")
    report = check_checkpoint(
        args.checkpoint, args.ingredients, coordinates=args.coordinates, log=print
    )
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    report.raise_if_violating()


if __name__ == "__main__":
    main()
