"""The one grading path, shared by fast_eval and the hidden final.

Both stages sample molecules with the pinned DiGress tree baked into the image at
/opt/harness/digress and score them through this file, so a fast_eval number and a
final number are produced by the same code.

Three things in here are worth reading before trusting the output.

**Upstream stays the authority for the headline numbers.** `parse_metrics` reads
DiGress's own stdout with the regex set the reference protocol used -- that set produced
every recorded number in [metadata], so it is proven rather than plausible.

**The split numbers are auxiliary only.** A generative task has no rows to partition,
so score(P) and score(F\\P) are computed from the sampled molecules themselves. The
upstream stdout remains authoritative for the headline NLL and whole-set rates; the
repository-computed split is additional reporting and never gates, suppresses, or
changes the benchmark score.

**Nothing here loads a model.** The artifact description works on a state dict's shapes
and the checkpoint's stored hyper-parameters, both passed in as plain data. Keep it
that way: it is what makes the check testable without a GPU, and it is why
`describe_checkpoint` and `describe_artifact` take a mapping of names to shapes rather
than a checkpoint path.

Python 3.9 -- the image is conda-based and pins python=3.9, so no tomllib, no
`zip(strict=)`, no match statement.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

UPSTREAM_REVISION = "780242b8d3e7d78316bb5cf90c639fb0cd4c6079"
UPSTREAM_ROOT = Path("/opt/harness/digress")

# QM9 with hydrogens removed. Four atom types and five bond classes, and these
# two widths are what the artifact check pins -- see check_artifact.
QM9_NO_H_ATOM_TYPES = ("C", "N", "O", "F")
QM9_NO_H_BOND_CLASSES = ("none", "single", "double", "triple", "aromatic")
EXPECTED_ATOM_CATEGORIES = len(QM9_NO_H_ATOM_TYPES)
EXPECTED_BOND_CATEGORIES = len(QM9_NO_H_BOND_CLASSES)

# The five metrics the reference protocol required of every complete record.
REQUIRED_METRICS = ("validity", "relaxed_validity", "uniqueness", "novelty", "nll")


# --------------------------------------------------------------------------- #
# reading DiGress's own output
# --------------------------------------------------------------------------- #


def last_metric(text: str, pattern: str, percent: bool = False) -> float | None:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if not matches:
        return None
    value = float(matches[-1])
    return value / 100.0 if percent else value


def parse_metrics(text: str) -> dict[str, float | None]:
    """Read the five metrics out of DiGress's stdout.

    Carried over verbatim from the reference protocol's eval/evaluate.py, including the
    empty-set convention below, because every number in task.toml's [metadata] was
    produced by these exact patterns. Changing one would silently move the
    baseline an improvement is compared against.
    """

    metrics: dict[str, float | None] = {
        "validity": last_metric(text, r"Validity over \d+ molecules:\s*([0-9.]+)%", percent=True),
        "relaxed_validity": last_metric(
            text, r"Relaxed validity over \d+ molecules:\s*([0-9.]+)%", percent=True
        ),
        "uniqueness": last_metric(
            text, r"Uniqueness over \d+ valid molecules:\s*([0-9.]+)%", percent=True
        ),
        "novelty": last_metric(
            text, r"Novelty over \d+ unique valid molecules:\s*([0-9.]+)%", percent=True
        ),
        "nll": last_metric(text, r"(?:Test|Val) NLL\s+(-?[0-9.]+)"),
    }
    # Upstream omits the uniqueness and novelty lines entirely when a checkpoint
    # generates no valid molecules. The reference protocol voided a real training-start run
    # over this before deciding that an empty valid set means a rate of 0.0 rather
    # than an evaluator failure -- see evidence.json invalid_evidence, run
    # digress-training-start-public10000-s43-d401620-g3-v1. A catastrophically bad
    # candidate still gets a complete, comparable record.
    if metrics["validity"] == 0.0:
        if metrics["uniqueness"] is None:
            metrics["uniqueness"] = 0.0
        if metrics["novelty"] is None:
            metrics["novelty"] = 0.0
    return metrics


def observed_sample_count(text: str) -> int | None:
    """How many molecules upstream actually generated.

    The reference protocol shipped a proxy that declared 512 samples, overrode only
    `general.samples_to_generate`, and generated 10000 -- upstream's test path reads
    `general.final_model_samples_to_generate`. The wrong number was recorded as a
    score before stdout was read closely. Both fields are set now, and this is the
    fail-closed half: the declared count and the observed count must agree.
    """

    matches = re.findall(r"Validity over (\d+) molecules:", text, flags=re.MULTILINE)
    return int(matches[-1]) if matches else None


# --------------------------------------------------------------------------- #
# the composite the proxy optimises
# --------------------------------------------------------------------------- #


def composite(metrics: Mapping[str, float | None]) -> float | None:
    """validity x uniqueness x novelty, the old [proxy].metric.

    The reference protocol named this metric `validity_uniqueness_novelty` and never wrote
    down the combining rule, so the rule is chosen here and stated rather than
    inferred silently. The product is the natural reading because upstream's three
    rates are already nested: uniqueness is over *valid* molecules and novelty is
    over *unique valid* ones, so

        validity x uniqueness x novelty

    is the fraction of all drawn samples that are valid, distinct and absent from
    the training set. A sum or a mean would not be a fraction of anything.

    Direction is max, and that is the opposite of what ranks the task. See
    task.toml [metadata] -- the mismatch is real and is not resolved here.
    """

    values = [metrics.get(name) for name in ("validity", "uniqueness", "novelty")]
    if any(value is None for value in values):
        return None
    product = 1.0
    for value in values:
        product *= float(value)
    return product


# --------------------------------------------------------------------------- #
# molecule metrics recomputed from the dumped SMILES, for the split only
# --------------------------------------------------------------------------- #


def canonical_smiles(smiles: str) -> str | None:
    """RDKit's canonical form, or None when the string is not a valid molecule.

    Imported lazily. Nothing at module import time should need RDKit, so that this
    file stays importable for its self-check on a host without it.
    """

    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    text = (smiles or "").strip()
    if not text:
        return None
    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        return None
    try:
        return Chem.MolToSmiles(molecule)
    except Exception:  # noqa: BLE001 -- an unserialisable molecule is an invalid one
        return None


def molecule_metrics(
    smiles: Sequence[str],
    train_smiles: Iterable[str] | None = None,
    canonicalize: Any | None = None,
) -> dict[str, float | None]:
    """validity, uniqueness and novelty over one list of generated SMILES.

    Set-level, exactly as upstream reports them: uniqueness is the share of the
    valid molecules that are distinct, novelty the share of those distinct valid
    molecules absent from the training set. Neither is a per-row average, which is
    why a subset score cannot be recovered arithmetically from the full-set score
    and the molecules have to be carried around.

    `train_smiles` of None leaves novelty None rather than guessing it. An empty
    valid set gives 0.0 for both rates, matching parse_metrics' convention.

    `canonicalize` defaults to RDKit and is injectable so that the set arithmetic --
    which is the part with an off-by-one in it -- can be tested on a host with no
    RDKit. The chemistry and the counting are separate concerns and only one of them
    needs a conda environment.
    """

    total = len(smiles)
    if total == 0:
        raise ValueError("no molecules to score")
    if canonicalize is None:
        canonicalize = canonical_smiles
    canonical = [canonicalize(item) for item in smiles]
    valid = [item for item in canonical if item is not None]
    validity = len(valid) / total
    if not valid:
        return {"validity": 0.0, "uniqueness": 0.0, "novelty": 0.0, "n": float(total)}

    unique = set(valid)
    uniqueness = len(unique) / len(valid)
    novelty: float | None = None
    if train_smiles is not None:
        known = {canonicalize(item) for item in train_smiles}
        known.discard(None)
        novelty = sum(1 for item in unique if item not in known) / len(unique)
    return {
        "validity": validity,
        "uniqueness": uniqueness,
        "novelty": novelty,
        "n": float(total),
    }


def split_scores(
    smiles: Sequence[str],
    proxy_samples: int,
    train_smiles: Iterable[str] | None = None,
    canonicalize: Any | None = None,
) -> dict[str, Any]:
    """The spec's three numbers, plus the one that makes them comparable.

    For this generative task, "the proxy rows" means the first N of the final run's
    own sampled list. That is well defined and free: draw the final's 10000, then
    slice the list in memory. It does not claim that a separate fast-eval run can
    reproduce that slice; the pinned tree's declared seed is inert.

        F     all 10000
        P     the first 2000
        F\\P   the remaining 8000

    `score(P) - score(F\\P)` is the overfitting measurement.

    The fourth number exists because uniqueness and novelty are set-level rates and
    therefore *size-dependent*: a 2000-molecule set has fewer collisions to find
    than an 8000-molecule one, so P scores higher than F\\P on uniqueness by set
    size alone, with nothing overfitted. `rest_head` scores the first 2000 of F\\P,
    which is size-matched to P. Compare P against that; the P-vs-F\\P difference
    mixes the effect with an artefact of counting.
    """

    total = len(smiles)
    if proxy_samples <= 0 or proxy_samples >= total:
        raise ValueError(
            f"proxy_samples must be a proper prefix of the stream: got {proxy_samples} of {total}"
        )
    proxy = list(smiles[:proxy_samples])
    rest = list(smiles[proxy_samples:])
    full_metrics = molecule_metrics(smiles, train_smiles, canonicalize)
    proxy_metrics = molecule_metrics(proxy, train_smiles, canonicalize)
    rest_metrics = molecule_metrics(rest, train_smiles, canonicalize)
    head_metrics = molecule_metrics(rest[:proxy_samples], train_smiles, canonicalize)

    result: dict[str, Any] = {
        "final": full_metrics,
        "proxy": proxy_metrics,
        "rest": rest_metrics,
        "rest_head": head_metrics,
        "proxy_samples": proxy_samples,
        "final_samples": total,
        "overlap_fraction": proxy_samples / total,
        "composite_final": composite(full_metrics),
        "composite_proxy": composite(proxy_metrics),
        "composite_rest": composite(rest_metrics),
        "composite_rest_head": composite(head_metrics),
    }
    if result["composite_proxy"] is not None and result["composite_rest"] is not None:
        result["overfitting"] = result["composite_proxy"] - result["composite_rest"]
    else:
        result["overfitting"] = None
    if result["composite_proxy"] is not None and result["composite_rest_head"] is not None:
        result["overfitting_size_matched"] = (
            result["composite_proxy"] - result["composite_rest_head"]
        )
    else:
        result["overfitting_size_matched"] = None
    return result


# --------------------------------------------------------------------------- #
# artifact description
# --------------------------------------------------------------------------- #

class ArtifactViolation(ValueError):
    """The checkpoint has no state that the frozen scorer can attempt to load."""


def describe_checkpoint(
    shapes: Mapping[str, Sequence[int]], hparams: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Summarise a state dict's structure without judging it.

    Parameter names and shapes are retained as diagnostics only. They describe the
    shipped DiGress checkpoint well, but are not a reliable way to decide which
    candidate algorithms are valid.
    """

    hparams = hparams or {}
    node_like: list[str] = []
    edge_like: list[str] = []
    attention_like: list[str] = []
    schedules: dict[str, int] = {}
    output_widths: dict[str, int] = {}

    for name, shape in shapes.items():
        dims = list(int(value) for value in shape)
        lowered = name.lower()
        # DiGress's denoiser carries separate node and edge streams end to end, and
        # names them X and E throughout -- mlp_in_X, mlp_out_E, the attention block's
        # e_add and e_mul. An autoregressive SMILES decoder has one token stream and
        # no per-edge tensor at all, which is the substitution this check exists to
        # catch.
        #
        # The single letters have to be matched as whole tokens. An earlier version
        # used `(^|[._])(e|edge)`, which matched `pos_embedding` -- so a SMILES
        # decoder was reported as having two "edge-like" tensors. It was still
        # rejected, for having no node stream, but a check whose diagnostic names the
        # wrong reason is a check nobody will trust the next time it fires.
        if re.search(r"(?:^|[._])(?:x|node)(?:$|[._0-9])", lowered):
            node_like.append(name)
        if re.search(r"(?:^|[._])(?:e|edge)(?:$|[._0-9])", lowered):
            edge_like.append(name)
        if any(token in lowered for token in ("attn", "attention", "self_att")):
            attention_like.append(name)
        # A 1-D float buffer with many entries and no matching weight is a
        # per-timestep schedule: alphas, betas, or their cumulative products.
        if len(dims) == 1 and dims[0] > 1 and any(
            token in lowered for token in ("alpha", "beta", "sigma", "gamma", "schedule")
        ):
            schedules[name] = dims[0]
        if len(dims) == 2:
            output_widths[name] = dims[0]
        elif len(dims) == 1:
            output_widths.setdefault(name, dims[0])

    widths = sorted(set(output_widths.values()))
    return {
        "tensors": len(shapes),
        "node_like": sorted(node_like),
        "edge_like": sorted(edge_like),
        "attention_like": sorted(attention_like),
        "schedule_lengths": schedules,
        "distinct_widths": widths,
        "has_atom_category_width": EXPECTED_ATOM_CATEGORIES in widths,
        "has_bond_category_width": EXPECTED_BOND_CATEGORIES in widths,
        "recorded_diffusion_steps": _recorded_diffusion_steps(hparams),
    }


def _recorded_diffusion_steps(hparams: Mapping[str, Any]) -> int | None:
    """Pull diffusion_steps out of a Lightning hparams blob, however it is nested.

    Not used as evidence on its own -- it is a number the training run wrote, so it
    is exactly the kind of self-reported field the spec says not to trust. It is
    cross-checked against the schedule buffer length, which is a tensor and is not
    a claim.
    """

    stack: list[Any] = [hparams]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            for key, value in current.items():
                if key == "diffusion_steps" and isinstance(value, int):
                    return int(value)
                if isinstance(value, (Mapping, list, tuple)):
                    stack.append(value)
        elif isinstance(current, (list, tuple)):
            stack.extend(current)
    return None


def describe_artifact(
    shapes: Mapping[str, Sequence[int]], hparams: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Return structural diagnostics and reject only a tensor-free checkpoint.

    Actual compatibility is established by the frozen upstream scorer loading and
    evaluating the checkpoint. Tensor-name heuristics do not gate candidates.
    """

    structure = describe_checkpoint(shapes, hparams)
    if not shapes:
        raise ArtifactViolation("the checkpoint holds no tensors")

    return structure


def state_dict_shapes(state: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    """Reduce a torch state dict to names and shapes, dropping everything else.

    The whole artifact check runs on this, so it never touches tensor values and
    never needs a GPU -- and the check is testable from a plain dict of tuples,
    which is what smoke() does below.
    """

    shapes: dict[str, tuple[int, ...]] = {}
    for name, value in state.items():
        shape = getattr(value, "shape", None)
        if shape is None:
            continue
        shapes[str(name)] = tuple(int(dim) for dim in shape)
    return shapes


# A DiGress-shaped state dict, reduced to names and shapes for diagnostic smoke tests.
def _in_family_shapes(diffusion_steps: int = 500) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {
        "model.mlp_in_X.0.weight": (256, 4),
        "model.mlp_in_E.0.weight": (128, 5),
        "model.tf_layers.0.self_attn.q.weight": (256, 256),
        "model.tf_layers.0.self_attn.k.weight": (256, 256),
        "model.tf_layers.0.self_attn.e_add.weight": (128, 256),
        "model.tf_layers.0.self_attn.x_out.weight": (256, 256),
        # The two categorical heads: 4 atom types, 5 bond classes.
        "model.mlp_out_X.2.weight": (4, 256),
        "model.mlp_out_X.2.bias": (4,),
        "model.mlp_out_E.2.weight": (5, 128),
        "model.mlp_out_E.2.bias": (5,),
        # DiGress stores one more boundary than it has steps.
        "noise_schedule.betas": (diffusion_steps + 1,),
        "noise_schedule.alphas_bar": (diffusion_steps + 1,),
    }
    return shapes


def smoke() -> None:
    """Self-check the split arithmetic and structural diagnostics."""

    hparams = {"cfg": {"model": {"diffusion_steps": 500}}}
    structure = describe_artifact(_in_family_shapes(), hparams)
    assert structure["has_atom_category_width"], structure
    assert structure["has_bond_category_width"], structure
    assert structure["recorded_diffusion_steps"] == 500, structure

    token_model = {
        "model.token_embedding.weight": (40, 256),
        "model.layers.0.self_attn.q.weight": (256, 256),
        "model.lm_head.weight": (40, 256),
    }
    assert describe_artifact(token_model, {})["tensors"] == 3
    try:
        describe_artifact({}, {})
    except ArtifactViolation:
        pass
    else:  # pragma: no cover
        raise AssertionError("an empty checkpoint was accepted")

    # --- the split ------------------------------------------------------------
    # 10 distinct valid molecules repeated, so uniqueness is exactly computable
    # by hand and the size-matched control is visibly different from the naive one.
    # `identity` stands in for RDKit: this half is set arithmetic, and the host
    # running the self-check has no conda environment.
    def identity(value: str) -> str | None:
        return value or None

    stream = ["C" * (index % 10 + 1) for index in range(100)]
    split = split_scores(stream, 20, train_smiles=["C"], canonicalize=identity)
    assert split["final_samples"] == 100, split
    assert split["proxy_samples"] == 20, split
    assert abs(split["overlap_fraction"] - 0.2) < 1e-12, split
    assert split["proxy"]["n"] == 20.0, split
    assert split["rest"]["n"] == 80.0, split
    assert split["rest_head"]["n"] == 20.0, split
    # Every one of the ten molecules appears in each slice, so uniqueness falls as
    # the slice grows -- which is exactly the size effect rest_head controls for.
    assert split["proxy"]["uniqueness"] > split["rest"]["uniqueness"], split
    assert abs(split["proxy"]["uniqueness"] - split["rest_head"]["uniqueness"]) < 1e-12, split
    assert abs(split["overfitting_size_matched"]) < 1e-12, split
    assert split["overfitting"] > 0.0, split

    assert abs(composite({"validity": 0.5, "uniqueness": 0.5, "novelty": 0.5}) - 0.125) < 1e-12
    assert composite({"validity": 0.5, "uniqueness": None, "novelty": 0.5}) is None

    # --- upstream's own output ------------------------------------------------
    log = (
        "Validity over 10000 molecules: 98.01%\n"
        "Relaxed validity over 10000 molecules: 98.77%\n"
        "Uniqueness over 9801 valid molecules: 97.18%\n"
        "Novelty over 9524 unique valid molecules: 50.71%\n"
        "Test NLL 69.71\n"
    )
    parsed = parse_metrics(log)
    assert observed_sample_count(log) == 10000, log
    assert abs(parsed["validity"] - 0.9801) < 1e-12, parsed
    assert abs(parsed["nll"] - 69.71) < 1e-12, parsed
    # The empty-valid-set convention, which voided a real run before it existed.
    empty = parse_metrics("Validity over 512 molecules: 0.00%\nVal NLL 162.46\n")
    assert empty["uniqueness"] == 0.0 and empty["novelty"] == 0.0, empty

    print("grade.py smoke passed")


if __name__ == "__main__":
    smoke()
