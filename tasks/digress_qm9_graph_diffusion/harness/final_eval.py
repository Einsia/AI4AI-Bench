"""Score 10,000 sampled molecules and formal test-set NLL.

Checkpoint names and shapes are recorded as diagnostics; actual compatibility is
established by the frozen upstream scorer loading the artifact. Proxy and complement
metrics are slices of the same generated stream. Exploration and
retraining see a validation-backed compatibility tensor, while formal scoring
alone mounts the complete asset and reports test NLL. The conda image uses
Python 3.9.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from grade import (  # noqa: E402
    REQUIRED_METRICS,
    UPSTREAM_REVISION,
    ArtifactViolation,
    composite,
    describe_artifact,
    describe_checkpoint,
    observed_sample_count,
    parse_metrics,
    split_scores,
    state_dict_shapes,
)

TASK_ID = "digress_qm9_graph_diffusion"
METRIC = "nll"
DIRECTION = "minimize"
# The old [proxy].metric, reported here too. Direction max -- opposite to METRIC.
COMPOSITE_METRIC = "validity_uniqueness_novelty"

# The old public tier's seed and count. Kept, because task.toml's recorded public
# metrics were measured at exactly this pair and moving either would silently move
# the number an improvement is compared against.
FINAL_SAMPLES = 10000
FINAL_SEED = 20269701
# The final partitions its own one-run stream: 2000/10000 = 20%. This does not
# imply that a separate fast_eval invocation overlaps the final; sampling is unseeded.
PROXY_SAMPLES = 2000

UPSTREAM_ROOT = Path("/opt/harness/digress")
REWARD_PATH = Path("/logs/verifier/reward.txt")


def _allow_unpickling_the_pinned_tree() -> None:
    """Put the pinned grader tree on sys.path so a real checkpoint can be torch.loaded.

    MEASURED, not anticipated. Without this, `torch.load` of any real DiGress checkpoint
    dies with `ModuleNotFoundError: No module named 'datasets'` before a GPU is touched --
    so every score run on this task failed at inspect_artifact. Lightning pickles the
    hyper_parameters, and DiGress's are live objects:

        cfg                 omegaconf.dictconfig.DictConfig
        dataset_infos       datasets.qm9_dataset.QM9infos
        visualization_tools analysis.visualization.MolecularVisualization
        extra_features      diffusion.extra_features.ExtraFeatures
        domain_features     diffusion.extra_features_molecular.ExtraMolecularFeatures

    Unpickling imports the defining module by the name it had when it was pickled. The
    training run is launched as `python3 <tree>/src/main.py`, so its sys.path[0] was
    <tree>/src and those names are TOP-LEVEL there. final_eval.py runs as
    /opt/harness/final_eval.py, so sys.path[0] is /opt/harness and none of them resolve.

    BOTH entries are needed and the second is easy to miss: `datasets/qm9_dataset.py`
    itself does `import src.utils`, which needs the tree ROOT rather than its src/. That
    is what the image's own ENV PYTHONPATH=/workspace/digress provides for the training
    run. Adding only <tree>/src moves the error from 'datasets' to 'src'.

    APPENDED, not inserted at 0. This tree carries modules with ordinary names -- utils,
    models, metrics, analysis -- and putting them ahead of site-packages would let them
    shadow a real dependency inside the scoring path. Appending means only names that
    resolve nowhere else fall through to here, which is exactly the unpickling case.

    The tree is /opt/harness/digress, the read-only grader copy -- never
    /workspace/digress, which is the copy the candidate owns.
    """

    for entry in (UPSTREAM_ROOT / "src", UPSTREAM_ROOT):
        text = str(entry)
        if entry.is_dir() and text not in sys.path:
            sys.path.append(text)


_allow_unpickling_the_pinned_tree()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def split_output(
    candidate: dict[str, Any] | None,
    error: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    """Publish repository-computed split metrics as an auxiliary diagnostic.

    The benchmark's ranking score is upstream test NLL. This split is additional
    repository-side reporting and never gates or changes that score.
    """

    available = candidate is not None and error is None
    report: dict[str, Any] = {
        "status": "available" if available else "unavailable",
        "score_final": candidate["composite_final"] if available else None,
        "score_proxy": candidate["composite_proxy"] if available else None,
        "score_rest": candidate["composite_rest"] if available else None,
        "score_rest_head": candidate["composite_rest_head"] if available else None,
        "overfitting": candidate["overfitting"] if available else None,
        "overfitting_size_matched": (
            candidate["overfitting_size_matched"] if available else None
        ),
        "note": (
            "the composite is validity x uniqueness x novelty, maximised. NLL -- "
            "the metric that ranks -- is a test-split quantity and does not "
            "partition over generated samples, so it is reported once. Repository-"
            "computed split values are published when the generated-SMILES dump is "
            "available; they never gate the upstream NLL."
        ),
    }
    diagnostic = {
        "status": "available" if available else "unavailable",
        "error": error,
        "reason": "auxiliary split; upstream NLL and whole-set metrics are authoritative",
        "candidate_split": candidate,
    }
    return report, candidate if available else None, diagnostic


# --------------------------------------------------------------------------- #
# the artifact
# --------------------------------------------------------------------------- #


def resolve_checkpoint(checkpoint: Path) -> Path:
    """Accept either the checkpoint file or the directory holding it."""

    if checkpoint.is_file():
        return checkpoint
    candidate = checkpoint / "last.ckpt"
    if candidate.is_file():
        return candidate
    found = sorted(checkpoint.glob("*.ckpt"), key=lambda path: path.stat().st_mtime)
    if found:
        return found[-1]
    raise FileNotFoundError(
        f"no .ckpt under {checkpoint}; expected last.ckpt as written by solution/run.sh"
    )


def inspect_artifact(model: Path) -> dict[str, Any]:
    """Load the checkpoint's structure for diagnostics.

    `map_location="cpu"` and nothing is loaded into a module: the check needs names
    and shapes, not values, so it costs a read and no device memory.

    Lightning nests the weights under "state_dict" and the config under
    "hyper_parameters". A bare state dict is also accepted; the frozen upstream
    scorer is the authority on whether it is loadable.
    """

    import torch

    blob = torch.load(str(model), map_location="cpu")
    if not isinstance(blob, dict):
        raise ArtifactViolation("checkpoint is not a dict, so it is not a Lightning checkpoint")
    state = blob.get("state_dict")
    if state is None:
        # A bare state dict is still checkable; only the hparams cross-check is lost.
        state = {key: value for key, value in blob.items() if hasattr(value, "shape")}
        if not state:
            keys = sorted(str(key) for key in blob)[:20]
            raise ArtifactViolation(
                f"checkpoint holds no state_dict and no tensors: keys are {keys}"
            )
    hparams = blob.get("hyper_parameters") or {}
    shapes = state_dict_shapes(state)
    structure = describe_artifact(shapes, hparams)
    structure["lightning_checkpoint"] = "state_dict" in blob
    structure["global_step"] = blob.get("global_step")
    structure["epoch"] = blob.get("epoch")
    return structure


# --------------------------------------------------------------------------- #
# driving the pinned upstream
# --------------------------------------------------------------------------- #


def patch_upstream_main(text: str) -> str:
    """Apply the two exact changes required for checkpoint-only evaluation."""

    replacements = (
        (
            "os.chdir(cfg.general.test_only.split('checkpoints')[0])",
            "os.chdir(os.path.dirname(cfg.general.test_only))",
            "checkpoint cwd",
        ),
        (
            "cfg = model.cfg\n    cfg.general.test_only = resume",
            "cfg = model.cfg\n"
            "    cfg.general.gpus = saved_cfg.general.gpus\n"
            "    cfg.general.final_model_samples_to_generate = "
            "saved_cfg.general.final_model_samples_to_generate\n"
            "    cfg.general.final_model_samples_to_save = "
            "saved_cfg.general.final_model_samples_to_save\n"
            "    cfg.general.final_model_chains_to_save = "
            "saved_cfg.general.final_model_chains_to_save\n"
            "    cfg.general.test_only = resume",
            "checkpoint sampling config",
        ),
    )
    for old, new, label in replacements:
        if text.count(old) != 1:
            raise RuntimeError(
                f"pinned DiGress {label} patch no longer applies exactly once"
            )
        text = text.replace(old, new, 1)
    return text


def prepare_upstream(output: Path) -> Path:
    """Copy the read-only pinned tree somewhere writable and patch the copy.

    DiGress writes hydra output and sampling chains relative to what it is pointed
    at, so it cannot run out of a read-only directory. The reference protocol's first
    evaluator let it write below the read-only upstream mount and the run was
    voided; every write here stays below /out.

    The tree comes from /opt/harness/digress, not from /workspace/digress. That is
    the whole reason the image carries it twice: on this task the sampler and the
    metrics *are* the grader, so scoring through the copy the candidate can edit
    would let a candidate compute its own score.
    """

    if not UPSTREAM_ROOT.is_dir():
        raise FileNotFoundError(
            f"the pinned grader tree is missing at {UPSTREAM_ROOT}. It is baked into the image by "
            "environment/Dockerfile; without it there is no evaluator."
        )
    upstream = output / "work/upstream"
    shutil.copytree(
        str(UPSTREAM_ROOT), str(upstream), ignore=shutil.ignore_patterns(".git", "__pycache__")
    )
    # MAKE THE COPY WRITABLE. copytree preserves mode, and the image runs `chmod -R a-w`
    # over /opt/harness so the grader cannot be edited in place -- so the copy arrives
    # read-only too, and the patch below died with
    #
    #     PermissionError: [Errno 13] Permission denied: '/out/work/upstream/src/main.py'
    #
    # Measured on the B300, not anticipated: it is the second of two bugs that made every
    # score run on this task fail before reaching a GPU. This does not weaken the
    # read-only grader: /opt/harness itself stays unwritable, and the only thing made
    # writable is the private copy under /out, which exists precisely so the original does
    # not have to be written to. Directories need the bit as well as files -- a read-only
    # directory refuses a write to a file inside it whatever that file's own mode says.
    for path in (upstream, *upstream.rglob("*")):
        try:
            path.chmod(path.stat().st_mode | stat.S_IWUSR)
        except OSError:
            # Best effort per entry. Only src/main.py has to be writable for the patch;
            # failing the whole copy over one odd entry would trade a working score run
            # for a tidier loop.
            pass
    main_source = upstream / "src/main.py"
    if not main_source.is_file():
        raise FileNotFoundError(main_source)
    main_source.write_text(patch_upstream_main(main_source.read_text()))
    return upstream


def run_upstream(
    upstream: Path,
    runtime_checkpoint: Path,
    data: Path,
    output: Path,
    samples: int,
    seed: int,
    label: str,
    cache_root: Path | None = None,
) -> str:
    """Sample `samples` molecules from the checkpoint and return upstream's stdout.

    Both sampling fields are overridden. Setting only `samples_to_generate` is what
    made the reference protocol record a 512-sample proxy that had actually generated 10000
    -- upstream's test path reads `final_model_samples_to_generate`.

    `final_model_samples_to_save` is set to `samples` rather than 0, which is the one
    substantive change to the old command: the three-number split needs the
    molecules themselves, and this is what writes them out.

    `cache_root` is where HOME, TMPDIR and the two cache directories go, and it defaults
    to `output` so the final's behaviour is unchanged. fast_eval passes /out instead,
    because its `output` is a fresh per-call work directory: this image reaches the card
    only through its compute_37 PTX entry, so a fresh HOME means the driver re-JITs every
    kernel -- 62.6 s cold against 0.5 s warm, measured, and ~200 MiB of cache written per
    call. With the caches under /out and the work directory still per-call, repeated
    fast_eval calls share one warm cache without their sampling output colliding.
    """

    caches = cache_root if cache_root is not None else output
    command = [
        sys.executable,
        str(upstream / "src/main.py"),
        "+experiment=qm9_no_h",
        "hydra.run.dir={}".format(output / "hydra"),
        f"general.name={TASK_ID}-{label}-checkpoint-only",
        "general.wandb=disabled",
        f"general.test_only={runtime_checkpoint.resolve()}",
        "general.evaluate_all_checkpoints=false",
        f"general.samples_to_generate={samples}",
        f"general.final_model_samples_to_generate={samples}",
        # The molecules are the artifact the split is computed from.
        f"general.final_model_samples_to_save={samples}",
        "general.final_model_chains_to_save=0",
        f"train.seed={seed}",
        f"dataset.datadir={data}",
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "WANDB_MODE": "offline",
            "HYDRA_FULL_ERROR": "1",
            "PYTHONPATH": str(upstream),
            # /tmp is a tmpfs (8 GiB under the current ContainerSpec), and RDKit,
            # matplotlib and torch all cache below it. Send them to the output mount.
            "TMPDIR": str(caches / "tmp"),
            "HOME": str(caches / "tmp/home"),
            "XDG_CACHE_HOME": str(caches / "tmp/cache"),
            "MPLCONFIGDIR": str(caches / "tmp/matplotlib"),
        }
    )
    for key in ("TMPDIR", "HOME", "XDG_CACHE_HOME", "MPLCONFIGDIR"):
        Path(environment[key]).mkdir(parents=True, exist_ok=True)
    log_path = output / f"{label}-stdout.log"
    with log_path.open("w") as log:
        subprocess.run(
            command,
            cwd=str(upstream),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return log_path.read_text(errors="replace")


# --------------------------------------------------------------------------- #
# the generated molecules
# --------------------------------------------------------------------------- #

# Prefer the known checkpoint-local final_smiles.txt. Fallback globs exclude the copied
# upstream source tree, which contains paper sample dumps unrelated to the current run.
SMILES_DUMP_GLOBS = (
    "final_smiles*.txt",
    "final_smiles*.pkl",
    "generated_smiles*.txt",
    "*_smiles*.txt",
    "smiles*.txt",
    "graphs/**/*.txt",
    "**/final_smiles*",
    "**/generated_smiles*",
)


def find_smiles_dump(
    roots: Sequence[Path], exclude: Sequence[Path] = (), preferred: Sequence[Path] = ()
) -> Path | None:
    """This run's generated-SMILES dump, or None.

    `preferred` exact paths are checked before fallback globbing.

    `exclude` names subtrees to skip, by path rather than by omission from `roots`: the
    upstream tree copy sits UNDER the output root, so `**/...` patterns rooted at /out
    reach into it regardless of what `roots` lists.
    """

    for path in preferred:
        if path.is_file() and path.stat().st_size > 0:
            return path

    excluded = []
    for entry in exclude:
        try:
            excluded.append(entry.resolve())
        except OSError:
            continue

    def is_excluded(path: Path) -> bool:
        try:
            resolved = path.resolve()
        except OSError:
            return False
        return any(resolved == root or root in resolved.parents for root in excluded)

    candidates: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for pattern in SMILES_DUMP_GLOBS:
            for path in root.glob(pattern):
                if path.is_file() and path.stat().st_size > 0 and not is_excluded(path):
                    candidates.append(path)
    if not candidates:
        return None
    # Largest wins among what is left: a per-epoch or per-chain dump is smaller than the
    # final one. This is only safe because the pinned tree's own sample files are excluded.
    return sorted(set(candidates), key=lambda path: (path.stat().st_size, str(path)))[-1]


def read_smiles_dump(path: Path) -> list[str]:
    """Read one SMILES per line, or a pickled list.

    Tolerant about the line format on purpose -- upstream writes a plain list, but a
    leading index or a trailing score would both be plausible, and neither should
    take down a 17-minute run. Anything after the first whitespace-separated token
    is dropped.
    """

    if path.suffix == ".pkl":
        import pickle

        with path.open("rb") as handle:
            payload = pickle.load(handle)
        return [str(item) for item in payload]

    molecules: list[str] = []
    for line in path.read_text(errors="replace").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        molecules.append(text.split()[0])
    return molecules


def load_train_smiles(data: Path) -> list[str] | None:
    """The training SMILES novelty is measured against, from the frozen export.

    Part of the preprocessed dataset rather than something computed here: the pinned
    DiGress preprocessing writes it, and the 13-file export in assets.lock.yaml
    includes it. None when it is absent, which leaves the *split's* novelty
    unreported -- upstream's own full-set novelty still comes from stdout.
    """

    for pattern in ("train_smiles*.npy", "**/train_smiles*.npy", "train_smiles*.txt"):
        for path in sorted(data.glob(pattern)):
            if not path.is_file():
                continue
            if path.suffix == ".npy":
                import numpy

                loaded = numpy.load(str(path), allow_pickle=True)
                return [str(item) for item in loaded.tolist()]
            return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return None


# --------------------------------------------------------------------------- #
# the reward
# --------------------------------------------------------------------------- #


def write_reward(nll: float, reward_path: Path) -> None:
    """Harbor's verifier contract: one scalar, higher is better.

    NLL is minimised, so the scalar written is **-NLL**. That is not an arbitrary
    sign flip: the quantity is a negative log-likelihood, so its negation is the log
    likelihood, and higher likelihood is better. Writing raw NLL here would order
    every candidate backwards, and a reward file that is wrong in the sign is the
    kind of thing that reads fine in a log.

    summary.json carries the raw `nll` under metrics, so nothing downstream has to
    know about this.
    """

    reward_path.parent.mkdir(parents=True, exist_ok=True)
    reward_path.write_text(f"{-nll:.10f}\n", encoding="utf-8")


def evaluate(
    checkpoint: Path,
    assets: Path,
    output: Path,
    reward_path: Path,
    samples: int = FINAL_SAMPLES,
    proxy_samples: int = PROXY_SAMPLES,
    seed: int = FINAL_SEED,
    smiles_file: Path | None = None,
) -> dict[str, Any]:
    model = resolve_checkpoint(checkpoint)
    data = assets / "data/qm9_no_h"
    if not data.is_dir():
        raise FileNotFoundError(data)
    output.mkdir(parents=True, exist_ok=True)

    checkpoint_digest = file_sha256(model)
    # Record structure before scoring. This is diagnostic only; the upstream load
    # below determines compatibility.
    structure = inspect_artifact(model)

    upstream = prepare_upstream(output)
    # A byte-identical copy below the output mount, so the submitted checkpoint mount
    # stays read-only while upstream writes beside what it is pointed at.
    runtime_checkpoint = output / "work/checkpoint/last.ckpt"
    runtime_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(model), str(runtime_checkpoint))
    if file_sha256(runtime_checkpoint) != checkpoint_digest:
        raise RuntimeError("runtime checkpoint copy does not match the submitted checkpoint")

    started = time.monotonic()
    log_text = run_upstream(
        upstream, runtime_checkpoint, data, output, samples, seed, "final"
    )
    metrics = parse_metrics(log_text)
    missing = [name for name in REQUIRED_METRICS if metrics.get(name) is None]
    if missing:
        raise RuntimeError(
            f"DiGress emitted incomplete molecule metrics, missing {missing}: {metrics}"
        )
    observed = observed_sample_count(log_text)
    if observed != samples:
        # Fail closed. The reference protocol recorded a "512-sample" score that had
        # generated 10000 because only one of the two sampling fields was set.
        raise RuntimeError(
            f"sample-count mismatch: declared {samples}, upstream generated {observed}"
        )

    # --- the three numbers ----------------------------------------------------
    # The checkpoint's own directory is where the patched main.py chdirs and writes, so it
    # is both the preferred exact path and the first root. `upstream` is EXCLUDED rather
    # than merely left out of the roots: it lives under `output`, and it carries the pinned
    # repo's own sample dumps, one of which is larger than anything a real run produces.
    dump = smiles_file or find_smiles_dump(
        [runtime_checkpoint.parent, output, output / "hydra"],
        exclude=[upstream],
        preferred=[runtime_checkpoint.parent / "final_smiles.txt"],
    )
    split_candidate: dict[str, Any] | None = None
    split_error: str | None = None
    train_smiles: list[str] | None = None
    if dump is None:
        listing = sorted(
            str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()
        )
        split_error = (
            "no generated-SMILES dump found, so score(P) and score(F\\P) cannot be "
            "computed. Looked for {} under {} and {}. Files written:\n  {}\n"
            "Pass --smiles-file to name it explicitly, or add its name to "
            "SMILES_DUMP_GLOBS in this file.".format(
                list(SMILES_DUMP_GLOBS), output, upstream, "\n  ".join(listing[:60])
            )
        )
    else:
        try:
            molecules = read_smiles_dump(dump)
            if len(molecules) < samples:
                split_error = (
                    f"the SMILES dump at {dump} holds {len(molecules)} molecules, fewer "
                    f"than the {samples} sampled. The split needs the whole stream."
                )
            else:
                molecules = molecules[:samples]
                train_smiles = load_train_smiles(data)
                split_candidate = split_scores(molecules, proxy_samples, train_smiles)
                (output / "generated_smiles.txt").write_text(
                    "\n".join(molecules) + "\n", encoding="utf-8"
                )
        except Exception as exc:  # The split is diagnostic; upstream NLL remains valid.
            split_error = f"{type(exc).__name__}: {exc}"

    three_number_report, split, split_diagnostic = split_output(split_candidate, split_error)

    nll = float(metrics["nll"])
    summary: dict[str, Any] = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "passed",
        "benchmark": f"DiGress QM9-no-H, {samples} sampled molecules",
        "metric": METRIC,
        "direction": DIRECTION,
        "nll_split": "test",
        "metrics": dict(metrics),
        # The composite the proxy optimises, reported beside the metric that ranks.
        # Different directions; see task.toml [metadata].
        "composite_metric": COMPOSITE_METRIC,
        "composite_direction": "max",
        "score": nll,
        "reward": -nll,
        "samples": samples,
        "observed_samples": observed,
        "evaluator_seed": seed,
        "proxy_samples": proxy_samples,
        "overlap_fraction": proxy_samples / samples,
        "three_number_report": three_number_report,
        "split_detail": split,
        "split_diagnostic": split_diagnostic,
        "upstream_agreement": split_diagnostic["status"],
        "smiles_dump": str(dump) if dump is not None else None,
        "checkpoint_sha256": checkpoint_digest,
        "artifact_description": structure,
        "upstream_revision": UPSTREAM_REVISION,
        "metric_conventions": {
            "empty_valid_set_uniqueness": 0.0,
            "empty_unique_valid_set_novelty": 0.0,
        },
        "novelty_reference": "train split SMILES" if train_smiles else "unavailable",
        "wall_seconds": time.monotonic() - started,
        "checkpoint_only": True,
        "offline": True,
        "image_digest": os.environ.get("IMAGE_DIGEST"),
    }
    atomic_json(output / "summary.json", summary)
    write_reward(nll, reward_path)
    return summary


# --------------------------------------------------------------------------- #
# mock and smoke
# --------------------------------------------------------------------------- #


def mock(output: Path, reward_path: Path) -> dict[str, Any]:
    """The real output shape, from synthetic molecules. No GPU, no dataset, no RDKit.

    Exercises the part a consumer parses and the part with arithmetic in it: the
    three-number split, the overlap fraction, and the reward sign.
    """

    output.mkdir(parents=True, exist_ok=True)

    def identity(value: str) -> str | None:
        return value or None

    # 1200 distinct molecules cycled through 10000 draws, so uniqueness is well
    # under 1 and the size effect the size-matched control exists for is visible.
    molecules = [f"C{index % 1200}" for index in range(FINAL_SAMPLES)]
    split = split_scores(
        molecules, PROXY_SAMPLES, train_smiles=["C0", "C1"], canonicalize=identity
    )
    nll = 69.71
    metrics = {
        "validity": split["final"]["validity"],
        "relaxed_validity": split["final"]["validity"],
        "uniqueness": split["final"]["uniqueness"],
        "novelty": split["final"]["novelty"],
        "nll": nll,
    }
    three_number_report, published_split, split_diagnostic = split_output(split, None)
    summary = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "passed",
        "metric": METRIC,
        "direction": DIRECTION,
        "nll_split": "test",
        "metrics": metrics,
        "composite_metric": COMPOSITE_METRIC,
        "composite_direction": "max",
        "score": nll,
        "reward": -nll,
        "samples": FINAL_SAMPLES,
        "observed_samples": FINAL_SAMPLES,
        "evaluator_seed": FINAL_SEED,
        "proxy_samples": PROXY_SAMPLES,
        "overlap_fraction": PROXY_SAMPLES / FINAL_SAMPLES,
        "three_number_report": three_number_report,
        "split_detail": published_split,
        "split_diagnostic": split_diagnostic,
        "upstream_agreement": "not_applicable",
        "mock": True,
    }
    atomic_json(output / "summary.json", summary)
    write_reward(nll, reward_path)
    return summary


def smoke() -> None:
    """Self-check with no filesystem and no torch."""

    def identity(value: str) -> str | None:
        return value or None

    molecules = [f"C{index % 1200}" for index in range(FINAL_SAMPLES)]
    split = split_scores(molecules, PROXY_SAMPLES, train_smiles=[], canonicalize=identity)
    if split["final_samples"] != FINAL_SAMPLES or split["proxy_samples"] != PROXY_SAMPLES:
        raise RuntimeError(f"unexpected split shape: {split}")
    if abs(split["overlap_fraction"] - 0.2) > 1e-12:
        raise RuntimeError("overlap must be 20%: {}".format(split["overlap_fraction"]))
    # 1200 distinct molecules: the 2000-molecule prefix holds all of them, the
    # 8000-molecule remainder also does, so uniqueness differs by set size alone --
    # which is why the size-matched control is the one to read.
    if split["composite_proxy"] is None or split["composite_rest"] is None:
        raise RuntimeError(f"the composite must be computable on every slice: {split}")
    if abs(split["overfitting_size_matched"]) > 1e-9:
        raise RuntimeError(
            "identical size-matched slices must show no overfitting, got {}".format(
                split["overfitting_size_matched"]
            )
        )
    available_report, available_split, available_diagnostic = split_output(split, None)
    if available_report["status"] != "available" or available_split is not split:
        raise RuntimeError("an available split must be published as an auxiliary")
    if available_diagnostic["status"] != "available":
        raise RuntimeError("an available split must carry an auxiliary diagnostic")
    described = describe_artifact(
        {
            "model.token_embedding.weight": (40, 256),
            "model.layers.0.self_attn.q.weight": (256, 256),
            "model.lm_head.weight": (40, 256),
        },
        {},
    )
    if described["tensors"] != 3:
        raise RuntimeError("checkpoint structure diagnostic lost tensors")
    if composite({"validity": 1.0, "uniqueness": 1.0, "novelty": 1.0}) != 1.0:
        raise RuntimeError("composite is not a product")
    print(json.dumps({"final_eval_smoke": "passed", "n": split["final_samples"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--assets", type=Path, default=Path("/assets"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reward-path", type=Path, default=REWARD_PATH)
    parser.add_argument(
        "--samples",
        type=int,
        default=FINAL_SAMPLES,
        help="molecules to draw; changing the default changes the formal protocol",
    )
    parser.add_argument(
        "--proxy-samples",
        type=int,
        default=PROXY_SAMPLES,
        help="prefix of the same stream that counts as the proxy. 2000 of 10000.",
    )
    parser.add_argument("--seed", type=int, default=FINAL_SEED)
    parser.add_argument(
        "--smiles-file",
        type=Path,
        default=None,
        help="the generated-SMILES dump, if the search does not find it",
    )
    parser.add_argument(
        "--describe-checkpoint",
        action="store_true",
        help="print the checkpoint's structure and exit, without asserting anything. "
        "Use this to confirm the artifact check against a real baseline artifact.",
    )
    parser.add_argument("--mock", action="store_true", help="synthetic molecules, no GPU")
    parser.add_argument("--smoke", action="store_true", help="self-check and exit")
    args = parser.parse_args()

    if args.smoke:
        smoke()
        return
    if args.describe_checkpoint:
        if args.checkpoint is None:
            parser.error("--describe-checkpoint needs --checkpoint")
        model = resolve_checkpoint(args.checkpoint.resolve())
        import torch

        blob = torch.load(str(model), map_location="cpu")
        state = blob.get("state_dict", blob) if isinstance(blob, dict) else {}
        hparams = blob.get("hyper_parameters", {}) if isinstance(blob, dict) else {}
        print(
            json.dumps(
                describe_checkpoint(state_dict_shapes(state), hparams),
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
        return
    if args.output is None:
        parser.error("--output is required")
    if args.mock:
        print(json.dumps(mock(args.output.resolve(), args.reward_path), sort_keys=True))
        return
    if args.checkpoint is None:
        parser.error("--checkpoint is required outside --mock/--smoke")
    print(
        json.dumps(
            evaluate(
                args.checkpoint.resolve(),
                args.assets.resolve(),
                args.output.resolve(),
                args.reward_path,
                args.samples,
                args.proxy_samples,
                args.seed,
                args.smiles_file,
            ),
            sort_keys=True,
            default=str,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"DiGress final failed: {exc}", file=sys.stderr)
        raise
