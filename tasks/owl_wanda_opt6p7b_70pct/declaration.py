"""What each phase of this task mounts, runs, and exports.

Read tasks/opd_math_1p5b/declaration.py first. This file is the same idea with the
one structural difference the refactor was for: **OWL prunes and never trains.**

There is no training loop, no optimizer, and no learning rate. The shipped retrain
phase makes one activation-aware pruning pass over the dense model and publishes
one candidate, while the formal interface also permits a candidate method to
publish as many as three construction attempts.

  reserve_sec   600. run.sh consumes the host wall clock and stops the pruning process
                before the outer container deadline. A partial artifact remains invalid.
  output_glob   a numbered formal-artifact series, using construction progress.
  free_gib      set, but derived rather than measured. See RETRAIN_FREE_GIB.

The mount lists are the load-bearing part, and their order is part of the contract.
That explore has no `test` entry is what makes "the Agent cannot tune against the
final text" true; that retrain has no wikitext entry at all is what stops the
pruning pass reading either evaluation split. Neither is enforced by a check or a
file permission.
"""

from __future__ import annotations

from dataclasses import replace

from container import Phase, checkpoint_validation_phase

# The dense fp16 model is 13,318,532,446 bytes on disk (recorded in
# environment/assets.lock.yaml) and the export is the same dtype and the same
# shapes: unstructured sparsity writes zeros, it does not compress a tensor. So the
# artifact is ~12.4 GiB, plus a decompressed C4 shard and its arrow cache under /out.
#
# DERIVED FROM A RECORDED SIZE, NOT MEASURED. The reference protocol recorded no disk figure
# for this task. 40 leaves room for a candidate that widens the calibration set and
# for the HF caches beside the artifact.
RETRAIN_FREE_GIB = 64

EXPLORE = Phase(
    name="explore",
    timeout_sec=14400,
    command=("bash",),
    mounts=(
        # THE TARGET PATH HAS TO CONTAIN "opt", AND THAT IS NOT COSMETIC.
        #
        # Upstream OWL selects between the OPT and Llama module layouts by
        # substring-matching the path it was handed -- `if "opt" in args.model:
        # layers = model.model.decoder.layers else: layers = model.model.layers` --
        # at ELEVEN sites across main.py and lib/prune_all.py. A neutral path takes
        # the else branch everywhere and dies on `'OPTModel' object has no attribute
        # 'layers'`, after the 13 GiB model has loaded. Measured, in the first two
        # baseline runs of this task: the first failed on the same sniff in the
        # tokenizer dispatch, the second here.
        #
        # Naming it here rather than patching the eleven sites, for two reasons. It
        # is one host-controlled line instead of eleven edits to a pinned tree; and
        # /workspace/owl is the Agent's to edit, so patched sniffs could be undone by
        # a candidate while this cannot.
        #
        # `dense` is kept in the name because that is the role the rest of the task
        # refers to -- the alias under [assets] is still models/dense, and only the
        # mount target changes.
        ("asset:models/dense", "/assets/models/opt-dense", True),
        # The whole C4 train shard, not the 128 sequences the old recipe drew from
        # it. How many sequences to calibrate on is the Agent's choice now, and it
        # cannot be a choice if the container only holds the old sample.
        ("asset:data/c4", "/assets/data/c4", True),
        # fast_eval's text: WikiText2 *validation*. Same evaluator as the final,
        # different split.
        ("asset:data/wikitext2/validation", "/assets/data/wikitext2/validation", True),
        # No test split. WikiText2 test is public, so this is not secrecy -- it stops
        # four hours being spent tuning the layer allocation against the exact 140
        # blocks the result is reported on.
        #
        # /logs read-only carries the deadline the host writes; /logs/agent is
        # writable underneath it. Two mounts, opposite directions, one tree.
        ("run:logs", "/logs", True),
        ("run:logs/agent", "/logs/agent", False),
        ("run:out", "/out", False),
    ),
    exports={
        "DENSE_MODEL": "/assets/models/opt-dense",
        "CALIBRATION_DATA": "/assets/data/c4",
    },
    hooks=("write_deadline", "report_submission"),
    # The Agent writes /workspace in the image layer, so the root filesystem cannot
    # be read-only.
    read_only_root=False,
)

# Formal replay uses the benchmark-wide 12-hour ceiling. The pristine B300 recipe
# completed in 157-161 seconds, so this is a budget boundary rather than an expected
# duration; the process exits as soon as pruning and export complete.
RETRAIN_TIMEOUT_SEC = 43200

RETRAIN = Phase(
    name="retrain",
    timeout_sec=RETRAIN_TIMEOUT_SEC,
    command=("bash", "/workspace/run.sh"),
    mounts=(
        # "opt" in the target path, for the reason spelled out on EXPLORE's mount of
        # the same asset: upstream picks the OPT module layout by substring-matching
        # this path at eleven sites. Both phases mount it, so both need the name.
        ("asset:models/dense", "/assets/models/opt-dense", True),
        ("asset:data/c4", "/assets/data/c4", True),
        # The patch file, not the directory holding it. That directory is the explore
        # phase's output and contains whatever the Agent pruned during its 4 h;
        # binding it would put those at /patch/, where a candidate could export one
        # of its own artifacts instead of pruning the fixed dense model.
        ("run:patch", "/patch/candidate.patch", True),
        ("run:out", "/out", False),
    ),
    exports={
        # The two fixed inputs. run.sh reads ${DENSE_MODEL:-...}, so this beats an
        # edited default -- but it is a guard-rail, not a boundary. A candidate that
        # writes a literal path into prune.py never reads the variable. What actually
        # fixes the model is that the container has nowhere else to load from:
        # /assets is the only weights mount, the patch cannot carry weights, and
        # there is no network.
        "DENSE_MODEL": "/assets/models/opt-dense",
        "CALIBRATION_DATA": "/assets/data/c4",
    },
    # No collect_checkpoints: see the module docstring. inspect_patch still applies --
    # a 12 GiB pruned model must not ride in through candidate.patch.
    hooks=("inspect_patch",),
    apply_patch=True,
    reserve_sec=600,
    free_gib=RETRAIN_FREE_GIB,
    # The shipped method publishes one construction at progress zero; candidate
    # methods may publish up to three independently loadable constructions.
    checkpoint_glob="checkpoints/checkpoint-*",
    checkpoint_payload=".",
    artifact_limit=3,
    checkpoint_kind="model",
    read_only_root=False,
)

# 2x the old [proxy].timeout_seconds of 1800, which was itself a timeout and not a
# measurement, for the same evaluator on the slightly smaller validation split. The
# final reads 140 blocks of 2048 tokens through one fp16 forward pass each; the model
# load dominates. NOT MEASURED -- measure one final and replace this.
SCORE_TIMEOUT_SEC = 3600

SCORE = Phase(
    name="score",
    timeout_sec=SCORE_TIMEOUT_SEC,
    command=(
        "python3",
        "/opt/harness/final_eval.py",
        "--output",
        "/out",
        "--reward-path",
        "/logs/verifier/reward.txt",
        "--checkpoint",
        "/ckpt",
        "--assets",
        "/assets",
    ),
    mounts=(
        ("run:out", "/out", False),
        ("run:logs/verifier", "/logs/verifier", False),
        ("asset:data/wikitext2/test", "/assets/data/wikitext2/test", True),
        ("run:checkpoint", "/ckpt", True),
    ),
    # No report_reward. That hook reads payload["correct"] and payload["n"] out of
    # summary.json and prints "<metric> = <score> (<correct>/<n>)", which is an
    # accuracy shape. A perplexity summary has no correct count and no n, so the hook
    # raises KeyError. See the report: it needs to read the metric shape rather than
    # assume one. final_eval.py prints its own one-line summary instead.
    hooks=(),
    # Nothing in this phase writes the root filesystem.
    read_only_root=True,
    pass_image_digest=True,
)

VALIDATE = checkpoint_validation_phase(SCORE, "owl_wanda_opt6p7b_70pct")

# Same container and mounts, synthetic per-block NLLs instead of a model. For
# Checking the plumbing without a GPU or mounted test split. The pinned test asset
# exists, but SCORE_MOCK deliberately needs neither it nor a checkpoint.
SCORE_MOCK = replace(
    SCORE,
    name="score-mock",
    command=(
        "python3",
        "/opt/harness/final_eval.py",
        "--output",
        "/out",
        "--reward-path",
        "/logs/verifier/reward.txt",
        "--mock",
    ),
    mounts=(
        ("run:out", "/out", False),
        ("run:logs/verifier", "/logs/verifier", False),
    ),
)

PHASES: dict[str, Phase] = {
    "explore": EXPLORE,
    # Named "retrain" for consistency with the other nine tasks and with
    # runner.py's documented CLI, not because anything is retrained. It is the
    # replay-the-candidate-in-a-clean-container phase; here that replay prunes.
    "retrain": RETRAIN,
    "score": SCORE,
    "checkpoint-validate": VALIDATE,
    "score-mock": SCORE_MOCK,
}
