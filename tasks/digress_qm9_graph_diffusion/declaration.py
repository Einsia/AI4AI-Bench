"""Container phases for DiGress training and isolated QM9 evaluation.

The image has separate editable and evaluator copies of the pinned DiGress tree. Training
receives a derived train/validation asset whose required test-named tensor aliases validation;
formal score alone receives the real test tensor. Fast and final sampling are independent
draws because the pinned trainer does not consume the configured seed.
"""

from __future__ import annotations

from dataclasses import replace

from container import Phase, checkpoint_validation_phase

# Reserve enough time for Lightning to finish an epoch and write last.ckpt.
RETRAIN_RESERVE_SEC = 600
# Covers the checkpoint, optimizer state, Hydra output and sampling artifacts.
RETRAIN_FREE_GIB = 20

QM9_TRAIN_MOUNT = ("asset:data/qm9_train_val", "/assets/data/qm9_no_h", True)
QM9_SCORE_MOUNT = ("asset:data/qm9_no_h", "/assets/data/qm9_no_h", True)

EXPLORE = Phase(
    name="explore",
    timeout_sec=14400,
    command=("bash",),
    mounts=(
        QM9_TRAIN_MOUNT,
        # /logs read-only carries the deadline the host writes: the Agent can see
        # how long it has and cannot move it. /logs/agent is writable underneath
        # it, because the agent's instruction copy and transcript land there. Two
        # mounts, opposite directions, one tree.
        ("run:logs", "/logs", True),
        ("run:logs/agent", "/logs/agent", False),
        ("run:out", "/out", False),
    ),
    hooks=("write_deadline", "report_submission"),
    # The Agent writes /workspace in the image layer, so the root filesystem
    # cannot be read-only.
    read_only_root=False,
)

RETRAIN = Phase(
    name="retrain",
    timeout_sec=43200,
    command=("bash", "/workspace/run.sh"),
    mounts=(
        QM9_TRAIN_MOUNT,
        # The patch file, not the directory holding it. That directory is the
        # explore phase's output and holds the checkpoints the Agent trained in
        # its 4 h; binding it would put those at /patch/checkpoint, where a
        # candidate could warm-start from its own weights instead of from a fresh
        # initialisation.
        ("run:patch", "/patch/candidate.patch", True),
        ("run:out", "/out", False),
    ),
    exports={
        # The one fixed input. run.sh reads ${QM9_DATA:-...}, so this beats an
        # edited default -- but it is a guard-rail, not a boundary. A candidate
        # that writes a literal path into the Hydra line never reads the variable.
        # What actually fixes the data is that the container has nowhere else to
        # load from: /assets is the only data mount, the patch cannot carry a
        # dataset (inspect_patch refuses one), and there is no network.
        "QM9_DATA": "/assets/data/qm9_no_h",
    },
    # DiGress writes one last.ckpt rather than numbered global_step directories.
    hooks=("inspect_patch",),
    apply_patch=True,
    reserve_sec=RETRAIN_RESERVE_SEC,
    free_gib=RETRAIN_FREE_GIB,
    checkpoint_glob="checkpoints/checkpoint-*",
    checkpoint_payload="model.ckpt",
    artifact_limit=3,
    checkpoint_kind="file",
    read_only_root=False,
)

# Conservative ceiling for formal molecule sampling and test NLL.
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
        QM9_SCORE_MOUNT,
        ("run:checkpoint", "/ckpt", True),
    ),
    # The evaluator writes the transformed reward directly.
    hooks=(),
    # THE GRADER'S IMPORT PATH, and it is an integrity boundary rather than a convenience.
    #
    # Dockerfile.base sets ENV PYTHONPATH=/workspace/digress -- the tree the CANDIDATE
    # owns and edits. The score phase does not run out of /workspace, but its process
    # inherits that path, and scoring genuinely has to import the pinned tree: Lightning
    # pickles DiGress's hyper_parameters as live objects (datasets.qm9_dataset.QM9infos and
    # four more), so torch.load of any real checkpoint imports their defining modules.
    #
    # harness/final_eval.py appends the grader tree to sys.path for exactly that reason.
    # Setting PYTHONPATH here removes /workspace/digress from the scoring process entirely.
    #
    # Both entries, in this order: <tree>/src makes datasets/analysis/diffusion resolve as
    # the top-level names the pickle carries, and <tree> makes the `src.` prefixed imports
    # inside them resolve.
    exports={"PYTHONPATH": "/opt/harness/digress/src:/opt/harness/digress"},
    # DiGress writes hydra output and sampling chains next to whatever it is
    # pointed at, so this phase does need a writable root -- but only /out and
    # /logs/verifier are writable mounts, and final_eval keeps every write below /out.
    read_only_root=False,
    pass_image_digest=True,
)

VALIDATE = checkpoint_validation_phase(SCORE, "digress_qm9_graph_diffusion")

# Same container and mounts, synthetic molecules instead of sampling. For checking
# the plumbing, the three-number split and the artifact check on a machine with
# neither a GPU nor the dataset.
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
    "retrain": RETRAIN,
    "score": SCORE,
    "checkpoint-validate": VALIDATE,
    "score-mock": SCORE_MOCK,
}
