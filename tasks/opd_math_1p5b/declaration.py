"""OPD phase mounts, commands and exports.

Mount sources use `asset:<relative>` for fixed assets and `run:<name>` for trial
artifacts. Exploration excludes AIME, retraining excludes Math500, and score-mock
inherits the formal score mounts through Phase.with_command().
"""

from __future__ import annotations

from dataclasses import replace

from container import Phase, checkpoint_validation_phase

# Reserve before the wall clock, so the trainer finishes its step and writes a
# complete checkpoint rather than being killed mid-write.
RETRAIN_RESERVE_SEC = 1200
# One checkpoint is ~27 GiB with optimizer state and the run keeps three. Allow for a
# fourth checkpoint while the oldest is pruned, plus rollout dumps and Ray state.
RETRAIN_FREE_GIB = 160

EXPLORE = Phase(
    name="explore",
    timeout_sec=14400,
    command=("bash",),
    mounts=(
        ("asset:models/student", "/assets/models/student", True),
        ("asset:models/teacher", "/assets/models/teacher", True),
        ("asset:data/train.parquet", "/assets/data/train.parquet", True),
        # fast_eval's question set. Same JustRL tar as the final's grader, so both
        # stages score through one path.
        ("asset:data/math500", "/assets/data/math500", True),
        # No aime. The final questions are public, so this is not secrecy -- it stops
        # four hours being spent tuning against the exact set the result is reported
        # on.
        #
        # /logs read-only carries the deadline the host writes: the Agent can see how
        # long it has and cannot move it. /logs/agent is writable underneath it,
        # because the agent's instruction copy and transcript land there. Two mounts,
        # opposite directions, one tree.
        ("run:logs", "/logs", True),
        ("run:logs/agent", "/logs/agent", False),
        ("run:out", "/out", False),
    ),
    hooks=("write_deadline", "report_submission"),
    # The Agent writes /workspace in the image layer, so the root filesystem cannot
    # be read-only.
    read_only_root=False,
)

RETRAIN = Phase(
    name="retrain",
    timeout_sec=43200,
    command=("bash", "/workspace/run.sh"),
    mounts=(
        ("asset:models/student", "/assets/models/student", True),
        ("asset:models/teacher", "/assets/models/teacher", True),
        ("asset:data/train.parquet", "/assets/data/train.parquet", True),
        # The patch file, not the directory holding it. That directory is phase A's
        # output and contains the checkpoints the Agent trained in its 4 h; binding
        # it put those at /patch/checkpoints, where a candidate could warm-start
        # from its own weights instead of the fixed student.
        ("run:patch", "/patch/candidate.patch", True),
        ("run:out", "/out", False),
    ),
    exports={
        # The three fixed inputs. run.sh reads ${STUDENT_MODEL:-...}, so this beats
        # an edited default -- but it is a guard-rail, not a boundary. A candidate
        # that rewrites the Hydra line to a literal path never reads the variable.
        # What actually fixes the model is that the container has nowhere else to
        # load from: /assets is the only weights mount, the patch cannot carry
        # weights, and there is no network.
        "STUDENT_MODEL": "/assets/models/student",
        "TEACHER_MODEL": "/assets/models/teacher",
        "TRAIN_DATA": "/assets/data/train.parquet",
    },
    # inspect_patch reads the diff before a GPU is claimed; verify_fixed_inputs reads
    # what Hydra actually resolved after the run. Two readings of one claim, at the two
    # ends where they can each catch what the other cannot.
    hooks=("inspect_patch", "collect_checkpoints", "verify_fixed_inputs"),
    apply_patch=True,
    reserve_sec=RETRAIN_RESERVE_SEC,
    free_gib=RETRAIN_FREE_GIB,
    checkpoint_glob="checkpoints/checkpoint-*",
    checkpoint_payload=".",
    artifact_limit=3,
    checkpoint_kind="model",
    read_only_root=False,
)

# B300 single-device scores have taken 41-54 minutes with a batch width of 240.
# 9000 seconds is at least 2.7x those measurements and leaves explicit verifier margin.
SCORE_TIMEOUT_SEC = 9000

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
        "--gpus",
        "$gpus",
    ),
    mounts=(
        ("run:out", "/out", False),
        ("run:logs/verifier", "/logs/verifier", False),
        ("asset:data/aime", "/assets/data/aime", True),
        ("asset:models/verifier", "/assets/models/verifier", True),
        ("run:checkpoint", "/ckpt", True),
    ),
    hooks=("report_reward",),
    # Nothing in this phase writes the root filesystem.
    read_only_root=True,
    pass_image_digest=True,
)

VALIDATE = checkpoint_validation_phase(SCORE, "opd_math_1p5b")

# Same container and mounts, synthetic rows instead of generation. For checking the
# plumbing on a machine with neither a GPU nor the question set.
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
