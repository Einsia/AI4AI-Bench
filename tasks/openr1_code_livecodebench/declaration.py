"""Container phases for OpenR1 training and isolated LiveCodeBench evaluation.

Exploration receives only the public evaluation projection. Retraining receives no
LiveCodeBench rows, and the formal row set is mounted only for score.
"""

from __future__ import annotations

from dataclasses import replace

from container import Phase, checkpoint_validation_phase

# Reserve before the wall clock, so the trainer finishes its optimizer step and
# writes a complete checkpoint rather than being killed mid-save. HF Trainer's
# save writes model shards, optimizer state and the RNG state as separate files;
# a kill between them leaves a directory that exists and cannot be resumed or
# loaded.
RETRAIN_RESERVE_SEC = 600
# Covers two retained checkpoints, the final export, tokenizer files, dataset cache and
# evaluator output without risking a partial checkpoint write.
RETRAIN_FREE_GIB = 40

EXPLORE = Phase(
    name="explore",
    timeout_sec=14400,
    command=("bash",),
    mounts=(
        ("asset:models/training_start", "/assets/models/training_start", True),
        # The complete materialized corpus: 8005 training rows and 128 validation
        # rows. Reallocating the fixed rows is allowed; there are no spare rows.
        ("asset:data/codeforces_cots", "/assets/data/codeforces_cots", True),
        # The v4/v5 rows fast_eval scores: the 64-row proxy and the 204-row
        # disjoint confirmation slice both come out of these two files.
        ("asset:data/livecodebench_public", "/assets/data/livecodebench_public", True),
        # No data/livecodebench_final. The v6 rows are the final's, and they are
        # not in this container.
        #
        # /logs read-only carries the deadline the host writes: the Agent can see
        # how long it has and cannot move it. /logs/agent is writable underneath
        # it for the instruction copy and transcript. Two mounts, opposite
        # directions, one tree.
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
        ("asset:models/training_start", "/assets/models/training_start", True),
        ("asset:data/codeforces_cots", "/assets/data/codeforces_cots", True),
        # No LiveCodeBench mount, public or final. The training phase has no
        # reason to read an evaluation row and now cannot.
        #
        # The patch file, not the directory holding it. That directory is the
        # explore phase's output and contains whatever the Agent trained in its
        # 4 h; binding it would put those checkpoints at /patch/..., where a
        # candidate could warm-start from its own weights instead of the fixed
        # training start.
        ("run:patch", "/patch/candidate.patch", True),
        ("run:out", "/out", False),
    ),
    exports={
        # The two fixed inputs. run.sh reads ${TRAINING_START:-...}, so this
        # beats an edited default -- but it is a guard-rail, not a boundary. A
        # candidate that hard-codes a path never reads the variable. What
        # actually fixes the model is that the container has nowhere else to load
        # from: /assets is the only weights mount, the patch cannot carry
        # weights, and there is no network.
        "TRAINING_START": "/assets/models/training_start",
        "TRAIN_DATA": "/assets/data/codeforces_cots",
    },
    hooks=("inspect_patch", "collect_checkpoints"),
    apply_patch=True,
    reserve_sec=RETRAIN_RESERVE_SEC,
    free_gib=RETRAIN_FREE_GIB,
    # HF Trainer writes checkpoint-<N>; run.sh exports the one being handed on as
    # checkpoints/global_step_<N> because that is the shape the shared
    # collect_checkpoints hook parses. See run.sh's export step.
    checkpoint_glob="checkpoints/checkpoint-*",
    checkpoint_payload=".",
    artifact_limit=3,
    checkpoint_kind="model",
    read_only_root=False,
)

# Conservative ceiling for generation and execution over the fixed final set.
SCORE_TIMEOUT_SEC = 10800

SCORE = Phase(
    name="score",
    timeout_sec=SCORE_TIMEOUT_SEC,
    command=(
        "python3",
        "/opt/harness/final_eval.py",
        "--output",
        "/out",
        "--reward-path",
        "/logs/verifier/reward",
        "--checkpoint",
        "/ckpt",
        "--assets",
        "/assets",
    ),
    mounts=(
        ("run:out", "/out", False),
        ("run:logs/verifier", "/logs/verifier", False),
        # The evaluator verifies this score-only asset and never falls back to public rows.
        ("asset:data/livecodebench_final", "/assets/data/livecodebench_final", True),
        ("run:checkpoint", "/ckpt", True),
    ),
    hooks=("report_reward",),
    # Nothing in this phase writes the root filesystem. The evaluator's
    # code-execution workers write to /dev/shm, which docker mounts separately
    # and --read-only does not cover; see [x-ai4ai.container].shm_size.
    read_only_root=True,
    pass_image_digest=True,
)

VALIDATE = checkpoint_validation_phase(SCORE, "openr1_code_livecodebench")

# Same container and mounts, synthetic rows instead of generation and execution.
# For checking the plumbing on a machine with neither a GPU nor the v6 rows --
# which, until the asset above exists, is every machine.
SCORE_MOCK = replace(
    SCORE,
    name="score-mock",
    command=(
        "python3",
        "/opt/harness/final_eval.py",
        "--output",
        "/out",
        "--reward-path",
        "/logs/verifier/reward",
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
