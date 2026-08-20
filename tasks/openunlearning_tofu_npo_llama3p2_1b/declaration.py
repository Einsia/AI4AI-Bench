"""Container phases for the official Llama TOFU NPO protocol."""

from __future__ import annotations

from dataclasses import replace

from container import Phase, checkpoint_validation_phase

RETRAIN_RESERVE_SEC = 900
RETRAIN_FREE_GIB = 40

EXPLORE = Phase(
    name="explore",
    timeout_sec=14400,
    command=("bash",),
    mounts=(
        ("asset:models/training_start", "/assets/models/training_start", True),
        ("asset:data/train", "/assets/data/train", True),
        ("run:logs", "/logs", True),
        ("run:logs/agent", "/logs/agent", False),
        ("run:out", "/out", False),
    ),
    hooks=("write_deadline", "report_submission"),
    read_only_root=False,
)

RETRAIN = Phase(
    name="retrain",
    timeout_sec=43200,
    command=("bash", "/workspace/run.sh"),
    mounts=(
        ("asset:models/training_start", "/assets/models/training_start", True),
        ("asset:data/train", "/assets/data/train", True),
        ("run:patch", "/patch/candidate.patch", True),
        ("run:out", "/out", False),
    ),
    exports={
        "TRAINING_START": "/assets/models/training_start",
        "TRAIN_DATA": "/assets/data/train",
    },
    hooks=("inspect_patch",),
    apply_patch=True,
    reserve_sec=RETRAIN_RESERVE_SEC,
    free_gib=RETRAIN_FREE_GIB,
    checkpoint_glob="checkpoints/checkpoint-*",
    checkpoint_payload=".",
    artifact_limit=3,
    checkpoint_kind="model",
    read_only_root=False,
)

SCORE = Phase(
    name="score",
    timeout_sec=14400,
    command=(
        "python3",
        "/opt/harness/final_eval.py",
        "--checkpoint",
        "/ckpt",
        "--assets",
        "/assets",
        "--output",
        "/out",
        "--reward-path",
        "/logs/verifier/reward",
    ),
    mounts=(
        ("run:out", "/out", False),
        ("run:logs/verifier", "/logs/verifier", False),
        ("asset:models/training_start", "/assets/models/training_start", True),
        ("asset:models/retain_reference", "/assets/models/retain_reference", True),
        ("asset:data/final", "/assets/data/final", True),
        ("run:checkpoint", "/ckpt", True),
    ),
    hooks=(),
    read_only_root=True,
    pass_image_digest=True,
)

VALIDATE = checkpoint_validation_phase(SCORE, "openunlearning_tofu_npo_llama3p2_1b")

SCORE_MOCK = replace(
    SCORE,
    name="score-mock",
    command=(
        "python3",
        "/opt/harness/final_eval.py",
        "--mock",
        "--output",
        "/out",
        "--reward-path",
        "/logs/verifier/reward",
    ),
    mounts=(
        ("run:out", "/out", False),
        ("run:logs/verifier", "/logs/verifier", False),
    ),
)

PHASES = {
    "explore": EXPLORE,
    "retrain": RETRAIN,
    "score": SCORE,
    "checkpoint-validate": VALIDATE,
    "score-mock": SCORE_MOCK,
}
