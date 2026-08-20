"""What each phase of this task mounts, runs, and exports.

Values only. The Phase class lives in orchestrator/container.py, because it is
format rather than content.

The mount lists are the load-bearing part. Three things they hold, none of which is
enforced by a checker or a file permission:

  * explore has no `data/rewardbench` entry. It gets `data/rewardbench_proxy`, the
    512 stratified pairs, and nothing else -- so 2473 of the final's 2985 rows are
    unreachable during the 4 h. This is what makes the held-out number held out.
  * retrain has no RewardBench entry at all, proxy or otherwise. The 12 h run
    trains; it does not measure.
  * score mounts the base model so the shipped adapter can be assembled. A candidate
    may instead export a complete scalar reward model from the same fresh replay.

## A note on the proxy mount, because it looks like a subset mount

The spec's rule is that a mount must be the whole de-contaminated source, so a
candidate is not stuck with rows the container does not hold. That rule is about
the *training* input, and `data/pairs.jsonl` obeys it -- the whole projection is
mounted and `TRAIN_PAIRS` is a default rather than a frozen row count.

`data/rewardbench_proxy` is deliberately a subset, and of the *evaluation* set. A
proxy that held all 2985 rows would let 4 h be spent tuning against the exact rows
the result is reported on, which is the one thing the two-tier design exists to
prevent. OPD achieves the same separation by using a different question set for
fast_eval; here the proxy is a subset of the final instead, which is what the spec
prefers -- it maximises correlation and makes the overfitting measurable, at the
cost of 17.2% of the final being visible.

## score mounts the training pool, and that is on purpose

`data/pairs.jsonl` appears in the score phase as well. It is the only way to
actually check that the 8192 training pairs are disjoint from RewardBench: that
needs both files in one container, and before this split RewardBench was mounted
nowhere. The reference protocol shipped the claim (`decontamination_status: "passed"`)
and kept the evidence judge-side. It leaks nothing -- the Agent already has the
whole pool during exploration -- and `final_eval.verify_disjoint` recomputes the
overlap rather than reading a manifest field.
"""

from __future__ import annotations

from dataclasses import replace

from container import Phase, checkpoint_validation_phase

# Stop training with time left to write a complete artifact rather than being killed
# mid-write. The shipped adapter is small, but a candidate may export a full model.
RETRAIN_RESERVE_SEC = 600
# Covers the tokenized dataset cache and trainer output under /out.
RETRAIN_FREE_GIB = 20

EXPLORE = Phase(
    name="explore",
    timeout_sec=14400,
    command=("bash",),
    mounts=(
        ("asset:models/base", "/assets/models/base", True),
        # The mounted file contains exactly 8192 materialized pairs; load_pairs
        # raises rather than silently truncating requests beyond that boundary.
        ("asset:data/pairs.jsonl", "/assets/data/pairs.jsonl", True),
        # The proxy's 512 pairs. No full RewardBench entry: that is what keeps
        # 2473 rows of the final unseen.
        ("asset:data/rewardbench_proxy", "/assets/data/rewardbench_proxy", True),
        # /logs read-only carries the deadline the host writes: the Agent can see
        # how long it has and cannot move it. /logs/agent is writable underneath,
        # because the instruction copy and transcript land there.
        ("run:logs", "/logs", True),
        ("run:logs/agent", "/logs/agent", False),
        ("run:out", "/out", False),
    ),
    hooks=("write_deadline", "report_submission"),
    # The Agent writes /workspace in the image layer.
    read_only_root=False,
)

RETRAIN = Phase(
    name="retrain",
    timeout_sec=43200,
    command=("bash", "/workspace/run.sh"),
    mounts=(
        ("asset:models/base", "/assets/models/base", True),
        # 8192 rows, the whole file. See the explore phase's note: this is not the
        # full projection and TRAIN_PAIRS above 8192 raises.
        ("asset:data/pairs.jsonl", "/assets/data/pairs.jsonl", True),
        # The patch file, not the directory holding it. That directory is the
        # explore phase's output and holds the checkpoints the Agent trained in its
        # 4 h; binding it would put those at /patch/checkpoint, where a candidate
        # could warm-start from its own weights instead of the pinned base.
        ("run:patch", "/patch/candidate.patch", True),
        ("run:out", "/out", False),
    ),
    exports={
        # The two fixed inputs. run.sh reads ${BASE_MODEL:-...}, so this beats an
        # edited default -- but it is a guard-rail, not a boundary. A candidate that
        # writes a literal path into train.py never reads the variable. What fixes
        # the model is that the container has nowhere else to load from: /assets is
        # the only weights mount, the patch cannot carry weights, and there is no
        # network.
        "BASE_MODEL": "/assets/models/base",
        "TRAIN_DATA": "/assets/data/pairs.jsonl",
    },
    hooks=("inspect_patch",),
    apply_patch=True,
    reserve_sec=RETRAIN_RESERVE_SEC,
    free_gib=RETRAIN_FREE_GIB,
    # The shared artifact resolver treats this exact directory as the sole output.
    # The shipped recipe exports one adapter there; candidates may export a full model.
    checkpoint_glob="checkpoints/checkpoint-*",
    checkpoint_payload=".",
    artifact_limit=3,
    checkpoint_kind="model",
    read_only_root=False,
)

# Current B300 final receipts complete in roughly three minutes for 2,985 pairs.
# The 1,800-second ceiling leaves room for model loading and fleet contention; it
# is a safety limit rather than a target runtime.
SCORE_TIMEOUT_SEC = 1800

SCORE = Phase(
    name="score",
    timeout_sec=SCORE_TIMEOUT_SEC,
    command=(
        "python3",
        "/opt/harness/final_eval.py",
        "--output",
        "/out",
        # `reward`, not `reward.txt`. orchestrator/runner.py:report_reward reads
        # `<logs>/verifier/reward`, so OPD's `--reward-path .../reward.txt` writes a
        # file the hook never finds and the reward line is silently absent from every
        # score run. Matching the orchestrator here rather than matching OPD.
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
        # The base model is needed when the artifact is a parameter-efficient delta.
        ("asset:models/base", "/assets/models/base", True),
        ("asset:data/rewardbench", "/assets/data/rewardbench", True),
        # For the decontamination recomputation only; see the module docstring.
        ("asset:data/pairs.jsonl", "/assets/data/pairs.jsonl", True),
        ("run:checkpoint", "/ckpt", True),
    ),
    hooks=("report_reward",),
    # Nothing in this phase writes the root filesystem.
    read_only_root=True,
    pass_image_digest=True,
)

VALIDATE = checkpoint_validation_phase(SCORE, "ultrafeedback_bt_rm_rewardbench")

# Same container, synthetic rows over the real subset structure instead of a model.
# For checking the plumbing and the three-number report on a machine with neither a
# GPU nor the question set.
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
