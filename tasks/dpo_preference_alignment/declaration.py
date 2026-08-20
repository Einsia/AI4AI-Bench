"""What each phase of this task mounts, runs, and exports.

Values only. The Phase class lives in orchestrator/container.py, because it is format
rather than content.

The mount lists are the load-bearing part, and on this task they carry the *whole*
constraint model -- there is no artifact-side check at all. Four things they hold,
none of which is enforced by a checker or a file permission:

  * explore has no `data/ifeval_final` entry. It gets `data/ifeval_proxy`, the 128
    proxy rows, and nothing else -- so 285 of the final's 413 rows are unreachable
    during the 4 h. This is what makes the held-out number held out.
  * retrain has no IFEval entry at all, proxy or otherwise. The 12 h run trains; it
    does not measure.
  * every phase gets `models/policy_start` and only `models/policy_start`. It is the
    training start, the frozen DPO reference policy and the backbone the score phase
    applies an adapter to, all one read-only tree. Nothing else in any container can
    supply weights, and there is no network.
  * retrain and explore get the whole UltraFeedback snapshot, not the 8192 rows the
    shipped default reads.

## What this replaces, and why nothing here is a check

The reference protocol's `[algorithm].invariants` were three lines:

    "the reference policy, preference pairs, chat format and target model remain frozen"
    "the objective remains pairwise sigmoid DPO with QLoRA adapters"
    "SFT, ORPO, SimPO, KTO, PPO, reward-model training and reference-free objectives
     are forbidden"

The second and third are exactly what v1 relaxes, so they are gone, along with
`eval/selection.toml`'s `require_algorithm_family = "pairwise_dpo_qlora"` and
`require_reference_policy = "frozen_zephyr_7b_sft_start"`, the `[agent]`
`mutable_recipe`/`mutable_method_globs`/`forbidden_capabilities` allowlists, and
`environment/check_candidate.py` with its `known-invalid/` corpus.

The first line survives, and the four bullets above are the whole of how. There is no
`harness/final_eval.py` artifact check to go with them -- unlike OWL, whose 70%
sparsity is a property of the weights that no mount can hold, everything frozen here
is frozen by having nowhere else to come from.

## score mounts the reference policy, and that is on purpose

`models/policy_start` appears in the score phase. It is not there to check anything;
it is there so a candidate can hand back a LoRA adapter instead of a merged 14.5 GiB
model. `trainer.save_model()` on a PEFT model writes the adapter alone -- the merge in
the shipped train.py is a separate step after it -- so an adapter is the most likely
artifact a modified trainer produces, and voiding a trial over that would be voiding
it over a serialization detail. Taking that path also makes the backbone provably come
from this mount rather than from the submission. See harness/checkpoint.py.
"""

from __future__ import annotations

from dataclasses import replace

from container import Phase, checkpoint_validation_phase

# Reserve before the wall clock, so the trainer stops in time to write a complete
# artifact rather than being killed mid-write.
#
# 1200 s, matching OPD rather than BT-RM's 600. The export is not a save: with
# EXPORT_MODE=merged, train.py reloads the frozen start in bf16, merges the adapter and
# writes ~14.5 GiB in 4 GiB shards. BT-RM writes a few tens of MiB and can afford
# less. Not measured -- the reference protocol never timed the export separately -- so
# this is sized to the artifact it has to write.
RETRAIN_RESERVE_SEC = 1200
# The merged checkpoint is 14.5 GiB, and beside it the run holds up to
# SAVE_TOTAL_LIMIT intermediate adapters at 674 MB each (measured on the previous
# branch), the HF datasets cache for the tokenized pool, and the repro copy. About
# 20 GiB in the shipped configuration.
#
# 40 is therefore a deliberate over-estimate rather than a measurement: peak disk was
# never recorded, and a candidate that raises LORA_R or SAVE_TOTAL_LIMIT moves the
# intermediate term. A truncated export is worse than a refusal at startup.
RETRAIN_FREE_GIB = 40

EXPLORE = Phase(
    name="explore",
    timeout_sec=14400,
    command=("bash",),
    mounts=(
        ("asset:models/policy_start", "/assets/models/policy_start", True),
        # The whole snapshot, both splits, not the 8192 + 128 rows the shipped default
        # reads. "How much data" is one of the things v1 hands back.
        ("asset:data/ultrafeedback", "/assets/data/ultrafeedback", True),
        # The proxy's 128 rows. No data/ifeval_final entry: that is what keeps the
        # other 285 rows of the final unseen.
        ("asset:data/ifeval_proxy", "/assets/data/ifeval_proxy", True),
        # /logs read-only carries the deadline the host writes: the Agent can see how
        # long it has and cannot move it. /logs/agent is writable underneath, because
        # the instruction copy and transcript land there.
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
        ("asset:models/policy_start", "/assets/models/policy_start", True),
        ("asset:data/ultrafeedback", "/assets/data/ultrafeedback", True),
        # The patch file, not the directory holding it. That directory is the explore
        # phase's output and holds whatever the Agent trained in its 4 h; binding it
        # would put those weights at /patch/checkpoint, where a candidate could
        # warm-start from its own adapter instead of the pinned start.
        ("run:patch", "/patch/candidate.patch", True),
        ("run:out", "/out", False),
    ),
    exports={
        # The two fixed inputs. run.sh reads ${POLICY_START:-...}, so this beats an
        # edited default -- but it is a guard-rail, not a boundary. A candidate that
        # writes a literal path into train.py never reads the variable. What fixes the
        # model is that the container has nowhere else to load from: /assets is the
        # only weights mount, the patch cannot carry weights, and there is no network.
        "POLICY_START": "/assets/models/policy_start",
        "TRAIN_DATA": "/assets/data/ultrafeedback",
    },
    hooks=("inspect_patch",),
    apply_patch=True,
    reserve_sec=RETRAIN_RESERVE_SEC,
    free_gib=RETRAIN_FREE_GIB,
    # The publication helper converts the final export and retained trainer saves
    # into the common numbered formal-artifact protocol.
    checkpoint_glob="checkpoints/checkpoint-*",
    checkpoint_payload=".",
    artifact_limit=3,
    checkpoint_kind="model",
    read_only_root=False,
)

# 14400, which is what the reference protocol's judge-side `final.toml` allocated for
# this exact 413-row final at gpus = 1.
#
# The arithmetic behind it, since three independent routes agree: 128 rows measured
# 1827 s / 1716 s / 1932 s on one device, i.e. about 14.3 s per row; the 541-row final
# measured 8454 s, i.e. 15.6 s per row; and the 413-row hidden run measured 2946 s on
# *two* devices. All three put a one-device 413-row pass near 5900 s. 14400 is about
# 2.4x that.
#
# The multiple stays generous rather than being tightened to the arithmetic, because
# 5900 s is an implication and not a measurement -- no single-device 413-row run
# exists -- and because the recorded failure mode here is a timeout: two runs are in
# evidence.json as "invalid_runtime_timeout", where a single-GPU IFEval pass blew a
# four-hour limit and produced no summary at all. Replace this with a measurement, not
# with a smaller guess.
SCORE_TIMEOUT_SEC = 14400

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
        # file the hook never finds and its reward line is silently absent from every
        # score run. Matching the orchestrator here rather than matching OPD.
        "--reward-path",
        "/logs/verifier/reward",
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
        # The frozen start, because the submission may be an adapter and the backbone
        # it applies to must come from the pinned source. See the module docstring.
        ("asset:models/policy_start", "/assets/models/policy_start", True),
        # All 541 source rows. The tiers are cut here rather than mounted pre-sliced,
        # so the 128/413/285 arithmetic is recomputed in the container that uses it.
        ("asset:data/ifeval_final", "/assets/data/ifeval_final", True),
        ("run:checkpoint", "/ckpt", True),
    ),
    hooks=("report_reward",),
    # Nothing in this phase writes the root filesystem.
    read_only_root=True,
    pass_image_digest=True,
)

VALIDATE = checkpoint_validation_phase(SCORE, "dpo_preference_alignment")

# Same container, synthetic rows over the real split structure instead of a model. For
# checking the plumbing and the three-number report on a machine with neither a GPU nor
# the question set.
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
