"""What each phase of this task mounts, runs, and exports.

Values only. The Phase class lives in orchestrator/container.py beside Mount, because
it is format rather than content.

The mount lists are the load-bearing part, and their order is part of the contract: it
is the order the `--mount` flags appear in. Two entries deserve attention.

**There is no training-data mount, and that is not an omission.** This task's dataset
is generated: the policy samples its own images and the reward scores them. The only
data-like input is the prompt distribution, `simple_animals`, which is a 45-line text
file read by `random.choice` inside the pinned upstream tree -- so it is baked into the
image rather than mounted, and both the candidate's copy under /workspace and the
harness's copy under /opt/harness hold the whole of it. There is no subset to widen.

**The final's generation seed is a mount.** `asset:data/final_reference.json` appears
in score and not in explore, and it carries the final's sample count, its generation
seed, and same-tier training-start diagnostics. It exists because
/opt/harness is in the image, so the 4 h container can read every line of
final_eval.py; a seed written there would be published. On a generated dataset the seed
*is* the held-out set -- 45 prompts and one latent per row -- so an Agent that knows it
can tune against the exact 256 images the result is reported on. This is the same
mechanism the reference task uses for its AIME questions, applied to the only thing
here that plays the same role.
"""

from __future__ import annotations

from dataclasses import replace

from container import Phase, checkpoint_validation_phase

EXPLORE = Phase(
    name="explore",
    timeout_sec=14400,
    command=("bash",),
    mounts=(
        ("asset:models/stable-diffusion-v1-5", "/assets/models/stable-diffusion-v1-5", True),
        # The reward's CLIP half. Mounted into training as well as scoring, because
        # the aesthetic score IS the training reward -- unlike the reference task,
        # where the grader is a scoring-time input only. What keeps the reward honest
        # is not that the candidate cannot reach CLIP, it is that the scoring path
        # loads its scorer from /opt/harness and runs in a container with no patch
        # applied. See harness/grade.py:import_scorer.
        ("asset:models/clip", "/assets/models/clip", True),
        # No data/final_reference.json. That is what makes "the Agent cannot tune
        # against the exact images the result is reported on" true.
        #
        # /logs read-only carries the deadline the host writes: the Agent can see how
        # long it has and cannot move it. /logs/agent is writable underneath it.
        ("run:logs", "/logs", True),
        ("run:logs/agent", "/logs/agent", False),
        ("run:out", "/out", False),
    ),
    hooks=("write_deadline", "report_submission"),
    # The Agent writes /workspace in the image layer, so the root filesystem cannot be
    # read-only.
    read_only_root=False,
)

RETRAIN = Phase(
    name="retrain",
    timeout_sec=43200,
    # The public driver launches the inner trainer and exports the selected complete
    # adapter to /out/checkpoint. Direct and formal runs therefore share one entry point.
    command=("bash", "/workspace/run.sh"),
    mounts=(
        ("asset:models/stable-diffusion-v1-5", "/assets/models/stable-diffusion-v1-5", True),
        ("asset:models/clip", "/assets/models/clip", True),
        # The patch file, not the directory holding it. That directory is the explore
        # phase's output and contains the adapters the Agent trained in its 4 h;
        # binding it would put those where a candidate could warm-start from its own
        # weights instead of the fixed base model.
        ("run:patch", "/patch/candidate.patch", True),
        ("run:out", "/out", False),
    ),
    exports={
        # The two fixed inputs. run.sh reads ${DDPO_MODEL:-...}, so this beats an
        # edited default -- but it is a guard-rail, not a boundary. A candidate that
        # writes a literal path into ddpo_config.py never reads the variable. What
        # actually fixes the model is that the container has nowhere else to load
        # from: /assets is the only weights mount, the patch cannot carry weights, and
        # there is no network.
        "DDPO_MODEL": "/assets/models/stable-diffusion-v1-5",
        "DDPO_CLIP_PATH": "/assets/models/clip",
    },
    hooks=("inspect_patch",),
    apply_patch=True,
    # The driver stops its process group before the outer deadline, then exports the
    # latest complete epoch checkpoint. Reserve space for three published adapters and
    # the generated-image/cache working set.
    reserve_sec=300,
    free_gib=20,
    #
    # DECLARED rather than left empty, because this task has two directories that both
    # look like the artifact and only one is. train.py's final export is out/checkpoint;
    # upstream's own saves sit four levels deeper at
    # out/logs/<run>_<timestamp>/checkpoints/checkpoint_<n>, and with
    # NUM_CHECKPOINT_LIMIT=2 the last two survive. A host-side search has to pick
    # between them by traversal order, and the operator baseline runner documents getting that
    # order wrong in both directions.
    #
    # At run.sh's default NUM_EPOCHS=13 the choice is harmless -- the highest retained
    # save IS what train.py copied, byte-identical, verified across the landed seeds
    # (43 c1bb8c5b, 50 aa46c0d2, 52 eebd7736). It stops being harmless as soon as a
    # candidate changes the schedule. Then "the deepest checkpoint_<n>" and
    # "the epoch the run finished on" are different weights and both are plausible.
    #
    # A plain path is taken as given, so this names the export and ends the guessing.
    checkpoint_glob="checkpoints/checkpoint-*",
    checkpoint_payload=".",
    artifact_limit=3,
    checkpoint_kind="weights",
    read_only_root=False,
)

# Carried from the reference protocol's judge-side final.toml, which declared 7200 s for a
# 256-sample run. It is a declared ceiling, NOT a measurement: no wall clock was ever
# recorded for this task -- I grepped the whole of baseline/evidence.json for
# time|sec|dur|wall|elapsed and got nothing. The `samples: 88` in its resource profile
# is a GPU-monitor sample count with no stated interval, so it yields no duration
# either. Measure one score run and replace this.
SCORE_TIMEOUT_SEC = 7200

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
        "--reference",
        "/assets/data/final_reference.json",
    ),
    mounts=(
        ("run:out", "/out", False),
        ("run:logs/verifier", "/logs/verifier", False),
        ("asset:models/stable-diffusion-v1-5", "/assets/models/stable-diffusion-v1-5", True),
        ("asset:models/clip", "/assets/models/clip", True),
        # Final-only: sample count, generation seed, and the training-start reference
        # alignment and diversity diagnostics are compared against.
        ("asset:data/final_reference.json", "/assets/data/final_reference.json", True),
        ("run:checkpoint", "/ckpt", True),
    ),
    hooks=("report_reward",),
    # Nothing in this phase writes the root filesystem.
    read_only_root=True,
    pass_image_digest=True,
)

VALIDATE = checkpoint_validation_phase(SCORE, "ddpo_sd15_aesthetic")

# Same container, synthetic rows instead of generation. For checking the plumbing on a
# machine with neither a GPU nor the mounted protocol.
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

# An end-to-end diagnostic case without a GPU: synthetic rows that collapse
# diversity. It remains scoreable and must report alignment/diversity alerts.
SCORE_MOCK_COLLAPSED = replace(
    SCORE_MOCK,
    name="score-mock-collapsed",
    command=(
        "python3",
        "/opt/harness/final_eval.py",
        "--output",
        "/out",
        "--reward-path",
        "/logs/verifier/reward",
        "--mock-collapsed",
    ),
)

PHASES: dict[str, Phase] = {
    "explore": EXPLORE,
    "retrain": RETRAIN,
    "score": SCORE,
    "checkpoint-validate": VALIDATE,
    "score-mock": SCORE_MOCK,
    "score-mock-collapsed": SCORE_MOCK_COLLAPSED,
}
