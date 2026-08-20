"""What each phase of this task mounts, runs, and exports.

This task is the reason the Phase dataclass has to be about containers rather
than about training. There is no training loop anywhere in it: the candidate
selects or weights 72 frozen checkpoints and averages them, so the artifact is
produced by a search over fixed weights rather than by gradient descent. Three
fields on Phase are shaped by OPD's training loop and land differently here.

  output_glob      a numbered formal-artifact series, using construction progress.

  reserve_sec      Kept, and this is a deliberate departure from "a task with no
                   training loop has no wall clock to reserve against". The
                   a prior greedy search took 10362 s (2.88 h) at two validation
                   images per class; a candidate that raises
                   that count or widens the merge trials can walk past 12 h, and
                   a search killed at the wall exports nothing at all. So the
                   phase hands the search a deadline and solution/soup.py stops
                   one trial short of it and exports the best soup it has. The
                   mechanism is the same as OPD's -- Phase.exports_with_wall_clock
                   turns reserve_sec into MAX_WALL_TIME_SECONDS and
                   DEADLINE_RESERVE_SECONDS -- and the reason is the same too: a
                   partial result beats no result, and both are better than a
                   truncated write.

  free_gib         Kept but small. One soup is ~456 MiB, not OPD's 27 GiB.

Values only. The Phase class itself lives in orchestrator/container.py beside
Mount, because it is format rather than content.

Mount sources are `asset:<relative>` for things under the asset root and
`run:<name>` for paths supplied on the command line, so no host path appears here.

The mount lists are the load-bearing part. Two entries carry the whole
proxy/final boundary:

  explore mounts   asset:data/imagenetv2_proxy   2000 images, offsets 0-1
  score mounts     asset:data/imagenetv2_final   10000 images, offsets 0-9

The proxy tree is a materialized projection of the final one rather than a view
of it, and that is the point. ImageNetV2 arrives as 1000 class directories of 10
files; if the exploration container held all 10000 files, "the proxy is offsets
0-1" would be a request rather than a fact, because 80% of the final rows would
be sitting in a readable mount. Mounting a 2000-file tree makes the split
structural, the same way OPD's explore has a math500 entry and no aime entry.

What that costs is written down where it belongs: environment/assets.lock.yaml
carries the selection rule, and harness/final_eval.py re-derives the same rule
rather than trusting a manifest, so the two agree by construction instead of by
convention.
"""

from __future__ import annotations

from dataclasses import replace

from container import Phase, checkpoint_validation_phase

# Reserve before the wall clock, so the search abandons its current trial and
# writes a complete soup rather than being killed mid-export. 900 s is a whole
# trial plus the export: the shipped search's 143 scoring passes took 10362 s,
# about 72 s each, and torch.save of a 456 MiB state dict is seconds. A candidate
# whose single trial costs more than 900 s gets a truncated last trial rather than
# a truncated file, which is the trade worth making.
BUILD_RESERVE_SEC = 900
# One exported soup is ~456 MiB. The headroom is for a candidate that keeps
# several trial soups, not for the shipped method, which keeps one.
BUILD_FREE_GIB = 8

EXPLORE = Phase(
    name="explore",
    timeout_sec=14400,
    command=("bash",),
    mounts=(
        # All 72 ingredients, read-only. Not a subset: a candidate that could
        # only see some of them could not evaluate the uniform baseline it is
        # measured against.
        ("asset:models/ingredients", "/assets/models/ingredients", True),
        # The CLIP ViT-B/32 base. `get_model_from_sd` needs it to turn a state
        # dict into a module, so both the search and the evaluator load it.
        ("asset:models/clip", "/assets/models/clip", True),
        # 2000 images, offsets 0-1. The other 8000 final rows are not in this
        # container. See the module docstring.
        ("asset:data/imagenetv2_proxy", "/assets/data/imagenetv2_proxy", True),
        # /logs read-only carries the deadline the host writes: the Agent can see
        # how long it has and cannot move it. /logs/agent is writable underneath
        # it, because the agent's instruction copy and transcript land there.
        ("run:logs", "/logs", True),
        ("run:logs/agent", "/logs/agent", False),
        ("run:out", "/out", False),
    ),
    hooks=("write_deadline", "report_submission"),
    # The Agent writes /workspace in the image layer, so the root filesystem
    # cannot be read-only.
    read_only_root=False,
)

# 12 h, from the reference protocol's [budget].formal_seconds. The shipped greedy search
# uses 10362 s of it, so a candidate has roughly 4x the shipped search to spend --
# which is the budget fact worth telling the Agent, and instruction.md does.
BUILD_TIMEOUT_SEC = 43200

BUILD = Phase(
    name="retrain",
    timeout_sec=BUILD_TIMEOUT_SEC,
    command=("bash", "/workspace/run.sh"),
    mounts=(
        ("asset:models/ingredients", "/assets/models/ingredients", True),
        ("asset:models/clip", "/assets/models/clip", True),
        # The proxy rows, again. The soup search needs something to score
        # candidate merges against, and this is the same 2000 images the Agent
        # tuned on -- so the 12 h run reproduces the search it watched, rather
        # than running the same code against a row set it has never seen.
        ("asset:data/imagenetv2_proxy", "/assets/data/imagenetv2_proxy", True),
        # The patch file, not the directory holding it. That directory is the
        # exploration phase's output and contains any soups the Agent built in
        # its 4 h; binding it would put those at /patch/, where a candidate could
        # export a checkpoint it had already produced instead of building one
        # from the ingredients.
        ("run:patch", "/patch/candidate.patch", True),
        ("run:out", "/out", False),
    ),
    exports={
        # The three fixed inputs. run.sh reads ${INGREDIENTS:-...}, so this beats
        # an edited default -- but it is a guard-rail, not a boundary. What fixes
        # the ingredient set is that the container has nowhere else to load
        # weights from: /assets is the only weights mount, the patch cannot carry
        # weights, there is no network, and harness/final_eval.py checks the
        # exported soup against the ingredient basis afterwards.
        "INGREDIENTS": "/assets/models/ingredients",
        "CLIP_CACHE": "/assets/models/clip",
        "PROXY_DATA": "/assets/data/imagenetv2_proxy",
    },
    hooks=("inspect_patch",),
    apply_patch=True,
    reserve_sec=BUILD_RESERVE_SEC,
    free_gib=BUILD_FREE_GIB,
    # A plain path is taken as given by the artifact resolver. There is no
    # global_step_N sequence to sort.
    checkpoint_glob="checkpoints/checkpoint-*",
    checkpoint_payload="model.pt",
    artifact_limit=3,
    checkpoint_kind="file",
    read_only_root=False,
)

# NOT MEASURED. Two costs sit under this number and neither has a measurement
# behind it: 10000 images through one CLIP ViT-B/32, which is cheap, and two
# streaming passes over the 30.6 GiB ingredient set for the artifact check, which
# is disk-bound. The reference protocol's 3600 s for its 2000-row proxy was itself a
# ceiling rather than a measured cost, so there is nothing to scale from. 7200 is
# a ceiling; measure one score phase and replace it.
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
        "/logs/verifier/reward.txt",
        "--checkpoint",
        "/ckpt",
        "--assets",
        "/assets",
    ),
    mounts=(
        ("run:out", "/out", False),
        ("run:logs/verifier", "/logs/verifier", False),
        # All 10000 images, offsets 0-9. The proxy rows are 2000 of these, which
        # is what lets final_eval.py report score(P) and score(F\P) from one pass.
        ("asset:data/imagenetv2_final", "/assets/data/imagenetv2_final", True),
        ("asset:models/clip", "/assets/models/clip", True),
        # The ingredients are mounted into the SCORING phase, which no other task
        # here does with a training input. They are the basis the artifact check
        # solves against: "this checkpoint is an affine combination of the 72"
        # cannot be answered without the 72. Read-only, and the phase never
        # writes anything but /out and /logs/verifier.
        ("asset:models/ingredients", "/assets/models/ingredients", True),
        ("run:checkpoint", "/ckpt", True),
    ),
    hooks=("report_reward",),
    # Nothing in this phase writes the root filesystem.
    read_only_root=True,
    pass_image_digest=True,
)

VALIDATE = checkpoint_validation_phase(SCORE, "model_soup_clip_imagenetv2")

# Same container and mounts, synthetic rows instead of a forward pass. For
# checking the plumbing on a machine with neither a GPU nor the images. --mock
# also skips the artifact check, which needs the real ingredient set; the check
# has its own self-test, `python3 harness/soup_check.py --smoke`, which builds a
# small synthetic basis and proves the check separates a combination from a
# perturbation.
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
    # Named "retrain" for the lifecycle position rather than for what it does:
    # fresh container, candidate patch applied, up to three artifacts produced, 12 h. It
    # retrains nothing. runner.py's CLI documents the three phase names and two of
    # the nine tasks in this repository do not train, so the name is the
    # orchestrator's vocabulary and this task borrows it rather than inventing a
    # fourth one.
    "retrain": BUILD,
    "score": SCORE,
    "checkpoint-validate": VALIDATE,
    "score-mock": SCORE_MOCK,
}
