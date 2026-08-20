"""What each phase of this task mounts, runs, and exports.

Same role as tasks/opd_math_1p5b/declaration.py; read that one first. Values only --
the Phase class lives in orchestrator/container.py beside Mount, because it is
format rather than content.

Mount sources are `asset:<relative>` for things under the asset root and
`run:<name>` for paths supplied on the command line, so no host path appears here.

Two things about this task's mount lists are worth stating, because both differ
from every other task in the repository.

**There is no data mount.** Sokoban boards are generated from a seed, not read from
a file. The generator is baked into the image twice: an editable copy at
/workspace/ragen that training uses, and a frozen read-only copy at
/opt/harness/ragen that the evaluators use. So the thing a data mount would fix is
instead fixed by the image, and harness/final_eval.py checks it from the artifact
rather than trusting it.

**The final's environment seed is not in the image.** It is the last argument of
the score command below. Every other task withholds its final by not mounting the
question set; here the questions are computable, so what has to be absent from the
exploration container is the seed that computes them. This file is host-side --
the Dockerfile copies solution/ and harness/ and nothing else -- so a value written
here never reaches a container the Agent can read. That is the whole mechanism, and
it is why FINAL_ENVIRONMENT_SEED is a literal here rather than a constant in
harness/final_eval.py.
"""

from __future__ import annotations

from dataclasses import replace

from container import Phase, checkpoint_validation_phase

# Reserve before the wall clock. RAGEN has no wall-clock stop of its own -- unlike
# OPD, nothing in the trainer ends a step early -- so this only buys the merge step
# room to finish. A candidate that raises the update count past the budget loses the
# last partial save and scores the previous complete one; see instruction.md.
RETRAIN_RESERVE_SEC = 1800
# An FSDP checkpoint of a 3B actor is model + optimizer + extra. The selected
# recipe keeps two and writes a merged HF export for each; leave additional room
# for rollout dumps, Ray's session directory and merge-time temporary files.
RETRAIN_FREE_GIB = 200

# The four proxy banks. Public, cheap to regenerate, and deliberately visible to
# the Agent -- they are what fast_eval scores.
PROXY_ENVIRONMENT_SEEDS = (4242, 4243, 4244, 4245)
# The final seed stays host-side so exploration cannot reconstruct the held-out boards.
FINAL_ENVIRONMENT_SEED = 123

# Where the policy is mounted, and why the path spells out the architecture.
#
# RAGEN's train.py decides whether the response mask is supported by reading the
# model path as a string:
#
#     assert ("qwen" in config.model_path.lower()
#             or "llama-3" in config.model_path.lower()) \
#            or (not config.enable_response_mask)
#
# Upstream never trips it, because upstream's default model_path is the hub id
# `Qwen/Qwen2.5-3B-Instruct`. Mounting the same weights at a role-named path --
# /assets/models/policy -- made the sniff answer "not qwen", and retrain died in
# config validation before the first board was generated:
#
#     AssertionError: response mask is currently only supported for qwen and
#     llama-3 models
#
# The weights ARE Qwen2.5-3B-Instruct, and `enable_response_mask: True` is
# upstream's default in both config/base.yaml and config/eval.yaml -- it is also
# part of the pinned algorithm, so switching it off to satisfy the assert would
# change which tokens enter the loss. The path changes instead and states the fact the
# sniff is asking about.
#
# Only the container path changes. The asset alias stays `models/policy`, so the
# host-side asset tree is untouched and still matches environment/assets.lock.yaml.
#
# Host-side on purpose, and this is the load-bearing part: run.sh lives in
# /workspace, so its POLICY_MODEL default is inside candidate.patch and a candidate
# can edit it. The mount and the exports below are the only places that pin this.
# Same fix shape as OWL's /assets/models/dense, for the same reason.
POLICY_MOUNT = "/assets/models/policy-qwen2.5-3b-instruct"

# Triton targets compute capability 10.3 and requires CUDA 12.9 ptxas for the
# corresponding PTX ISA. The Dockerfile installs this assembler and exports
# TRITON_PTXAS_PATH; check_image.py verifies the target at build time.
PTXAS_IN_IMAGE = "/opt/cuda-ptxas/ptxas"

EXPLORE = Phase(
    name="explore",
    timeout_sec=14400,
    command=("bash",),
    mounts=(
        ("asset:models/policy", POLICY_MOUNT, True),
        # No data mount, and no proxy-board mount: fast_eval generates its 256
        # boards from seeds 4242-4245 using the frozen generator in the image.
        #
        # /logs read-only carries the deadline the host writes: the Agent can see
        # how long it has and cannot move it. /logs/agent is writable underneath
        # it, for the instruction copy and the transcript. Two mounts, opposite
        # directions, one tree.
        ("run:logs", "/logs", True),
        ("run:logs/agent", "/logs/agent", False),
        ("run:out", "/out", False),
    ),
    exports={
        # THE SAME EXPORT RETRAIN CARRIES, AND FOR THE SAME REASON: run.sh's default
        # is /assets/models/policy and the mount is POLICY_MOUNT. Without this the
        # Agent's first `bash /workspace/run.sh` -- the command instruction.md opens
        # with -- dies in config validation, before a board is generated, on
        #
        #     AssertionError: response mask is currently only supported for qwen and
        #     llama-3 models
        #
        # which names neither the path nor the mount and reads like a model-support
        # limitation rather than a stale default. Measured in this container, and the
        # one recorded agent run on this task spent its first two commands on it.
        #
        # This works on the agent path only because runner.py's run_with_agent now
        # sets a phase's exports on the CONTAINER rather than folding them into the
        # command: `docker exec` inherits the container's environment, and
        # Phase.with_command -- which the agent path uses to make PID 1 wait -- clears
        # exports by design. That is why the first attempt at this fix, an export
        # here alone, did nothing.
        #
        # It is a guard-rail, not a boundary, exactly as in RETRAIN below: a candidate
        # is free to edit the default in run.sh, and what actually fixes the policy is
        # that /assets is the only weights mount and there is no network.
        "POLICY_MODEL": POLICY_MOUNT,
    },
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
        ("asset:models/policy", POLICY_MOUNT, True),
        # The patch file, not the directory holding it. That directory is the
        # explore phase's output and holds whatever the Agent trained in its 4 h;
        # binding it would put those weights inside this container, where a
        # candidate could warm-start from them instead of the fixed policy.
        ("run:patch", "/patch/candidate.patch", True),
        ("run:out", "/out", False),
    ),
    exports={
        # The one fixed input that has a path. run.sh reads ${POLICY_MODEL:-...},
        # so this beats an edited default -- but it is a guard-rail, not a boundary.
        # A candidate that writes a literal path into the Hydra line never reads the
        # variable. What actually fixes the policy is that the container has nowhere
        # else to load from: /assets is the only weights mount, the patch cannot
        # carry weights, and there is no network.
        #
        # It is also what makes run.sh's stale default harmless: the default still
        # reads /assets/models/policy, which is no longer mounted, and this export
        # beats it. See POLICY_MOUNT above.
        "POLICY_MODEL": POLICY_MOUNT,
    },
    hooks=("inspect_patch", "collect_checkpoints"),
    apply_patch=True,
    reserve_sec=RETRAIN_RESERVE_SEC,
    free_gib=RETRAIN_FREE_GIB,
    # finalize.py merges the FSDP shards into loadable HF exports here. Scoring an
    # unmerged work/verl_checkpoints/global_step_* directory is always invalid.
    checkpoint_glob="checkpoints/checkpoint-*",
    checkpoint_payload=".",
    artifact_limit=3,
    checkpoint_kind="model",
    read_only_root=False,
)

# Four B300 finals took 175-445 s before the service-teardown repair. 1800 s is at
# least 4x the slowest receipt and leaves room for a candidate with longer turns;
# the exact artifact predicate still refuses a partial rollout.
SCORE_TIMEOUT_SEC = 1800

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
        # The seed the boards come from. Supplied here rather than defaulted in the
        # harness, so the exploration container -- same image -- cannot read it.
        "--environment-seed",
        str(FINAL_ENVIRONMENT_SEED),
    ),
    mounts=(
        ("run:out", "/out", False),
        ("run:logs/verifier", "/logs/verifier", False),
        # /ckpt MUST BE ONE MERGED EXPORT, NOT THE CHECKPOINTS ROOT, and the harness
        # cannot tell you which you passed until it has already refused the run.
        #
        # final_eval.py's check_checkpoint_carries_no_code scans the whole mount
        # recursively and refuses any file with a code suffix, .sh among them. run.sh
        # writes its reproducibility copy to ${CKPT_DIR}/repro/run.sh -- inside the
        # checkpoints root. So mounting .../out/checkpoints, which is the path
        # instruction.md names and the one resolve_checkpoint is built to accept, is
        # refused. Measured on a landed seed:
        #
        #   /out/checkpoints           -> the checkpoint carries executable code ...
        #                                 code files ['repro/run.sh']
        #   /out/checkpoints/checkpoint-100 -> audit passed
        #
        # The refusal message says the candidate shipped code that could redefine the
        # environment, and that the trial is invalid. Nothing of the sort happened: the
        # file is the HARNESS's own, written by the harness's own launcher. A run lost
        # this way would be misread as cheating, which is worse than a crash.
        #
        # The operator contract remains: pass the leaf checkpoint-<N>. run.sh now
        # writes its reproducibility copy under ${OUTPUT_DIR}/repro, outside CKPT_DIR,
        # so the checkpoints root contains merged artifacts only after finalization.
        ("run:checkpoint", "/ckpt", True),
        # No model mount and no data mount. The checkpoint is the only input: the
        # policy being scored is the checkpoint's own weights, and the boards come
        # from the frozen generator in the image. The assembler this phase also needs
        # is in the image (PTXAS_IN_IMAGE), so it is not a mount either.
    ),
    hooks=("report_reward",),
    # final_eval.py writes only /out and /logs/verifier. Everything the rollout
    # needs -- hydra's run directory, the triton and vLLM caches -- is redirected
    # under /out, because /tmp is a 256 MiB tmpfs and /opt/harness is read-only.
    read_only_root=True,
    pass_image_digest=True,
)

VALIDATE = checkpoint_validation_phase(SCORE, "ragen_sokoban_grpo")

# Same container and mounts, synthetic rows instead of a rollout. For checking the
# plumbing on a machine with neither a GPU nor the frozen tree.
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
