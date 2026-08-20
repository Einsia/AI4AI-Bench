# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e
#
# Build (network disabled, four named contexts):
#
#   docker buildx build --network=none \
#     --build-context ragen_source=<RAGEN@20daedc4 tree, verl submodule included> \
#     --build-context git_deb=<git .deb directory> \
#     --build-context wheelhouse=<pinned wheel directory> \
#     --build-context cuda_ptxas=<directory holding a CUDA 12.9 ptxas> \
#     -f tasks/ragen_sokoban_grpo/environment/Dockerfile .
#
# These four lines are not only documentation: the operator image builder reads the
# context names out of this comment block to build the --build-context arguments, so a
# context used below and not listed here fails the build with "missing build context".
#
# Layout, by owner and lifetime:
#
#   /workspace       everything the Agent owns. Writable, and the only thing that
#                    enters candidate.patch. Holds run.sh, finalize.py and ragen/.
#   /opt/harness     read-only: the three .sh entry points, the evaluators, and a
#                    second pristine copy of the RAGEN tree under ragen/.
#   /assets          read-only mounts (the policy). Not in the image.
#   /out /logs       mount points. Not in the image.
#
# The two copies of the RAGEN tree are the point of this file. RAGEN is both the
# training framework and the Sokoban environment, so "the framework is editable" and
# "the environment is frozen" are claims about the same source. One editable copy
# under /workspace and one frozen copy under /opt/harness is what lets both be true:
# training uses the first, the evaluators use the second, and harness/final_eval.py
# checks at score time that the rollout resolved to the second. The tree is 37.7 MB,
# so the duplication is free.
#
# The base follows RAGEN's pinned verl revision.
# RAGEN@20daedc4 pins verl at d62da495 = verl's release/v0.6.1 head, and verl 0.6.1's
# reference recipe directory is `verl0.6-cu128-torch2.8.0-fa2.7.4`: cu128, torch
# 2.8.0. This base is cu128 + torch 2.8.0+cu128. So the runtime here is the one
# upstream verl 0.6.1 targets. `vllm==0.8.2` in RAGEN's setup.py is stale metadata;
# verl 0.6.1 itself asks for
# `vllm>=0.8.5,<=0.11.0`.
#
# The selected CUDA 12.8 stack also passes the repository's kernel and collective gates
# on the target accelerator.
#
# What is NOT changed: the two-tree layout, the compat patch, the per-request seed
# scheme, the pristine .git baseline, and every path in instruction.md.

ARG PYTORCH_IMAGE="pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel@sha256:a7103283ea7113e10ae5d014bd2342acebda0bc53164b2f7b1dd6eb7a766bdb6"
FROM ${PYTORCH_IMAGE}

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH=/workspace/ragen:/workspace/ragen/verl \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    WANDB_MODE=disabled \
    # The container runs as the host's uid, which has no /etc/passwd entry here.
    # getpass.getuser() falls back to pwd.getpwuid() and raises KeyError, and torch
    # calls it while setting up its inductor cache -- so without USER set, importing
    # torch fails outright. getuser() checks these env vars first. Still true on this
    # base: it has no entry for an arbitrary host uid either.
    USER=agent \
    LOGNAME=agent \
    HOME=/tmp \
    # /workspace/.git is created by root at build time and read by the host uid at
    # run time, which trips git's dubious-ownership check. This is the only way to
    # pre-authorize it without a writable global gitconfig.
    GIT_CONFIG_COUNT=1 \
    GIT_CONFIG_KEY_0=safe.directory \
    GIT_CONFIG_VALUE_0=/workspace

# git, without which submit.sh cannot produce a candidate patch at all, and without
# which the `git apply` of the compat patch below cannot run either -- from a build
# context, because this build has no network and apt would need one.
#
# The PyTorch base carries no git. The runtime tool gate verifies this pinned install
# before any phase starts.
RUN --mount=type=bind,from=git_deb,target=/opt/build/git,readonly \
    dpkg -i /opt/build/git/*.deb \
    && git --version

COPY tasks/ragen_sokoban_grpo/environment/runtime-requirements.lock /opt/build/runtime-requirements.lock
COPY tasks/ragen_sokoban_grpo/environment/ragen_runtime_compat.patch /opt/build/ragen_runtime_compat.patch
COPY tasks/ragen_sokoban_grpo/environment/check_image.py /opt/build/check_image.py

RUN --mount=type=bind,from=wheelhouse,target=/wheelhouse,readonly \
    --mount=type=bind,from=ragen_source,target=/opt/build/src/ragen,readonly \
    # Remove any conflicting framework copies before installing the pinned tree.
    rm -rf /opt/ragen /opt/RAGEN /opt/llm-algobench \
    && mkdir -p /workspace /opt/harness \
    && cp -a /opt/build/src/ragen /workspace/ragen \
    # Apply the compatibility and per-request-seed patch before the copy, so both
    # trees are byte-identical. --check first, so a context drift fails the build
    # rather than accepting a partially applicable patch.
    && git -C /workspace/ragen apply --check -p1 /opt/build/ragen_runtime_compat.patch \
    && git -C /workspace/ragen apply -p1 /opt/build/ragen_runtime_compat.patch \
    && rm -rf /workspace/ragen/.git /workspace/ragen/verl/.git \
    && find /workspace/ragen -name '__pycache__' -prune -exec rm -rf {} + \
    # The frozen copy. This is the environment, the solve reward and the rollout
    # code the evaluators use, and the reason "the environment does not change" is a
    # property of the image rather than of a mount.
    && cp -a /workspace/ragen /opt/harness/ragen \
    # --force-reinstall so numpy lands at 1.26.4 over the base's 2.3.2: verl 0.6.1
    # requires numpy<2.0.0 and gym 0.26.2 refuses NumPy 2 at import. It is the only
    # base package the wheelhouse replaces -- vllm 0.11.0 requires torch==2.8.0, which
    # the base already is, so the torch/vLLM stack is added to, not overwritten.
    && python -m pip install --no-compile --no-cache-dir --no-index --find-links=/wheelhouse \
      --only-binary=:all: --no-deps --require-hashes --force-reinstall \
      -r /opt/build/runtime-requirements.lock

# The PTX assembler, and the one ENV this image sets for a reason outside the recipe.
#
# The B300 card reports compute capability (10, 3). Triton asks ptxas for `sm_103a`,
# and the CUDA 12.8 in this base -- both /usr/local/cuda-12.8 and the copy inside
# triton -- does not have that name:
#
#     ptxas -lineinfo -v --gpu-name=sm_103a /out/tmp/tmpXXXX.ptx -o ...
#     ptxas fatal   : Value 'sm_103a' is not defined for option 'gpu-name'
#     torch._inductor.exc.InductorError: NoTritonConfigsError: No valid triton configs.
#
# which killed every torch.compile in the run: vLLM's sampler helper
# (vllm/v1/sample/ops/logprobs.py:batched_count_greater_than, a module-level
# @torch.compile that enforce_eager does not govern) and verl's entropy_from_logits,
# which this recipe runs with actor_rollout_ref.actor.use_torch_compile=True.
#
# 12.9 is not an arbitrary "newer": it is the only version that works. sm_103 first
# exists in CUDA 12.9, and triton 3.4.0 refuses anything above major 12 --
# compiler.ptx_get_version raises "Triton only support CUDA 10.0 or higher, but got
# CUDA version: 13.0" for the host's 13.0 ptxas. The version also decides the PTX
# `.version` directive triton emits (compiler.make_ptx rewrites it from the parsed
# ptxas version), and sm_103 needs PTX ISA 8.8; a 12.8 claim produces
# "error : PTX .version 8.7 does not support .target sm_103a". 12.9 maps to 8.8.
#
# Keep the path in the image environment so every phase, including an agent-replaced
# command, receives the same compiler.
#
# A candidate can still override or unset it. It is toolchain, not a fixed input --
# the only consequence of pointing it elsewhere is that its own kernels stop
# compiling.
COPY --from=cuda_ptxas ptxas /opt/cuda-ptxas/ptxas
ENV TRITON_PTXAS_PATH=/opt/cuda-ptxas/ptxas

COPY tasks/ragen_sokoban_grpo/solution/ /workspace/
COPY tasks/ragen_sokoban_grpo/harness/ /opt/harness/
COPY orchestrator/lifecycle.py /opt/harness/lifecycle.py
COPY orchestrator/container_harness/ /opt/harness/
COPY LICENSE THIRD_PARTY_NOTICES.md /licenses/

# Fixed so the baseline commit hash is a function of the tree alone. Without it two
# builds minutes apart produce different commit objects from an identical tree, and
# the image digest stops being reproducible.
ARG SOURCE_DATE_EPOCH=1767225600
ARG AI4AI_AGENT_UID=20003
ARG AI4AI_AGENT_GID=20003

RUN chmod 0755 /workspace/run.sh /opt/harness/*.sh \
    # Capture the pristine tree as one commit, then move .git out of the Agent's
    # reach. submit.sh copies it back to produce a single diff covering run.sh,
    # finalize.py and every edit to ragen/ together.
    #
    # -f defeats RAGEN's and verl's .gitignore. Without it an Agent file matching an
    # ignore rule would be silently absent from candidate.patch and would not exist
    # in the 12 h container.
    && find /workspace -mindepth 2 -name .git -prune -exec rm -rf {} + \
    && git -C /workspace init -q \
    && git -C /workspace config user.email harness@localhost \
    && git -C /workspace config user.name "ragen harness" \
    && git -C /workspace config commit.gpgsign false \
    && git -C /workspace add -A -f \
    && GIT_AUTHOR_DATE="@${SOURCE_DATE_EPOCH} +0000" \
       GIT_COMMITTER_DATE="@${SOURCE_DATE_EPOCH} +0000" \
       git -C /workspace -c core.hooksPath=/dev/null commit -q -m "pristine baseline" \
    && mkdir -p /opt/harness/git-base \
    # The index caches per-file stat data -- mtime, ctime, inode, device -- so it
    # differs across otherwise identical builds. submit.sh rebuilds it with
    # `git add -A -f`, and the diff is against HEAD either way.
    && rm -f /workspace/.git/index \
    && mv /workspace/.git /opt/harness/git-base/.git \
    && mkdir -p /assets /out /logs \
    # Ownership lets the non-root Agent record executable-bit changes as well as
    # content changes in candidate.patch.
    && chmod -R a+rwX /workspace /out \
    && chown -R "${AI4AI_AGENT_UID}:${AI4AI_AGENT_GID}" /workspace \
    && chmod -R a+rX,a-w /opt/harness \
    && chmod 0555 /opt/harness/save_checkpoint.py /opt/harness/validate_checkpoint.py \
    && chmod 0555 /opt/harness/gpu_phase_lock.py \
    && python /opt/build/check_image.py \
    # Build residue. /opt should hold nothing but harness, so the layout a reader
    # sees is the layout the docs describe.
    && rm -rf /opt/build \
    && rm -rf /tmp/* /tmp/.[!.]* /tmp/..?* /root/.cache /root/.wget-hsts \
    # Scope the tolerated no-match status to find so an earlier self-check failure
    # remains fatal.
    && { find /opt/harness /workspace -name '__pycache__' -prune \
           -exec rm -rf {} + 2>/dev/null || true; }

LABEL org.opencontainers.image.title="ragen_sokoban_grpo" \
      org.ai4ai.task="ragen_sokoban_grpo" \
      org.ai4ai.ragen="mll-lab-nu/RAGEN@20daedc47558e000f7de912b060646bf2e8026bd" \
      org.ai4ai.verl="volcengine/verl@d62da4950573d7a4b7ef2362337952e7ab59e78d"

WORKDIR /workspace
CMD ["bash"]
