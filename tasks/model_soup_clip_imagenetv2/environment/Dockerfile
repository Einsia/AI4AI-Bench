# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e
#
# Offline task-layer build with pinned source and package contexts:
#
#   docker buildx build --network=none \
#     --build-context model_soups_source=<mlfoundations/model-soups@d5398f18 tree> \
#     --build-context wheelhouse=<pinned wheel directory, incl. the CLIP wheel> \
#     --build-context git_deb=<git .deb directory> \
#     -f tasks/model_soup_clip_imagenetv2/environment/Dockerfile .
#
# Layout, by owner and lifetime:
#
#   /workspace           everything the Agent owns. Writable, and the only thing
#                        that enters candidate.patch. Holds run.sh and soup.py.
#   /opt/harness         read-only: the entry points, the evaluators, the artifact
#                        check, and the pinned model-soups tree.
#   /assets              read-only mounts (ingredients, CLIP cache, images).
#   /out /logs           mount points. Not in the image.

# The digest-pinned CUDA 12.8 base passes both the generic kernel gate and a real CLIP
# forward pass on the target accelerator. The task-specific forward pass also checks any
# compiled extensions that a generic matmul would miss.
ARG BASE_IMAGE="pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel@sha256:a7103283ea7113e10ae5d014bd2342acebda0bc53164b2f7b1dd6eb7a766bdb6"
FROM ${BASE_IMAGE}

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    WANDB_MODE=disabled \
    # The container runs as the host's uid, which has no /etc/passwd entry here.
    # getpass.getuser() falls back to pwd.getpwuid() and raises KeyError, and torch
    # calls it while setting up its inductor cache. getuser() checks these first.
    USER=agent \
    LOGNAME=agent \
    HOME=/tmp \
    # /workspace/.git is created by root at build time and read by the host uid at
    # run time, which trips git's dubious-ownership check.
    GIT_CONFIG_COUNT=1 \
    GIT_CONFIG_KEY_0=safe.directory \
    GIT_CONFIG_VALUE_0=/workspace

# git, without which submit.sh cannot produce a candidate patch at all, and without
# which the `git -C /workspace init` below fails at build time. From a build context
# because this image builds with --network=none and apt would need the network.
#
# The PyTorch base does not contain git, so the offline build installs it from the
# pinned build context. The runtime tool gate verifies it before any phase starts.
RUN --mount=type=bind,from=git_deb,target=/opt/build/git,readonly \
    dpkg -i /opt/build/git/*.deb \
    && git --version

COPY tasks/model_soup_clip_imagenetv2/environment/runtime-requirements.lock \
     /opt/build/runtime-requirements.lock
COPY tasks/model_soup_clip_imagenetv2/environment/check_image.py /opt/build/check_image.py

RUN --mount=type=bind,from=wheelhouse,target=/wheelhouse,readonly \
    --mount=type=bind,from=model_soups_source,target=/opt/build/src/model-soups,readonly \
    # The model-soups tree is code, not data, and baking it in read-only is what
    # makes "the architecture does not change" true: utils.get_model_from_sd decides
    # which module a state dict becomes and which 1000 logits come out. The search
    # under /workspace imports it from here and cannot edit it, so a soup that
    # scores well during exploration means the same thing at score time.
    #
# Only utils.py is required. Upstream's datasets package pulls in
# `imagenetv2_pytorch`, while this task selects rows directly through
# harness/grade.py's `class_offsets`.
    #
    # main.py is also left out: it downloads checkpoints, and it would be the one
    # piece of this tree with a use for a network.
    mkdir -p /opt/harness/model_soups /licenses/third-party \
    && cp /opt/build/src/model-soups/utils.py /opt/harness/model_soups/ \
    && cp /opt/build/src/model-soups/LICENSE \
      /licenses/third-party/model-soups-MIT-LICENSE \
    && test -s /licenses/third-party/model-soups-MIT-LICENSE \
    && python3 -m pip install --no-compile --no-cache-dir --no-index \
      --find-links=/wheelhouse --only-binary=:all: --no-deps --require-hashes \
      -r /opt/build/runtime-requirements.lock \
    && python3 -m pip check

COPY tasks/model_soup_clip_imagenetv2/solution/ /workspace/
COPY tasks/model_soup_clip_imagenetv2/harness/ /opt/harness/
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
    # reach. submit.sh copies it back to produce one diff over /workspace.
    && find /workspace -mindepth 2 -name .git -prune -exec rm -rf {} + \
    && git -C /workspace init -q \
    && git -C /workspace config user.email harness@localhost \
    && git -C /workspace config user.name "model soup harness" \
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
    # The container runs with --user <host uid>:<host gid>, not known at build time,
    # Ownership lets the non-root Agent record executable-bit changes as well as
    # content changes in candidate.patch.
    && chmod -R a+rwX /workspace /out \
    && chown -R "${AI4AI_AGENT_UID}:${AI4AI_AGENT_GID}" /workspace \
    && chmod -R a+rX,a-w /opt/harness \
    && chmod 0555 /opt/harness/save_checkpoint.py /opt/harness/validate_checkpoint.py \
    && chmod 0444 /opt/harness/forward.py /opt/harness/grade.py \
    && python3 /opt/build/check_image.py \
    # Build residue. /opt should hold nothing but harness.
    && rm -rf /opt/build \
    && rm -rf /tmp/* /tmp/.[!.]* /tmp/..?* /root/.cache /root/.wget-hsts \
    # Scope the tolerated no-match status to find so an earlier self-check failure
    # remains fatal.
    && { find /opt/harness /workspace -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true; }

LABEL org.opencontainers.image.title="model_soup_clip_imagenetv2" \
      org.ai4ai.task="model_soup_clip_imagenetv2" \
      org.ai4ai.model_soups="mlfoundations/model-soups@d5398f181ea51c5cd9d95ebacc6ea7132bb108ec" \
      org.ai4ai.clip="openai/CLIP@a1d071733d7111c9c014f024669f959182114e33"

WORKDIR /workspace
CMD ["bash"]
