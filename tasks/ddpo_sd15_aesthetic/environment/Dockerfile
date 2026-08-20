# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e
#
# Build (network disabled, two named contexts):
#
#   docker buildx build --network=none \
#     --build-context ddpo_source=<kvablack/ddpo-pytorch@1958463f tree> \
#     --build-context git_deb=<git .deb directory> \
#     --build-context wheelhouse=<pinned wheel directory> \
#     -f tasks/ddpo_sd15_aesthetic/environment/Dockerfile .
#
# Layout, by owner and lifetime:
#
#   /workspace     everything the Agent owns. Writable, and the only thing that
#                  enters candidate.patch. Holds run.sh, train.py, ddpo_config.py
#                  and the ddpo-pytorch tree.
#   /opt/harness   read-only: the three .sh entry points, the evaluators, a SECOND
#                  pinned copy of ddpo-pytorch holding the reward, and the pristine
#                  .git baseline.
#   /assets        read-only mounts (base model, CLIP, the final's protocol).
#   /out /logs     mount points. Not in the image.
#
# The upstream tree is copied TWICE, and that is the central decision in this file.
# The aesthetic scorer is not only the evaluator, it is the training reward, so a
# single shared copy would mean the candidate edits the thing it is scored on. The
# /opt/harness copy is what harness/grade.py loads; the /workspace copy is the
# candidate's to change. 4.1 MiB, so the duplication costs nothing.

# The CUDA 12.8 base is digest-pinned and passes the repository's real kernel gate on the
# target accelerator. Package pins that affect the method remain in runtime-requirements.in.
ARG PYTORCH_IMAGE="pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel@sha256:a7103283ea7113e10ae5d014bd2342acebda0bc53164b2f7b1dd6eb7a766bdb6"
FROM ${PYTORCH_IMAGE}

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    WANDB_MODE=disabled \
    PYTHONHASHSEED=0 \
    # The container runs as the host's uid, which has no /etc/passwd entry here.
    # getpass.getuser() falls back to pwd.getpwuid() and raises KeyError; accelerate
    # and wandb both call it during setup. getuser() checks these env vars first.
    USER=agent \
    LOGNAME=agent \
    HOME=/tmp \
    # /workspace/.git is created by root at build time and read by the host uid at
    # run time, which trips git's dubious-ownership check. This is the only way to
    # pre-authorize it without a writable global gitconfig.
    GIT_CONFIG_COUNT=1 \
    GIT_CONFIG_KEY_0=safe.directory \
    GIT_CONFIG_VALUE_0=/workspace

# Deliberately NOT set: PYTHONPATH. An editable install would expose both source copies:
# `ddpo_pytorch` has no __init__.py, so it is a namespace package, and with both trees
# reachable `ddpo_pytorch.__path__` holds both directories. The harness would then
# import a package merged with the candidate's copy.
# run.sh sets cwd to the tree instead, which is all upstream needs, and grade.py
# scrubs /workspace off sys.path before it imports anything.
#
# An editable install would also fight the fixed runtime pins: upstream's setup.py declares
# accelerate==0.17 while the task pins 0.20.3.

# submit.sh requires git. Install it from a pinned context so the task layer remains
# buildable with --network=none.
RUN --mount=type=bind,from=git_deb,target=/opt/build/git,readonly \
    dpkg -i /opt/build/git/*.deb \
    && git --version

COPY tasks/ddpo_sd15_aesthetic/environment/runtime-requirements.lock /opt/build/runtime-requirements.lock
COPY tasks/ddpo_sd15_aesthetic/environment/offline_assets_and_epoch_zero_save.patch /opt/build/upstream.patch
COPY tasks/ddpo_sd15_aesthetic/environment/check_image.py /opt/build/check_image.py

RUN --mount=type=bind,from=wheelhouse,target=/wheelhouse,readonly \
    --mount=type=bind,from=ddpo_source,target=/opt/build/src/ddpo,readonly \
    mkdir -p /workspace /opt/harness \
    && cp -a /opt/build/src/ddpo /workspace/ddpo-pytorch \
    && cp -a /opt/build/src/ddpo /opt/harness/ddpo-pytorch \
    # The patch does three things: point the aesthetic scorer's CLIP at the read-only
    # mount instead of the hub, move the randn_tensor import to where diffusers
    # 0.17.1 keeps it, and drop the `epoch != 0` half of upstream's save condition so
    # save_freq means what it says. --check first, so a context drift fails the build
    # instead of accepting a partially applicable patch.
    && for tree in /workspace/ddpo-pytorch /opt/harness/ddpo-pytorch; do \
         git -C "$tree" init -q . \
         && git -C "$tree" apply --check -p1 /opt/build/upstream.patch \
         && git -C "$tree" apply -p1 /opt/build/upstream.patch \
         && rm -rf "$tree/.git"; \
       done \
    # The lock, not the .in: --require-hashes makes a substituted wheel a build failure,
    # and --no-deps is safe because the lock is the whole closure (25 packages, resolved
    # inside this base image so it adds to the base's torch rather than replacing it).
    # Two of the 25 are wheels built on the host from sdists PyPI never published as
    # wheels -- runtime-requirements.lock's header says which and why.
    #
    # 25 and not 24 because one member, numpy==1.26.4, DOWNGRADES the base's numpy 2.3.2.
    # That is the one place this install does not merely add to the base, and it is
    # deliberate: wandb 0.15.12 touches np.float_ at import time, which numpy 2 removed,
    # and diffusers imports wandb transitively through accelerate.tracking. See
    # runtime-requirements.in. `pip check` below is what catches it if that downgrade ever
    # breaks the base's torch or torchvision.
    && python3 -m pip install --no-compile --no-cache-dir --no-index \
      --find-links=/wheelhouse --only-binary=:all: --no-deps --require-hashes \
      -r /opt/build/runtime-requirements.lock \
    && python3 -m pip check

COPY tasks/ddpo_sd15_aesthetic/solution/ /workspace/
COPY tasks/ddpo_sd15_aesthetic/harness/ /opt/harness/
COPY orchestrator/lifecycle.py /opt/harness/lifecycle.py
COPY orchestrator/container_harness/ /opt/harness/
COPY LICENSE THIRD_PARTY_NOTICES.md /licenses/

# Fixed so the baseline commit hash is a function of the tree alone. Without it two
# builds minutes apart produce different commit objects from an identical tree, and
# the image digest stops being reproducible.
ARG SOURCE_DATE_EPOCH=1767225600
ARG AI4AI_AGENT_UID=20003
ARG AI4AI_AGENT_GID=20003

RUN chmod 0755 /workspace/run.sh /workspace/launch_training.sh /opt/harness/*.sh \
    # Capture the pristine tree as one commit, then move .git out of the Agent's
    # reach. submit.sh copies it back to produce a single diff covering run.sh,
    # train.py, ddpo_config.py and the framework together.
    #
    # -f defeats the upstream tree's .gitignore. Without it an Agent file matching an
    # ignore rule would be silently absent from candidate.patch and would not exist
    # in the 12 h container.
    && find /workspace -mindepth 2 -name .git -prune -exec rm -rf {} + \
    && git -C /workspace init -q \
    && git -C /workspace config user.email harness@localhost \
    && git -C /workspace config user.name "ddpo harness" \
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
    # Production runs use the non-root Agent identity declared above. Ownership is
    # required in addition to write permission so executable-bit changes are
    # representable in candidate.patch.
    && chmod -R a+rwX /workspace /out \
    && chown -R "${AI4AI_AGENT_UID}:${AI4AI_AGENT_GID}" /workspace \
    && chmod -R a+rX,a-w /opt/harness \
    && chmod 0555 /opt/harness/save_checkpoint.py /opt/harness/validate_checkpoint.py \
    && python3 /opt/build/check_image.py \
    # Build residue. /opt should hold nothing but harness, so the layout a reader sees
    # is the layout the docs describe.
    && rm -rf /opt/build \
    # check_image exercises a mock evaluator and libraries may create cache roots.
    # None is a runtime input; keep the published image free of smoke receipts/cache.
    && rm -rf /tmp/* /tmp/.[!.]* /tmp/..?* /root/.cache /root/.wget-hsts \
    # Scope the tolerated no-match status to find so an earlier self-check failure
    # remains fatal.
    && { find /opt/harness /workspace -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true; }

LABEL org.opencontainers.image.title="ddpo_sd15_aesthetic" \
      org.ai4ai.task="ddpo_sd15_aesthetic" \
      org.ai4ai.upstream="kvablack/ddpo-pytorch@1958463f020112c9a7bc85768d296daacc2e1b4b" \
      org.ai4ai.base_model="stable-diffusion-v1-5/stable-diffusion-v1-5@451f4fe16113bff5a5d2269ed5ad43b0592e9a14" \
      org.ai4ai.reward_clip="openai/clip-vit-large-patch14@32bd64288804d66eefd0ccbe215aa642df71cc41"

WORKDIR /workspace
CMD ["bash"]
