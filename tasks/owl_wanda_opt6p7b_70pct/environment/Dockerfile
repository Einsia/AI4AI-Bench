# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e
#
# Build (network disabled, two named contexts):
#
#   docker buildx build --network=none \
#     --build-context owl_source=<luuyin/OWL@dddb7a4 tree, .git removed> \
#     --build-context git_deb=<git .deb directory> \
#     --build-context wheelhouse=<pinned wheel directory> \
#     -f tasks/owl_wanda_opt6p7b_70pct/environment/Dockerfile .
#
# Layout, by owner and lifetime:
#
#   /workspace     everything the Agent owns. Writable, and the only thing that
#                  enters candidate.patch. Holds run.sh, prune.py, owl_opt.py and
#                  the OWL tree.
#   /opt/harness   read-only: the three .sh entry points and the two evaluators.
#   /assets        read-only mounts (dense model, C4, WikiText2). Not in the image.
#   /out /logs     mount points. Not in the image.
#
# The base is the reference protocol's, unchanged: OWL dddb7a4 uses the 2023 Hugging Face API
# and needs transformers 4.28.1, which the pytorch 2.8.0 image tolerates.

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
    WANDB_MODE=offline \
    # The container runs as the host's uid, which has no /etc/passwd entry here.
    # getpass.getuser() falls back to pwd.getpwuid() and raises KeyError, and torch
    # calls it while setting up caches. getuser() checks these env vars first.
    USER=agent \
    LOGNAME=agent \
    HOME=/tmp \
    # /workspace/.git is created by root at build time and read by the host uid at
    # run time, which trips git's dubious-ownership check. This is the only way to
    # pre-authorize it without a writable global gitconfig.
    GIT_CONFIG_COUNT=1 \
    GIT_CONFIG_KEY_0=safe.directory \
    GIT_CONFIG_VALUE_0=/workspace

# submit.sh requires git. Install it from a pinned context so the task layer remains
# buildable with --network=none.
RUN --mount=type=bind,from=git_deb,target=/opt/build/git,readonly \
    dpkg -i /opt/build/git/*.deb \
    && git --version

COPY tasks/owl_wanda_opt6p7b_70pct/environment/runtime-requirements.lock /opt/build/runtime-requirements.lock
COPY tasks/owl_wanda_opt6p7b_70pct/environment/check_image.py /opt/build/check_image.py
COPY tasks/owl_wanda_opt6p7b_70pct/environment/patch_validator_device_map.py /opt/build/patch_validator_device_map.py

RUN --mount=type=bind,from=wheelhouse,target=/wheelhouse,readonly \
    --mount=type=bind,from=owl_source,target=/opt/build/src/owl,readonly \
    mkdir -p /workspace \
    # The OWL tree goes under /workspace, not into a read-only /opt. That is the
    # constraint change: on the reference protocol it was a read-only runtime mount and the
    # pruning implementation was out of reach. Here lib/prune_all.py is editable and
    # submit.sh's single diff covers it.
    && cp -a /opt/build/src/owl /workspace/owl \
    && python -m pip install --no-compile --no-cache-dir --no-index --find-links=/wheelhouse \
      --only-binary=:all: --require-hashes \
      -r /opt/build/runtime-requirements.lock \
    # accelerate 0.18.0 pins an older torch than the base image ships, so its
    # dependency graph is deliberately not resolved. Same as the reference protocol.
    && python -m pip install --no-compile --no-cache-dir --no-index --find-links=/wheelhouse \
      --no-deps accelerate==0.18.0

COPY tasks/owl_wanda_opt6p7b_70pct/solution/ /workspace/
COPY tasks/owl_wanda_opt6p7b_70pct/harness/ /opt/harness/
COPY orchestrator/lifecycle.py /opt/harness/lifecycle.py
COPY orchestrator/container_harness/ /opt/harness/
COPY LICENSE THIRD_PARTY_NOTICES.md /licenses/

# Fixed so the baseline commit hash is a function of the tree alone. Without it two
# builds minutes apart produce different commit objects from an identical tree, and
# the image digest stops being reproducible.
ARG SOURCE_DATE_EPOCH=1767225600
ARG AI4AI_AGENT_UID=20003
ARG AI4AI_AGENT_GID=20003

RUN python /opt/build/patch_validator_device_map.py /opt/harness/validate_checkpoint.py \
    && chmod 0755 /workspace/run.sh /opt/harness/*.sh \
    # Capture the pristine tree as one commit, then move .git out of the Agent's
    # reach. submit.sh copies it back to produce a single diff that covers run.sh,
    # prune.py, owl_opt.py and the OWL tree together.
    #
    # -f defeats the OWL tree's .gitignore. Without it an Agent file matching an
    # ignore rule would be silently absent from candidate.patch and would not exist
    # in the pruning container.
    #
    # Remove nested repositories before creating the submission baseline. Otherwise git
    # stages a gitlink and omits edits made inside the vendored OWL tree.
    && find /workspace -mindepth 2 -name .git -prune -exec rm -rf {} + \
    && git -C /workspace init -q \
    && git -C /workspace config user.email harness@localhost \
    && git -C /workspace config user.name "owl harness" \
    && git -C /workspace config commit.gpgsign false \
    && git -C /workspace add -A -f \
    && GIT_AUTHOR_DATE="@${SOURCE_DATE_EPOCH} +0000" \
       GIT_COMMITTER_DATE="@${SOURCE_DATE_EPOCH} +0000" \
       git -C /workspace -c core.hooksPath=/dev/null commit -q -m "pristine baseline" \
    && mkdir -p /opt/harness/git-base \
    # The index caches per-file stat data -- mtime, ctime, inode, device -- so it is
    # different on every build even from an identical tree, and on OPD it was the last
    # thing keeping the image digest from being reproducible. submit.sh rebuilds it
    # with `git add -A -f`, and the diff is against HEAD either way.
    && rm -f /workspace/.git/index \
    && mv /workspace/.git /opt/harness/git-base/.git \
    && mkdir -p /assets /out /logs \
    # The container runs with --user <host uid>:<host gid>, which is not known at
    # Ownership lets the non-root Agent record executable-bit changes as well as
    # content changes in candidate.patch.
    && chmod -R a+rwX /workspace /out \
    && chown -R "${AI4AI_AGENT_UID}:${AI4AI_AGENT_GID}" /workspace \
    && chmod -R a+rX,a-w /opt/harness \
    && chmod 0555 /opt/harness/save_checkpoint.py /opt/harness/validate_checkpoint.py \
    && python /opt/build/check_image.py \
    # Build residue. /opt should hold nothing but harness, so the layout a reader sees
    # is the layout the docs describe.
    && rm -rf /opt/build \
    && rm -rf /out/* /out/.[!.]* /out/..?* \
         /tmp/* /tmp/.[!.]* /tmp/..?* /root/.cache /root/.wget-hsts \
    # Scope the tolerated no-match status to find so an earlier self-check failure
    # remains fatal.
    && { find /opt/harness /workspace -name '__pycache__' -prune \
           -exec rm -rf {} + 2>/dev/null || true; }

LABEL org.opencontainers.image.title="owl_wanda_opt6p7b_70pct" \
      org.ai4ai.task="owl_wanda_opt6p7b_70pct" \
      org.ai4ai.owl="luuyin/OWL@dddb7a4bffe27c73e4c8cf692b3a5e36401532c8" \
      org.ai4ai.base_model="facebook/opt-6.7b@a45aa65bbeb77c1558bc99bedc6779195462dab0"

WORKDIR /workspace
CMD ["bash"]
