# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e
#
# Build (network disabled, four named contexts):
#
#   docker buildx build --network=none \
#     --build-context lcb_source=<LiveCodeBench@28fef95e checkout> \
#     --build-context wheelhouse=<pinned wheel directory> \
#     --build-context git_deb=<git .deb directory> \
#     --build-context tini_deb=<tini .deb directory> \
#     -f tasks/openr1_code_livecodebench/environment/Dockerfile .
#
# Layout, by owner and lifetime:
#
#   /workspace     everything the Agent owns. Writable, and the only thing that
#                  enters candidate.patch. Holds run.sh and train.py.
#   /opt/harness   read-only: the three .sh entry points, the evaluators, the
#                  pinned LiveCodeBench tree, and the pristine .git baseline.
#   /assets        read-only mounts (model, corpus, LiveCodeBench rows). Not in
#                  the image.
#   /out /logs     mount points. Not in the image.
#
# The evaluator and pinned LiveCodeBench source are baked into read-only
# /opt/harness, which makes evaluator immutability a property of the image layout.

# The CUDA 12.8 base is digest-pinned and passes an actual kernel launch on the target
# accelerator. Training and generation package pins remain in runtime-requirements.in.
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
    # The container runs as the host's uid, which has no /etc/passwd entry here.
    # getpass.getuser() falls back to pwd.getpwuid() and raises KeyError, and
    # torch calls it while setting up its inductor cache.
    USER=agent \
    LOGNAME=agent \
    HOME=/tmp \
    # /workspace/.git is created by root at build time and read by the host uid at
    # run time, which trips git's dubious-ownership check. This is the only way to
    # pre-authorize it without a writable global gitconfig.
    GIT_CONFIG_COUNT=1 \
    GIT_CONFIG_KEY_0=safe.directory \
    GIT_CONFIG_VALUE_0=/workspace

# git for submit.sh's diff; tini as PID 1 so a signal reaches the trainer and
# zombie code-execution workers get reaped during the final.
#
# Install both from pinned .deb contexts because the task layer builds with
# --network=none.
#
# tini gets its own context rather than being added to the shared git_deb: the other five
# tasks mount git_deb, and changing its contents would move their layer fingerprints.
RUN --mount=type=bind,from=git_deb,target=/opt/build/git,readonly \
    --mount=type=bind,from=tini_deb,target=/opt/build/tini,readonly \
    dpkg -i /opt/build/git/*.deb /opt/build/tini/*.deb \
    && git --version \
    && /usr/bin/tini --version

COPY tasks/openr1_code_livecodebench/environment/runtime-requirements.lock /opt/build/runtime-requirements.lock
COPY tasks/openr1_code_livecodebench/environment/check_image.py /opt/build/check_image.py

RUN --mount=type=bind,from=wheelhouse,target=/wheelhouse,readonly \
    --mount=type=bind,from=lcb_source,target=/opt/build/src/livecodebench,readonly \
    python -m pip install --no-compile --no-cache-dir --no-index --find-links=/wheelhouse \
      --only-binary=:all: --no-deps --require-hashes --force-reinstall \
      -r /opt/build/runtime-requirements.lock \
    && python -m pip check \
    # The LiveCodeBench evaluator is code, not data: baking it in is what makes
    # "the evaluator does not change" true, and it is what lets the proxy and the
    # final share one path. The release .jsonl files are deliberately NOT copied
    # -- those are mounts, and the v6 ones reach only the scoring container.
    && mkdir -p /opt/harness/livecodebench /licenses/third-party \
    && cp -a /opt/build/src/livecodebench/lcb_runner /opt/harness/livecodebench/lcb_runner \
    && cp /opt/build/src/livecodebench/LICENSE \
      /licenses/third-party/LiveCodeBench-MIT-LICENSE \
    && test -s /licenses/third-party/LiveCodeBench-MIT-LICENSE \
    # The prompt module reads its few-shot examples relative to the working
    # directory, so whatever sits beside lcb_runner has to come too.
    && if [ -d /opt/build/src/livecodebench/assets ]; then \
         cp -a /opt/build/src/livecodebench/assets /opt/harness/livecodebench/assets; \
       fi

COPY tasks/openr1_code_livecodebench/solution/ /workspace/
COPY tasks/openr1_code_livecodebench/harness/ /opt/harness/
COPY orchestrator/lifecycle.py /opt/harness/lifecycle.py
COPY orchestrator/container_harness/ /opt/harness/
COPY LICENSE THIRD_PARTY_NOTICES.md /licenses/

# Fixed so the baseline commit hash is a function of the tree alone. Without it,
# two builds minutes apart produce different commit objects from an identical tree
# and the image digest stops being reproducible.
ARG SOURCE_DATE_EPOCH=1767225600
ARG AI4AI_AGENT_UID=20003
ARG AI4AI_AGENT_GID=20003

RUN chmod 0755 /workspace/run.sh /opt/harness/*.sh \
    # Capture the pristine tree as one commit, then move .git out of the Agent's
    # reach. submit.sh copies it back to produce a single diff covering run.sh,
    # train.py and anything new.
    && find /workspace -mindepth 2 -name .git -prune -exec rm -rf {} + \
    && git -C /workspace init -q \
    && git -C /workspace config user.email harness@localhost \
    && git -C /workspace config user.name "openr1 harness" \
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
    && python /opt/build/check_image.py \
    && rm -rf /opt/build \
    && rm -rf /tmp/* /tmp/.[!.]* /tmp/..?* /root/.cache /root/.wget-hsts \
    # Scope the tolerated no-match status to find so an earlier self-check failure
    # remains fatal.
    && { find /opt/harness /workspace -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true; }

LABEL org.opencontainers.image.title="openr1_code_livecodebench" \
      org.ai4ai.task="openr1_code_livecodebench" \
      org.ai4ai.recipe_source="huggingface/open-r1@1416fa0cf21595d2083b399a2a0bbddd7f6e9563" \
      org.ai4ai.evaluator="LiveCodeBench/LiveCodeBench@28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"

WORKDIR /workspace
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
