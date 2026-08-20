# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e
#
# Build (network disabled, one named context):
#
#   docker buildx build --network=none \
#     --build-context git_deb=<git .deb directory> \
#     --build-context wheelhouse=<pinned wheel directory> \
#     -f tasks/ultrafeedback_bt_rm_rewardbench/environment/Dockerfile .
#
# Layout, by owner and lifetime:
#
#   /workspace     everything the Agent owns. Writable, and the only thing that
#                  enters candidate.patch. Holds run.sh and train.py.
#   /opt/harness   read-only: the three .sh entry points, the evaluators, the
#                  shared grader, and the pristine .git baseline.
#   /assets        read-only mounts (base model, pairs, RewardBench). Not in the image.
#   /out /logs     mount points. Not in the image.
#
# One task-local image contains the editable solution and read-only evaluator, so
# training and scoring use one pinned dependency set.

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
    # The reference protocol left WANDB_MODE=offline, which still writes run
    # directories and moving debug-log symlinks. train.py passes report_to=[], so
    # nothing should reach wandb at all; disabled is the setting that makes that
    # true if a candidate re-enables the integration without thinking about it.
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

# git, without which submit.sh cannot produce a candidate patch at all -- from a build
# context, because this image builds with --network=none and apt would need the network.
#
# The PyTorch base carries no git. The runtime tool gate verifies this pinned install
# before any phase starts.
RUN --mount=type=bind,from=git_deb,target=/opt/build/git,readonly \
    dpkg -i /opt/build/git/*.deb \
    && git --version

COPY tasks/ultrafeedback_bt_rm_rewardbench/environment/runtime-requirements.lock /opt/build/runtime-requirements.lock
COPY tasks/ultrafeedback_bt_rm_rewardbench/environment/check_image.py /opt/build/check_image.py

RUN --mount=type=bind,from=wheelhouse,target=/wheelhouse,readonly \
    python -m pip install --no-compile --no-cache-dir --no-index --find-links=/wheelhouse \
      --only-binary=:all: --no-deps --require-hashes --force-reinstall \
      -r /opt/build/runtime-requirements.lock \
    && python -m pip check

COPY tasks/ultrafeedback_bt_rm_rewardbench/solution/ /workspace/
COPY tasks/ultrafeedback_bt_rm_rewardbench/harness/ /opt/harness/
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
    # reach. submit.sh copies it back to produce a single diff covering run.sh and
    # train.py together.
    && git -C /workspace init -q \
    && git -C /workspace config user.email harness@localhost \
    && git -C /workspace config user.name "btrm harness" \
    && git -C /workspace config commit.gpgsign false \
    && git -C /workspace add -A -f \
    && GIT_AUTHOR_DATE="@${SOURCE_DATE_EPOCH} +0000" \
       GIT_COMMITTER_DATE="@${SOURCE_DATE_EPOCH} +0000" \
       git -C /workspace -c core.hooksPath=/dev/null commit -q -m "pristine baseline" \
    && mkdir -p /opt/harness/git-base \
    # The index caches per-file stat data -- mtime, ctime, inode, device -- so it is
    # different on every build even from an identical tree. submit.sh rebuilds it
    # with `git add -A -f`, and the diff is against HEAD either way.
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
    # Build residue. /opt should hold nothing but harness, so the layout a reader
    # sees is the layout the docs describe.
    && rm -rf /opt/build \
    && rm -rf /tmp/* /tmp/.[!.]* /tmp/..?* /root/.cache /root/.wget-hsts \
    # Scope the tolerated no-match status to find so an earlier self-check failure
    # remains fatal.
    && { find /opt/harness /workspace -name '__pycache__' -prune \
           -exec rm -rf {} + 2>/dev/null || true; }

LABEL org.opencontainers.image.title="ultrafeedback_bt_rm_rewardbench" \
      org.ai4ai.task="ultrafeedback_bt_rm_rewardbench" \
      org.ai4ai.base_model="mistralai/Mistral-7B-Instruct-v0.2@63a8b081895390a26e140280378bc85ec8bce07a" \
      org.ai4ai.train_data="HuggingFaceH4/ultrafeedback_binarized@3949bf5f8c17c394422ccfab0c31ea9c20bdeb85" \
      org.ai4ai.final_data="allenai/reward-bench@168d848cdbbea9764fae4a544dc9ca1e6cca4931" \
      org.ai4ai.algorithm_source="RLHFlow/RLHF-Reward-Modeling@fc39179f5a9a4dd75047a4c0311672905b9d9a04"

WORKDIR /workspace
CMD ["bash"]
