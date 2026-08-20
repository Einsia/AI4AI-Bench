# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e
# Offline named contexts discovered by the operator image builder:
# --build-context git_deb=
# --build-context wheelhouse=
# --build-context open_unlearning_source=

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
    USER=agent \
    LOGNAME=agent \
    HOME=/tmp \
    GIT_CONFIG_COUNT=1 \
    GIT_CONFIG_KEY_0=safe.directory \
    GIT_CONFIG_VALUE_0=/workspace

RUN --mount=type=bind,from=git_deb,target=/opt/build/git,readonly \
    dpkg -i /opt/build/git/*.deb \
    && git --version

COPY tasks/openunlearning_tofu_npo_llama3p2_1b/environment/runtime-requirements.lock /opt/build/runtime-requirements.lock
COPY tasks/openunlearning_tofu_npo_llama3p2_1b/environment/apply_bf16_compat.py /opt/build/apply_bf16_compat.py
COPY tasks/openunlearning_tofu_npo_llama3p2_1b/environment/check_image.py /opt/build/check_image.py
COPY tasks/openunlearning_tofu_npo_llama3p2_1b/environment/verify_source_context.py /opt/build/verify_source_context.py

RUN --mount=type=bind,from=wheelhouse,target=/wheelhouse,readonly \
    python3 -m pip install --no-compile --no-cache-dir --no-index --find-links=/wheelhouse \
      --only-binary=:all: --no-deps --require-hashes --force-reinstall \
      -r /opt/build/runtime-requirements.lock \
    && python3 -m pip check

COPY tasks/openunlearning_tofu_npo_llama3p2_1b/solution/ /workspace/
COPY tasks/openunlearning_tofu_npo_llama3p2_1b/harness/ /opt/harness/
COPY orchestrator/lifecycle.py /opt/harness/lifecycle.py
COPY orchestrator/container_harness/ /opt/harness/
COPY LICENSE THIRD_PARTY_NOTICES.md /licenses/
# The checkout metadata and upstream data never enter an immutable COPY layer:
# they are removed in the same RUN that materializes the two code trees.
RUN --mount=type=bind,from=open_unlearning_source,target=/context-open-unlearning,readonly \
    python /opt/build/verify_source_context.py /context-open-unlearning \
    && cp -a /context-open-unlearning /workspace/open-unlearning \
    && cp -a /context-open-unlearning /opt/harness/open-unlearning \
    && printf '%s\n' '4ad738aaf60f6a4385f6e2506d01da99e76c31f3' \
         > /opt/harness/open-unlearning.revision \
    && find /workspace/open-unlearning /opt/harness/open-unlearning -name .git -prune -exec rm -rf {} + \
    && rm -rf /workspace/open-unlearning/data /opt/harness/open-unlearning/data \
    && python3 /opt/build/apply_bf16_compat.py /opt/harness/open-unlearning

ARG SOURCE_DATE_EPOCH=1767225600
ARG AI4AI_AGENT_UID=20003
ARG AI4AI_AGENT_GID=20003
RUN chmod 0755 /workspace/run.sh /opt/harness/*.sh \
    && mkdir -p /assets/models /assets/data/train /assets/data/final /out /logs /opt/harness/git-base \
    && find /workspace -mindepth 2 -name .git -prune -exec rm -rf {} + \
    && git -C /workspace init -q \
    && git -C /workspace config user.name ai4ai \
    && git -C /workspace config user.email ai4ai@invalid \
    && git -C /workspace add -A \
    && GIT_AUTHOR_DATE="@${SOURCE_DATE_EPOCH} +0000" GIT_COMMITTER_DATE="@${SOURCE_DATE_EPOCH} +0000" \
       git -C /workspace commit -q -m pristine \
    && mv /workspace/.git /opt/harness/git-base/.git \
    && python3 /opt/build/check_image.py \
    && chmod -R a+rX,a-w /opt/harness \
    && chmod 0555 /opt/harness/save_checkpoint.py /opt/harness/validate_checkpoint.py \
    && chmod -R a+rwX /workspace /out /logs \
    && chown -R "${AI4AI_AGENT_UID}:${AI4AI_AGENT_GID}" /workspace \
    && rm -rf /opt/build \
    && rm -rf /tmp/* /tmp/.[!.]* /tmp/..?* /root/.cache /root/.wget-hsts \
    && { find /workspace /opt/harness -type d -name __pycache__ -prune -exec rm -rf {} + || true; }

LABEL org.opencontainers.image.title="openunlearning_tofu_npo_llama3p2_1b" \
      org.ai4ai.task="openunlearning_tofu_npo_llama3p2_1b" \
      org.ai4ai.upstream="locuslab/open-unlearning@4ad738aaf60f6a4385f6e2506d01da99e76c31f3" \
      org.ai4ai.base_model="meta-llama/Llama-3.2-1B-Instruct"

WORKDIR /workspace
