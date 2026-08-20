# Use the daemon's built-in Dockerfile frontend. Pinning an external
# docker/dockerfile image makes an otherwise offline build contact Docker Hub
# before it can consume the four local named contexts below.
#
# Build (network disabled, four named contexts):
#
#   docker buildx build --network=none --no-cache --provenance=false \
#     --output type=docker,rewrite-timestamp=true \
#     --build-context verl_source=tasks/opd_math_1p5b/environment/verl \
#     --build-context justrl_source=<JustRL@cf444e99 tree> \
#     --build-context wheelhouse=<pinned wheel directory> \
#     --build-context agent_tools=<directory containing pinned rg binary> \
#     -f tasks/opd_math_1p5b/environment/Dockerfile .
#
# Layout, by owner and lifetime:
#
#   /workspace     everything the Agent owns. Writable, and the only thing that
#                  enters candidate.patch. Holds run.sh, train.py and verl.
#   /opt/harness   read-only: the three .sh entry points, the evaluators, the
#                  pinned JustRL grader, and the pristine .git baseline.
#   /assets        read-only mounts (models, data). Not in the image.
#   /out /logs     mount points. Not in the image.

ARG VERL_IMAGE="verlai/verl:vllm017.latest@sha256:4c43bbf17e90284b1102008399240b25406e8d34fea178d86272231b333b7cb6"
FROM ${VERL_IMAGE}

ARG SOURCE_DATE_EPOCH=1767225600
ARG AI4AI_AGENT_UID=20003
ARG AI4AI_AGENT_GID=20003

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH=/workspace/verl \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    WANDB_MODE=disabled \
    SWANLAB_MODE=disabled \
    DO_NOT_TRACK=1 \
    VLLM_NO_USAGE_STATS=1 \
    VLLM_HOST_IP=127.0.0.1 \
    RAY_USAGE_STATS_ENABLED=0 \
    GLOO_SOCKET_IFNAME=lo \
    NO_PROXY=127.0.0.1,localhost \
    no_proxy=127.0.0.1,localhost \
    # The container runs as the host's uid, which has no /etc/passwd entry here.
    # getpass.getuser() falls back to pwd.getpwuid() and raises KeyError, and
    # torch calls it while setting up its inductor cache -- so without USER set,
    # `import verl` fails outright. getuser() checks these env vars first.
    USER=agent \
    LOGNAME=agent \
    HOME=/tmp \
    # /workspace/.git is created by root at build time and read by the host uid
    # at run time, which trips git's dubious-ownership check. This is the only
    # way to pre-authorize it without a writable global gitconfig.
    GIT_CONFIG_COUNT=1 \
    GIT_CONFIG_KEY_0=safe.directory \
    GIT_CONFIG_VALUE_0=/workspace

RUN --mount=type=bind,target=/context,readonly \
    --mount=type=bind,from=wheelhouse,target=/wheelhouse,readonly \
    --mount=type=bind,from=verl_source,target=/context-verl,readonly \
    --mount=type=bind,from=justrl_source,target=/context-justrl,readonly \
    --mount=type=bind,from=agent_tools,target=/agent-tools,readonly \
    # COPY creates a separate layer whose parent-directory timestamps vary across
    # no-cache builds. Copy every input inside this one normalized layer instead.
    mkdir -p /opt/build /workspace \
    && cp /context/tasks/opd_math_1p5b/environment/runtime-requirements.lock \
          /opt/build/runtime-requirements.lock \
    && cp /context/tasks/opd_math_1p5b/environment/single_gpu_wall_clock.patch \
          /opt/build/single_gpu_wall_clock.patch \
    && cp /agent-tools/rg /usr/local/bin/rg \
    # The base image ships its own verl at /opt/verl and OPD at /opt/opd. Both
    # go: the pinned tree below is the one that runs, and OPD's bundled verl
    # has no trainer/distillation at all, so it was never imported.
    && rm -rf /opt/verl /opt/opd /opt/llm-algobench /workspace/verl /opt/harness \
    && mkdir -p /workspace \
    && cp -a /context-verl /workspace/verl \
    # Colocate the teacher with the actor on one device, add the wall-clock
    # deadline, and teach the teacher manager to sleep/wake its KV cache.
    # Single-GPU distillation does not work without this.
    # The patch names paths as verl/trainer/... , so -p1 from the repo root
    # resolves to /workspace/verl/verl/trainer/... . --check first so a context
    # drift fails the build instead of half-applying.
    && git -C /workspace/verl apply --check -p1 /opt/build/single_gpu_wall_clock.patch \
    && git -C /workspace/verl apply -p1 /opt/build/single_gpu_wall_clock.patch \
    # The JustRL grader is code, not data: baking it in is what makes "the
    # evaluator does not change" true. fast_eval and the hidden final call the
    # same grade_answer_verl. AIME questions are deliberately NOT copied -- they
    # are mounted only into the final container.
    && mkdir -p /opt/harness/justrl/evals \
    && cp /context-justrl/evals/grade.py /context-justrl/evals/utils.py \
          /opt/harness/justrl/evals/ \
    && python -m pip install --no-compile --no-cache-dir --no-index --find-links=/wheelhouse \
      --only-binary=:all: --no-deps --require-hashes \
      -r /opt/build/runtime-requirements.lock \
    && cp -a /context/tasks/opd_math_1p5b/solution/. /workspace/ \
    && mkdir -p /opt/harness \
    && cp -a /context/tasks/opd_math_1p5b/harness/. /opt/harness/ \
    && cp /context/orchestrator/lifecycle.py /opt/harness/lifecycle.py \
    && cp -a /context/orchestrator/container_harness/. /opt/harness/ \
    && mkdir -p /licenses \
    && cp /context/LICENSE /context/THIRD_PARTY_NOTICES.md /licenses/ \
    && chmod 0755 /workspace/run.sh /opt/harness/*.sh /usr/local/bin/rg \
    # Capture the pristine tree as one commit, then move .git out of the
    # Agent's reach. submit.sh copies it back to produce a single diff that
    # covers run.sh, train.py and verl together.
    #
    # -f defeats verl's .gitignore. Without it an Agent file matching an
    # ignore rule would be silently absent from candidate.patch and would not
    # exist in the 12 h container.
    && find /workspace -mindepth 2 -name .git -prune -exec rm -rf {} + \
    && git -C /workspace init -q \
    && git -C /workspace config user.email harness@localhost \
    && git -C /workspace config user.name "opd harness" \
    && git -C /workspace config commit.gpgsign false \
    && git -C /workspace add -A -f \
    && GIT_AUTHOR_DATE="@${SOURCE_DATE_EPOCH} +0000" \
       GIT_COMMITTER_DATE="@${SOURCE_DATE_EPOCH} +0000" \
       git -C /workspace -c core.hooksPath=/dev/null commit -q -m "pristine baseline" \
    && mkdir -p /opt/harness/git-base \
    # The index caches per-file stat data -- mtime, ctime, inode, device -- so it
    # is different on every build even from an identical tree, and it was the
    # last thing keeping the image digest from being reproducible. submit.sh
    # rebuilds it with `git add -A -f`, and the diff is against HEAD either way.
    && rm -f /workspace/.git/index \
    && mv /workspace/.git /opt/harness/git-base/.git \
    && mkdir -p /assets /out /logs \
    # Ownership lets the non-root Agent record executable-bit changes as well as
    # content changes in candidate.patch.
    && chmod -R a+rwX /workspace /out \
    && chown -R "${AI4AI_AGENT_UID}:${AI4AI_AGENT_GID}" /workspace \
    && chmod -R a+rX,a-w /opt/harness \
    && chmod 0555 /opt/harness/save_checkpoint.py /opt/harness/validate_checkpoint.py \
                  /opt/harness \
    # Build residue. /opt should hold nothing but harness, so the layout a reader
    # sees is the layout the docs describe.
    && rm -rf /opt/build \
    && python /context/tasks/opd_math_1p5b/environment/check_image.py \
    # Remove smoke/cache state and VCS metadata inherited from the public base.
    # DeepEP itself is retained; only its unused repository metadata is removed.
    && rm -rf /home/dpsk_a2a/DeepEP/.git /root/.cache /root/.launchpadlib \
         /root/.wget-hsts /tmp/* /tmp/.[!.]* /tmp/..?* \
    # The braces scope the tolerated cleanup failure. A bare trailing `|| true`
    # would also mask an earlier self-check failure and attest a broken image.
    && { find /opt/harness /workspace -name '__pycache__' -prune \
           -exec rm -rf {} + 2>/dev/null || true; } \
    # Normalize every path created by this layer, including parent directories
    # whose mtimes change when children are copied or removed.
    && find /workspace /opt/harness /usr/local/bin/rg \
         -exec touch -h -d "@${SOURCE_DATE_EPOCH}" {} + \
    && touch -h -d "@${SOURCE_DATE_EPOCH}" \
         / /tmp /opt /usr/local/bin /assets /out /logs

LABEL org.opencontainers.image.title="opd_math_1p5b" \
      org.ai4ai.task="opd_math_1p5b" \
      org.ai4ai.verl="volcengine/verl@7aed6b230776f963fa09509c10d9c3a767d1102c" \
      org.ai4ai.justrl="JustRL@cf444e9920b599c865ee6742c88dc162dcabcec2" \
      org.ai4ai.algorithm_source="thunlp/OPD@4532fd35ccfdde82adc918b265e4c964534e83d1 verl_example/opd.sh"

WORKDIR /workspace
CMD ["bash"]
