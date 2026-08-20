# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e
#
# Build (network disabled, one named context):
#
#   docker buildx build --network=none \
#     --build-context digress_source=<DiGress@780242b8 tree> \
#     -f tasks/digress_qm9_graph_diffusion/environment/Dockerfile .
#
# Layout, by owner and lifetime:
#
#   /workspace       everything the Agent owns. Writable, and the only thing that
#                    enters candidate.patch. Holds run.sh, train.py and digress/.
#   /opt/harness     read-only: the three .sh entry points, the evaluators, AND a
#                    second copy of the pinned DiGress tree.
#   /assets          read-only mounts (the QM9 export). Not in the image.
#   /out /logs       mount points. Not in the image.
#
# THE TREE IS IN THE IMAGE TWICE, ON PURPOSE. On this task the sampler and the
# metrics are the grader: validity, uniqueness, novelty and NLL all come out of
# DiGress's own code. If the evaluator imported /workspace/digress, a candidate
# would own the code computing its own score. /opt/harness/digress is what the
# evaluators use and the Agent cannot write it; the score phase additionally runs
# from the original image layer with the patch unapplied.
#
# THE CONDA ENVIRONMENT IS NOT BUILT HERE. It is Dockerfile.base, built separately
# WITH network and consumed below by digest.
#
# Network-dependent package resolution belongs to Dockerfile.base. This task layer
# consumes the verified base as fixed bytes and builds offline.
#
# The base still carries the two consequences of being conda-based, and both are
# checked there rather than left to run time: `git` is a RUNTIME dependency because
# submit.sh shells out to it, and PATH is set explicitly because a `bash -lc` from
# `docker exec` gets no micromamba hook -- `python=3.9` leaves `python` and
# `python3.9` but no `python3`, which every harness entry point invokes by name.

# The caller must provide the local base by immutable config digest and verify it
# against environment.base_image_config_digest in task.toml before this offline build.
ARG DIGRESS_BASE_IMAGE
FROM ${DIGRESS_BASE_IMAGE}

COPY tasks/digress_qm9_graph_diffusion/environment/check_image.py /opt/build/check_image.py
COPY tasks/digress_qm9_graph_diffusion/environment/lightning_gloo_fallback.py \
     /opt/build/lightning_gloo_fallback.py
COPY tasks/digress_qm9_graph_diffusion/environment/patch_preprocessed_only.py \
     /opt/build/patch_preprocessed_only.py
COPY tasks/digress_qm9_graph_diffusion/environment/patch_bounded_training.py \
     /opt/build/patch_bounded_training.py

# Lightning defaults to NCCL for CUDA, but the pinned single-device runtime cannot
# execute its NCCL collective on the target capability. The task uses Gloo only for
# that process-group control path; model computation remains on CUDA.
#
# It matters at devices=1 because upstream hardcodes
# strategy="ddp_find_unused_parameters_true", so a process group is created regardless, on
# BOTH paths -- retrain through solution/run.sh and score through
# harness/final_eval.py:run_upstream, which reruns the same src/main.py under test_only.
#
# Keep the fallback in the task layer so source_sha256 attests it. The task layer is
# offline and therefore cannot move the RDKit or Lightning dependency closure.
#
# Its own step, before the chain below, so a failed patch is its own build error with its
# own message rather than one line inside a twelve-command &&-chain. The script asserts the
# upstream text appears exactly once and unit-tests both branches of the result without a
# GPU; check_image.py then re-asserts the patch is present in the finished image.
RUN python3 /opt/build/lightning_gloo_fallback.py

RUN --mount=type=bind,from=digress_source,target=/opt/build/src/digress,readonly \
    mkdir -p /workspace /opt/harness \
    # The editable copy: the Agent's, and the only one that enters candidate.patch.
    && cp -a /opt/build/src/digress /workspace/digress \
    # The grader copy. Same bytes, read-only, imported by the evaluators. This is
    # what makes "the evaluator does not change" checkable by reading one file.
    && cp -a /opt/build/src/digress /opt/harness/digress \
    && rm -rf /workspace/digress/.git /opt/harness/digress/.git \
    && python3 /opt/build/patch_preprocessed_only.py \
       /workspace/digress /opt/harness/digress \
    && python3 /opt/build/patch_bounded_training.py \
       /workspace/digress /opt/harness/digress

COPY tasks/digress_qm9_graph_diffusion/solution/ /workspace/
COPY tasks/digress_qm9_graph_diffusion/harness/ /opt/harness/
COPY orchestrator/lifecycle.py /opt/harness/lifecycle.py
COPY orchestrator/container_harness/ /opt/harness/
COPY LICENSE THIRD_PARTY_NOTICES.md /licenses/

# Fixed so the baseline commit hash is a function of the tree alone. Without it two
# builds minutes apart produce different commit objects from an identical tree and
# the image digest stops being reproducible.
ARG SOURCE_DATE_EPOCH=1767225600
ARG AI4AI_AGENT_UID=20003
ARG AI4AI_AGENT_GID=20003

RUN chmod 0755 /workspace/run.sh /opt/harness/*.sh \
    # Capture the pristine tree as one commit, then move .git out of the Agent's
    # reach. submit.sh copies it back to produce a single diff covering run.sh,
    # train.py and digress/ together.
    #
    # -f defeats any .gitignore in the pinned tree. Without it an Agent file
    # matching an ignore rule would be silently absent from candidate.patch and
    # would not exist in the 12 h container.
    && find /workspace -mindepth 2 -name .git -prune -exec rm -rf {} + \
    # patch_preprocessed_only.py compiles the patched upstream module as a syntax
    # check. Do not commit that build residue into the pristine candidate base.
    && { find /workspace -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true; } \
    && git -C /workspace init -q \
    && git -C /workspace config user.email harness@localhost \
    && git -C /workspace config user.name "digress harness" \
    && git -C /workspace config commit.gpgsign false \
    && git -C /workspace add -A -f \
    && GIT_AUTHOR_DATE="@${SOURCE_DATE_EPOCH} +0000" \
       GIT_COMMITTER_DATE="@${SOURCE_DATE_EPOCH} +0000" \
       git -C /workspace -c core.hooksPath=/dev/null commit -q -m "pristine baseline" \
    && mkdir -p /opt/harness/git-base \
    # The index caches per-file stat data -- mtime, ctime, inode, device -- so it is
    # different across otherwise identical builds. submit.sh rebuilds it with
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
    && python3 /opt/build/check_image.py \
    # Build residue. /opt should hold nothing but harness, so the layout a reader
    # sees is the layout the docs describe.
    && rm -rf /opt/build \
    && rm -rf /tmp/* /tmp/.[!.]* /tmp/..?* /root/.cache /root/.wget-hsts \
    # Scope the tolerated no-match status to find so an earlier self-check failure
    # remains fatal.
    && { find /opt/harness /workspace -name '__pycache__' -prune \
           -exec rm -rf {} + 2>/dev/null || true; }

LABEL org.opencontainers.image.title="digress_qm9_graph_diffusion" \
      org.ai4ai.task="digress_qm9_graph_diffusion" \
      org.ai4ai.digress="cvignac/DiGress@780242b8d3e7d78316bb5cf90c639fb0cd4c6079" \
      org.ai4ai.dataset="QM9 hydrogens-removed, digress-qm9-noh-v1"

WORKDIR /workspace
CMD ["bash"]
