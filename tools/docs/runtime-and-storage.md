# Runtime, storage, and mounts

All host storage locations are operator-configurable. The repository contains no required
site-specific storage, fixed user-home, or lab-host path. Executable discovery may use the
current account's `$HOME`, but no account name is embedded. Container paths are a small
protocol ABI: they are intentionally fixed so a patch cannot change where trusted inputs
arrive.

## Host prerequisites

- Linux `amd64`, Python 3.10+, and a modern Docker Engine/BuildKit supporting named build
  contexts, BuildKit secrets, internal bridge networks, bind mounts, and GPU requests.
- A local Docker daemon in the same mount and bridge-network environment as the
  orchestrator. `DOCKER_HOST` must not point at an unrelated remote machine: the daemon must
  see the same host paths and its internal-bridge gateway must reach the host-side egress
  proxy. Remote and rootless daemons generally do not provide this contract and are not
  supported for official runs unless the operator validates every mount, bridge, and GPU
  behavior.
- An NVIDIA driver and NVIDIA Container Toolkit compatible with the selected task image.
  `docker run --gpus` must work before starting a trial.
- Free host storage for the task's declared `storage_mb` (up to 512 GiB), plus runtime
  assets, Docker layers, build contexts, and receipts. The runner checks phase output space,
  but it cannot reserve the daemon's image store.
- A native Codex or Claude executable. Discovery checks the explicit
  `CODEX_BINARY`/`CLAUDE_BINARY` override first, then `PATH`, then
  `$HOME/.local/bin`. Check the selected CLI before a long run:

```bash
codex --version
claude --version
```

## Host configuration

CLI values take precedence over environment variables.

| Purpose | CLI | Environment | Default |
|---|---|---|---|
| One task's assets | `--assets` | `AI4AI_ASSETS_ROOT` | required; none |
| Asset download staging | `--staging-root` on `prepare_assets.py` | `AI4AI_ASSET_STAGING_ROOT` | hidden directory beside task asset root |
| Image build contexts | `--context-root` on release tools | `AI4AI_BUILD_CONTEXT_ROOT` | required; none |
| Image build receipts | `--receipt-root` on `build_images.py` | `AI4AI_IMAGE_RECEIPT_ROOT` | required; none |
| Agent run/output parent | `--root` | `AI4AI_RUN_ROOT` | required; none |
| Checkpoint-only evaluation parent | `--root` | `AI4AI_EVALUATION_ROOT` | required; none |
| Task directory | `--task` | `AI4AI_TASK` | `tasks/opd_math_1p5b` |
| GPU index | `--gpu` | `AI4AI_GPU` | `0` |
| Image override | `--image` | `AI4AI_IMAGE` | task declaration |
| Source identity policy | `--source-check` | `AI4AI_SOURCE_CHECK` | `warn` |
| Image identity policy | `--image-check` | `AI4AI_IMAGE_CHECK` | `warn` |
| Hardware contract policy | `--hardware-check` | `AI4AI_HARDWARE_CHECK` | `warn` |
| Missing-image policy | `--image-pull-policy` | `AI4AI_IMAGE_PULL_POLICY` | `missing` (pull once) |
| Docker command | — | `AI4AI_DOCKER` | `docker` |
| Docker daemon | — | `DOCKER_HOST` | Docker client default |
| API semaphore directory | `--agent-api-concurrency-root` | `AI4AI_AGENT_API_CONCURRENCY_ROOT` | `/tmp/ai4ai-agent-api` |
| Recovery GPU-claim directory | — | `AI4AI_GPU_CLAIM_ROOT` | `${TMPDIR:-/tmp}/ai4ai-gpu-claims` |
| Codex/Claude executable | — | `CODEX_BINARY` / `CLAUDE_BINARY` | `PATH`, then `$HOME/.local/bin/...` |
| OpenAI endpoint | — | `OPENAI_BASE_URL` | `https://api.openai.com/v1` |
| Anthropic endpoint | — | `ANTHROPIC_BASE_URL` | `https://api.anthropic.com` |
| Upstream CONNECT proxy | — | `AI4AI_UPSTREAM_PROXY`, `AI4AI_EGRESS_UPSTREAM`, or `HTTPS_PROXY` | direct; unauthenticated HTTP CONNECT only |

Example with all large files on a separate volume:

```bash
export AI4AI_ASSETS_ROOT=/data/ai4ai/assets/ragen_sokoban_grpo
export AI4AI_ASSET_STAGING_ROOT=/data/ai4ai/asset-staging
export AI4AI_RUN_ROOT=/data/ai4ai/runs
export DOCKER_HOST=unix:///run/docker.sock
bash orchestrator/trial.sh demo \
  --task tasks/ragen_sokoban_grpo --agent codex --gpu 1
```

The asset staging default deliberately stays on the asset filesystem so a verified alias can
be published with an atomic rename instead of copying a large model through `/tmp`. Docker's
own image/layer storage is configured on the selected Docker daemon, not by AI4AI; use the
daemon configuration or choose another daemon through `DOCKER_HOST`. Named image build
contexts are passed explicitly by `tools/build_images.py --context-root` and therefore
have no repository-specific host default.

All recovery/scoring watchers on one host must share `AI4AI_GPU_CLAIM_ROOT`; its small
`flock` files coordinate device selection and contain no model or experiment payload. Put it
on a host-local filesystem that supports `flock`, not in a per-process temporary directory.
The legacy unscoped `CLAIMS` variable remains a compatibility fallback, but new deployments
should use the documented AI4AI name.

No internal host path is used as a fallback. Absolute paths can appear in private run logs,
Docker error messages, and raw manifests because Docker bind mounts necessarily resolve the
host source. Sanitize those operational records before publishing them.

API base URLs must be explicit HTTPS URLs. A non-default port and URL path are allowed, but
userinfo/embedded credentials, query strings, and fragments are rejected; egress permits
only the exact resolved `host:port`. An upstream proxy is optional and must be a plain,
unauthenticated HTTP CONNECT endpoint such as `http://proxy.example:3128`. Proxy
authentication and HTTPS-to-proxy transport are not implemented. Sites requiring either
should provide a compatible local unauthenticated CONNECT relay rather than embedding
credentials in a URL.

## Container mount contract

| Container path | Source | Access | Meaning |
|---|---|---|---|
| `/workspace` | pristine image layer | writable in Explore/retrain | candidate-owned source tree |
| `/assets/...` | aliases below `--assets` | read-only | fixed models and data; phase-specific |
| `/out` | subdirectory below `--root/<run>` | writable | checkpoints and evaluator receipts |
| `/logs` | subdirectory below the run | phase-dependent | deadline, Agent, and verifier logs |
| `/patch/candidate.patch` | Explore output | read-only | the only Explore-to-formal handoff |
| `/ckpt` | selected formal checkpoint | read-only | validation/final input |
| `/opt/harness` | frozen image layer | read-only | lifecycle and evaluator code |
| `/tmp` | container tmpfs | writable, ephemeral | caches that must not survive a phase |

Changing the host directories needs only the CLI/environment settings above. Changing these
container destinations changes the benchmark contract and requires editing the task
declaration, the task scripts, and the frozen image together; it is not a deployment setting.

Task TOML records CPU, RAM, shared-memory, process, GPU-memory, and nominal storage needs.
Individual phases also declare a minimum free-space preflight so a checkpoint is not left
half-written. These are resource requirements, not hidden paths. The current official
declarations target one NVIDIA B300; notably OpenR1 and RAGEN declare more than 200 GiB of
free GPU memory. A compatible accelerator can be used for development or a locally modified
configuration, but it is not a like-for-like official run. The runner does not silently
reduce model size, batch size, or training budget to fit smaller hardware.

The default `--hardware-check warn` records GPU type/capacity mismatches in `preflight.json`,
warns, and continues as non-official. `off` skips only those two contract checks. GPU
reservations and existing-process occupancy are independent safety gates and fail closed in
every mode. Official `strict` additionally requires the declared type and capacity; all
modes require an idle, unreserved device.

## Identity and hardware modes

The runner does **not** require a Git commit or a clean worktree. `git_head` is provenance in
the manifest only. The independent gates are:

- `--source-check warn` (default) reports that repository files differ from the image build
  receipt but does not block ordinary use. `--source-check off` is available for deliberate
  local source development. Official replay uses `--source-check strict`. This one policy
  covers two independent content checks: the image-source receipt and a locked host-side
  benchmark contract. The latter hashes each task's `task.toml`, `declaration.py`,
  `instruction.md`, and `environment/assets.lock.yaml`, plus the orchestration phase code,
  and is verified before a task
  declaration is imported. It records no Git commit, branch, worktree status, path, or
  modification time.
- Official `--source-check strict` also reads and verifies every asset alias mounted by the
  current phase. `run-config.json` binds the path-independent lock/content digest;
  `preflight.json` records per-alias expected and observed hashes plus the local root as
  provenance. Identical assets may move to another host path, but mutating bytes in place
  invalidates official replay. Phase scoping preserves the hidden-final boundary.
- `--image-check warn` (default) reports that a locally built or retagged image differs from
  the declared release artifact but does not block ordinary use. It compares both immutable
  filesystem layers and the behavior-relevant OCI execution configuration. Official replay
  uses `--image-check strict`.
- `--hardware-check warn` (default) reports a GPU type or free-capacity mismatch but permits
  a non-official development run. Official replay uses `--hardware-check strict`.
  `--hardware-check off` skips only the model/type and peak-capacity contract.
- Tool-floor, GPU reservations/existing-process occupancy, basic GPU visibility, free-space,
  and real CUDA-kernel probes always fail closed. Neither `--hardware-check off` nor
  deprecated `--skip-digest-check` disables them.

Missing images are pulled once by default (`--image-pull-policy missing`), after which every
container uses `--pull=never` so a tag cannot drift during one lifecycle. Air-gapped sites use
`--image-pull-policy never` after loading the images themselves.

Fixed read-only assets, no-network formal replay, hidden/final-only mounts, source-only patch
submission, wall-clock limits, checkpoint validation, and failure accounting are benchmark
integrity constraints. They are deliberately not removed by local image options.

For an official run, enable all three gates explicitly:

```bash
export AI4AI_SOURCE_CHECK=strict
export AI4AI_IMAGE_CHECK=strict
export AI4AI_HARDWARE_CHECK=strict
```

These checks compare source content and OCI image content, not a Git commit, branch name,
remote, or clean-worktree state. `git_head` is recorded only as provenance. Local forks can
rebuild and run without editing a recorded hash; their manifest records that the image was
not officially verified. Strict mode also fails when the declaration lacks either expected
layer or OCI-config identity; it does not silently treat an absent identity as verified.
Official hardware strict mode similarly requires the declared GPU model/count and free
capacity to match. A run cannot switch task, image, policy, or timeouts and then resume under
the same name: `run-config.json` prevents outputs from different configurations being spliced
together. In strict mode the asset root is the deliberate exception: it may move when the
locked content identity is unchanged, and the runner rehashes the aliases mounted by the next
phase before accepting them. Warn/off lifecycles retain the literal asset path as their only
continuity signal, so moving those assets requires a new run name.
