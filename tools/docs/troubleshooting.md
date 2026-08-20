# Troubleshooting

Start with the structured host check. Add `--json` for automation and `--hash-assets` when
you need an exact (and potentially slow) asset verification.

```bash
python tools/check_setup.py \
  --task ddpo_sd15_aesthetic \
  --assets /data/ai4ai/assets/ddpo_sd15_aesthetic \
  --root /data/ai4ai/runs \
  --gpu 0 --agent codex --mode local
```

## Docker or GPU checks fail

- Confirm `docker version` reaches the same local daemon that can see your bind-mount paths.
- Confirm `docker info` lists the `nvidia` runtime and that `nvidia-smi` reports the selected
  GPU index.
- Run `bash tools/smoke.sh --root /data/ai4ai/smoke --gpu 0`. It pulls the immutable DDPO
  image if needed and exercises an actual CUDA kernel, host mounts, and the synthetic score
  phase without an API key or task assets.
- AI4AI refuses an occupied or container-reserved GPU in every verification mode. Choose an
  idle device rather than disabling the hardware identity check.

## Assets are missing or invalid

Pass the task-specific directory—not the parent containing all tasks—to `--assets`. Re-run
`tools/prepare_assets.py`, then use `tools/verify_assets.py --hash` to distinguish a
partial download from a content mismatch. Preparation never overwrites an existing alias;
move an invalid alias aside before rebuilding it. Upstream gated repositories require an
accepted license and `HF_TOKEN`.

Published images do not require Git submodules. The OPD `verl` submodule is an image build
input only; initialize it only when following the image rebuild procedure.

## Agent or API checks fail

The executable lookup order is `CODEX_BINARY`/`CLAUDE_BINARY`, `PATH`, then
`$HOME/.local/bin`. An explicit but invalid override fails rather than silently selecting a
different CLI. Set `OPENAI_API_KEY` for Codex, or `ANTHROPIC_API_KEY`/
`ANTHROPIC_AUTH_TOKEN` for Claude. The setup checker reports whether a credential exists but
never prints its value.

Custom endpoints must be HTTPS URLs without embedded credentials, query parameters, or
fragments. Use `OPENAI_BASE_URL` or `ANTHROPIC_BASE_URL`; do not place keys in the URL.
The credential must be issued for the selected endpoint: a gateway key is not an OpenAI
Platform key, even when both services expose an OpenAI-compatible API.

## Local versus official verification

Local mode records source, image, or GPU-contract mismatches as non-official diagnostics
where safe. It does not bypass missing assets, broken CUDA kernels, occupied GPUs, or invalid
artifacts. Official mode additionally requires the frozen source, image, asset hashes, and
declared B300 hardware contract. See [runtime and storage](runtime-and-storage.md) for the
three verification policies and configurable paths.

## Disk or interrupted runs

Each task declares an output budget of up to 512 GiB, in addition to assets and Docker's
image store. Put `--root`, `--assets`, and Docker storage on volumes with enough free space.
Do not resume a partially written formal run under a new configuration; use a new run name
and preserve the original receipts for failure accounting.
