# Runtime assets

AI4AI does not bake models or datasets into its task images. Each run receives one
task-specific host directory through `--assets`; declarations map relative aliases such as
`models/policy` to read-only container paths below `/assets`. Runtime assets, image build
contexts, and run outputs are three different stores. Do not copy one store wholesale into
another.

## Layout and verification

Use one parent directory with one subdirectory per task:

```bash
export AI4AI_ASSET_STORE=/data/ai4ai/assets
```

```text
$AI4AI_ASSET_STORE/
  ddpo_sd15_aesthetic/models/stable-diffusion-v1-5/
  ddpo_sd15_aesthetic/models/clip/
  ...
  ragen_sokoban_grpo/models/policy/
```

Pass the task directory, not the parent:

```bash
export AI4AI_ASSETS_ROOT="$AI4AI_ASSET_STORE/ragen_sokoban_grpo"
python3 tools/verify_assets.py \
  --task ragen_sokoban_grpo --assets "$AI4AI_ASSETS_ROOT"
python3 tools/verify_assets.py \
  --task ragen_sokoban_grpo --assets "$AI4AI_ASSETS_ROOT" --hash
```

The first command checks required aliases, file counts, and byte totals. `--hash` reads the
full payload and checks recorded SHA-256 identities. To check a complete installation, use
`--all --assets-root "$AI4AI_ASSET_STORE" --hash`. The full reference asset set is roughly
110 GiB. Local warn/off runs do not pay that full cost. An official strict lifecycle hashes
the aliases actually mounted by each phase before that phase starts, records a path-independent
receipt in `preflight.json`, and requires every receipt to match the immutable run configuration.
Explore therefore never has to read a score-only/hidden alias.

Each task's authoritative inventory is
`tasks/<task>/environment/assets.lock.yaml`. Revisions are immutable commit hashes, not
moving branches. For `tree_manifest_sha256`, the verifier hashes compact, sorted-key JSON
records `{path, sha256, size_bytes}`; `.cache` is excluded. Do not use `du` as a substitute
for `size_bytes`, which is the sum of regular-file sizes.

The absolute asset root is operational provenance only and may differ across hosts. Scientific
identity is the locked alias/content digest. `environment/assets.lock.yaml` is itself part of
the release-owned host contract, so changing a declared asset hash also requires a reviewed
host-contract lock refresh; editing the lock and the bytes together cannot silently preserve
official status.

## Prepare public snapshots

Install the project with its asset-provider dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[assets]'
export HF_TOKEN=...                 # only when an upstream repository requires it
```

First inspect which aliases have a released exact materializer:

```bash
python3 tools/prepare_assets.py \
  --task ragen_sokoban_grpo \
  --assets "$AI4AI_ASSET_STORE/ragen_sokoban_grpo"
```

Then prepare either all ready aliases or a selected alias:

```bash
export AI4AI_ASSET_STAGING_ROOT=/data/ai4ai/asset-staging
python3 tools/prepare_assets.py \
  --task ragen_sokoban_grpo \
  --assets "$AI4AI_ASSET_STORE/ragen_sokoban_grpo" \
  --execute

python3 tools/prepare_assets.py \
  --task owl_wanda_opt6p7b_70pct \
  --assets "$AI4AI_ASSET_STORE/owl_wanda_opt6p7b_70pct" \
  --alias data/wikitext2/test --execute
```

Without `--alias`, the command visits every required alias. Dependencies are built first,
every completed alias is hash-verified, and an existing valid alias is reused without being
overwritten. Prepare and verify all ten tasks with:

```bash
python3 tools/prepare_assets.py \
  --all --assets-root "$AI4AI_ASSET_STORE" --execute
python3 tools/verify_assets.py \
  --all --assets-root "$AI4AI_ASSET_STORE" --hash
```

The all-task command keeps independent tasks moving after a failure and returns nonzero if
any alias failed. Re-running it verifies and skips completed immutable aliases.

`--staging-root` overrides `AI4AI_ASSET_STAGING_ROOT`. With neither set, the tool creates a
hidden staging directory beside the task asset root, on the same filesystem. A provider
download always lands in a fresh temporary staging directory. The tool then copies only the
declared files, performs task-specific flattening/sidecars, verifies the complete SHA-256
identity, and atomically renames the verified alias into place. It never treats a Hugging
Face cache, nested provider directory, or partially downloaded alias as the final asset and
never overwrites an existing alias.

Publication prefers Linux `renameat2(RENAME_NOREPLACE)`. On a shared filesystem that does
not support that flag, the tool serializes the destination check and same-filesystem rename
with a persistent `flock` file outside the scientific alias. The staging root and final
asset root must be on the same filesystem; the tool never falls back to an unlocked
existence-check overwrite or a cross-filesystem copy.

This staging rule matters for both correctness and privacy. Provider caches can contain
metadata that is not part of the benchmark identity, and an internal asset tree can contain
host paths, trainer state, or site-specific sidecars. Do not publish or distribute an
existing cluster, cache, or experiment tree wholesale. Build a clean asset store
from the public lock and publish only aliases that pass the complete hash check.

For a pinned Git source, clone it outside the runtime asset root and checkout the exact
revision in the lock. Those repositories and offline wheelhouses are image build contexts;
they are not runtime assets and must not be mounted into an Agent run. Git submodules are
not required when using the published task images; the OPD source submodule is needed only
when rebuilding that image.

Follow lock-level `materialize` steps after a provider download. For example, OWL copies the
single WikiText parquet from `wikitext-2-raw-v1/` to the alias root because its evaluator uses
a non-recursive glob. The RewardBench task writes the recorded revision plus a newline to the
deterministic `.llmab-revision` sidecar; that sidecar is part of the pinned tree hash. The
verifier rejects a raw provider layout that has not undergone these steps.

Some model providers require accepting terms before download. The repository does not
redistribute those weights and does not bypass provider access controls. Review each lock's
`license`, `license_source`, and `restrictions` fields before mirroring or publishing data.

## Derived assets and receipts

All ten tasks have a public materialization plan. Direct model and dataset snapshots come
from immutable upstream revisions. More involved aliases are reproduced as follows:

- DiGress preprocesses public QM9 into the frozen hydrogen-free tensors, then derives the
  train/validation-only view.
- DPO merges the pinned Zephyr QLoRA adapter into its Mistral base and projects IFEval.
- Model Soup downloads the 72 numbered release checkpoints, safely expands ImageNetV2,
  and selects proxy offsets 0--1.
- OPD exports MATH-500 and AIME from pinned sources and applies the frozen DAPO/AIME
  zero-overlap projector.
- Reward modeling applies the frozen RewardBench decontamination rule to produce the
  paper protocol's complete 8,192-pair training asset and its 512-row proxy.

Task-specific Python builders run with `--network none` inside the task image pinned by
registry digest. Their script digest is recorded in the asset lock. Downloads remain in a
temporary host staging tree; only their read-only, provider-neutral inputs enter the builder.
Every output must match the existing exact file or tree digest before publication.

For each newly built alias, the tool writes a provenance receipt below
`.ai4ai-materialization/` in the task asset root. Receipts record the task, alias, lock-entry
digest, builder image and builder-script digest. They are deliberately outside the alias, so
provenance metadata cannot alter the scientific content hash.

DDPO's `data/final_reference.json` is generated from the inline lock content using canonical
indented JSON. RAGEN has no dataset alias because its Sokoban boards are generated by pinned
code and seeds.

## Final-only assets

Final-only does not mean secret. Several final splits, including OpenR1 v6, are public
upstream data but are mounted only during scoring to prevent exploration-time tuning. A
missing final-only alias is reported as unavailable, never converted to a zero score.

Two deployment profiles should be distinguished in published results:

- **Self-hosted reproduction:** download every pinned alias, including public score-only
  splits. Phase-specific mounts still keep final data out of Explore/formal containers.
- **Blind evaluation (not yet available):** a future service can retain final-only aliases
  and give users only Explore/formal assets while reporting the same asset hashes in its
  receipt.

The benchmark boundary comes from mounts and receipts, not from pretending public datasets
are secret. This release supports self-hosted reproduction and requires disclosure that the
final assets were locally available.
