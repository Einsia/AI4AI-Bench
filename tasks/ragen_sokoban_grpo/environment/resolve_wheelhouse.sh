#!/usr/bin/env bash
# Runs ON the B300 host. Resolve this task's wheelhouse against its own base image and
# write runtime-requirements.lock with hashes.
#
# WHY THIS EXISTS RATHER THAN the operator wheelhouse resolver.
#
# The shared resolver is the right tool and this is a copy of it, with one stage added.
# It resolves with `--only-binary :all:`, because a wheelhouse holds wheels and the
# image build has no network to compile an sdist with. This task has two requirements
# PyPI publishes as an sdist and nothing else:
#
#   gym==0.26.2                    ragen/env/sokoban/env.py does `import gym` and
#                                  gym_sokoban depends on it. Not substitutable by
#                                  gymnasium: that would mean editing the frozen
#                                  environment tree, whose bytes are what the final's
#                                  guarantee rests on.
#
#   antlr4-python3-runtime==4.9.3   pinned as `==4.9.*` by BOTH hydra-core (every 1.x,
#                                  including 1.3.4) and omegaconf 2.3.x, whose grammar
#                                  parser is antlr-generated.
#
# The second one is why this stage is load-bearing rather than a convenience. Without
# it, `--only-binary :all:` drops antlr 4.9, which drops every hydra 1.x, and the
# resolver backtracks to hydra-core 0.11.3 and omegaconf 1.4.1 -- a self-consistent
# 2019 pair that installs cleanly, passes `pip check`, and then dies at RAGEN's first
# `@hydra.main(version_base=None, ...)`. A silently ancient resolve is worse than a
# failed one.
#
# So stage 0 below builds both wheels, once, with network, into a directory beside the
# wheelhouse, and every later stage adds it to --find-links. Both are pure Python, so
# building them needs no toolchain. The image build stays --network=none and
# --only-binary=:all:.
#
# Everything else is the shared script's logic, including the reason it is not
# `pip download`: that resolves in a vacuum, does not know what the base has, and on
# DPO produced 68 wheels / 2.7 GB that installed torch 2.13.0 over the base's 2.8.0 and
# failed `pip check`. Resolving with `pip install --dry-run --report` INSIDE the base
# answers the question actually being asked -- what must be ADDED to this base.
#
# Usage: bash tasks/ragen_sokoban_grpo/environment/resolve_wheelhouse.sh [base-image]
set -uo pipefail

TASK=ragen_sokoban_grpo
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}
CTX=${CTX:-$REPO/.local/docker-context}
OUT="$CTX/wheelhouse/$TASK"
PREBUILT="$CTX/wheelhouse/${TASK}_prebuilt"
ENVDIR="$REPO/tasks/$TASK/environment"
PROXY=${PROXY:-}
AI4AI_DOCKER=${AI4AI_DOCKER:-docker}
read -r -a DOCKER_CMD < <(printf '%s\n' "$AI4AI_DOCKER")
docker_cmd() { "${DOCKER_CMD[@]}" "$@"; }

[ -d "$ENVDIR" ] || { echo "$TASK: no environment directory at $ENVDIR" >&2; exit 2; }
SRC="$ENVDIR/runtime-requirements.in"
[ -f "$SRC" ] || { echo "$TASK: no runtime-requirements.in" >&2; exit 2; }

BASE=${1:-}
if [ -z "$BASE" ]; then
  BASE=$(grep -oE '^ARG [A-Z_]*IMAGE="[^"]+"' "$ENVDIR/Dockerfile" | head -1 | sed 's/.*="//;s/"$//')
fi
[ -n "$BASE" ] || { echo "$TASK: could not read a base image from the Dockerfile" >&2; exit 2; }
echo "=== $TASK ==="
echo "  base: $BASE"
docker_cmd image inspect "$BASE" >/dev/null 2>&1 \
  || { echo "  base image is not present locally -- pull it first" >&2; exit 3; }

echo "--- 0. pre-build the sdist-only wheels (network) ---"
sudo mkdir -p "$PREBUILT" && sudo chmod 0777 "$PREBUILT"
SDIST_ONLY=("gym==0.26.2" "antlr4-python3-runtime==4.9.3")
for spec in "${SDIST_ONLY[@]}"; do
  # setuptools normalises a dash to an underscore in the wheel name, so match on both.
  stem=$(printf '%s' "${spec%%==*}" | tr '-' '_')
  version=${spec##*==}
  if ls "$PREBUILT/${stem}-${version}-"*.whl >/dev/null 2>&1 \
     || ls "$PREBUILT/${spec%%==*}-${version}-"*.whl >/dev/null 2>&1; then
    echo "  already staged: $spec"
    continue
  fi
  echo "  building: $spec"
  docker_cmd run --rm --network=host \
    -e HTTPS_PROXY="$PROXY" -e HTTP_PROXY="$PROXY" \
    -e PIP_DISABLE_PIP_VERSION_CHECK=1 \
    -v "$PREBUILT:/prebuilt" "$BASE" \
    bash -lc "python -m pip wheel --no-deps --no-cache-dir -w /prebuilt '$spec' 2>&1 | tail -3" \
    || { echo "  could not build $spec" >&2; exit 4; }
done
ls "$PREBUILT"/*.whl >/dev/null 2>&1 || { echo "  nothing staged" >&2; exit 4; }
echo "  staged: $(ls "$PREBUILT"/*.whl | xargs -n1 basename | tr '\n' ' ')"

mkdir -p "$OUT"
# Start clean. A stale wheel from an earlier vacuum-resolved run is exactly the kind of
# thing that survives into a build and is never noticed. $PREBUILT is a separate
# directory precisely so this cannot delete the gym wheel.
rm -f "$OUT"/*.whl "$OUT"/*.tar.gz "$OUT"/resolution.json 2>/dev/null

grep -vE '^\s*(#|$)' "$SRC" | sed 's/ *\\$//' | grep -vE '^\s*--hash' > "$OUT/requirements.txt"
echo "  $(wc -l < "$OUT/requirements.txt" | tr -d ' ') direct requirements"

echo "--- 1. resolve inside the base image ---"
docker_cmd run --rm --network=host \
  -e HTTPS_PROXY="$PROXY" -e HTTP_PROXY="$PROXY" \
  -e PIP_DISABLE_PIP_VERSION_CHECK=1 \
  -v "$OUT:/wh" -v "$PREBUILT:/prebuilt:ro" "$BASE" \
  bash -lc 'python -m pip install --dry-run --only-binary :all: --find-links /prebuilt \
      --report /wh/resolution.json -r /wh/requirements.txt -q 2>&1 | tail -25' \
  || { echo "  RESOLVE FAILED -- see the pip output above" >&2; exit 4; }
[ -s "$OUT/resolution.json" ] || { echo "  no resolution report was written" >&2; exit 4; }

echo "--- 2. read the plan ---"
python3.12 - "$OUT/resolution.json" "$OUT/plan.txt" <<'PY'
import json, pathlib, sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text())
rows = []
for item in report.get("install", []):
    meta = item.get("metadata", {})
    url = item.get("download_info", {}).get("url", "")
    digest = item.get("download_info", {}).get("archive_info", {}).get("hashes", {}).get("sha256", "")
    rows.append((meta.get("name"), meta.get("version"), url, digest, bool(item.get("requested"))))
rows.sort(key=lambda r: r[0].lower())
pathlib.Path(sys.argv[2]).write_text(
    "\n".join(f"{n}\t{v}\t{u}\t{d}\t{'direct' if q else 'transitive'}" for n, v, u, d, q in rows) + "\n"
)
print(f"  {len(rows)} packages to add ({sum(1 for r in rows if r[4])} direct, "
      f"{sum(1 for r in rows if not r[4])} transitive)")
PY
[ -s "$OUT/plan.txt" ] || { echo "  the plan is empty" >&2; exit 4; }

echo "--- 3. fetch exactly those wheels ---"
# A locally built wheel has a file:// URL in the report; copy those rather than asking
# an index for them.
cut -f3 "$OUT/plan.txt" | grep -E '^https?://' > "$OUT/urls.txt"
cut -f3 "$OUT/plan.txt" | grep -E '^file://' | sed 's|^file://||' > "$OUT/local.txt" || true
while read -r path; do
  [ -n "$path" ] && cp -f "$PREBUILT/$(basename "$path")" "$OUT/" 2>/dev/null \
    && echo "  local: $(basename "$path")"
done < "$OUT/local.txt"
if [ -s "$OUT/urls.txt" ]; then
  docker_cmd run --rm --network=host \
    -e HTTPS_PROXY="$PROXY" -e HTTP_PROXY="$PROXY" \
    -e PIP_DISABLE_PIP_VERSION_CHECK=1 \
    -v "$OUT:/wh" -w /wh "$BASE" \
    bash -lc 'python -m pip download --quiet --no-deps --only-binary :all: --dest /wh \
        $(sed "s/^/ /" /wh/urls.txt | tr "\n" " ") 2>&1 | tail -5' \
    || { echo "  download failed" >&2; exit 5; }
fi
wheels=$(ls "$OUT"/*.whl 2>/dev/null | wc -l | tr -d ' ')
echo "  $wheels wheels, $(du -sh "$OUT" | cut -f1)"

echo "--- 4. write the lock, with the hashes the index served ---"
python3.12 - "$OUT/plan.txt" "$ENVDIR/runtime-requirements.lock" "$OUT" <<'PY'
import hashlib, pathlib, sys

plan = pathlib.Path(sys.argv[1]).read_text().splitlines()
lock = pathlib.Path(sys.argv[2])
wheelhouse = pathlib.Path(sys.argv[3])

# Verify against the bytes on disk rather than trusting the report: these are the bytes
# the build will install, and if the two disagree it is the disk that is authoritative.
on_disk = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(wheelhouse.glob("*.whl"))}

lines = [
    "# Compiled from runtime-requirements.in by environment/resolve_wheelhouse.sh, on the",
    "# B300 host, INSIDE this task's own base image -- so these are the packages that must",
    "# be ADDED to that base, not a standalone closure. Resolving without the base in scope",
    "# produces a set that overwrites the base's own torch.",
    "#",
    "# Each hash is the sha256 of the bytes in the wheelhouse, checked against what the",
    "# index reported. --require-hashes makes any later substitution a build failure.",
    "#",
    "# gym==0.26.2 and antlr4-python3-runtime==4.9.3 are the two entries PyPI has no wheel",
    "# for; their hashes are of the wheels stage 0 of that script builds from the published",
    "# sdists. Both are pure Python, so the wheel is a repack of the sdist's contents.",
    "",
]
mismatched, missing = [], []
for row in plan:
    if not row.strip():
        continue
    name, version, url, digest, kind = row.split("\t")
    filename = url.rsplit("/", 1)[-1]
    actual = on_disk.get(filename)
    if actual is None:
        missing.append(f"{name}=={version}")
        continue
    if digest and actual != digest:
        mismatched.append(f"{name}=={version}")
        continue
    if kind == "transitive":
        lines.append(f"# {kind}")
    lines.append(f"{name}=={version} \\")
    lines.append(f"    --hash=sha256:{actual}")

if missing:
    print(f"  MISSING from the wheelhouse: {', '.join(missing[:8])}", file=sys.stderr)
    sys.exit(6)
if mismatched:
    print(f"  HASH MISMATCH vs the index: {', '.join(mismatched[:8])}", file=sys.stderr)
    sys.exit(6)

lock.write_text("\n".join(lines) + "\n")
print(f"  lock written: {sum(1 for line in lines if line.endswith(' \\'))} pinned requirements")
PY
status=$?
[ $status -eq 0 ] || exit $status

echo "--- 5. does the result actually satisfy the base? ---"
# Validate the resolved wheelhouse against the declared base environment.
docker_cmd run --rm --network=none \
  -v "$OUT:/wh:ro" "$BASE" \
  bash -lc 'python -m pip install --quiet --no-index --find-links /wh --no-deps /wh/*.whl 2>&1 | tail -3
    echo "--- pip check ---"; python -m pip check 2>&1 | tail -12
    echo "--- torch must still be the base'"'"'s 2.8.0+cu128 ---"
    python -c "import torch, numpy; print(\"torch\", torch.__version__, \"numpy\", numpy.__version__)"'

echo "=== $TASK: wheelhouse resolved ==="
echo "  wheelhouse: $OUT ($wheels wheels)"
echo "  prebuilt:   $PREBUILT"
echo "  lock:       $ENVDIR/runtime-requirements.lock"
