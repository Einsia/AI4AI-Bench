#!/usr/bin/env bash
# Resolve this task's wheelhouse against its declared base and write a hash lock.
# Two required packages are published only as sdists:
#
#   rouge_score==0.1.2   a *core* dependency of every lighteval ever published. sdist only.
#   langdetect==1.0.9    required at call time by lighteval's ifeval_prompt, which is
#                        decorated @requires("langdetect"). sdist only.
#
# Step 0 builds deterministic wheels into `prebuilt/`; the lock records both the
# upstream sdist hashes and the wheel byte hashes used by the offline image build.
#
# Usage: bash tasks/dpo_preference_alignment/environment/resolve_wheelhouse.sh [base-image]
set -uo pipefail

TASK=dpo_preference_alignment
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}
CTX=${CTX:-$REPO/.local/docker-context}
OUT="$CTX/wheelhouse/$TASK"
ENVDIR="$REPO/tasks/$TASK/environment"
PROXY=${PROXY:-}
AI4AI_DOCKER=${AI4AI_DOCKER:-docker}
read -r -a DOCKER_CMD < <(printf '%s\n' "$AI4AI_DOCKER")
docker_cmd() { "${DOCKER_CMD[@]}" "$@"; }

# Pure-Python sdists, so the wheel is a repackaging rather than a compile. Fixed for the
# zip mtimes inside the built wheel, without which its sha256 differs on every run.
export SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH:-1767225600}
SDIST_ONLY=("rouge_score==0.1.2" "langdetect==1.0.9")

[ -d "$ENVDIR" ] || { echo "$TASK: no environment directory at $ENVDIR" >&2; exit 2; }
SRC="$ENVDIR/runtime-requirements.in"
[ -f "$SRC" ] || { echo "$TASK: no runtime-requirements.in" >&2; exit 2; }

BASE=${1:-}
if [ -z "$BASE" ]; then
  BASE=$(grep -oE '^ARG [A-Z_]*IMAGE="[^"]+"' "$ENVDIR/Dockerfile" | head -1 | sed 's/.*="//;s/"$//')
fi
[ -n "$BASE" ] || { echo "$TASK: could not read a base image from the Dockerfile" >&2; exit 2; }
echo "=== $TASK: resolving against $BASE ==="

docker_cmd image inspect "$BASE" >/dev/null 2>&1 \
  || { echo "  base image is not present locally -- pull it first" >&2; exit 3; }

mkdir -p "$OUT/prebuilt"
# Start clean, so a wheel left by an earlier vacuum-resolved run cannot survive into a
# build unnoticed. prebuilt/ is rebuilt from scratch too.
rm -f "$OUT"/*.whl "$OUT"/*.tar.gz "$OUT"/resolution.json "$OUT"/plan.txt "$OUT"/urls.txt 2>/dev/null
rm -rf "$OUT/prebuilt" "$OUT/sdist" 2>/dev/null
mkdir -p "$OUT/prebuilt" "$OUT/sdist"
chmod -R a+rwX "$OUT"

grep -vE '^\s*(#|$)' "$SRC" | sed 's/ *\\$//' | grep -vE '^\s*--hash' > "$OUT/requirements.txt"
echo "  $(wc -l < "$OUT/requirements.txt" | tr -d ' ') direct requirements"

echo "--- 0. build the two sdist-only wheels ---"
docker_cmd run --rm --network=host \
  -e HTTPS_PROXY="$PROXY" -e HTTP_PROXY="$PROXY" \
  -e PIP_DISABLE_PIP_VERSION_CHECK=1 -e SOURCE_DATE_EPOCH="$SOURCE_DATE_EPOCH" \
  -v "$OUT:/wh" "$BASE" \
  bash -lc "set -e
    python -m pip download --no-deps --no-binary :all: --dest /wh/sdist ${SDIST_ONLY[*]} -q
    for archive in /wh/sdist/*.tar.gz; do
      # No filtering pipeline here. \`... | grep -v DEPRECATION\` exits 1 when the
      # deprecation notice was the only line pip wrote, and under \`set -e\` that aborts
      # the build of a wheel that had in fact succeeded.
      python -m pip wheel --no-deps --wheel-dir /wh/prebuilt \"\$archive\" -q
    done
    python - <<'PY'
import hashlib, pathlib
for label, pattern in (('sdist', '/wh/sdist/*'), ('wheel', '/wh/prebuilt/*.whl')):
    for path in sorted(pathlib.Path('/').glob(pattern.lstrip('/'))):
        print('  %-6s %-46s sha256=%s'
              % (label, path.name, hashlib.sha256(path.read_bytes()).hexdigest()))
PY" \
  || { echo "  could not build the sdist-only wheels" >&2; exit 4; }

echo "--- 1. resolve inside the base image ---"
# --find-links is the only difference from the shared script: it is what lets pip see a
# wheel for rouge_score and langdetect, and so resolve at all.
docker_cmd run --rm --network=host \
  -e HTTPS_PROXY="$PROXY" -e HTTP_PROXY="$PROXY" \
  -e PIP_DISABLE_PIP_VERSION_CHECK=1 \
  -v "$OUT:/wh" "$BASE" \
  bash -lc 'python -m pip install --dry-run --only-binary :all: --find-links /wh/prebuilt \
      --report /wh/resolution.json -r /wh/requirements.txt -q 2>&1 | tail -20' \
  || { echo "  RESOLVE FAILED -- see the pip output above" >&2; exit 4; }
[ -s "$OUT/resolution.json" ] || { echo "  no resolution report was written" >&2; exit 4; }

echo "--- 2. read the plan ---"
python3.12 - "$OUT/resolution.json" "$OUT/plan.txt" <<'PY'
import json, pathlib, sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text())
rows = []
for item in report.get("install", []):
    meta = item.get("metadata", {})
    info = item.get("download_info", {})
    rows.append((
        meta.get("name"), meta.get("version"), info.get("url", ""),
        info.get("archive_info", {}).get("hashes", {}).get("sha256", ""),
        bool(item.get("requested")),
    ))

# Nothing here may replace the base's torch. A wheelhouse that does is the failure mode
# the shared script's header documents, and it is silent: the build succeeds and the
# image is broken.
overwrites = sorted(
    f"{name}=={version}" for name, version, _, _, _ in rows
    if name and name.lower().split("[")[0] in {"torch", "torchvision", "torchaudio", "triton"}
)
if overwrites:
    print("  REFUSING: the plan replaces the base's torch: " + ", ".join(overwrites),
          file=sys.stderr)
    raise SystemExit(7)

rows.sort(key=lambda r: r[0].lower())
pathlib.Path(sys.argv[2]).write_text(
    "\n".join(f"{n}\t{v}\t{u}\t{d}\t{'direct' if req else 'transitive'}" for n, v, u, d, req in rows) + "\n"
)
print(f"  {len(rows)} packages to add "
      f"({sum(1 for r in rows if r[4])} direct, {sum(1 for r in rows if not r[4])} transitive)")
print("  torch is untouched")
PY
status=$?
[ $status -eq 0 ] || exit $status

echo "--- 3. fetch exactly those wheels ---"
# The two locally built wheels have file:// URLs in the report, so they are not fetched
# here; they are copied in from prebuilt/ instead.
cut -f3 "$OUT/plan.txt" | grep -E '^https?://' > "$OUT/urls.txt"
echo "  $(wc -l < "$OUT/urls.txt" | tr -d ' ') from the index, $(ls "$OUT/prebuilt"/*.whl | wc -l | tr -d ' ') built locally"
docker_cmd run --rm --network=host \
  -e HTTPS_PROXY="$PROXY" -e HTTP_PROXY="$PROXY" \
  -v "$OUT:/wh" -w /wh "$BASE" \
  bash -lc 'python -m pip download --quiet --no-deps --only-binary :all: --dest /wh \
      $(sed "s/^/ /" /wh/urls.txt | tr "\n" " ") 2>&1 | tail -5' \
  || { echo "  download failed" >&2; exit 5; }
cp -f "$OUT/prebuilt"/*.whl "$OUT"/

wheels=$(ls "$OUT"/*.whl 2>/dev/null | wc -l | tr -d ' ')
echo "  $wheels wheels, $(du -sh "$OUT" | cut -f1)"

echo "--- 4. write the lock, hashed against the bytes on disk ---"
python3.12 - "$OUT/plan.txt" "$ENVDIR/runtime-requirements.lock" "$OUT" <<'PY'
import hashlib, pathlib, sys

plan = pathlib.Path(sys.argv[1]).read_text().splitlines()
lock = pathlib.Path(sys.argv[2])
wheelhouse = pathlib.Path(sys.argv[3])

on_disk = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(wheelhouse.glob("*.whl"))}
sdists = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted((wheelhouse / "sdist").glob("*"))}
built = {p.name for p in sorted((wheelhouse / "prebuilt").glob("*.whl"))}

lines = [
    "# Resolved on the B300 host by environment/resolve_wheelhouse.sh, INSIDE this task's",
    "# own base image -- so these are the packages that must be ADDED to that base, not a",
    "# standalone closure. Resolving without the base in scope produces a set that",
    "# overwrites the base's own torch; see that script's header.",
    "#",
    "# Each hash is the sha256 of the bytes in the wheelhouse, checked against what the",
    "# index reported. --require-hashes makes any later substitution a build failure.",
    "#",
    "# Two entries are marked `locally built`: PyPI has no wheel for them, so the wheel was",
    "# built from the upstream sdist and the hash below is that wheel's. The sdist each came",
    "# from, by its own upstream hash:",
]
for name, digest in sdists.items():
    lines.append(f"#   {name}  sha256:{digest}")
lines.append("")

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
    # A locally built wheel has no index hash to agree with, which is exactly why it is
    # called out in the lock rather than passed off as an index artifact.
    if filename in built:
        lines.append("# locally built")
    elif digest and actual != digest:
        mismatched.append(f"{name}=={version}")
        continue
    elif kind == "transitive":
        lines.append(f"# {kind}")
    lines.append(f"{name}=={version} \\")
    lines.append(f"    --hash=sha256:{actual}")

if missing:
    print(f"  MISSING from the wheelhouse: {', '.join(missing[:5])}", file=sys.stderr)
    raise SystemExit(6)
if mismatched:
    print(f"  HASH MISMATCH vs the index: {', '.join(mismatched[:5])}", file=sys.stderr)
    raise SystemExit(6)

lock.write_text("\n".join(lines) + "\n")
print(f"  lock written: {sum(1 for line in lines if line.endswith(' \\'))} pinned requirements")
PY
status=$?
[ $status -eq 0 ] || exit $status

echo "--- 5. does the result actually satisfy the base? ---"
docker_cmd run --rm --network=none \
  -v "$OUT:/wh:ro" "$BASE" \
  bash -lc 'python -m pip install --quiet --no-index --find-links /wh /wh/*.whl 2>&1 | tail -3
    python -m pip check 2>&1 | tail -5
    python -c "import torch; print(\"  torch still\", torch.__version__)"' \
  && echo "  pip check clean" \
  || echo "  NOTE: pip check reported something -- read it above before building"

echo "=== $TASK: wheelhouse resolved ==="
echo "  wheelhouse: $OUT ($wheels wheels)"
echo "  lock:       $ENVDIR/runtime-requirements.lock"
