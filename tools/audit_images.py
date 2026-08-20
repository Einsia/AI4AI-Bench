#!/usr/bin/env python3
"""Read-only release audit for AI4AI task images.

The audit checks declared layer identity, OCI metadata/history for credential-like
values, and the final filesystem for build/smoke/cache residue. It never pulls,
builds, tags, pushes, or starts a GPU workload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
TASKS = REPO / "tasks"
SECRET = re.compile(
    r"(?i)(?:api[_-]?key|auth[_-]?token|access[_-]?token|secret|password)"
    r"\s*[:=]\s*[^\s,;]{6,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|"
    r"\b(?:sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9_-]{16,}|"
    r"github_pat_[A-Za-z0-9_-]{16,}|glpat-[A-Za-z0-9_-]{16,}|"
    r"hf_[A-Za-z0-9]{20,}|(?:AKIA|ASIA)[0-9A-Z]{16})"
)
INTERNAL_PATH = re.compile(
    r"/(?:shared|cluster)/(?:home|scratch|poc)/|"
    r"\b(?:[a-z]{2,}-gpu[0-9]+|[a-z][0-9]+-instan-[0-9]+)\b"
)
HIGH_CONFIDENCE_SECRET = re.compile(
    rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|"
    rb"\b(?:sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9_-]{16,}|"
    rb"github_pat_[A-Za-z0-9_-]{16,}|glpat-[A-Za-z0-9_-]{16,}|"
    rb"hf_[A-Za-z0-9]{20,}|(?:AKIA|ASIA)[0-9A-Z]{16})|"
    rb"(?i:https?://[^\s/:]{1,64}:[^\s/@]{6,128}@)"
)
SUSPICIOUS_LAYER_PATH = re.compile(
    r"(^|/)(?:\.env(?:\..*)?|id_(?:rsa|ed25519)|credentials(?:\.json)?|"
    r"99proxy|auth\.json)$"
)


def task_config(task: Path) -> dict[str, Any]:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as tomllib  # type: ignore[no-redef]
    with (task / "task.toml").open("rb") as stream:
        return tomllib.load(stream)


def run(command: list[str], *, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout)


def layer_fingerprint(runtime: list[str], image: str) -> str | None:
    result = run(
        [
            *runtime,
            "image",
            "inspect",
            image,
            "--format",
            "{{range .RootFS.Layers}}{{.}} {{end}}",
        ]
    )
    if result.returncode != 0:
        return None
    return "layers:" + hashlib.sha256("\n".join(result.stdout.split()).encode()).hexdigest()


def stream_findings(stream: Any, *, chunk_size: int = 1024 * 1024) -> tuple[bool, bool]:
    """Scan a binary stream without excluding large files.

    The overlap preserves tokens split across chunk boundaries. The expressions are
    intentionally high confidence, so applying them to binary payloads is preferable
    to silently exempting large wheels, archives, or model-support binaries.
    """

    secret = False
    internal = False
    tail = b""
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        window = tail + chunk
        secret = secret or HIGH_CONFIDENCE_SECRET.search(window) is not None
        internal = (
            internal or INTERNAL_PATH.search(window.decode("utf-8", errors="ignore")) is not None
        )
        if secret and internal:
            break
        tail = window[-512:]
    return secret, internal


def config_fingerprint(metadata: dict[str, Any]) -> str:
    fields = (
        "User",
        "Env",
        "Entrypoint",
        "Cmd",
        "WorkingDir",
        "Shell",
        "Healthcheck",
        "StopSignal",
        "ExposedPorts",
        "Volumes",
        "Labels",
        "OnBuild",
    )
    raw = metadata.get("Config") or {}
    canonical = {
        "os": metadata.get("Os"),
        "architecture": metadata.get("Architecture"),
        "variant": metadata.get("Variant"),
        "config": {field: raw.get(field) for field in fields},
    }
    encoded = json.dumps(canonical, separators=(",", ":"), sort_keys=True).encode()
    return "config:" + hashlib.sha256(encoded).hexdigest()


def scan_saved_layers(runtime: list[str], image: str) -> tuple[list[str], list[str]]:
    """Scan every OCI/Docker layer, including files deleted by later layers."""

    problems: list[str] = []
    warnings: list[str] = []
    process = subprocess.Popen(
        [*runtime, "save", image],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|*") as archive:
            for outer in archive:
                if not outer.isfile() or not outer.name.endswith("/layer.tar"):
                    continue
                stream = archive.extractfile(outer)
                if stream is None:
                    continue
                with tarfile.open(fileobj=stream, mode="r|*") as layer:
                    for member in layer:
                        name = member.name.lstrip("./")
                        if SUSPICIOUS_LAYER_PATH.search(name):
                            problems.append(f"suspicious file exists in layer history: /{name}")
                        if name.endswith("/.git/config") and not name.startswith(
                            "opt/harness/git-base/.git/"
                        ):
                            warnings.append(f"VCS metadata exists in layer history: /{name}")
                        if not member.isfile():
                            continue
                        payload = layer.extractfile(member)
                        if payload is None:
                            continue
                        secret, internal = stream_findings(payload)
                        if secret:
                            problems.append(f"credential-like bytes exist in layer: /{name}")
                        if internal:
                            problems.append(f"internal host/path bytes exist in layer: /{name}")
    except (OSError, tarfile.TarError) as error:
        problems.append(f"could not scan saved image layers: {error}")
    finally:
        process.stdout.close()
    stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
    returncode = process.wait()
    if returncode != 0:
        problems.append(f"docker save failed: {stderr.strip()[:500]}")
    return sorted(set(problems)), sorted(set(warnings))


HOME_RESIDUE_SCAN = r"""
for user_home in ${AI4AI_HOME_AUDIT_ROOTS:-/root /home/*}; do
  test -d "$user_home" || continue

  find "$user_home" -xdev -type d -name .git -print -prune 2>/dev/null |
    while IFS= read -r path; do printf 'HOME_GIT:%s\n' "$path"; done

  find "$user_home" -xdev -type d \( \
      -name .cache -o -name .pytest_cache -o \
      -name .mypy_cache -o -name .ruff_cache -o -name .nv -o -name .triton \
    \) -print -prune 2>/dev/null |
    while IFS= read -r path; do
      if find "$path" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
        printf 'HOME_CACHE:%s\n' "$path"
      fi
    done

  for relative in \
      .ssh .aws .docker .kube .gnupg .huggingface \
      .config/gcloud .config/gh .config/huggingface; do
    path="$user_home/$relative"
    if test -d "$path" && find "$path" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
      printf 'HOME_SENSITIVE_DIR:%s\n' "$path"
    fi
  done

  find "$user_home" -xdev \
    \( -type d \( \
       -name .git -o -name .cache -o -name .pytest_cache -o \
       -name .mypy_cache -o -name .ruff_cache \
    \) -prune \) -o \
    \( -type f \( \
       -name .netrc -o -name .git-credentials -o -name .gitconfig -o \
       -name .npmrc -o -name .pypirc -o -name pip.conf -o \
       -name auth.json -o -name credentials.json -o \
       -name id_rsa -o -name id_ed25519 -o -name .bash_history -o \
       -name .python_history -o -name .wget-hsts -o -name .lesshst -o \
       -name .curlrc \
    \) -print \) 2>/dev/null |
    while IFS= read -r path; do printf 'HOME_SENSITIVE_FILE:%s\n' "$path"; done
done | sort -u
"""


FILESYSTEM_PROBE = (
    r"""
set -eu
problems=''
add() { problems="${problems}${problems:+\n}$1"; }
test ! -e /opt/build || add '/opt/build remains'
test ! -e /etc/apt/apt.conf.d/99proxy || add 'build-time apt proxy remains'
test -s /licenses/LICENSE || add '/licenses/LICENSE is missing'
test -s /licenses/THIRD_PARTY_NOTICES.md || add '/licenses/THIRD_PARTY_NOTICES.md is missing'
for path in /root/.launchpadlib; do
  test ! -e "$path" || add "$path remains"
done
for root in /tmp /out /logs; do
  test ! -d "$root" || \
    ! find "$root" -mindepth 1 -print -quit | grep -q . || \
    add "$root is not empty"
done
test ! -d /assets || ! find /assets -type f -print -quit | grep -q . || add '/assets contains files'
# Do not let image metadata narrow the audit roots. AI4AI_HOME_AUDIT_ROOTS exists
# only so the standalone fragment can be exercised against temporary trees in tests.
unset AI4AI_HOME_AUDIT_ROOTS
home_residue=$(
"""
    + HOME_RESIDUE_SCAN
    + r"""
)
test -z "$home_residue" || add "$home_residue"
find /workspace /opt/harness -type d -name .git -print 2>/dev/null | while IFS= read -r gitdir; do
  test "$gitdir" = /opt/harness/git-base/.git || printf 'UNEXPECTED_GIT:%s\n' "$gitdir"
done > /tmp/ai4ai-git-audit
test ! -s /tmp/ai4ai-git-audit || add "$(cat /tmp/ai4ai-git-audit)"
rm -f /tmp/ai4ai-git-audit
if test -f /opt/harness/git-base/.git/config && \
   grep -qE '^\s*\[remote ' /opt/harness/git-base/.git/config; then
  add 'pristine git baseline contains a remote'
fi
if test -n "$problems"; then printf '%b\n' "$problems"; exit 1; fi
printf 'FILESYSTEM CLEAN\n'
"""
)


def audit(
    task: Path,
    runtime: list[str],
    *,
    image_override: str | None,
    skip_digest: bool,
    scan_layers: bool,
) -> dict[str, Any]:
    config = task_config(task).get("environment", {})
    image = image_override or str(config.get("image", ""))
    problems: list[str] = []
    inspect = run([*runtime, "image", "inspect", image])
    if inspect.returncode != 0:
        detail = (inspect.stderr or inspect.stdout).strip()
        missing = "No such image" in detail or "not found" in detail.lower()
        return {
            "task": task.name,
            "image": image,
            "status": "missing" if missing else "runtime-error",
            "problems": [detail],
        }
    try:
        metadata = json.loads(inspect.stdout)[0]
    except (IndexError, TypeError, ValueError):
        return {"task": task.name, "image": image, "status": "invalid-inspect"}
    platform = f"{metadata.get('Os')}/{metadata.get('Architecture')}"
    if platform != "linux/amd64":
        problems.append(f"unexpected platform {platform}")
    observed = layer_fingerprint(runtime, image)
    expected = str(config.get("digest", ""))
    observed_config = config_fingerprint(metadata)
    expected_config = str(config.get("config_digest", ""))
    if observed is None:
        problems.append("could not compute image layer fingerprint")
    if not skip_digest and expected and observed != expected:
        problems.append(f"layer fingerprint {observed} != {expected}")
    if not skip_digest and expected_config and observed_config != expected_config:
        problems.append(f"config fingerprint {observed_config} != {expected_config}")
    if not skip_digest and not expected_config:
        problems.append("task.toml records no config_digest")

    serialised = json.dumps(
        {
            "env": metadata.get("Config", {}).get("Env", []),
            "labels": metadata.get("Config", {}).get("Labels", {}),
        },
        sort_keys=True,
    )
    history = run([*runtime, "history", "--no-trunc", "--format", "{{.CreatedBy}}", image])
    if history.returncode != 0:
        problems.append(
            "could not inspect image history: " + (history.stderr or history.stdout).strip()[:500]
        )
    metadata_text = serialised + "\n" + history.stdout
    if SECRET.search(metadata_text):
        problems.append("credential-like value appears in image metadata/history")
    if INTERNAL_PATH.search(metadata_text):
        problems.append("internal host/path value appears in image metadata/history")

    labels = metadata.get("Config", {}).get("Labels") or {}
    if labels.get("org.ai4ai.task") != task.name:
        problems.append("org.ai4ai.task label is absent or incorrect")
    probe = run(
        [
            *runtime,
            "run",
            "--rm",
            "--pull=never",
            "--network=none",
            "--entrypoint",
            "bash",
            image,
            "-lc",
            FILESYSTEM_PROBE,
        ],
        timeout=600,
    )
    if probe.returncode != 0 or "FILESYSTEM CLEAN" not in probe.stdout:
        detail = (probe.stdout + "\n" + probe.stderr).strip()
        problems.append(f"filesystem audit failed: {detail[:2000]}")
    layer_warnings: list[str] = []
    if scan_layers:
        layer_problems, layer_warnings = scan_saved_layers(runtime, image)
        problems.extend(layer_problems)
    return {
        "task": task.name,
        "image": image,
        "status": "clean" if not problems else "not-clean",
        "platform": platform,
        "image_id": metadata.get("Id"),
        "layer_fingerprint": observed,
        "config_fingerprint": observed_config,
        "layer_warnings": layer_warnings,
        "problems": problems,
    }


def resolve_task(value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = TASKS / value
    candidate = candidate.resolve()
    if not (candidate / "task.toml").is_file():
        raise SystemExit(f"unknown task: {value}")
    return candidate


def write_receipt(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    """Atomically publish one audit receipt without overwriting prior evidence."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit(f"refusing to overwrite audit receipt: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise SystemExit(f"refusing to overwrite audit receipt: {path}") from error
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--task")
    selector.add_argument("--all", action="store_true")
    parser.add_argument("--runtime", default=os.environ.get("AI4AI_DOCKER", "docker"))
    parser.add_argument("--image", help="override image (single-task audit only)")
    parser.add_argument("--skip-digest", action="store_true", help="for a newly built image")
    parser.add_argument(
        "--skip-layer-scan",
        action="store_true",
        help="skip docker-save scan of deleted/lower-layer content (not for release sign-off)",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--receipt",
        type=Path,
        help="atomically write JSON evidence here and refuse an existing path",
    )
    args = parser.parse_args()
    if args.all and args.image:
        parser.error("--image is only valid with --task")
    runtime = shlex.split(args.runtime)
    if not runtime:
        parser.error("--runtime is empty")
    tasks = (
        [resolve_task(args.task)]
        if args.task
        else sorted(path for path in TASKS.iterdir() if (path / "task.toml").is_file())
    )
    reports = [
        audit(
            task,
            runtime,
            image_override=args.image,
            skip_digest=args.skip_digest,
            scan_layers=not args.skip_layer_scan,
        )
        for task in tasks
    ]
    payload = reports if args.all else reports[0]
    if args.receipt:
        write_receipt(args.receipt, payload)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for report in reports:
            print(f"{report['task']}: {report['status']} - {report['image']}")
            for problem in report.get("problems", []):
                print(f"  {problem}")
            for warning in report.get("layer_warnings", []):
                print(f"  warning: {warning}")
    return 0 if all(report["status"] == "clean" for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
