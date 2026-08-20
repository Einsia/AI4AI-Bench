#!/usr/bin/env python3
"""Build and audit release-candidate task images without uploading them.

This driver is intentionally offline for task-image builds.  It verifies every
named build context first, derives the candidate tag from the verified
``build_inputs_sha256``, builds with BuildKit and no network/cache/pull, and
writes an atomic release receipt.  It never logs in, pushes, or invokes a
registry command.

DiGress is the sole exception with a separately built dependency base.  The
base build is networked only when the operator explicitly supplies
``--build-digress-base``; otherwise the already-local base must match the
identity recorded in ``task.toml``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib  # type: ignore[no-redef]

try:
    import yaml
except ModuleNotFoundError as error:  # pragma: no cover - declared dependency
    raise SystemExit("build_images.py requires PyYAML (`pip install PyYAML`)") from error


SCRIPT = Path(__file__).resolve()
DEFAULT_REPO = SCRIPT.parents[1]
HEX_64 = re.compile(r"^[0-9a-f]{64}$")
SENSITIVE_ARGUMENT = re.compile(
    r"(?:PASSWORD|PASSWD|SECRET|TOKEN|CREDENTIAL|API[_-]?KEY|PRIVATE[_-]?KEY)", re.I
)
SECRET_VALUE = re.compile(
    r"https?://[^\s/:]{1,64}:[^\s/@]{6,128}@|"
    r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|"
    r"\b(?:sk-|ghp_|github_pat_|glpat-|hf_)[A-Za-z0-9_-]{16,}",
    re.I,
)
CONFIG_FIELDS = (
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
ARG_DECLARATION = re.compile(r"^\s*ARG\s+([A-Za-z_][A-Za-z0-9_]*)(?:=(.*))?\s*$")
FROM_DECLARATION = re.compile(
    r"^\s*FROM(?:\s+--platform=\S+)?\s+(\S+)(?:\s+AS\s+([A-Za-z0-9_.-]+))?\s*$",
    re.I,
)
VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")
SYNTAX_DIRECTIVE = re.compile(r"^#\s*syntax\s*=\s*(\S+)\s*$", re.I)
PINNED_IMAGE_REFERENCE = re.compile(r"^\S+@sha256:[0-9a-f]{64}$")
REPOSITORY_PREFIX = re.compile(
    r"^[a-z0-9]+(?:[._-][a-z0-9]+)*(?::[0-9]+)?"
    r"(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)*$"
)


class ReleaseBuildError(RuntimeError):
    """A fail-closed release preparation error."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


@contextlib.contextmanager
def exclusive_receipt_root(receipt_root: Path, tasks: list[Path]):
    """Reserve a new evidence root and reject reruns/concurrent writers."""

    receipt_root.mkdir(parents=True, exist_ok=True)
    existing = [receipt_root / "receipt.json"]
    existing.extend(receipt_root / task.name for task in tasks)
    occupied = [str(path) for path in existing if path.exists()]
    if occupied:
        raise ReleaseBuildError(
            "refusing to overwrite existing release evidence: " + ", ".join(occupied)
        )
    lock = receipt_root / ".build-images.lock"
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ReleaseBuildError(f"receipt root is already locked: {lock}") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {"pid": os.getpid(), "host": socket.gethostname(), "started_at": utc_now()},
                    sort_keys=True,
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        yield
    finally:
        lock.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ReleaseBuildError(f"expected a mapping in {path}")
    return value


def resolve_tasks(repo: Path, task_value: str | None, all_tasks: bool) -> list[Path]:
    task_root = repo / "tasks"
    if all_tasks:
        tasks = sorted(path for path in task_root.iterdir() if (path / "task.toml").is_file())
        if not tasks:
            raise ReleaseBuildError(f"no tasks found under {task_root}")
        return tasks
    assert task_value is not None
    candidate = Path(task_value)
    if not candidate.is_absolute():
        candidate = task_root / candidate
    candidate = candidate.resolve()
    if not (candidate / "task.toml").is_file():
        raise ReleaseBuildError(f"unknown task: {task_value}")
    try:
        candidate.relative_to(task_root.resolve())
    except ValueError as error:
        raise ReleaseBuildError(f"task must be under {task_root}: {candidate}") from error
    return [candidate]


def context_names(task: Path) -> list[str]:
    lock_path = task / "environment" / "assets.lock.yaml"
    lock = load_yaml(lock_path)
    names = (lock.get("build_contexts") or {}).get("names")
    if not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names):
        raise ReleaseBuildError(f"{lock_path} has no non-empty build_contexts.names list")
    if len(set(names)) != len(names):
        raise ReleaseBuildError(f"{lock_path} contains duplicate build-context names")
    return names


def declared_build_args(task_config: dict[str, Any]) -> dict[str, str]:
    raw = (task_config.get("environment") or {}).get("build_args") or {}
    if not isinstance(raw, dict):
        raise ReleaseBuildError("environment.build_args must be a table")
    result: dict[str, str] = {}
    for key, value in raw.items():
        key = str(key)
        if SENSITIVE_ARGUMENT.search(key):
            raise ReleaseBuildError(
                f"refusing sensitive build argument {key!r}; use a BuildKit secret explicitly"
            )
        if not isinstance(value, (str, int, float, bool)):
            raise ReleaseBuildError(f"build argument {key!r} is not scalar")
        text = str(value).lower() if isinstance(value, bool) else str(value)
        if SECRET_VALUE.search(text) or any(ord(character) < 32 for character in text):
            raise ReleaseBuildError(
                f"refusing credential-like or control-character value for build argument {key!r}"
            )
        result[key] = text
    return result


def validate_repository_prefix(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    if value != value.strip() or any(character.isspace() for character in value):
        raise ReleaseBuildError("image repository prefix contains whitespace")
    if "@" in value:
        raise ReleaseBuildError("image repository prefix must not contain a digest or credentials")
    if SECRET_VALUE.search(value):
        raise ReleaseBuildError("image repository prefix contains a credential-like value")
    if not REPOSITORY_PREFIX.fullmatch(value):
        raise ReleaseBuildError(
            "invalid image repository prefix; expected a lowercase namespace such as "
            "docker.io/my-org or registry.example:5000/my-org"
        )
    return value


def candidate_tag(
    task_config: dict[str, Any],
    build_inputs_sha256: str,
    repository_prefix: str | None = None,
) -> str:
    if not HEX_64.fullmatch(build_inputs_sha256):
        raise ReleaseBuildError("context verifier returned an invalid build_inputs_sha256")
    environment = task_config.get("environment") or {}
    declared = str(environment.get("image", ""))
    if not declared:
        raise ReleaseBuildError("task.toml has no environment.image")
    # After publication `[environment].image` carries an immutable digest
    # reference, `repo@sha256:...`, whose last path segment contains a colon that
    # is part of the digest rather than a tag separator. Splitting on that colon
    # yields `repo@sha256` and a candidate tag Docker rejects as an invalid
    # reference, so the digest is stripped first and only then is a trailing tag
    # removed.
    declared_repository = declared.split("@", 1)[0]
    if ":" in declared_repository.rsplit("/", 1)[-1]:
        declared_repository = declared_repository.rsplit(":", 1)[0]
    repository = declared_repository
    if repository_prefix:
        repository = f"{repository_prefix}/{declared_repository.rsplit('/', 1)[-1]}"
    schema = str(task_config.get("schema_version", "1.5"))
    # One shared repository holds every task image, so the repository name alone no
    # longer says which task a tag belongs to. When the repository's last segment is
    # not the task name, carry the task name in the tag instead.
    name = str(task_config.get("name") or "")
    leaf = repository.rsplit("/", 1)[-1]
    if name and leaf != name:
        return f"{repository}:{name}-v{schema}-{build_inputs_sha256[:12]}"
    return f"{repository}:v{schema}-{build_inputs_sha256[:12]}"


def source_hash(repo: Path, task: Path) -> str:
    """Call the orchestrator's canonical source identity implementation."""

    module_path = repo / "orchestrator" / "task.py"
    module_name = f"_ai4ai_release_task_{hashlib.sha256(str(repo).encode()).hexdigest()[:12]}"
    previous_path = list(sys.path)
    try:
        sys.path.insert(0, str(repo / "orchestrator"))
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ReleaseBuildError(f"could not import source_hash from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        value = str(module.source_hash(task))
    finally:
        sys.path[:] = previous_path
    if not HEX_64.fullmatch(value):
        raise ReleaseBuildError("canonical source_hash returned an invalid value")
    return value


def task_context_root(context_root: Path, task: Path, all_tasks: bool) -> Path:
    return context_root / task.name if all_tasks else context_root


def run_process(
    argv: list[str],
    *,
    cwd: Path,
    timeout: int,
    env: dict[str, str] | None = None,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    if stdout_path is None or stderr_path is None:
        return subprocess.run(
            argv,
            cwd=cwd,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            env=env,
            check=False,
            stdout=stdout,
            stderr=stderr,
            text=True,
            timeout=timeout,
        )
    return subprocess.CompletedProcess(argv, completed.returncode, "", "")


def verify_contexts(
    *,
    repo: Path,
    task: Path,
    context_root: Path,
    receipt_path: Path,
    timeout: int,
) -> dict[str, Any]:
    verifier = repo / "tools" / "verify_build_contexts.py"
    if not verifier.is_file():
        raise ReleaseBuildError(f"required context verifier is missing: {verifier}")
    argv = [
        sys.executable,
        str(verifier),
        "--task",
        task.name,
        "--context-root",
        str(context_root),
        "--receipt",
        str(receipt_path),
        "--json",
    ]
    completed = run_process(argv, cwd=repo, timeout=timeout)
    if not receipt_path.is_file():
        detail = (completed.stderr or completed.stdout).strip()[:1000]
        raise ReleaseBuildError(
            f"context verifier wrote no receipt (exit {completed.returncode}): {detail}"
        )
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, ValueError) as error:
        raise ReleaseBuildError(f"invalid context receipt {receipt_path}: {error}") from error
    records = receipt.get("tasks")
    matching = (
        [
            record
            for record in records
            if isinstance(record, dict) and record.get("task") == task.name
        ]
        if isinstance(records, list)
        else []
    )
    if completed.returncode != 0 or receipt.get("status") != "ok" or len(matching) != 1:
        raise ReleaseBuildError(
            "context verification failed: "
            f"exit={completed.returncode}, status={receipt.get('status')!r}, "
            f"task_records={len(matching)}"
        )
    record = matching[0]
    if record.get("status") != "ok":
        raise ReleaseBuildError(f"context verification failed for {task.name}")
    digest = str(record.get("build_inputs_sha256", ""))
    if not HEX_64.fullmatch(digest):
        raise ReleaseBuildError("context receipt has no valid build_inputs_sha256")
    return {
        "argv": argv,
        "exit_code": completed.returncode,
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "status": "ok",
        "task_record": record,
    }


def task_build_argv(
    *,
    runtime: list[str],
    repo: Path,
    task: Path,
    contexts_root: Path,
    contexts: list[str],
    build_args: dict[str, str],
    tag: str,
) -> list[str]:
    argv = [
        *runtime,
        "build",
        "--network=none",
        "--no-cache",
        "--pull=false",
        "--platform=linux/amd64",
        "--progress=plain",
    ]
    for name in sorted(contexts):
        argv.extend(["--build-context", f"{name}={contexts_root / name}"])
    for name, value in sorted(build_args.items()):
        if SENSITIVE_ARGUMENT.search(name):
            raise ReleaseBuildError(f"refusing sensitive build argument {name!r}")
        argv.extend(["--build-arg", f"{name}={value}"])
    argv.extend(
        [
            "-f",
            str(task / "environment" / "Dockerfile"),
            "-t",
            tag,
            str(repo),
        ]
    )
    return argv


def dockerfile_frontend(path: Path) -> dict[str, Any]:
    """Return the frontend contract and reject mutable external references."""

    first_line = path.read_text().splitlines()[0] if path.stat().st_size else ""
    match = SYNTAX_DIRECTIVE.fullmatch(first_line)
    if match is None:
        return {
            "mode": "builtin",
            "reference": "dockerfile.v0",
            "offline": True,
            "note": "no syntax directive; the BuildKit daemon's bundled frontend is used",
        }
    reference = match.group(1)
    if not PINNED_IMAGE_REFERENCE.fullmatch(reference):
        raise ReleaseBuildError(
            f"external Dockerfile frontend must be digest-pinned, got {reference!r} in {path}"
        )
    return {
        "mode": "external-pinned",
        "reference": reference,
        "digest": reference.rsplit("@", 1)[1],
        "offline": "only after explicit prefetch into this builder",
        "note": (
            "frontend resolution precedes RUN --network=none; a clean builder must explicitly "
            "prefetch this exact digest before the task build"
        ),
    }


def resolve_docker_variables(token: str, values: dict[str, str], dockerfile: Path) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        if name not in values:
            raise ReleaseBuildError(f"unresolved FROM variable {name!r} in {dockerfile}")
        return values[name]

    return VARIABLE.sub(replace, token)


def dockerfile_base_references(task: Path, build_args: dict[str, str]) -> list[str]:
    """Resolve each external ``FROM`` without asking Docker to pull it."""

    dockerfile = task / "environment" / "Dockerfile"
    defaults: dict[str, str] = {}
    stages: set[str] = set()
    references: list[str] = []
    for raw_line in dockerfile.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        argument = ARG_DECLARATION.fullmatch(line)
        if argument:
            value = argument.group(2)
            if value is not None:
                try:
                    parsed = shlex.split(value, comments=False)
                except ValueError as error:
                    raise ReleaseBuildError(
                        f"invalid ARG declaration in {dockerfile}: {line}"
                    ) from error
                if len(parsed) != 1:
                    raise ReleaseBuildError(f"invalid ARG declaration in {dockerfile}: {line}")
                defaults[argument.group(1)] = parsed[0]
            continue
        declaration = FROM_DECLARATION.fullmatch(line)
        if not declaration:
            continue
        token, stage = declaration.groups()
        values = {**defaults, **build_args}
        resolved = resolve_docker_variables(token, values, dockerfile)
        if "$" in resolved:
            raise ReleaseBuildError(f"unresolved FROM expression {resolved!r} in {dockerfile}")
        if resolved.lower() != "scratch" and resolved.lower() not in stages:
            references.append(resolved)
        if stage:
            stages.add(stage.lower())
    if not references:
        raise ReleaseBuildError(f"{dockerfile} declares no external base image")
    return references


def require_local_base_images(
    *,
    runtime: list[str],
    repo: Path,
    task: Path,
    build_args: dict[str, str],
    timeout: int,
) -> list[dict[str, Any]]:
    """Fail before Docker can implicitly pull a missing ``FROM`` image."""

    records: list[dict[str, Any]] = []
    for reference in dockerfile_base_references(task, build_args):
        if "@sha256:" not in reference and task.name != "digress_qm9_graph_diffusion":
            raise ReleaseBuildError(
                "task base image is not digest-pinned and will not be used for release: "
                f"{reference}"
            )
        try:
            identity = inspect_image(runtime, reference, cwd=repo, timeout=min(timeout, 300))
        except ReleaseBuildError as error:
            raise ReleaseBuildError(
                "required base image is not present locally; refusing an implicit pull: "
                f"{reference}: {error}"
            ) from error
        records.append({"reference": reference, **identity})
    return records


def digress_base_argv(
    *,
    runtime: list[str],
    repo: Path,
    task: Path,
    image: str,
    apt_proxy_env: str | None,
) -> list[str]:
    argv = [
        *runtime,
        "build",
        "--network=default",
        "--no-cache",
        "--pull=false",
        "--platform=linux/amd64",
        "--progress=plain",
    ]
    if apt_proxy_env:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", apt_proxy_env):
            raise ReleaseBuildError("--digress-apt-proxy-env is not an environment name")
        argv.extend(["--secret", f"id=apt_proxy,env={apt_proxy_env}"])
    argv.extend(
        [
            "-f",
            str(task / "environment" / "Dockerfile.base"),
            "-t",
            image,
            str(repo),
        ]
    )
    return argv


def inspect_image(runtime: list[str], image: str, *, cwd: Path, timeout: int) -> dict[str, Any]:
    completed = run_process([*runtime, "image", "inspect", image], cwd=cwd, timeout=timeout)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[:1000]
        raise ReleaseBuildError(f"could not inspect local image {image}: {detail}")
    try:
        metadata = json.loads(completed.stdout)[0]
    except (IndexError, TypeError, ValueError) as error:
        raise ReleaseBuildError(f"runtime returned invalid inspect data for {image}") from error
    layers = (metadata.get("RootFS") or {}).get("Layers") or []
    if not isinstance(layers, list) or not layers:
        raise ReleaseBuildError(f"image {image} has no RootFS layer identity")
    layer_digest = "layers:" + hashlib.sha256("\n".join(layers).encode()).hexdigest()
    config = metadata.get("Config") or {}
    canonical_config = {
        "os": metadata.get("Os"),
        "architecture": metadata.get("Architecture"),
        "variant": metadata.get("Variant"),
        "config": {field: config.get(field) for field in CONFIG_FIELDS},
    }
    config_digest = (
        "config:"
        + hashlib.sha256(
            json.dumps(canonical_config, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()
    )
    return {
        "image_id": metadata.get("Id"),
        "repo_digests": metadata.get("RepoDigests") or [],
        "platform": {
            "os": metadata.get("Os"),
            "architecture": metadata.get("Architecture"),
            "variant": metadata.get("Variant"),
        },
        "layer_fingerprint": layer_digest,
        "config_fingerprint": config_digest,
    }


def inspect_candidate_if_present(
    runtime: list[str], image: str, *, cwd: Path, timeout: int
) -> dict[str, Any] | None:
    """Return an existing candidate identity without allowing an implicit pull."""

    completed = run_process(
        [*runtime, "image", "inspect", image], cwd=cwd, timeout=min(timeout, 300)
    )
    if completed.returncode == 0:
        return inspect_image(runtime, image, cwd=cwd, timeout=min(timeout, 300))
    detail = (completed.stderr or completed.stdout).strip()
    lowered = detail.lower()
    if "no such image" in lowered or "not found" in lowered:
        return None
    raise ReleaseBuildError(f"could not determine whether candidate tag exists: {detail[:1000]}")


def ensure_digress_base(
    *,
    runtime: list[str],
    repo: Path,
    task: Path,
    task_config: dict[str, Any],
    image: str,
    build: bool,
    apt_proxy_env: str | None,
    timeout: int,
    log_root: Path,
    plan: bool,
) -> dict[str, Any]:
    environment = task_config.get("environment") or {}
    expected_layers = str(environment.get("base_image_digest", ""))
    expected_config = str(environment.get("base_image_config_digest", ""))
    if not expected_layers.startswith("layers:"):
        raise ReleaseBuildError("DiGress task.toml has no portable base_image_digest")
    if not expected_config.startswith("config:"):
        raise ReleaseBuildError("DiGress task.toml has no portable base_image_config_digest")
    result: dict[str, Any] = {
        "image": image,
        "expected_layer_fingerprint": expected_layers,
        "expected_config_fingerprint": expected_config,
        "explicit_build_requested": build,
        "status": "pending",
        "dockerfile_frontend": dockerfile_frontend(task / "environment" / "Dockerfile.base"),
    }
    if build:
        argv = digress_base_argv(
            runtime=runtime,
            repo=repo,
            task=task,
            image=image,
            apt_proxy_env=apt_proxy_env,
        )
        result["build_argv"] = argv
        if plan:
            result["status"] = "planned"
            return result
        started = time.monotonic()
        result["started_at"] = utc_now()
        try:
            completed = run_process(
                argv,
                cwd=repo,
                timeout=timeout,
                env={**os.environ, "DOCKER_BUILDKIT": "1"},
                stdout_path=log_root / "digress-base.stdout.log",
                stderr_path=log_root / "digress-base.stderr.log",
            )
        except subprocess.TimeoutExpired:
            result["finished_at"] = utc_now()
            result["duration_sec"] = round(time.monotonic() - started, 3)
            result["status"] = "timeout"
            result["problem"] = f"DiGress base build exceeded {timeout} seconds"
            return result
        result["finished_at"] = utc_now()
        result["duration_sec"] = round(time.monotonic() - started, 3)
        result["exit_code"] = completed.returncode
        result["stdout_log"] = str(log_root / "digress-base.stdout.log")
        result["stderr_log"] = str(log_root / "digress-base.stderr.log")
        if completed.returncode != 0:
            result["status"] = "build-failed"
            return result
    elif plan:
        # Planning is still a release preflight: detect whether the fixed base is
        # actually present rather than promising a build that cannot start.
        result["inspection_during_plan"] = True
    try:
        identity = inspect_image(runtime, image, cwd=repo, timeout=min(timeout, 300))
    except ReleaseBuildError as error:
        result["status"] = "inspection-failed"
        result["problem"] = str(error)
        return result
    result["observed"] = identity
    mismatches = []
    if identity["layer_fingerprint"] != expected_layers:
        mismatches.append(f"layers {identity['layer_fingerprint']} != expected {expected_layers}")
    if identity["config_fingerprint"] != expected_config:
        mismatches.append(f"config {identity['config_fingerprint']} != expected {expected_config}")
    if mismatches:
        result["status"] = "identity-mismatch"
        result["problem"] = f"DiGress base {image} identity mismatch: {'; '.join(mismatches)}"
        return result
    result["status"] = "ok"
    return result


def run_audit(
    *, runtime: list[str], repo: Path, task: Path, image: str, timeout: int
) -> dict[str, Any]:
    argv = [
        sys.executable,
        str(repo / "tools" / "audit_images.py"),
        "--task",
        task.name,
        "--runtime",
        shlex.join(runtime),
        "--image",
        image,
        "--skip-digest",
        "--json",
    ]
    try:
        completed = run_process(argv, cwd=repo, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {
            "argv": argv,
            "exit_code": None,
            "status": "timeout",
            "problem": f"release image audit exceeded {timeout} seconds",
        }
    try:
        report = json.loads(completed.stdout)
    except ValueError:
        detail = (completed.stderr or completed.stdout).strip()[:1000]
        return {
            "argv": argv,
            "exit_code": completed.returncode,
            "status": "invalid-output",
            "problem": f"image audit returned invalid JSON: {detail}",
        }
    return {
        "argv": argv,
        "exit_code": completed.returncode,
        "status": report.get("status"),
        "report": report,
    }


def require_audit_identity(audit: dict[str, Any], expected: dict[str, Any]) -> None:
    """Bind a clean audit to the image inspected immediately after the build."""

    report = audit.get("report")
    if not isinstance(report, dict):
        raise ReleaseBuildError("release image audit has no structured report")
    comparisons = {
        "image_id": expected.get("image_id"),
        "layer_fingerprint": expected.get("layer_fingerprint"),
        "config_fingerprint": expected.get("config_fingerprint"),
    }
    mismatches = [
        f"{field}={report.get(field)!r} != {value!r}"
        for field, value in comparisons.items()
        if not value or report.get(field) != value
    ]
    if mismatches:
        raise ReleaseBuildError("release image audit identity mismatch: " + "; ".join(mismatches))


def new_task_receipt(task: Path) -> dict[str, Any]:
    return {
        "task": task.name,
        "status": "pending",
        "started_at": utc_now(),
        "problems": [],
    }


def prepare_task(
    *,
    repo: Path,
    task: Path,
    context_root: Path,
    receipt_root: Path,
    runtime: list[str],
    verify_timeout: int,
    build_timeout: int,
    audit_timeout: int,
    audit: bool,
    plan: bool,
    build_digress_base: bool,
    digress_base_image: str | None,
    digress_apt_proxy_env: str | None,
    repository_prefix: str | None,
) -> dict[str, Any]:
    receipt = new_task_receipt(task)
    task_receipt_root = receipt_root / task.name
    context_receipt = task_receipt_root / "build-contexts.json"
    try:
        config = load_toml(task / "task.toml")
        names = context_names(task)
        source = source_hash(repo, task)
        receipt["source_sha256"] = source
        receipt["dockerfile_frontend"] = dockerfile_frontend(task / "environment" / "Dockerfile")
        try:
            receipt["context_verification"] = verify_contexts(
                repo=repo,
                task=task,
                context_root=context_root,
                receipt_path=context_receipt,
                timeout=verify_timeout,
            )
        except (OSError, ReleaseBuildError, subprocess.TimeoutExpired) as error:
            receipt["context_verification"] = {
                "status": "failed",
                "receipt_path": str(context_receipt),
                "problem": str(error),
            }
            raise
        verified = receipt["context_verification"]["task_record"]
        verified_source = str(
            (verified.get("repository_inputs") or {}).get("task_source_sha256", "")
        )
        if verified_source != source:
            raise ReleaseBuildError(
                "context receipt source identity disagrees with the canonical working-tree hash"
            )
        build_inputs = str(verified["build_inputs_sha256"])
        tag = candidate_tag(config, build_inputs, repository_prefix)
        receipt["build_inputs_sha256"] = build_inputs
        receipt["candidate_tag"] = tag
        receipt["repository_prefix_override"] = repository_prefix
        existing_candidate = inspect_candidate_if_present(
            runtime, tag, cwd=repo, timeout=build_timeout
        )
        receipt["candidate_tag_preflight"] = {
            "status": "exists" if existing_candidate else "available",
            "observed": existing_candidate,
        }
        if existing_candidate and not plan:
            raise ReleaseBuildError(
                f"candidate tag already exists and will not be overwritten: {tag}"
            )
        build_args = declared_build_args(config)

        if task.name == "digress_qm9_graph_diffusion":
            base = digress_base_image or build_args.get("DIGRESS_BASE_IMAGE")
            if not base:
                raise ReleaseBuildError("DiGress has no declared or requested base image")
            build_args["DIGRESS_BASE_IMAGE"] = base
            receipt["digress_base"] = ensure_digress_base(
                runtime=runtime,
                repo=repo,
                task=task,
                task_config=config,
                image=base,
                build=build_digress_base,
                apt_proxy_env=digress_apt_proxy_env,
                timeout=build_timeout,
                log_root=task_receipt_root,
                plan=plan,
            )
            accepted_base_statuses = {"ok"}
            if plan and build_digress_base:
                accepted_base_statuses.add("planned")
            if receipt["digress_base"]["status"] not in accepted_base_statuses:
                raise ReleaseBuildError(
                    "DiGress base preflight failed: "
                    f"{receipt['digress_base'].get('problem', receipt['digress_base']['status'])}"
                )

        if plan and task.name == "digress_qm9_graph_diffusion" and build_digress_base:
            receipt["base_images"] = {
                "status": "planned-after-explicit-base-build",
                "references": dockerfile_base_references(task, build_args),
            }
        else:
            receipt["base_images"] = {
                "status": "ok",
                "records": require_local_base_images(
                    runtime=runtime,
                    repo=repo,
                    task=task,
                    build_args=build_args,
                    timeout=build_timeout,
                ),
            }

        argv = task_build_argv(
            runtime=runtime,
            repo=repo,
            task=task,
            contexts_root=context_root,
            contexts=names,
            build_args=build_args,
            tag=tag,
        )
        receipt["build"] = {
            "argv": argv,
            "run_network": "none",
            "no_cache": True,
            "base_pull": False,
            "buildkit": True,
            "frontend_resolution": receipt["dockerfile_frontend"],
            "status": "planned" if plan else "pending",
        }
        if plan:
            receipt["status"] = "planned"
            return receipt

        started = time.monotonic()
        receipt["build"]["started_at"] = utc_now()
        try:
            completed = run_process(
                argv,
                cwd=repo,
                timeout=build_timeout,
                env={**os.environ, "DOCKER_BUILDKIT": "1"},
                stdout_path=task_receipt_root / "build.stdout.log",
                stderr_path=task_receipt_root / "build.stderr.log",
            )
        except subprocess.TimeoutExpired as error:
            receipt["build"].update(
                {
                    "finished_at": utc_now(),
                    "duration_sec": round(time.monotonic() - started, 3),
                    "status": "timeout",
                    "stdout_log": str(task_receipt_root / "build.stdout.log"),
                    "stderr_log": str(task_receipt_root / "build.stderr.log"),
                }
            )
            raise ReleaseBuildError(f"task image build exceeded {build_timeout} seconds") from error
        receipt["build"].update(
            {
                "finished_at": utc_now(),
                "duration_sec": round(time.monotonic() - started, 3),
                "exit_code": completed.returncode,
                "stdout_log": str(task_receipt_root / "build.stdout.log"),
                "stderr_log": str(task_receipt_root / "build.stderr.log"),
                "status": "ok" if completed.returncode == 0 else "failed",
            }
        )
        if completed.returncode != 0:
            raise ReleaseBuildError(f"task image build failed with exit {completed.returncode}")

        receipt["image_identity"] = inspect_image(
            runtime, tag, cwd=repo, timeout=min(build_timeout, 300)
        )
        if audit:
            receipt["audit"] = run_audit(
                runtime=runtime,
                repo=repo,
                task=task,
                image=tag,
                timeout=audit_timeout,
            )
            if receipt["audit"]["exit_code"] != 0 or receipt["audit"]["status"] != "clean":
                raise ReleaseBuildError("release image audit did not pass cleanly")
            require_audit_identity(receipt["audit"], receipt["image_identity"])
        else:
            receipt["audit"] = {"status": "not-requested"}
        receipt["status"] = "ok"
    except (OSError, ReleaseBuildError, subprocess.TimeoutExpired, ValueError) as error:
        receipt["status"] = "failed"
        receipt["problems"].append(str(error))
    except Exception as error:  # pragma: no cover - last-resort receipt preservation
        receipt["status"] = "failed"
        receipt["problems"].append(f"unexpected {type(error).__name__}: {error}")
    finally:
        receipt["finished_at"] = utc_now()
        atomic_json(task_receipt_root / "receipt.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--task")
    selector.add_argument("--all", action="store_true")
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument(
        "--context-root",
        type=Path,
        default=os.environ.get("AI4AI_BUILD_CONTEXT_ROOT"),
        help="staged build-context root (env: AI4AI_BUILD_CONTEXT_ROOT)",
    )
    parser.add_argument(
        "--receipt-root",
        type=Path,
        default=os.environ.get("AI4AI_IMAGE_RECEIPT_ROOT"),
        help="new no-clobber evidence root (env: AI4AI_IMAGE_RECEIPT_ROOT)",
    )
    parser.add_argument("--runtime", default=os.environ.get("AI4AI_DOCKER", "docker"))
    parser.add_argument(
        "--repository-prefix",
        default=os.environ.get("AI4AI_IMAGE_REPOSITORY_PREFIX"),
        help=(
            "optional registry/namespace replacing the declared ai4ai namespace "
            "(env: AI4AI_IMAGE_REPOSITORY_PREFIX)"
        ),
    )
    parser.add_argument("--plan", action="store_true", help="verify and write build argv only")
    parser.add_argument("--audit", action="store_true", help="run the full lower-layer image audit")
    parser.add_argument("--verify-timeout-sec", type=int, default=1800)
    parser.add_argument("--build-timeout-sec", type=int, default=14400)
    parser.add_argument("--audit-timeout-sec", type=int, default=7200)
    parser.add_argument(
        "--build-digress-base",
        action="store_true",
        help="explicitly perform the networked DiGress dependency-base build",
    )
    parser.add_argument("--digress-base-image")
    parser.add_argument(
        "--digress-apt-proxy-env",
        help="environment-variable name exposed to BuildKit as apt_proxy (value is never argv)",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.context_root is None:
        parser.error("--context-root or AI4AI_BUILD_CONTEXT_ROOT is required")
    if args.receipt_root is None:
        parser.error("--receipt-root or AI4AI_IMAGE_RECEIPT_ROOT is required")

    repo = args.repo.resolve()
    context_root = args.context_root.resolve()
    receipt_root = args.receipt_root.resolve()
    runtime = shlex.split(args.runtime)
    if not runtime:
        parser.error("--runtime is empty")
    if SECRET_VALUE.search("\n".join(runtime)):
        parser.error("--runtime contains a credential-like value that cannot enter a receipt")
    try:
        repository_prefix = validate_repository_prefix(args.repository_prefix)
    except ReleaseBuildError as error:
        parser.error(str(error))
    if args.digress_apt_proxy_env and not args.build_digress_base:
        parser.error("--digress-apt-proxy-env requires --build-digress-base")
    if args.build_digress_base and args.task not in (None, "digress_qm9_graph_diffusion"):
        parser.error("--build-digress-base is only relevant to DiGress or --all")

    try:
        tasks = resolve_tasks(repo, args.task, args.all)
    except (OSError, ReleaseBuildError) as error:
        parser.error(str(error))

    try:
        reservation = exclusive_receipt_root(receipt_root, tasks)
        reservation.__enter__()
    except ReleaseBuildError as error:
        parser.error(str(error))
    try:
        overall: dict[str, Any] = {
            "schema_version": 1,
            "kind": "ai4ai-image-release-preparation",
            "generated_at": utc_now(),
            "repository": str(repo),
            "context_root": str(context_root),
            "receipt_root": str(receipt_root),
            "host": socket.gethostname(),
            "runtime": runtime,
            "repository_prefix_override": repository_prefix,
            "upload_performed": False,
            "plan": args.plan,
            "audit_requested": args.audit,
            "tasks": [],
            "status": "pending",
        }
        receipt_path = receipt_root / "receipt.json"
        atomic_json(receipt_path, overall)
        for task in tasks:
            task_root = task_context_root(context_root, task, args.all)
            result = prepare_task(
                repo=repo,
                task=task,
                context_root=task_root,
                receipt_root=receipt_root,
                runtime=runtime,
                verify_timeout=args.verify_timeout_sec,
                build_timeout=args.build_timeout_sec,
                audit_timeout=args.audit_timeout_sec,
                audit=args.audit,
                plan=args.plan,
                build_digress_base=args.build_digress_base,
                digress_base_image=args.digress_base_image,
                digress_apt_proxy_env=args.digress_apt_proxy_env,
                repository_prefix=repository_prefix,
            )
            overall["tasks"].append(result)
            overall["updated_at"] = utc_now()
            atomic_json(receipt_path, overall)
        expected = "planned" if args.plan else "ok"
        overall["status"] = (
            expected
            if all(record["status"] == expected for record in overall["tasks"])
            else "failed"
        )
        overall["finished_at"] = utc_now()
        atomic_json(receipt_path, overall)
    finally:
        reservation.__exit__(None, None, None)
    if args.json:
        print(json.dumps(overall, indent=2, sort_keys=True))
    else:
        print(f"{overall['status']}: {receipt_path}")
        for record in overall["tasks"]:
            print(f"  {record['task']}: {record['status']}")
            for problem in record.get("problems", []):
                print(f"    {problem}")
    return 0 if overall["status"] == expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
