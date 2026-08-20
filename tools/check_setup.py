#!/usr/bin/env python3
"""Check whether this host is ready to run one AI4AI task.

The command is deliberately read-only. It does not pull images, download assets, or
start a container; ``tools/smoke.sh`` performs the destructive-free runtime probe.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "orchestrator"))
sys.path.insert(0, str(REPO / "tools"))

import agent as agent_runtime  # noqa: E402
import host_contract as host_contract_runtime  # noqa: E402
import task as task_runtime  # noqa: E402

MINIMUM_PYTHON = (3, 10)
SECRET_MARKERS = ("TOKEN", "KEY", "PASSWORD", "SECRET", "CREDENTIAL")


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


def _redact(value: str) -> str:
    redacted = value
    for key, secret in os.environ.items():
        if len(secret) >= 4 and any(marker in key.upper() for marker in SECRET_MARKERS):
            redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def _run(argv: list[str], *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return subprocess.CompletedProcess(argv, 127, "", str(error))


def _diagnostic(result: subprocess.CompletedProcess[str]) -> str:
    text = (result.stderr or result.stdout or f"exit {result.returncode}").strip()
    return _redact(text.splitlines()[-1][:400])


def _python_check() -> Check:
    version = sys.version_info[:3]
    if version < MINIMUM_PYTHON:
        return Check(
            "python",
            "fail",
            f"Python {version[0]}.{version[1]} is too old; Python 3.10+ is required",
        )
    missing = [
        module
        for module in ("yaml", "huggingface_hub")
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        return Check(
            "python",
            "fail",
            "missing asset-preparation dependencies: " + ", ".join(missing),
            {"install": "python -m pip install -e '.[assets]'"},
        )
    return Check(
        "python",
        "pass",
        f"Python {version[0]}.{version[1]}.{version[2]} and asset dependencies are available",
    )


def _docker_command() -> list[str]:
    try:
        command = shlex.split(os.environ.get("AI4AI_DOCKER", "docker"))
    except ValueError:
        return []
    return command


def _docker_check(command: list[str]) -> Check:
    if not command:
        return Check("docker", "fail", "AI4AI_DOCKER resolves to an empty or invalid command")
    if shutil.which(command[0]) is None:
        return Check("docker", "fail", f"container runtime is not on PATH: {command[0]}")
    result = _run([*command, "version", "--format", "{{.Server.Version}}"])
    if result.returncode != 0:
        return Check("docker", "fail", f"Docker daemon is unavailable: {_diagnostic(result)}")
    version = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "unknown"
    return Check("docker", "pass", f"Docker daemon {version} is reachable")


def _nvidia_runtime_check(command: list[str], docker_ready: bool) -> Check:
    if not docker_ready:
        return Check("nvidia_runtime", "fail", "cannot inspect NVIDIA runtime without Docker")
    result = _run([*command, "info", "--format", "{{json .Runtimes}}"])
    if result.returncode != 0:
        return Check(
            "nvidia_runtime", "fail", f"cannot inspect Docker runtimes: {_diagnostic(result)}"
        )
    try:
        runtimes = json.loads(result.stdout)
    except (TypeError, ValueError):
        runtimes = {}
    if not isinstance(runtimes, dict) or "nvidia" not in runtimes:
        return Check(
            "nvidia_runtime",
            "fail",
            "Docker does not report an NVIDIA runtime; install NVIDIA Container Toolkit",
        )
    return Check("nvidia_runtime", "pass", "Docker reports the NVIDIA container runtime")


def _gpu_check(gpu: int, config: dict[str, Any], mode: str) -> Check:
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return Check("gpu", "fail", "nvidia-smi is not on PATH")
    result = _run(
        [
            binary,
            "--query-gpu=index,name,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if result.returncode != 0:
        return Check("gpu", "fail", f"nvidia-smi failed: {_diagnostic(result)}")
    selected: tuple[str, int, int] | None = None
    for line in result.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 4:
            continue
        try:
            index, free_mib, total_mib = int(fields[0]), int(fields[2]), int(fields[3])
        except ValueError:
            continue
        if index == gpu:
            selected = (fields[1], free_mib, total_mib)
            break
    if selected is None:
        return Check("gpu", "fail", f"GPU index {gpu} was not reported by nvidia-smi")

    name, free_mib, total_mib = selected
    environment = config.get("environment", {})
    allowed = [str(item) for item in environment.get("gpu_types", [])]
    required_mib = (
        config.get("x-ai4ai", {}).get("gpu", {}).get("peak_memory_mib")
    )
    problems = []
    if allowed and not any(item.casefold() in name.casefold() for item in allowed):
        problems.append(f"{name} is not one of the declared types: {', '.join(allowed)}")
    if isinstance(required_mib, int) and free_mib < required_mib:
        problems.append(f"{free_mib} MiB free is below the declared {required_mib} MiB peak")
    if problems:
        status = "fail" if mode == "official" else "warn"
        return Check(
            "gpu",
            status,
            "; ".join(problems),
            {"index": gpu, "name": name, "free_mib": free_mib, "total_mib": total_mib},
        )
    return Check(
        "gpu",
        "pass",
        f"GPU {gpu} ({name}) has {free_mib} MiB free",
        {"index": gpu, "name": name, "free_mib": free_mib, "total_mib": total_mib},
    )


def _nearest_existing(path: Path) -> Path:
    candidate = path.expanduser().absolute()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _disk_check(assets: Path, root: Path | None, config: dict[str, Any]) -> Check:
    target = root if root is not None else assets
    location = _nearest_existing(target)
    try:
        free = shutil.disk_usage(location).free
    except OSError as error:
        return Check("disk", "fail", f"cannot inspect free space: {_redact(str(error))}")
    required_mb = config.get("environment", {}).get("storage_mb", 0)
    required = int(required_mb) * 1024 * 1024 if isinstance(required_mb, int) else 0
    free_gib = free / 1024**3
    if required and free < required:
        return Check(
            "disk",
            "fail",
            f"{free_gib:.1f} GiB free is below the task's "
            f"{required / 1024**3:.1f} GiB output budget",
        )
    label = (
        "output filesystem"
        if root is not None
        else "asset filesystem (used as output estimate)"
    )
    return Check("disk", "pass", f"{free_gib:.1f} GiB is free on the {label}")


def _source_check(task: Path, config: dict[str, Any], mode: str) -> Check:
    """Verify both task-image source identity and the host orchestration contract."""

    expected = str(config.get("environment", {}).get("source_sha256", ""))
    try:
        observed = task_runtime.source_hash(task)
        contract = host_contract_runtime.verify_contract(
            task,
            mode="strict" if mode == "official" else "warn",
        )
        if mode == "official":
            host_contract_runtime.require_contract(contract)
    except (OSError, task_runtime.ContainerError, host_contract_runtime.HostContractError) as error:
        return Check(
            "source_contract",
            "fail" if mode == "official" else "warn",
            _redact(str(error)),
        )

    problems = []
    if not expected:
        problems.append("task declaration has no image-source identity")
    elif observed != expected:
        problems.append("working task source differs from the published image source")
    if contract.get("status") != "match":
        problems.append(
            "host benchmark contract is " + str(contract.get("status", "unavailable"))
        )
    if problems:
        return Check(
            "source_contract",
            "fail" if mode == "official" else "warn",
            "; ".join(problems),
            {
                "image_source_match": bool(expected and observed == expected),
                "host_contract_status": contract.get("status"),
            },
        )
    return Check(
        "source_contract",
        "pass",
        "task source and host benchmark contract match the release identities",
    )


def _image_check(
    command: list[str],
    docker_ready: bool,
    image: str,
    config: dict[str, Any],
    mode: str,
) -> Check:
    if not docker_ready:
        return Check("image", "fail", "cannot inspect the task image without Docker")
    local = _run([*command, "image", "inspect", image, "--format", "{{.Id}}"])
    if local.returncode == 0:
        try:
            identity = task_runtime.image_identity(image, runtime=shlex.join(command))
            task_runtime.require_image_digest(
                image,
                str(config.get("environment", {}).get("digest", "")),
                str(config.get("environment", {}).get("config_digest", "")),
                runtime=shlex.join(command),
                identity=identity,
            )
        except (OSError, task_runtime.ContainerError) as error:
            return Check(
                "image",
                "fail" if mode == "official" else "warn",
                "local image is available but its release identity does not match: "
                + _redact(str(error)),
            )
        return Check(
            "image",
            "pass",
            "the local task image matches its published layer and execution-config identities",
        )
    remote = _run([*command, "manifest", "inspect", image], timeout=60)
    if remote.returncode == 0:
        if mode == "official":
            return Check(
                "image",
                "fail",
                "the task image is available in the registry but must be pulled before "
                "official layer/config verification",
            )
        return Check("image", "pass", "the immutable task image is available to pull")
    return Check(
        "image",
        "fail",
        "task image is neither local nor readable from its registry: " + _diagnostic(remote),
    )


def _assets_check(task: Path, assets: Path, include_hashes: bool) -> Check:
    if importlib.util.find_spec("yaml") is None:
        return Check(
            "assets",
            "fail",
            "cannot verify assets because PyYAML is not installed",
            {"install": "python -m pip install -e '.[assets]'"},
        )
    verify_assets = importlib.import_module("verify_assets")
    report = verify_assets.verify_task(task, assets, include_hashes=include_hashes)
    invalid = [row["alias"] for row in report["aliases"] if row["status"] != "ok"]
    if invalid:
        return Check(
            "assets",
            "fail",
            f"{len(invalid)} required asset aliases are missing or invalid",
            {"invalid_aliases": invalid, "hashes_checked": include_hashes},
        )
    suffix = " with exact hashes" if include_hashes else " (size/count checks)"
    return Check(
        "assets",
        "pass",
        f"all {len(report['aliases'])} required aliases passed{suffix}",
        {"hashes_checked": include_hashes},
    )


def _agent_checks(name: str) -> tuple[Check, Check]:
    try:
        spec = agent_runtime.resolve(name)
    except agent_runtime.AgentError as error:
        binary_check = Check("agent_cli", "fail", _redact(str(error)))
        try:
            metadata = agent_runtime.resolve_metadata(name)
        except agent_runtime.AgentError as metadata_error:
            return binary_check, Check("api", "fail", _redact(str(metadata_error)))
        spec = metadata
    else:
        result = _run([str(spec.binary), "--version"])
        if result.returncode != 0:
            binary_check = Check(
                "agent_cli", "fail", f"{name} --version failed: {_diagnostic(result)}"
            )
        else:
            output = (result.stdout or result.stderr).strip().splitlines()
            version = _redact(output[0][:200]) if output else "version unknown"
            binary_check = Check("agent_cli", "pass", f"{name} is runnable ({version})")

    try:
        api_key = spec.api_key()
    except agent_runtime.AgentError as error:
        return binary_check, Check("api", "fail", _redact(str(error)))
    if not api_key:
        variables = (
            "ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN"
            if name == "claude"
            else spec.api_key_env
        )
        return binary_check, Check("api", "fail", f"{variables} is not set")
    endpoint = spec.endpoint_identity()
    return binary_check, Check(
        "api",
        "pass",
        f"credentials are set for {endpoint['host']} (value not displayed)",
    )


def run_checks(
    *,
    task: Path,
    assets: Path,
    root: Path | None,
    gpu: int,
    agent_name: str,
    mode: str,
    hash_assets: bool,
) -> list[Check]:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
        import tomli as tomllib  # type: ignore[no-redef]

    with (task / "task.toml").open("rb") as stream:
        config = tomllib.load(stream)
    image = str(config.get("environment", {}).get("image", ""))
    docker_command = _docker_command()
    docker = _docker_check(docker_command)
    agent_cli, api = _agent_checks(agent_name)
    return [
        _python_check(),
        docker,
        _nvidia_runtime_check(docker_command, docker.status == "pass"),
        _gpu_check(gpu, config, mode),
        _disk_check(assets, root, config),
        _source_check(task, config, mode),
        _image_check(docker_command, docker.status == "pass", image, config, mode),
        _assets_check(task, assets, hash_assets or mode == "official"),
        agent_cli,
        api,
    ]


def _resolve_task(value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        direct = (REPO / candidate).resolve()
        candidate = direct if (direct / "task.toml").is_file() else REPO / "tasks" / value
    candidate = candidate.resolve()
    if not (candidate / "task.toml").is_file():
        raise ValueError(f"unknown task: {value}")
    return candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="task id or task directory")
    parser.add_argument("--assets", type=Path, required=True, help="one task's asset root")
    parser.add_argument(
        "--root",
        type=Path,
        help="planned run/output root; defaults to estimating on the asset filesystem",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--agent", choices=("codex", "claude"), required=True)
    parser.add_argument("--mode", choices=("local", "official"), default="local")
    parser.add_argument("--hash-assets", action="store_true", help="read and hash all assets")
    parser.add_argument("--json", action="store_true", help="emit a machine-readable report")
    args = parser.parse_args(argv)

    if args.gpu < 0:
        parser.error("--gpu must be non-negative")
    try:
        task = _resolve_task(args.task)
    except ValueError as error:
        parser.error(str(error))
    checks = run_checks(
        task=task,
        assets=args.assets,
        root=args.root,
        gpu=args.gpu,
        agent_name=args.agent,
        mode=args.mode,
        hash_assets=args.hash_assets,
    )
    failures = [check for check in checks if check.status == "fail"]
    warnings = [check for check in checks if check.status == "warn"]
    payload = {
        "schema_version": 1,
        "status": "ready" if not failures else "not_ready",
        "mode": args.mode,
        "task": task.name,
        "checks": [asdict(check) for check in checks],
        "failure_count": len(failures),
        "warning_count": len(warnings),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for check in checks:
            print(f"[{check.status.upper():4}] {check.name:16} {check.message}")
        if failures:
            print(f"\nSetup is not ready: {len(failures)} check(s) failed.")
        elif warnings:
            print(f"\nSetup is ready for local use with {len(warnings)} warning(s).")
        else:
            print("\nSetup is ready.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
