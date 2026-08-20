#!/usr/bin/env python3
"""Generate local SBOM and vulnerability evidence for one release image.

The helper never logs in, pulls, pushes, signs, or updates vulnerability databases.
It first proves that the image is already present in the configured local runtime,
then runs Syft against the Docker daemon and either Trivy in offline mode or Grype
against the generated SBOM.  Every outcome, including missing tools or databases,
is written to an atomic ``summary.json`` receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
SECRET_VALUE = re.compile(
    r"https?://[^\s/:]{1,64}:[^\s/@]{6,128}@|"
    r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|"
    r"\b(?:sk-|ghp_|github_pat_|glpat-|hf_)[A-Za-z0-9_-]{16,}",
    re.I,
)
SEVERITY_ORDER = {"UNKNOWN": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}


class ScanError(RuntimeError):
    """A release scan could not be completed safely."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_tool(value: str) -> str | None:
    if "/" in value:
        path = Path(value).expanduser().resolve()
        return str(path) if path.is_file() and os.access(path, os.X_OK) else None
    return shutil.which(value)


def run(
    argv: list[str],
    *,
    timeout: int,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def docker_scanner_environment(runtime: list[str]) -> tuple[dict[str, str], str]:
    """Resolve the Docker endpoint used by the runtime prefix for external scanners.

    Syft and Trivy do not execute the configured Docker CLI. They connect through
    ``DOCKER_HOST`` directly, so accepting an opaque wrapper could inspect one daemon
    and scan another. Support normal docker/sudo/env prefixes and fail closed for
    anything whose endpoint cannot be made identical.
    """

    docker_indexes = [
        index for index, token in enumerate(runtime) if Path(token).name in {"docker", "docker.exe"}
    ]
    if len(docker_indexes) != 1:
        raise ScanError("--runtime must contain exactly one docker executable")
    docker_index = docker_indexes[0]
    endpoint: str | None = None
    for token in runtime[:docker_index]:
        if token.startswith("DOCKER_HOST="):
            endpoint = token.split("=", 1)[1]
    suffix = runtime[docker_index + 1 :]
    index = 0
    while index < len(suffix):
        token = suffix[index]
        if token in {"-H", "--host"}:
            if index + 1 >= len(suffix):
                raise ScanError(f"{token} in --runtime has no value")
            endpoint = suffix[index + 1]
            index += 2
            continue
        if token.startswith("--host="):
            endpoint = token.split("=", 1)[1]
            index += 1
            continue
        if token.startswith("-H") and token != "-H":
            endpoint = token[2:]
            index += 1
            continue
        raise ScanError(
            "cannot prove scanner/runtime Docker endpoint equivalence with runtime suffix "
            f"{suffix!r}; use DOCKER_HOST or docker --host"
        )
    endpoint = endpoint or os.environ.get("DOCKER_HOST")
    if not endpoint:
        if os.environ.get("DOCKER_CONTEXT"):
            raise ScanError(
                "DOCKER_CONTEXT is set without DOCKER_HOST; explicitly set DOCKER_HOST so "
                "Syft/Trivy can be bound to the inspected daemon"
            )
        endpoint = "unix:///var/run/docker.sock"
    if any(character.isspace() for character in endpoint) or SECRET_VALUE.search(endpoint):
        raise ScanError("resolved DOCKER_HOST is invalid or contains credentials")
    environment = {**os.environ, "DOCKER_HOST": endpoint}
    environment.pop("DOCKER_CONTEXT", None)
    return environment, endpoint


def tool_version(path: str, *, env: dict[str, str] | None = None) -> dict[str, Any]:
    for flag in ("version", "--version"):
        try:
            completed = run([path, flag], timeout=20, env=env)
        except (OSError, subprocess.TimeoutExpired) as error:
            last_error = str(error)
            continue
        output = (completed.stdout + completed.stderr).strip()
        if completed.returncode == 0:
            return {
                "path": path,
                "available": True,
                "version_argv": [path, flag],
                "version": output[:2000],
            }
        last_error = output[:1000]
    return {"path": path, "available": False, "problem": last_error}


def discover_tools(
    args: argparse.Namespace, *, env: dict[str, str] | None = None
) -> dict[str, dict[str, Any]]:
    tools: dict[str, dict[str, Any]] = {}
    for name, requested in (
        ("syft", args.syft),
        ("trivy", args.trivy),
        ("grype", args.grype),
        ("cosign", args.cosign),
    ):
        resolved = resolve_tool(requested)
        tools[name] = (
            tool_version(resolved, env=env)
            if resolved
            else {"requested": requested, "available": False, "problem": "not found"}
        )
    return tools


def select_vulnerability_scanner(requested: str, tools: dict[str, dict[str, Any]]) -> str:
    if requested == "auto":
        for candidate in ("trivy", "grype"):
            if tools[candidate]["available"]:
                return candidate
        raise ScanError("neither Trivy nor Grype is installed")
    if not tools[requested]["available"]:
        raise ScanError(f"requested vulnerability scanner is unavailable: {requested}")
    return requested


def inspect_local_image(runtime: list[str], image: str, timeout: int) -> dict[str, Any]:
    completed = run([*runtime, "image", "inspect", image], timeout=min(timeout, 300))
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[:1000]
        raise ScanError(
            f"image is not available in the local runtime; refusing any registry fallback: {detail}"
        )
    try:
        metadata = json.loads(completed.stdout)[0]
    except (IndexError, TypeError, ValueError) as error:
        raise ScanError("runtime returned invalid image-inspect JSON") from error
    layers = (metadata.get("RootFS") or {}).get("Layers") or []
    if not isinstance(layers, list) or not layers:
        raise ScanError("local image has no RootFS layer identity")
    layer_fingerprint = "layers:" + hashlib.sha256("\n".join(layers).encode()).hexdigest()
    raw_config = metadata.get("Config") or {}
    canonical_config = {
        "os": metadata.get("Os"),
        "architecture": metadata.get("Architecture"),
        "variant": metadata.get("Variant"),
        "config": {field: raw_config.get(field) for field in CONFIG_FIELDS},
    }
    config_fingerprint = (
        "config:"
        + hashlib.sha256(
            json.dumps(canonical_config, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()
    )
    return {
        "image": image,
        "image_id": metadata.get("Id"),
        "repo_digests": metadata.get("RepoDigests") or [],
        "platform": {
            "os": metadata.get("Os"),
            "architecture": metadata.get("Architecture"),
            "variant": metadata.get("Variant"),
        },
        "layer_fingerprint": layer_fingerprint,
        "config_fingerprint": config_fingerprint,
    }


def run_to_temporary_json(
    argv: list[str],
    target: Path,
    *,
    timeout: int,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    started = time.monotonic()
    try:
        with temporary.open("w") as output:
            completed = subprocess.run(
                argv,
                check=False,
                stdout=output,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                env=env,
            )
        if completed.returncode != 0:
            raise ScanError(
                f"scanner exited {completed.returncode}: {completed.stderr.strip()[:2000]}"
            )
        try:
            json.loads(temporary.read_text())
        except (OSError, ValueError) as error:
            raise ScanError("scanner did not produce valid JSON") from error
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "argv": argv,
        "exit_code": 0,
        "duration_sec": round(time.monotonic() - started, 3),
        "path": str(target),
        "sha256": sha256_file(target),
        "size_bytes": target.stat().st_size,
    }


def run_trivy(
    tool: str,
    image: str,
    target: Path,
    *,
    timeout: int,
    env: dict[str, str],
) -> dict[str, Any]:
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    argv = [
        tool,
        "image",
        "--image-src",
        "docker",
        "--offline-scan",
        "--skip-db-update",
        "--skip-java-db-update",
        "--skip-check-update",
        "--format",
        "json",
        "--output",
        str(temporary),
        image,
    ]
    started = time.monotonic()
    try:
        completed = run(argv, timeout=timeout, env=env)
        if completed.returncode != 0:
            raise ScanError(
                f"Trivy exited {completed.returncode}: "
                f"{(completed.stderr or completed.stdout).strip()[:2000]}"
            )
        try:
            json.loads(temporary.read_text())
        except (OSError, ValueError) as error:
            raise ScanError("Trivy did not produce valid JSON") from error
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "argv": argv,
        "exit_code": 0,
        "duration_sec": round(time.monotonic() - started, 3),
        "path": str(target),
        "sha256": sha256_file(target),
        "size_bytes": target.stat().st_size,
        "network_policy": "offline-scan; database updates disabled",
    }


def summarise_sbom(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    packages = value.get("packages") if isinstance(value, dict) else None
    return {"spdx_packages": len(packages) if isinstance(packages, list) else None}


def summarise_vulnerabilities(path: Path, scanner: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    severities: Counter[str] = Counter()
    if scanner == "trivy":
        for result in value.get("Results", []) if isinstance(value, dict) else []:
            for finding in result.get("Vulnerabilities") or []:
                severities[str(finding.get("Severity") or "UNKNOWN").upper()] += 1
    else:
        for match in value.get("matches", []) if isinstance(value, dict) else []:
            vulnerability = match.get("vulnerability") or {}
            severities[str(vulnerability.get("severity") or "UNKNOWN").upper()] += 1
    return {"total": sum(severities.values()), "by_severity": dict(sorted(severities.items()))}


def audit_receipt_record(path: Path, identity: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise ScanError(f"audit receipt does not exist: {path}")
    try:
        payload = json.loads(path.read_text())
    except ValueError as error:
        raise ScanError(f"audit receipt is not valid JSON: {path}") from error
    report: Any = payload
    if isinstance(payload, dict) and payload.get("status") != "clean":
        nested = payload.get("audit")
        report = nested.get("report") if isinstance(nested, dict) else None
    if not isinstance(report, dict) or report.get("status") != "clean":
        raise ScanError("audit receipt does not contain a clean standalone or nested audit report")
    expected = {
        "image": identity.get("image"),
        "image_id": identity.get("image_id"),
        "layer_fingerprint": identity.get("layer_fingerprint"),
        "config_fingerprint": identity.get("config_fingerprint"),
    }
    mismatches = [
        f"{field}={report.get(field)!r} != {value!r}"
        for field, value in expected.items()
        if not value or report.get(field) != value
    ]
    if mismatches:
        raise ScanError("audit receipt is not bound to the scanned image: " + "; ".join(mismatches))
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "reported_status": "clean",
        "image_identity": expected,
    }


def _timestamp_values(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(child, str) and any(
                marker in str(key).lower()
                for marker in ("built", "created", "updated", "downloaded", "timestamp")
            ):
                records.append((name, child))
            else:
                records.extend(_timestamp_values(child, name))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            records.extend(_timestamp_values(child, f"{prefix}[{index}]"))
    return records


def _parse_timestamp(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def scanner_database_metadata(
    scanner: str, tool: str, *, timeout: int, env: dict[str, str]
) -> dict[str, Any]:
    commands = (
        [[tool, "version", "--format", "json"]]
        if scanner == "trivy"
        else [[tool, "db", "status", "-o", "json"], [tool, "db", "status", "--output", "json"]]
    )
    completed: subprocess.CompletedProcess[str] | None = None
    payload: Any = None
    for command in commands:
        completed = run(command, timeout=min(timeout, 60), env=env)
        if completed.returncode != 0:
            continue
        try:
            payload = json.loads(completed.stdout)
        except ValueError:
            continue
        break
    if payload is None or completed is None:
        detail = "" if completed is None else (completed.stderr or completed.stdout).strip()[:1000]
        return {"status": "unknown", "problem": f"could not read scanner DB metadata: {detail}"}
    timestamps = []
    now = datetime.now(timezone.utc)
    for name, raw in _timestamp_values(payload):
        parsed = _parse_timestamp(raw)
        timestamps.append(
            {
                "field": name,
                "value": raw,
                "age_hours": round((now - parsed).total_seconds() / 3600, 3) if parsed else None,
            }
        )
    nonnegative = [
        row["age_hours"]
        for row in timestamps
        if row["age_hours"] is not None and row["age_hours"] >= 0
    ]
    return {
        "status": "recorded",
        "command": completed.args,
        "metadata": payload,
        "timestamps": timestamps,
        "youngest_age_hours": min(nonnegative) if nonnegative else None,
    }


def severity_gate(summary: dict[str, Any], threshold: str | None) -> dict[str, Any]:
    if threshold is None:
        return {"enabled": False, "passed": None}
    minimum = SEVERITY_ORDER[threshold]
    violating = {
        severity: count
        for severity, count in (summary.get("by_severity") or {}).items()
        if SEVERITY_ORDER.get(severity, 0) >= minimum and count
    }
    return {
        "enabled": True,
        "threshold": threshold,
        "passed": not violating,
        "violating_findings": violating,
    }


def ensure_new_targets(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise ScanError(f"refusing to overwrite existing release evidence: {', '.join(existing)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime", default=os.environ.get("AI4AI_DOCKER", "docker"))
    parser.add_argument("--syft", default="syft")
    parser.add_argument("--trivy", default="trivy")
    parser.add_argument("--grype", default="grype")
    parser.add_argument("--cosign", default="cosign")
    parser.add_argument(
        "--vulnerability-scanner", choices=("auto", "trivy", "grype"), default="auto"
    )
    parser.add_argument("--audit-receipt", type=Path)
    parser.add_argument("--timeout-sec", type=int, default=7200)
    parser.add_argument(
        "--fail-on-severity",
        choices=("UNKNOWN", "LOW", "MEDIUM", "HIGH", "CRITICAL"),
        help="optional policy gate; the default records findings without calling the image clean",
    )
    parser.add_argument("--check-tools-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if not args.check_tools_only and not args.image:
        parser.error("--image is required unless --check-tools-only is used")

    output_dir = args.output_dir.resolve()
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        parser.error(f"refusing to overwrite existing receipt: {summary_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = shlex.split(args.runtime)
    if not runtime:
        parser.error("--runtime is empty")
    if SECRET_VALUE.search("\n".join(runtime)):
        parser.error("--runtime contains a credential-like value")
    try:
        scanner_env, docker_endpoint = docker_scanner_environment(runtime)
    except ScanError as error:
        parser.error(str(error))

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "kind": "ai4ai-local-image-security-evidence",
        "started_at": utc_now(),
        "host": socket.gethostname(),
        "runtime": runtime,
        "docker_endpoint": docker_endpoint,
        "image": args.image,
        "network_policy": "no pulls; scanner database updates disabled",
        "upload_performed": False,
        "signing_performed": False,
        "status": "pending",
        "problems": [],
    }
    try:
        receipt["tools"] = discover_tools(args, env=scanner_env)
        if args.check_tools_only:
            scanner = select_vulnerability_scanner(args.vulnerability_scanner, receipt["tools"])
            if not receipt["tools"]["syft"]["available"]:
                raise ScanError("Syft is not installed")
            receipt["selected_vulnerability_scanner"] = scanner
            receipt["status"] = "tools-available"
        else:
            if not receipt["tools"]["syft"]["available"]:
                raise ScanError("Syft is not installed")
            scanner = select_vulnerability_scanner(args.vulnerability_scanner, receipt["tools"])
            receipt["selected_vulnerability_scanner"] = scanner
            receipt["image_identity"] = inspect_local_image(runtime, args.image, args.timeout_sec)
            if args.audit_receipt:
                receipt["audit_receipt"] = audit_receipt_record(
                    args.audit_receipt, receipt["image_identity"]
                )

            sbom_path = output_dir / "sbom.spdx.json"
            vulnerabilities_path = output_dir / f"vulnerabilities.{scanner}.json"
            ensure_new_targets([sbom_path, vulnerabilities_path])
            syft = str(receipt["tools"]["syft"]["path"])
            receipt["sbom"] = run_to_temporary_json(
                [syft, f"docker:{args.image}", "-o", "spdx-json"],
                sbom_path,
                timeout=args.timeout_sec,
                env={**scanner_env, "SYFT_CHECK_FOR_APP_UPDATE": "false"},
            )
            receipt["sbom"]["network_policy"] = "application update checks disabled"
            receipt["sbom"]["summary"] = summarise_sbom(sbom_path)
            if scanner == "trivy":
                trivy = str(receipt["tools"]["trivy"]["path"])
                receipt["vulnerabilities"] = run_trivy(
                    trivy,
                    args.image,
                    vulnerabilities_path,
                    timeout=args.timeout_sec,
                    env=scanner_env,
                )
            else:
                grype = str(receipt["tools"]["grype"]["path"])
                receipt["vulnerabilities"] = run_to_temporary_json(
                    [grype, f"sbom:{sbom_path}", "-o", "json"],
                    vulnerabilities_path,
                    timeout=args.timeout_sec,
                    env={
                        **scanner_env,
                        "GRYPE_DB_AUTO_UPDATE": "false",
                        "GRYPE_CHECK_FOR_APP_UPDATE": "false",
                        "GRYPE_EXTERNAL_SOURCES_ENABLE": "false",
                    },
                )
                receipt["vulnerabilities"]["network_policy"] = (
                    "database/app updates and external sources disabled; existing local DB only"
                )
            receipt["vulnerabilities"]["summary"] = summarise_vulnerabilities(
                vulnerabilities_path, scanner
            )
            scanner_path = str(receipt["tools"][scanner]["path"])
            receipt["vulnerability_database"] = scanner_database_metadata(
                scanner, scanner_path, timeout=args.timeout_sec, env=scanner_env
            )
            receipt["vulnerability_policy"] = severity_gate(
                receipt["vulnerabilities"]["summary"], args.fail_on_severity
            )
            receipt["image_identity_after_scan"] = inspect_local_image(
                runtime, args.image, args.timeout_sec
            )
            if receipt["image_identity_after_scan"] != receipt["image_identity"]:
                raise ScanError("image identity changed while security evidence was generated")
            receipt["status"] = (
                "scan-complete"
                if receipt["vulnerability_policy"]["passed"] is not False
                else "policy-failed"
            )
    except (OSError, ScanError, subprocess.TimeoutExpired, ValueError) as error:
        receipt["status"] = "failed"
        receipt["problems"].append(str(error))
    finally:
        receipt["finished_at"] = utc_now()
        atomic_json(summary_path, receipt)

    if args.json:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        print(f"{receipt['status']}: {summary_path}")
        for problem in receipt["problems"]:
            print(f"  {problem}")
    expected = "tools-available" if args.check_tools_only else "scan-complete"
    return 0 if receipt["status"] == expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
