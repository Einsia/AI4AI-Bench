"""Write the durable receipt for one agent lifecycle.

The receipt is intentionally host-side and boring JSON.  It records both successful
explores, retrains, scores and fail-closed preflights, so a batch summary never has to
infer state from a missing log or from a Docker container that has already been removed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import socket
import subprocess
from pathlib import Path
from typing import Any

from host_contract import HostContractError, verify_contract
from phase_identity import content_digest
from task import load_task
from token_usage import TOKEN_FIELDS, summarize_token_usage, write_token_usage

MANIFEST_SCHEMA_VERSION = 9


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _bool(path: Path) -> bool:
    return path.exists()


def _first_fast_eval(run_dir: Path) -> Path | None:
    for candidate in sorted(run_dir.rglob("fast_eval*.json")):
        if candidate.is_file():
            return candidate
    return None


def _agent_state(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "explore/logs/agent/state.json"
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _task_source_sha256(task: Path) -> str | None:
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10 compatibility
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError:
            return None
    try:
        with (task / "task.toml").open("rb") as handle:
            value = tomllib.load(handle).get("environment", {}).get("source_sha256")
    except (OSError, ValueError):
        return None
    return str(value) if value else None


def _gpu_uuid(gpu: str | int) -> str | None:
    configured = os.environ.get("AI4AI_GPU_UUID")
    if configured:
        return configured
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--id",
                str(gpu),
                "--query-gpu=uuid",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _timestamp(path: Path) -> dt.datetime | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (OSError, ValueError):
        return None


def _phase_receipts(run_dir: Path) -> tuple[dict[str, bool], dict[str, int | None]]:
    complete: dict[str, bool] = {}
    elapsed: dict[str, int | None] = {}
    for phase in ("explore", "retrain", "score"):
        complete_path = run_dir / f".{phase}.complete"
        complete[phase] = complete_path.is_file()
        started = _timestamp(run_dir / f".{phase}.started")
        ended = _timestamp(complete_path)
        elapsed[phase] = int((ended - started).total_seconds()) if started and ended else None
    validation_roots = sorted((run_dir / "retrain/validation").glob("checkpoint-*"))
    validation_seconds = 0
    validation_complete = bool(validation_roots)
    for root in validation_roots:
        started = _timestamp(root / ".started")
        ended = _timestamp(root / ".complete")
        if not started or not ended:
            validation_complete = False
            continue
        validation_seconds += int((ended - started).total_seconds())
    complete["checkpoint_validation"] = validation_complete
    elapsed["checkpoint_validation"] = validation_seconds if validation_roots else None
    return complete, elapsed


def _latest_checkpoint(run_dir: Path) -> str | None:
    receipt = _read_json(run_dir / "retrain/artifacts.json") or {}
    accepted = receipt.get("accepted")
    if isinstance(accepted, list) and accepted:
        rows = [
            row
            for row in accepted
            if isinstance(row, dict) and isinstance(row.get("progress"), int)
        ]
        if rows:
            latest = max(rows, key=lambda row: int(row["progress"]))
            return str(
                latest.get("published_path") or latest.get("path") or latest.get("payload_path")
            )
    root = run_dir / "retrain/out/checkpoints"
    candidates = []
    for path in root.glob("global_step_*"):
        raw = path.name.removeprefix("global_step_")
        exported = path / "actor/huggingface/config.json"
        if raw.isdigit() and exported.is_file():
            candidates.append((int(raw), path / "actor/huggingface"))
    return str(max(candidates)[1]) if candidates else None


def _image_verification(run_dir: Path, *, selected_image: str | None) -> dict[str, Any]:
    """Summarize host-side phase receipts without trusting requested CLI flags."""

    all_paths = sorted(
        path
        for path in run_dir.rglob("preflight.json")
        if "out" not in path.relative_to(run_dir).parts
    )
    required: list[Path] = []
    if (run_dir / ".explore.complete").is_file():
        required.append(run_dir / "explore/preflight.json")
    if (run_dir / ".retrain.complete").is_file():
        required.append(run_dir / "retrain/preflight.json")
    for root in sorted((run_dir / "retrain/validation").glob("checkpoint-*")):
        if (root / ".complete").is_file():
            required.append(root / "preflight.json")
    for root in sorted((run_dir / "score").glob("artifact-*")):
        if (root / ".complete").is_file():
            required.append(root / "preflight.json")

    rows: list[dict[str, Any]] = []
    by_path: dict[Path, dict[str, Any]] = {}
    for path in all_paths:
        receipt = _read_json(path)
        if receipt is None:
            continue
        row = {**receipt, "receipt_path": path.relative_to(run_dir).as_posix()}
        rows.append(row)
        by_path[path] = row

    required_rows = [by_path.get(path) for path in required]
    complete = bool(required) and all(row is not None for row in required_rows)
    verified = complete
    hardware_verified = complete
    host_verified = complete
    asset_verified = complete
    run_config_bound = complete
    identities: set[tuple[str, str, str]] = set()
    hardware_contracts: set[str] = set()
    host_contracts: set[tuple[str, str, str]] = set()
    host_lock_hashes: set[str] = set()
    asset_contracts: set[tuple[str, str]] = set()
    run_config_hashes: set[str] = set()
    for row in required_rows:
        if row is None:
            verified = False
            hardware_verified = False
            host_verified = False
            asset_verified = False
            run_config_bound = False
            continue
        source = row.get("source") or {}
        image = row.get("image_identity") or {}
        hardware = row.get("hardware") or {}
        host = row.get("host_contract") or {}
        assets = row.get("asset_identity") or {}
        run_config = row.get("run_config") or {}
        hardware_devices = hardware.get("devices")
        occupancy_checks = hardware.get("occupancy_checks")
        if not (
            source.get("mode") == "strict"
            and source.get("status") == "match"
            and isinstance(source.get("expected"), str)
            and bool(source.get("expected"))
            and source.get("expected") == source.get("observed")
            and image.get("mode") == "strict"
            and image.get("status") == "match"
            and isinstance(image.get("expected_layers"), str)
            and bool(image.get("expected_layers"))
            and image.get("expected_layers") == image.get("observed_layers")
            and isinstance(image.get("expected_config"), str)
            and bool(image.get("expected_config"))
            and image.get("expected_config") == image.get("observed_config")
            and isinstance(row.get("image"), str)
            and bool(row.get("image"))
            and row.get("image") == selected_image
            and row.get("capability_checks") == "passed"
        ):
            verified = False
        if not (
            hardware.get("mode") == "strict"
            and hardware.get("status") == "match"
            and isinstance(hardware_devices, list)
            and bool(hardware_devices)
            and all(
                isinstance(device, dict)
                and device.get("type_match") is True
                and device.get("free_memory_match") is True
                for device in hardware_devices
            )
            and isinstance(occupancy_checks, list)
            and len(occupancy_checks) == 2
            and all(
                isinstance(check, dict)
                and check.get("mode") == "strict"
                and check.get("status") == "match"
                for check in occupancy_checks
            )
        ):
            hardware_verified = False
        if not (
            host.get("mode") == "strict"
            and host.get("status") == "match"
            and isinstance(host.get("algorithm"), str)
            and bool(host.get("algorithm"))
            and isinstance(host.get("expected"), str)
            and bool(host.get("expected"))
            and host.get("expected") == host.get("observed")
            and not host.get("changed_files")
        ):
            host_verified = False
        asset_rows = assets.get("aliases")
        if not (
            assets.get("mode") == "strict"
            and assets.get("status") == "match"
            and isinstance(assets.get("algorithm"), str)
            and bool(assets.get("algorithm"))
            and isinstance(assets.get("digest"), str)
            and bool(assets.get("digest"))
            and isinstance(asset_rows, list)
            and all(
                isinstance(asset, dict)
                and asset.get("status") == "match"
                and asset.get("expected_sha256") == asset.get("observed_sha256")
                for asset in asset_rows
            )
        ):
            asset_verified = False
        if not (
            run_config.get("status") == "present"
            and isinstance(run_config.get("sha256"), str)
            and bool(run_config.get("sha256"))
        ):
            run_config_bound = False
        hardware_contracts.add(
            json.dumps(
                {
                    "allowed_types": hardware.get("allowed_types"),
                    "required_peak_memory_mib": hardware.get("required_peak_memory_mib"),
                    "required_free_memory_mib": hardware.get("required_free_memory_mib"),
                },
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        identities.add(
            (
                str(row.get("image") or ""),
                str(image.get("observed_layers") or ""),
                str(image.get("observed_config") or ""),
            )
        )
        host_contracts.add(
            (
                str(host.get("algorithm") or ""),
                str(host.get("expected") or ""),
                str(host.get("observed") or ""),
            )
        )
        if isinstance(host.get("lock_sha256"), str) and host.get("lock_sha256"):
            host_lock_hashes.add(str(host["lock_sha256"]))
        asset_contracts.add(
            (str(assets.get("algorithm") or ""), str(assets.get("digest") or ""))
        )
        run_config_hashes.add(str(run_config.get("sha256") or ""))
    consistent = len(identities) == 1 and all(all(value for value in item) for item in identities)
    hardware_consistent = len(hardware_contracts) == 1
    host_consistent = len(host_contracts) == 1 and all(
        all(value for value in item) for item in host_contracts
    )
    asset_consistent = len(asset_contracts) == 1 and all(
        all(value for value in item) for item in asset_contracts
    )
    run_config_consistent = len(run_config_hashes) == 1 and all(run_config_hashes)
    selected = next(iter(identities)) if consistent else (None, None, None)
    selected_host = next(iter(host_contracts)) if host_consistent else (None, None, None)
    selected_assets = next(iter(asset_contracts)) if asset_consistent else (None, None)
    return {
        "official": bool(verified and consistent),
        "hardware_official": bool(hardware_verified and hardware_consistent),
        "host_contract_official": bool(host_verified and host_consistent),
        "host_contract_consistent": host_consistent,
        "asset_official": bool(asset_verified and asset_consistent),
        "asset_consistent": asset_consistent,
        "run_config_bound": bool(run_config_bound and run_config_consistent),
        "run_config_consistent": run_config_consistent,
        "run_config_sha256": next(iter(run_config_hashes)) if run_config_consistent else None,
        "hardware_consistent": hardware_consistent,
        "complete": complete,
        "consistent": consistent,
        "required_receipts": [path.relative_to(run_dir).as_posix() for path in required],
        "receipts": rows,
        "observed_image": selected[0],
        "observed_layers": selected[1],
        "observed_config": selected[2],
        "host_contract_algorithm": selected_host[0],
        "host_contract_expected": selected_host[1],
        "host_contract_observed": selected_host[2],
        # The whole lock also contains rows for unrelated tasks. Its hash is useful
        # provenance, but changing another task must not invalidate this task midway
        # through a lifecycle. The per-task expected/observed digest above is the gate.
        "host_contract_lock_sha256": (
            next(iter(host_lock_hashes)) if len(host_lock_hashes) == 1 else None
        ),
        "host_contract_lock_sha256es": sorted(host_lock_hashes),
        "asset_algorithm": selected_assets[0],
        "asset_digest": selected_assets[1],
        "_required_rows": required_rows,
    }


def _required_phase_key(run_dir: Path, row: dict[str, Any]) -> str | None:
    try:
        receipt = Path(str(row["receipt_path"]))
        parts = receipt.relative_to(run_dir).parts if receipt.is_absolute() else receipt.parts
    except (KeyError, ValueError):
        return None
    if parts[:1] == ("explore",):
        return "explore"
    if len(parts) >= 2 and parts[:2] == ("retrain", "validation"):
        return "checkpoint_validation"
    if parts[:1] == ("retrain",):
        return "retrain"
    if parts[:1] == ("score",):
        return "score"
    return None


def _run_config_verification(
    run_dir: Path,
    *,
    task: Path,
    instruction: Path | None,
    phase_verification: dict[str, Any],
) -> dict[str, Any]:
    """Verify the immutable config and the phase semantics actually executed."""

    path = run_dir / "run-config.json"
    value = _read_json(path)
    observed_sha256 = sha256_file(path)
    problems: list[str] = []
    if value is None or observed_sha256 is None:
        problems.append("missing_or_unreadable_run_config")
        return {
            "official": False,
            "status": "missing",
            "file": path.name,
            "sha256": observed_sha256,
            "receipt_sha256": phase_verification.get("run_config_sha256"),
            "problems": problems,
        }

    if value.get("schema_version") != 3:
        problems.append("unsupported_schema")
    execution_mode = value.get("execution_mode", "agent_explore")
    candidate_identity = value.get("candidate_patch_identity")
    if execution_mode == "external_patch_replay":
        lifecycle = _read_json(run_dir / "explore/out/lifecycle.json") or {}
        patch = run_dir / "explore/out/candidate.patch"
        if not (
            isinstance(candidate_identity, dict)
            and candidate_identity.get("provenance") == "external_patch"
            and candidate_identity.get("sha256") == sha256_file(patch)
            and candidate_identity.get("size_bytes")
            == (patch.stat().st_size if patch.is_file() else None)
            and lifecycle.get("submission_origin") == "external_patch"
            and lifecycle.get("candidate_patch_sha256") == candidate_identity.get("sha256")
            and lifecycle.get("candidate_patch_bytes") == candidate_identity.get("size_bytes")
            and not (run_dir / ".explore.complete").exists()
        ):
            problems.append("external_patch_lineage_mismatch")
    elif execution_mode != "agent_explore":
        problems.append("unknown_execution_mode")
    if not phase_verification.get("run_config_bound"):
        problems.append("phase_receipts_do_not_bind_one_run_config")
    if phase_verification.get("run_config_sha256") != observed_sha256:
        problems.append("run_config_hash_differs_from_phase_receipts")

    try:
        current_contract = verify_contract(task, mode="strict", instruction=instruction)
    except HostContractError:
        current_contract = {}
        problems.append("current_host_contract_unreadable")
    if not (
        current_contract.get("status") == "match"
        and current_contract.get("expected") == current_contract.get("observed")
        and isinstance(current_contract.get("expected"), str)
        and bool(current_contract.get("expected"))
    ):
        problems.append("current_host_contract_not_release_identical")

    recorded_contract = value.get("host_contract") or {}
    if not (
        recorded_contract.get("mode") == "strict"
        and recorded_contract.get("status") == "match"
        and recorded_contract.get("algorithm") == current_contract.get("algorithm")
        and recorded_contract.get("expected") == recorded_contract.get("observed")
        and recorded_contract.get("expected") == current_contract.get("expected")
        and recorded_contract.get("observed") == current_contract.get("observed")
        and phase_verification.get("host_contract_expected") == current_contract.get("expected")
        and phase_verification.get("host_contract_observed") == current_contract.get("observed")
        and phase_verification.get("host_contract_algorithm") == current_contract.get("algorithm")
    ):
        problems.append("host_contract_lineage_mismatch")

    recorded_assets = value.get("asset_identity") or {}
    locked_asset_rows = {
        str(asset.get("alias")): asset
        for asset in recorded_assets.get("aliases", [])
        if isinstance(asset, dict)
    }
    if not (
        recorded_assets.get("mode") == "strict"
        and recorded_assets.get("status") == "locked"
        and recorded_assets.get("algorithm") == phase_verification.get("asset_algorithm")
        and recorded_assets.get("digest") == phase_verification.get("asset_digest")
        and phase_verification.get("asset_official") is True
    ):
        problems.append("runtime_asset_lineage_mismatch")

    declared = value.get("declared_timeouts")
    effective = value.get("effective_timeouts")
    timeout_keys = {"explore", "retrain", "score", "checkpoint_validation"}
    if not (
        isinstance(declared, dict)
        and isinstance(effective, dict)
        and set(declared) == timeout_keys
        and set(effective) == timeout_keys
        and all(
            isinstance(declared[key], int)
            and not isinstance(declared[key], bool)
            and declared[key] > 0
            and isinstance(effective[key], int)
            and not isinstance(effective[key], bool)
            and effective[key] > 0
            for key in timeout_keys
        )
        and declared == effective
    ):
        problems.append("nonstandard_effective_timeouts")
    if value.get("score_phase") != "score":
        problems.append("nonstandard_score_phase")
    exports = value.get("retrain_exports") or {}
    if not (exports.get("names") == [] and exports.get("sha256") == content_digest({})):
        problems.append("retrain_export_overrides_present")
    if value.get("auto_retrain") is not True:
        problems.append("auto_retrain_disabled")

    phase_contract = value.get("phase_contract")
    if not (
        isinstance(phase_contract, dict)
        and set(phase_contract) == timeout_keys
        and value.get("phase_contract_sha256") == content_digest(phase_contract)
    ):
        problems.append("invalid_phase_contract")
        phase_contract = {}
    for row in phase_verification.get("_required_rows", []):
        if not isinstance(row, dict):
            continue
        key = _required_phase_key(run_dir, row)
        effective_phase = dict(row.get("effective_phase") or {})
        reported_sha256 = effective_phase.pop("sha256", None)
        if (
            key is None
            or reported_sha256 != content_digest(effective_phase)
            or effective_phase != phase_contract.get(key)
        ):
            problems.append(f"effective_phase_mismatch:{key or 'unknown'}")
        expected_aliases = sorted(
            str(mount.get("source"))[len("asset:") :]
            for mount in (phase_contract.get(key, {}).get("mounts", []) if key else [])
            if isinstance(mount, dict) and str(mount.get("source", "")).startswith("asset:")
        )
        asset_rows = (row.get("asset_identity") or {}).get("aliases") or []
        observed_aliases = sorted(
            str(asset.get("alias")) for asset in asset_rows if isinstance(asset, dict)
        )
        if observed_aliases != expected_aliases:
            problems.append(f"runtime_asset_alias_mismatch:{key or 'unknown'}")
        for asset in asset_rows:
            if not isinstance(asset, dict):
                continue
            locked = locked_asset_rows.get(str(asset.get("alias"))) or {}
            if not (
                asset.get("hash_kind") == locked.get("hash_kind")
                and asset.get("expected_sha256") == locked.get("sha256")
                and asset.get("observed_sha256") == locked.get("sha256")
            ):
                problems.append(f"runtime_asset_hash_mismatch:{key or 'unknown'}")

    return {
        "official": not problems,
        "status": "match" if not problems else "mismatch",
        "file": path.name,
        "sha256": observed_sha256,
        "receipt_sha256": phase_verification.get("run_config_sha256"),
        "score_phase": value.get("score_phase"),
        "declared_timeouts": declared,
        "effective_timeouts": effective,
        "retrain_exports": exports,
        "auto_retrain": value.get("auto_retrain"),
        "execution_mode": execution_mode,
        "candidate_patch_identity": candidate_identity,
        "problems": sorted(set(problems)),
    }


def _evidence(run_dir: Path) -> dict[str, Any]:
    fast_evals = sorted(str(path) for path in run_dir.rglob("fast_eval*.json") if path.is_file())
    checkpoint_receipts = sorted(
        str(path) for path in run_dir.rglob("checkpoint.ready.json") if path.is_file()
    )
    artifact_path = run_dir / "retrain/artifact.path"
    artifact = None
    if artifact_path.is_file():
        try:
            artifact = artifact_path.read_text(encoding="utf-8").strip() or None
        except OSError:
            artifact = None
    rejection = _read_json(run_dir / "retrain/rejection.json")
    artifacts_path = run_dir / "retrain/artifacts.json"
    artifacts = _read_json(artifacts_path)
    return {
        "fast_eval_receipt_count": len(fast_evals),
        "fast_eval_receipts": fast_evals,
        "checkpoint_receipt_count": len(checkpoint_receipts),
        "checkpoint_receipts": checkpoint_receipts,
        "artifact_path": artifact,
        "artifact_receipt": str(artifacts_path) if artifacts else None,
        "artifact_check": rejection
        or (
            {"status": "present", "accepted": len(artifacts.get("accepted", []))}
            if artifacts
            else None
        )
        or ({"status": "present"} if artifact else None),
    }


def make_manifest(
    *,
    task: Path,
    run_dir: Path,
    model: str,
    effort: str,
    gpu: str | int,
    status: str,
    start_time: str,
    end_time: str | None = None,
    exit_status: int | None = None,
    image_layers_digest: str | None = None,
    image_config_digest: str | None = None,
    image_archive_sha256: str | None = None,
    image: str | None = None,
    source_check: str = "warn",
    image_check: str = "warn",
    hardware_check: str = "warn",
    image_pull_policy: str = "missing",
    agent: str | None = None,
    agent_version: str | None = None,
    codex_version: str | None = None,
    auto_retrain: bool = False,
    gpu_uuid: str | None = None,
    slurm_job_id: str | None = None,
    failure_reason: str | None = None,
    source_root: Path | None = None,
    instruction: Path | None = None,
    agent_max_attempts: int = 0,
    agent_api_concurrency: int = 0,
    agent_api_concurrency_root: str | None = None,
) -> dict[str, Any]:
    """Build a complete manifest, using null/unknown rather than omitting fields."""

    run_dir = run_dir.resolve()
    source_root = (source_root or Path.cwd()).resolve()
    patch = run_dir / "explore/out/candidate.patch"
    fast_eval = _first_fast_eval(run_dir)
    run_config_value = _read_json(run_dir / "run-config.json") or {}
    execution_mode = run_config_value.get("execution_mode", "agent_explore")
    external_replay = execution_mode == "external_patch_replay"
    agent_state = _agent_state(run_dir)
    state_agent = agent_state.get("agent")
    if agent and state_agent and agent != state_agent:
        raise ValueError(f"manifest agent {agent!r} conflicts with state agent {state_agent!r}")
    recorded_agent = None if external_replay else str(agent or state_agent or "codex")
    if recorded_agent is not None and recorded_agent not in {"codex", "claude"}:
        raise ValueError(f"unsupported manifest agent: {recorded_agent!r}")
    recorded_agent_version = None if external_replay else (
        agent_version
        or agent_state.get("agent_version")
        or (
            codex_version or os.environ.get("CODEX_VERSION", "unknown")
            if recorded_agent == "codex"
            else os.environ.get("CLAUDE_VERSION", "unknown")
        )
    )
    recorded_backend = None if external_replay else str(
        agent_state.get("backend") or ("codex" if recorded_agent == "codex" else "anthropic")
    )
    recorded_provider = None if external_replay else str(
        agent_state.get("provider") or ("gateway" if recorded_agent == "codex" else "anthropic")
    )
    recorded_protocol = None if external_replay else str(
        agent_state.get("protocol") or ("responses" if recorded_agent == "codex" else "anthropic")
    )
    recorded_model = None if external_replay else str(agent_state.get("requested_model") or model)
    recorded_effort = (
        None if external_replay else str(agent_state.get("requested_effort") or effort)
    )
    token_usage = summarize_token_usage(
        run_dir / "explore/logs/agent",
        session_id=str(agent_state.get("session_id")) if agent_state.get("session_id") else None,
        agent=recorded_agent or "codex",
    )
    if external_replay:
        token_usage.update(
            agent=None,
            status="not_applicable",
            source="external_patch",
            cost_source=None,
            session_id=None,
        )
        for field in TOKEN_FIELDS:
            token_usage[field] = None
    if token_usage["status"] == "missing" and isinstance(agent_state.get("token_usage"), dict):
        token_usage = agent_state["token_usage"]
    effective_context_window = None
    model_usage = token_usage.get("model_usage")
    if isinstance(model_usage, dict):
        requested_model_usage = model_usage.get(recorded_model) if recorded_model else None
        if isinstance(requested_model_usage, dict):
            value = requested_model_usage.get("context_window")
            if isinstance(value, int) and value >= 0:
                effective_context_window = value
    phase_complete, phase_elapsed = _phase_receipts(run_dir)
    task_config = load_task(task)
    declared_image = str(task_config.get("environment", {}).get("image") or "") or None
    selected_image = image or declared_image
    verification = _image_verification(run_dir, selected_image=selected_image)
    run_config_verification = _run_config_verification(
        run_dir,
        task=task,
        instruction=instruction,
        phase_verification=verification,
    )
    formal_config = task_config.get("x-ai4ai", {}).get("formal", {})
    retrain_budget = formal_config.get("retrain_budget_sec")
    if not isinstance(retrain_budget, int) or isinstance(retrain_budget, bool):
        retrain_budget = None
    retrain_elapsed = phase_elapsed.get("retrain")
    artifact_receipt = _read_json(run_dir / "retrain/artifacts.json") or {}
    formal_artifacts = artifact_receipt.get("accepted", [])
    if not isinstance(formal_artifacts, list):
        formal_artifacts = []
    final_summary = _read_json(run_dir / "score/out/summary.json")
    lifecycle = _read_json(run_dir / "explore/out/lifecycle.json") or {
        "agent_exit_state": "running",
        "termination_reason": None,
        "submission_origin": "none",
        "candidate_state": "missing",
        "remaining_seconds_at_termination": None,
        "candidate_patch_bytes": None,
        "candidate_patch_sha256": None,
        "active_processes_at_termination": [],
        "active_gpu_processes_at_termination": [],
        "active_work_at_submit": None,
        "active_work_at_termination": None,
        "agent_session_active_at_termination": None,
        "candidate_rejection_reason": None,
    }
    rejection = _read_json(run_dir / "retrain/rejection.json")
    if rejection:
        lifecycle = {
            **lifecycle,
            "candidate_state": "rejected",
            "candidate_rejection_reason": rejection.get("reason") or rejection.get("error"),
        }
    recorded_status = str(agent_state.get("status", ""))
    payload: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "task": task.name,
        "task_path": str(task),
        "model": recorded_model,
        "effort": recorded_effort,
        "reasoning_effort": recorded_effort,
        "agent": recorded_agent,
        "agent_version": recorded_agent_version,
        "backend": recorded_backend,
        "provider": recorded_provider,
        "protocol": recorded_protocol,
        "endpoint_host": agent_state.get("endpoint_host"),
        "endpoint_port": agent_state.get("endpoint_port"),
        "endpoint_path": agent_state.get("endpoint_path"),
        "endpoint": agent_state.get("endpoint"),
        "context_window_requested": agent_state.get("context_window_requested"),
        "context_window_effective": effective_context_window,
        "git_head": os.environ.get("AI4AI_GIT_HEAD") or git_head(source_root),
        "source_sha256": _task_source_sha256(task),
        "declaration_sha256": sha256_file(task / "declaration.py"),
        "task_config_sha256": sha256_file(task / "task.toml"),
        "instruction_sha256": sha256_file(instruction or task / "instruction.md"),
        "host": socket.gethostname(),
        "gpu": str(gpu),
        "gpu_uuid": gpu_uuid or _gpu_uuid(gpu),
        "slurm_job_id": slurm_job_id or os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "codex_version": (
            codex_version
            or agent_state.get("codex_version")
            or os.environ.get("CODEX_VERSION", "unknown")
            if recorded_agent == "codex"
            else None
        ),
        "image_layers_digest": verification["observed_layers"]
        or image_layers_digest
        or os.environ.get("AI4AI_IMAGE_LAYERS_DIGEST", "unknown"),
        "image_config_digest": verification["observed_config"]
        or image_config_digest
        or os.environ.get("AI4AI_IMAGE_CONFIG_DIGEST", "unknown"),
        "image_archive_sha256": image_archive_sha256
        or os.environ.get("AI4AI_IMAGE_ARCHIVE_SHA256", "unknown"),
        "image": selected_image,
        "declared_image": declared_image,
        "image_override": bool(image and image != declared_image),
        "source_check": source_check,
        "image_check": image_check,
        "hardware_check": hardware_check,
        "image_pull_policy": image_pull_policy,
        "image_verification_receipts": verification["receipts"],
        "image_verification_complete": verification["complete"],
        "image_verification_consistent": verification["consistent"],
        "official_image_verification": verification["official"],
        "official_hardware_verification": verification["hardware_official"],
        "hardware_verification_consistent": verification["hardware_consistent"],
        "official_host_contract_verification": verification["host_contract_official"],
        "host_contract_verification_consistent": verification["host_contract_consistent"],
        "host_contract_algorithm": verification["host_contract_algorithm"],
        "host_contract_expected": verification["host_contract_expected"],
        "host_contract_observed": verification["host_contract_observed"],
        "host_contract_lock_sha256": verification["host_contract_lock_sha256"],
        "official_runtime_asset_verification": verification["asset_official"],
        "runtime_asset_verification_consistent": verification["asset_consistent"],
        "runtime_asset_algorithm": verification["asset_algorithm"],
        "runtime_asset_digest": verification["asset_digest"],
        "official_run_config_verification": run_config_verification["official"],
        "run_config_verification": run_config_verification,
        "execution_mode": run_config_verification.get("execution_mode", execution_mode),
        "candidate_provenance": lifecycle.get("submission_origin"),
        "official_replay": bool(
            verification["official"]
            and verification["hardware_official"]
            and verification["host_contract_official"]
            and verification["asset_official"]
            and run_config_verification["official"]
        ),
        "start_time": start_time,
        "end_time": end_time,
        "exit_status": exit_status,
        "status": status,
        "session_id": agent_state.get("session_id"),
        "resume_count": int(agent_state.get("resume_count", 0) or 0),
        "attempts": agent_state.get("attempts", []),
        "agent_max_attempts": agent_max_attempts,
        "agent_api_concurrency": agent_api_concurrency,
        "agent_api_concurrency_root": agent_api_concurrency_root,
        "agent_retry_policy": agent_state.get("retry_policy")
        or {
            "max_attempts": agent_max_attempts,
            "api_concurrency": agent_api_concurrency,
            "api_concurrency_root": agent_api_concurrency_root,
            "deadline_unix": None,
        },
        "last_failure_reason": agent_state.get("last_failure_reason") or failure_reason,
        "agent_state": recorded_status or None,
        "agent_exit_state": lifecycle.get("agent_exit_state"),
        "termination_reason": lifecycle.get("termination_reason"),
        "submission_origin": lifecycle.get("submission_origin"),
        "candidate_state": lifecycle.get("candidate_state"),
        "remaining_seconds_at_termination": lifecycle.get("remaining_seconds_at_termination"),
        "candidate_patch_bytes": (
            patch.stat().st_size if patch.is_file() else lifecycle.get("candidate_patch_bytes")
        ),
        "token_usage": token_usage,
        "candidate_patch_sha256": sha256_file(patch) or lifecycle.get("candidate_patch_sha256"),
        "candidate_patch": str(patch) if patch.is_file() else None,
        "active_processes_at_termination": lifecycle.get("active_processes_at_termination", []),
        "active_gpu_processes_at_termination": lifecycle.get(
            "active_gpu_processes_at_termination", []
        ),
        "active_work_at_submit": lifecycle.get("active_work_at_submit"),
        "active_work_at_termination": lifecycle.get("active_work_at_termination"),
        "agent_session_active_at_termination": lifecycle.get("agent_session_active_at_termination"),
        "candidate_rejection_reason": lifecycle.get("candidate_rejection_reason"),
        "evidence": _evidence(run_dir),
        "fast_eval_present": fast_eval is not None,
        "fast_eval_path": str(fast_eval) if fast_eval else None,
        "retrain_present": _bool(run_dir / "retrain"),
        "score_present": _bool(run_dir / "score"),
        "retrain_phase_created": _bool(run_dir / "retrain"),
        "score_phase_created": _bool(run_dir / "score"),
        "auto_retrain": auto_retrain,
        "phase_complete": phase_complete,
        "phase_elapsed_s": phase_elapsed,
        "retrain_budget_seconds": retrain_budget,
        "retrain_budget_utilization": (
            retrain_elapsed / retrain_budget
            if isinstance(retrain_elapsed, int) and retrain_budget
            else None
        ),
        "formal_artifacts": formal_artifacts,
        "formal_checkpoint_candidates": artifact_receipt.get("candidates", []),
        "formal_artifact_selection_rule": artifact_receipt.get("selection_rule"),
        "formal_artifact_publication_root": artifact_receipt.get("publication_root"),
        "latest_checkpoint": _latest_checkpoint(run_dir),
        "final_metric": final_summary.get("metric") if final_summary else None,
        "final_status": final_summary.get("status") if final_summary else None,
        "final_reason": (
            final_summary.get("reason")
            or final_summary.get("failure_reason")
            or final_summary.get("error")
            if final_summary
            else None
        ),
        "final_score": final_summary.get("score") if final_summary else None,
        "final_stderr": final_summary.get("stderr") if final_summary else None,
        "final_n": final_summary.get("n") if final_summary else None,
        "final_correct": final_summary.get("correct") if final_summary else None,
        "final_selection_rule": final_summary.get("selection_rule") if final_summary else None,
        "selected_artifact": final_summary.get("selected_artifact") if final_summary else None,
        "selected_progress": final_summary.get("selected_progress") if final_summary else None,
        "final_artifact_results": final_summary.get("artifact_results", [])
        if final_summary
        else [],
    }
    for field in TOKEN_FIELDS:
        payload[field] = token_usage.get(field)
    return payload


def write_manifest(path: Path, **kwargs: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = make_manifest(**kwargs)
    run_dir = Path(kwargs["run_dir"]).resolve()
    write_token_usage(
        run_dir / "explore/logs/agent/token_usage.json",
        payload["token_usage"],
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    usage = payload["token_usage"]
    print(
        f"{payload['agent']} token usage: "
        + " ".join(f"{field}={usage.get(field)}" for field in TOKEN_FIELDS)
        + f" status={usage.get('status')} source={usage.get('source')}"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--start-time", required=True)
    parser.add_argument("--end-time", default=None)
    parser.add_argument("--exit-status", type=int, default=None)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--instruction", type=Path, default=None)
    parser.add_argument("--image-layers-digest", default=None)
    parser.add_argument("--image-config-digest", default=None)
    parser.add_argument("--image-archive-sha256", default=None)
    parser.add_argument("--image", default=None)
    parser.add_argument("--source-check", choices=("warn", "strict", "off"), default="warn")
    parser.add_argument("--image-check", choices=("strict", "warn"), default="warn")
    parser.add_argument("--hardware-check", choices=("strict", "warn", "off"), default="warn")
    parser.add_argument("--image-pull-policy", choices=("missing", "never"), default="missing")
    parser.add_argument("--agent", choices=("codex", "claude"), default=None)
    parser.add_argument("--agent-version", default=None)
    parser.add_argument("--codex-version", default=None)
    parser.add_argument("--auto-retrain", type=int, choices=(0, 1), default=0)
    parser.add_argument("--gpu-uuid", default=None)
    parser.add_argument("--slurm-job-id", default=None)
    parser.add_argument("--failure-reason", default=None)
    parser.add_argument("--agent-max-attempts", type=int, default=0)
    parser.add_argument("--agent-api-concurrency", type=int, default=0)
    parser.add_argument("--agent-api-concurrency-root", default=None)
    args = parser.parse_args()
    write_manifest(
        args.path,
        task=args.task,
        run_dir=args.run_dir,
        model=args.model,
        effort=args.effort,
        gpu=args.gpu,
        status=args.status,
        start_time=args.start_time,
        end_time=args.end_time,
        exit_status=args.exit_status,
        image_layers_digest=args.image_layers_digest,
        image_config_digest=args.image_config_digest,
        image_archive_sha256=args.image_archive_sha256,
        image=args.image,
        source_check=args.source_check,
        image_check=args.image_check,
        hardware_check=args.hardware_check,
        image_pull_policy=args.image_pull_policy,
        agent=args.agent,
        agent_version=args.agent_version,
        codex_version=args.codex_version,
        auto_retrain=bool(args.auto_retrain),
        gpu_uuid=args.gpu_uuid,
        slurm_job_id=args.slurm_job_id,
        failure_reason=args.failure_reason,
        source_root=args.source_root,
        instruction=args.instruction,
        agent_max_attempts=args.agent_max_attempts,
        agent_api_concurrency=args.agent_api_concurrency,
        agent_api_concurrency_root=args.agent_api_concurrency_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
