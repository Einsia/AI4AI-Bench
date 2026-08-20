#!/usr/bin/env python3
"""Create or verify the immutable configuration of checkpoint-only evaluation."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from artifact import ArtifactError, read_receipt
from asset_identity import AssetIdentityError, aliases_from_phases, expected_contract
from host_contract import HostContractError, require_contract, verify_contract
from phase_identity import content_digest, phase_payload
from phase_loader import PhaseDeclarationError, load_phases
from task import load_task


def _timeout(value: str, declared: int) -> int:
    if not value:
        return declared
    try:
        resolved = int(value)
    except ValueError as error:
        raise SystemExit("--score-timeout must be a positive integer") from error
    if resolved <= 0:
        raise SystemExit("--score-timeout must be a positive integer")
    return resolved


def result_classification(
    *,
    source_check: str,
    image_check: str,
    hardware_check: str,
    score_phase: str,
    score_timeout_overridden: bool,
    declared_score_timeout: int,
    effective_score_timeout: int,
) -> tuple[str, dict[str, Any]]:
    verification = {
        "source_check": source_check,
        "image_check": image_check,
        "hardware_check": hardware_check,
        "score_phase": score_phase,
        "score_timeout_overridden": score_timeout_overridden,
        "declared_score_timeout": declared_score_timeout,
        "effective_score_timeout": effective_score_timeout,
        "score_timeout_matches_declared": (
            effective_score_timeout == declared_score_timeout
        ),
    }
    official = (
        all(
            verification[key] == "strict"
            for key in ("source_check", "image_check", "hardware_check")
        )
        and score_phase == "score"
        and not score_timeout_overridden
        and effective_score_timeout == declared_score_timeout
    )
    return (
        "official_self_hosted" if official else "non_official_local",
        verification,
    )


def canonical_config(args: argparse.Namespace) -> dict[str, Any]:
    task = args.task.resolve()
    contract = verify_contract(task, mode=args.source_check)
    require_contract(contract)
    phases = load_phases(task, namespace="checkpoint_evaluation_config")
    score_phase = args.score_phase
    missing = sorted({"checkpoint-validate", score_phase} - set(phases))
    if missing:
        raise SystemExit(f"task declaration is missing evaluation phases: {missing}")
    task_config = load_task(task)
    declared_image = task_config.get("environment", {}).get("image")
    declared_score_timeout = int(phases[score_phase].timeout_sec)
    effective_score_timeout = _timeout(args.score_timeout, declared_score_timeout)
    classification, verification = result_classification(
        source_check=args.source_check,
        image_check=args.image_check,
        hardware_check=args.hardware_check,
        score_phase=score_phase,
        score_timeout_overridden=bool(args.score_timeout),
        declared_score_timeout=declared_score_timeout,
        effective_score_timeout=effective_score_timeout,
    )
    effective_phases = {
        "checkpoint_validation": phases["checkpoint-validate"],
        "score": phases[score_phase].with_timeout(effective_score_timeout),
    }
    artifact_receipt = read_receipt(args.artifacts)
    candidates = artifact_receipt.get("candidates")
    if artifact_receipt.get("provenance") != "external_checkpoint" or not isinstance(
        candidates, list
    ):
        raise SystemExit("artifact receipt is not an external-checkpoint receipt")
    checkpoint_identity = [
        {
            "progress": row.get("progress"),
            "path": row.get("payload_path"),
            "provenance": row.get("provenance"),
            "content_identity": row.get("content_identity"),
        }
        for row in candidates
        if isinstance(row, dict)
    ]
    # runner.py verifies each phase's mounted subset against one full-task asset
    # contract (`contract_aliases=aliases_from_phases`). Bind that same union here so
    # validation and score receipts cannot splice two otherwise valid asset stores.
    all_aliases = aliases_from_phases(phases)
    asset_identity = expected_contract(task, all_aliases, mode=args.source_check)
    host_identity = {
        "mode": contract["mode"],
        "status": contract["status"],
        "algorithm": contract["algorithm"],
        "expected": None if args.source_check == "off" else contract["expected"],
        "observed": None if args.source_check == "off" else contract["observed"],
    }
    phase_contract = {name: phase_payload(phase) for name, phase in effective_phases.items()}
    return {
        "schema_version": 1,
        "execution_mode": "checkpoint_only_evaluation",
        "task": str(task),
        "assets": str(args.assets.resolve()),
        "asset_identity": asset_identity,
        "gpu": str(args.gpu),
        "image": args.image or declared_image,
        "source_check": args.source_check,
        "image_check": args.image_check,
        "hardware_check": args.hardware_check,
        "image_pull_policy": args.image_pull_policy,
        "score_phase": score_phase,
        "declared_score_timeout": declared_score_timeout,
        "effective_score_timeout": effective_score_timeout,
        "result_classification": classification,
        "verification": verification,
        "phase_contract": phase_contract,
        "phase_contract_sha256": content_digest(phase_contract),
        "host_contract": host_identity,
        "checkpoint_identity": checkpoint_identity,
    }


def initialize_or_validate(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise SystemExit(f"evaluation config is unreadable: {path}: {error}") from error
        portable_assets = (
            existing.get("asset_identity") == payload.get("asset_identity")
            and isinstance(payload.get("asset_identity"), dict)
            and payload["asset_identity"].get("mode") == "strict"
            and payload["asset_identity"].get("status") == "locked"
        )
        differences = {
            key: {"recorded": existing.get(key), "requested": payload.get(key)}
            for key in sorted(set(existing) | set(payload))
            if existing.get(key) != payload.get(key) and not (key == "assets" and portable_assets)
        }
        if differences:
            detail = "\n".join(
                f"  {key}: recorded={row['recorded']!r} requested={row['requested']!r}"
                for key, row in differences.items()
            )
            raise SystemExit(
                f"refusing to splice checkpoint evaluation across configurations:\n{detail}\n"
                "Use a new evaluation name."
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    unexpected = [
        entry.name
        for entry in path.parent.iterdir()
        if entry.name not in {".evaluation.lock", "artifacts.json"}
        and not entry.name.startswith(f".{path.name}.")
    ]
    if unexpected:
        raise SystemExit(
            "refusing to adopt a populated evaluation directory without an immutable "
            f"config: {', '.join(sorted(unexpected)[:8])}"
        )
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            initialize_or_validate(path, payload)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--image", default="")
    parser.add_argument("--source-check", choices=("warn", "strict", "off"), required=True)
    parser.add_argument("--image-check", choices=("warn", "strict"), required=True)
    parser.add_argument("--hardware-check", choices=("warn", "strict", "off"), required=True)
    parser.add_argument("--image-pull-policy", choices=("missing", "never"), required=True)
    parser.add_argument("--score-phase", choices=("score", "score-mock"), required=True)
    parser.add_argument("--score-timeout", default="")
    args = parser.parse_args()
    try:
        initialize_or_validate(args.path, canonical_config(args))
    except (
        ArtifactError,
        AssetIdentityError,
        HostContractError,
        PhaseDeclarationError,
    ) as error:
        parser.exit(2, f"evaluation-config: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
