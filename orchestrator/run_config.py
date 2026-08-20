#!/usr/bin/env python3
"""Create or validate the immutable configuration of one resumable lifecycle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from asset_identity import (
    AssetIdentityError,
    aliases_from_phases,
    expected_contract,
)
from external_patch import MAX_EXTERNAL_PATCH_BYTES
from host_contract import HostContractError, require_contract, verify_contract
from phase_identity import content_digest, phase_payload
from phase_loader import PhaseDeclarationError, load_phases
from task import load_task


def _positive_timeout(value: str | int | None, *, declared: int, name: str) -> int:
    if value in (None, ""):
        return declared
    try:
        resolved = int(value)
    except (TypeError, ValueError) as error:
        raise SystemExit(f"{name} timeout must be a positive integer, got {value!r}") from error
    if resolved <= 0:
        raise SystemExit(f"{name} timeout must be a positive integer, got {resolved}")
    return resolved


def _effective_exports(values: list[str]) -> dict[str, str]:
    """Resolve repeated KEY=VALUE arguments with the runner's last-wins semantics."""

    resolved: dict[str, str] = {}
    for item in values:
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise SystemExit(f"--retrain-export wants KEY=VALUE, got {item!r}")
        resolved[key] = value
    return resolved


def canonical_config(args: argparse.Namespace) -> dict[str, Any]:
    task = args.task.resolve()
    source_check = args.source_check
    try:
        contract = verify_contract(task, mode=source_check)
        require_contract(contract)
    except HostContractError as error:
        raise SystemExit(str(error)) from error
    try:
        phases = load_phases(task, namespace="run_config")
    except PhaseDeclarationError as error:
        raise SystemExit(str(error)) from error
    score_phase = args.score_phase
    required = ("explore", "retrain", score_phase, "checkpoint-validate")
    missing = sorted(set(required) - set(phases))
    if missing:
        raise SystemExit(f"task declaration is missing phases required by trial.sh: {missing}")

    try:
        asset_identity = expected_contract(
            task,
            aliases_from_phases(phases),
            mode=source_check,
        )
    except AssetIdentityError as error:
        raise SystemExit(str(error)) from error

    task_config = load_task(task)
    declared_image = task_config.get("environment", {}).get("image")
    declared_timeouts = {
        "explore": int(phases["explore"].timeout_sec),
        "retrain": int(phases["retrain"].timeout_sec),
        "score": int(phases[score_phase].timeout_sec),
        "checkpoint_validation": int(phases["checkpoint-validate"].timeout_sec),
    }
    effective_timeouts = {
        "explore": _positive_timeout(
            args.explore_timeout, declared=declared_timeouts["explore"], name="explore"
        ),
        "retrain": _positive_timeout(
            args.retrain_timeout, declared=declared_timeouts["retrain"], name="retrain"
        ),
        "score": _positive_timeout(
            args.score_timeout, declared=declared_timeouts["score"], name=score_phase
        ),
        "checkpoint_validation": declared_timeouts["checkpoint_validation"],
    }
    retrain_exports = _effective_exports(args.retrain_export)
    try:
        effective_phases = {
            "explore": phases["explore"].with_timeout(effective_timeouts["explore"]),
            "retrain": phases["retrain"]
            .with_timeout(effective_timeouts["retrain"])
            .with_extra_exports(retrain_exports),
            "checkpoint_validation": phases["checkpoint-validate"],
            "score": phases[score_phase].with_timeout(effective_timeouts["score"]),
        }
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    effective_phase_payload = {
        name: phase_payload(phase) for name, phase in effective_phases.items()
    }
    model = args.model
    effort = args.effort
    if args.agent:
        # Metadata resolution deliberately does not require the host CLI binary.  A
        # completed Explore may be resumed into formal retraining on a machine that
        # does not have Codex/Claude installed; binary validation belongs to the actual
        # agent launch, not immutable run-config validation.
        import agent as agent_runtime

        specification = agent_runtime.resolve_metadata(args.agent)
        model = model or specification.default_model or ""
        effort = effort or specification.reasoning_effort or ""
    host_identity = {
        "mode": contract["mode"],
        "status": contract["status"],
        "algorithm": contract["algorithm"],
        "expected": None if source_check == "off" else contract["expected"],
        "observed": None if source_check == "off" else contract["observed"],
    }
    concurrency = int(getattr(args, "agent_api_concurrency", 0))
    if concurrency < 0:
        raise SystemExit("--agent-api-concurrency must be zero or a positive integer")
    concurrency_root = getattr(args, "agent_api_concurrency_root", "")
    if concurrency > 0:
        concurrency_root = str(Path(concurrency_root).resolve())
    else:
        concurrency_root = None
    retrain_export_receipt = {
        "names": sorted(retrain_exports),
        "sha256": content_digest(retrain_exports),
    }
    candidate_patch = getattr(args, "candidate_patch", None)
    candidate_identity = None
    execution_mode = "agent_explore"
    if candidate_patch:
        candidate_path = Path(candidate_patch)
        if not candidate_path.is_file():
            raise SystemExit(
                f"--candidate-patch is not a regular file: {candidate_path}"
            )
        if candidate_path.stat().st_size > MAX_EXTERNAL_PATCH_BYTES:
            raise SystemExit(
                "--candidate-patch exceeds the 64 MiB public replay limit"
            )
        digest = hashlib.sha256()
        with candidate_path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        candidate_identity = {
            "provenance": "external_patch",
            "sha256": digest.hexdigest(),
            "size_bytes": candidate_path.stat().st_size,
        }
        execution_mode = "external_patch_replay"
    return {
        "schema_version": 3,
        "task": str(task),
        "assets": str(args.assets.resolve()),
        # The path is provenance only. Official identity is the locked content
        # digest, so an identical asset store may move between hosts/phases.
        "asset_identity": asset_identity,
        "gpu": str(args.gpu),
        "agent": args.agent,
        "model": model,
        "reasoning_effort": effort,
        "image": args.image or declared_image,
        "source_check": source_check,
        "image_check": args.image_check,
        "hardware_check": args.hardware_check,
        "image_pull_policy": args.image_pull_policy,
        "declared_timeouts": declared_timeouts,
        "effective_timeouts": effective_timeouts,
        "score_phase": score_phase,
        "retrain_exports": retrain_export_receipt,
        "phase_contract": effective_phase_payload,
        "phase_contract_sha256": content_digest(effective_phase_payload),
        "host_contract": host_identity,
        "auto_retrain": bool(int(getattr(args, "auto_retrain", 1))),
        "execution_mode": execution_mode,
        "candidate_patch_identity": candidate_identity,
        "agent_max_attempts": args.agent_max_attempts,
        "agent_api_concurrency": concurrency,
        "agent_api_concurrency_root": concurrency_root,
    }


def initialize_or_validate(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise SystemExit(f"run config is unreadable: {path}: {error}") from error
        # Hardware type was not checked before this field existed.  Preserve
        # resumability only for the new honest local default; requesting strict on
        # an old run still requires a new run name rather than retroactively calling
        # earlier phases official.
        existing.setdefault("hardware_check", "warn")
        existing.setdefault("image_pull_policy", "missing")
        # Schema-v3 ordinary Agent runs predate the explicit execution mode.  These
        # defaults describe exactly their historical semantics and preserve resume;
        # external-patch runs never existed without a bound patch identity.
        existing.setdefault("execution_mode", "agent_explore")
        existing.setdefault("candidate_patch_identity", None)
        portable_assets = (
            existing.get("asset_identity") == value.get("asset_identity")
            and isinstance(value.get("asset_identity"), dict)
            and value["asset_identity"].get("mode") == "strict"
            and value["asset_identity"].get("status") == "locked"
        )
        differences = {
            key: {"recorded": existing.get(key), "requested": value.get(key)}
            for key in sorted(set(existing) | set(value))
            if existing.get(key) != value.get(key) and not (key == "assets" and portable_assets)
        }
        if differences:
            detail = "\n".join(
                f"  {key}: recorded={row['recorded']!r} requested={row['requested']!r}"
                for key, row in differences.items()
            )
            raise SystemExit(
                f"refusing to splice a resumed lifecycle across configurations:\n{detail}\n"
                "Use a new run name for the changed configuration."
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Do not silently bless an old lifecycle with whatever arguments happened to be
    # supplied by its first invocation after this receipt was introduced.  Historical
    # phase stamps or output cannot prove task/assets/image continuity.
    temporary_prefix = f".{path.name}."
    existing_entries = [
        entry
        for entry in path.parent.iterdir()
        if entry != path
        and entry.name != ".trial.lock"
        and not entry.name.startswith(temporary_prefix)
    ]
    if existing_entries:
        preview = ", ".join(sorted(entry.name for entry in existing_entries)[:8])
        raise SystemExit(
            f"refusing to adopt populated legacy run without {path.name}: {preview}. "
            "Use a new run name, or perform an explicit evidence-backed migration."
        )
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=temporary_prefix, dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # link(2) publishes a fully-written inode only if the destination is still
        # absent. Unlike replace(), concurrent first invocations cannot overwrite one
        # another with different configurations; the loser validates the winner.
        try:
            os.link(temporary, path)
        except FileExistsError:
            initialize_or_validate(path, value)
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
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--agent", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--effort", default="")
    parser.add_argument("--image", default="")
    parser.add_argument("--source-check", choices=("warn", "strict", "off"), required=True)
    parser.add_argument("--image-check", choices=("warn", "strict"), required=True)
    parser.add_argument("--hardware-check", choices=("warn", "strict", "off"), required=True)
    parser.add_argument("--image-pull-policy", choices=("missing", "never"), required=True)
    parser.add_argument("--explore-timeout", default="")
    parser.add_argument("--retrain-timeout", default="")
    parser.add_argument("--score-timeout", default="")
    parser.add_argument("--score-phase", choices=("score", "score-mock"), required=True)
    parser.add_argument("--retrain-export", action="append", default=[])
    parser.add_argument("--agent-max-attempts", type=int, default=0)
    parser.add_argument("--agent-api-concurrency", type=int, default=0)
    parser.add_argument(
        "--agent-api-concurrency-root",
        default=os.environ.get("AI4AI_AGENT_API_CONCURRENCY_ROOT", "/tmp/ai4ai-agent-api"),
    )
    parser.add_argument("--auto-retrain", type=int, choices=(0, 1), default=1)
    parser.add_argument("--candidate-patch", type=Path, default=None)
    args = parser.parse_args()
    initialize_or_validate(args.path, canonical_config(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
