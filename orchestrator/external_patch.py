#!/usr/bin/env python3
"""Import and verify an operator-supplied patch for Formal-only replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from lifecycle import LifecycleError, atomic_json, read_json, sha256_file, utc_now

MAX_EXTERNAL_PATCH_BYTES = 64 * 1024 * 1024


def _stage_source(source: Path, out_dir: Path) -> tuple[Path, int, str]:
    """Copy and hash a bounded patch in one streaming pass."""

    if not source.is_file():
        raise LifecycleError(f"external candidate patch is not a regular file: {source}")
    out_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".candidate.patch.", suffix=".tmp", dir=out_dir
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    size = 0
    try:
        with os.fdopen(descriptor, "wb") as target, source.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                size += len(block)
                if size > MAX_EXTERNAL_PATCH_BYTES:
                    raise LifecycleError(
                        "external candidate patch exceeds the 64 MiB public replay limit"
                    )
                digest.update(block)
                target.write(block)
            target.flush()
            os.fsync(target.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    if size == 0:
        temporary.unlink(missing_ok=True)
        raise LifecycleError(
            "external candidate patch is empty; replay requires an explicitly submitted "
            "non-empty source patch"
        )
    return temporary, size, digest.hexdigest()


def _require_exact_receipt(
    path: Path,
    payload: dict[str, Any] | None,
    stable: dict[str, Any],
    variable_fields: set[str],
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    if payload is None or set(payload) != set(stable) | variable_fields:
        raise LifecycleError(
            "existing external-patch lifecycle is incomplete or malformed; use a new run name"
        )
    if any(payload.get(key) != value for key, value in stable.items()):
        raise LifecycleError(
            "existing external-patch lifecycle conflicts with the supplied patch; "
            "use a new run name"
        )
    if any(not isinstance(payload.get(key), str) or not payload[key] for key in variable_fields):
        raise LifecycleError(
            "existing external-patch lifecycle is incomplete or malformed; use a new run name"
        )
    return payload


def import_external_patch(source: Path, out_dir: Path) -> dict[str, Any]:
    """Copy a non-empty patch and bind a resumable lifecycle to its exact bytes."""

    staged, patch_bytes, digest = _stage_source(source, out_dir)
    patch = out_dir / "candidate.patch"
    provenance_path = out_dir / "external_patch.json"
    lifecycle_path = out_dir / "lifecycle.json"

    provenance_stable = {
        "schema_version": 1,
        "record_type": "external_patch",
        "provenance": "external_patch",
        "imported_path": str(patch.resolve()),
        "patch_bytes": patch_bytes,
        "patch_sha256": digest,
    }
    lifecycle_stable = {
        "schema_version": 1,
        "record_type": "exploration_lifecycle",
        "agent_exit_state": "not_run",
        "termination_reason": "external_patch_supplied",
        "submission_origin": "external_patch",
        "candidate_state": "nonempty",
        "remaining_seconds_at_termination": None,
        "candidate_patch_bytes": patch_bytes,
        "candidate_patch_sha256": digest,
        "active_processes_at_termination": [],
        "active_gpu_processes_at_termination": [],
        "active_work_at_submit": None,
        "active_work_at_termination": None,
        "agent_session_active_at_termination": None,
        "candidate_rejection_reason": None,
        "raw_agent_exit_status": None,
    }
    try:
        existing_provenance = _require_exact_receipt(
            provenance_path,
            read_json(provenance_path),
            provenance_stable,
            {"source_path", "imported_at"},
        )
        existing_lifecycle = _require_exact_receipt(
            lifecycle_path,
            read_json(lifecycle_path),
            lifecycle_stable,
            {"recorded_at"},
        )
        recorded_times: set[str] = set()
        if existing_provenance:
            recorded_times.add(str(existing_provenance["imported_at"]))
        if existing_lifecycle:
            recorded_times.add(str(existing_lifecycle["recorded_at"]))
        if len(recorded_times) > 1:
            raise LifecycleError(
                "existing external-patch lifecycle has conflicting timestamps; "
                "use a new run name"
            )
        imported_at = next(iter(recorded_times), utc_now())

        if patch.exists() or patch.is_symlink():
            if (
                patch.is_symlink()
                or not patch.is_file()
                or patch.stat().st_size != patch_bytes
                or sha256_file(patch) != digest
            ):
                raise LifecycleError(
                    "existing external-patch lifecycle conflicts with the supplied patch; "
                    "use a new run name"
                )
            staged.unlink(missing_ok=True)
        else:
            staged.replace(patch)
        provenance = existing_provenance or {
            **provenance_stable,
            "source_path": str(source.resolve()),
            "imported_at": imported_at,
        }
        lifecycle = existing_lifecycle or {
            **lifecycle_stable,
            "recorded_at": imported_at,
        }
        if existing_provenance is None:
            atomic_json(provenance_path, provenance)
        if existing_lifecycle is None:
            atomic_json(lifecycle_path, lifecycle)
        return lifecycle
    finally:
        staged.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = import_external_patch(args.source, args.out)
    except LifecycleError as error:
        parser.exit(2, f"external-patch: {error}\n")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
