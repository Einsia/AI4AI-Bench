"""Canonical, secret-safe identity for a resolved container phase."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def content_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def phase_payload(phase: Any) -> dict[str, Any]:
    exports = dict(sorted(phase.exports.items()))
    runtime_exports = {}
    for item in phase.exports_with_wall_clock:
        key, separator, value = item.partition("=")
        runtime_exports[key] = value if separator else ""
    runtime_exports = dict(sorted(runtime_exports.items()))
    return {
        "name": phase.name,
        "timeout_seconds": phase.timeout_sec,
        "command": list(phase.command),
        "mounts": [
            {"source": source, "target": target, "read_only": read_only}
            for source, target, read_only in phase.mounts
        ],
        "export_names": sorted(exports),
        # Values can carry credentials in local validation runs.  They still
        # participate in immutable identity, but only through this digest.
        "exports_sha256": content_digest(exports),
        "runtime_export_names": sorted(runtime_exports),
        "runtime_exports_sha256": content_digest(runtime_exports),
        "hooks": list(phase.hooks),
        "read_only_root": phase.read_only_root,
        "apply_patch": phase.apply_patch,
        "interactive": phase.interactive,
        "reserve_seconds": phase.reserve_sec,
        "free_gib": phase.free_gib,
        "output_glob": phase.output_glob,
        "artifact_limit": phase.artifact_limit,
        "artifact_kind": phase.artifact_kind,
        "checkpoint_glob": phase.checkpoint_glob,
        "checkpoint_payload": phase.checkpoint_payload,
        "checkpoint_kind": phase.checkpoint_kind,
        "pass_image_digest": phase.pass_image_digest,
    }
