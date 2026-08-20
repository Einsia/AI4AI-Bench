"""Content identity for the runtime assets mounted into benchmark phases.

Host paths are provenance, not identity.  Official replay instead hashes every
task-required asset alias against ``environment/assets.lock.yaml``.  This module is
kept host-side so it can run before any container receives a mount.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError as error:  # pragma: no cover - packaging diagnosis
    raise RuntimeError("runtime asset verification requires PyYAML") from error


ALGORITHM = "sha256-locked-runtime-assets-v1"
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class AssetIdentityError(RuntimeError):
    """A required runtime asset cannot be identified as the locked content."""


def aliases_from_phases(phases: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                source.removeprefix("asset:")
                for phase in phases.values()
                for source, _, _ in phase.mounts
                if source.startswith("asset:")
            }
        )
    )


def aliases_from_phase(phase: Any) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                source.removeprefix("asset:")
                for source, _, _ in phase.mounts
                if source.startswith("asset:")
            }
        )
    )


def _lock_entries(task: Path) -> dict[str, dict[str, Any]]:
    path = task / "environment/assets.lock.yaml"
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise AssetIdentityError(f"cannot read runtime asset lock {path}: {error}") from error
    if not isinstance(payload, dict):
        raise AssetIdentityError(f"runtime asset lock is not a mapping: {path}")
    if isinstance(payload.get("assets"), dict):
        return {
            str(name): value
            for name, value in payload["assets"].items()
            if isinstance(value, dict)
        }
    entries: dict[str, dict[str, Any]] = {}
    for group in ("models", "data"):
        values = payload.get(group) or {}
        if isinstance(values, dict):
            entries.update(
                {
                    f"{group}/{name}": value
                    for name, value in values.items()
                    if isinstance(value, dict)
                }
            )
    return entries


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _records(root: Path) -> list[dict[str, Any]]:
    if root.is_symlink():
        raise AssetIdentityError(f"runtime asset alias is a symlink: {root}")
    if root.is_file():
        return [{"path": root.name, "sha256": _sha256(root), "size_bytes": root.stat().st_size}]
    if not root.is_dir():
        raise AssetIdentityError(f"runtime asset alias does not exist: {root}")
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if ".cache" in relative.parts:
            continue
        if path.is_symlink():
            raise AssetIdentityError(f"runtime asset contains a symlink: {root / relative}")
        if path.is_file():
            records.append(
                {
                    "path": relative.as_posix(),
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
        elif not path.is_dir():
            raise AssetIdentityError(
                f"runtime asset contains a non-regular entry: {root / relative}"
            )
    return records


def _tree_digest(records: list[dict[str, Any]]) -> str:
    encoded = json.dumps(records, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _expected_count(entry: dict[str, Any]) -> int | None:
    for key in ("file_count", "expected_files"):
        value = entry.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def expected_contract(task: Path, aliases: Iterable[str], *, mode: str) -> dict[str, Any]:
    """Bind lock-owned content identities without reading hidden/future phase assets."""

    aliases = tuple(sorted(set(aliases)))
    if mode != "strict":
        return {
            "mode": mode,
            "status": "not_checked",
            "algorithm": ALGORITHM,
            "digest": None,
            "aliases": list(aliases),
        }
    entries = _lock_entries(task)
    rows: list[dict[str, str]] = []
    for alias in aliases:
        entry = entries.get(alias) or {}
        expected = entry.get("content_sha256")
        kind = entry.get("hash_kind")
        if not isinstance(expected, str) or not HEX_SHA256.fullmatch(expected):
            raise AssetIdentityError(f"{alias}: no complete content_sha256")
        if kind not in {"tree_manifest_sha256", "file_sha256"}:
            raise AssetIdentityError(f"{alias}: unsupported hash_kind {kind!r}")
        rows.append({"alias": alias, "hash_kind": kind, "sha256": expected})
    identity = {"algorithm": ALGORITHM, "task_id": task.name, "aliases": rows}
    digest = hashlib.sha256(
        json.dumps(identity, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    return {
        "mode": mode,
        "status": "locked",
        "algorithm": ALGORITHM,
        "digest": digest,
        "aliases": rows,
    }


def _file_digest(root: Path, records: list[dict[str, Any]], entry: dict[str, Any]) -> str:
    if root.is_file():
        return records[0]["sha256"]
    named = entry.get("file")
    if isinstance(named, str):
        matches = [row for row in records if row["path"] == named]
    else:
        required = entry.get("required_files")
        if isinstance(required, list) and len(required) == 1 and isinstance(required[0], str):
            matches = [
                row
                for row in records
                if row["path"] in {required[0], Path(required[0]).name}
            ]
        else:
            matches = records
    if len(matches) != 1:
        raise AssetIdentityError(f"file_sha256 does not identify one file below {root}")
    return str(matches[0]["sha256"])


def verify_assets(
    task: Path,
    assets: Path,
    aliases: Iterable[str],
    *,
    mode: str,
    contract_aliases: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return a path-independent receipt, hashing bytes only in strict mode."""

    if mode not in {"strict", "warn", "off"}:
        raise AssetIdentityError(f"unknown runtime asset verification mode {mode!r}")
    aliases = tuple(sorted(set(aliases)))
    contract_aliases = tuple(sorted(set(contract_aliases or aliases)))
    if mode != "strict":
        return {
            "mode": mode,
            "status": "not_checked",
            "algorithm": ALGORITHM,
            "digest": None,
            "aliases": list(aliases),
        }
    contract = expected_contract(task, contract_aliases, mode=mode)
    entries = _lock_entries(task)
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    for alias in aliases:
        relative = Path(alias)
        if relative.is_absolute() or ".." in relative.parts:
            problems.append(f"invalid alias {alias!r}")
            continue
        entry = entries.get(alias)
        if entry is None:
            problems.append(f"{alias}: absent from assets.lock.yaml")
            continue
        expected = entry.get("content_sha256")
        kind = entry.get("hash_kind")
        if not isinstance(expected, str) or not HEX_SHA256.fullmatch(expected):
            problems.append(f"{alias}: no complete content_sha256")
            continue
        try:
            records = _records(assets / relative)
            if kind == "tree_manifest_sha256":
                observed = _tree_digest(records)
            elif kind == "file_sha256":
                observed = _file_digest(assets / relative, records, entry)
            else:
                raise AssetIdentityError(f"unsupported hash_kind {kind!r}")
        except (OSError, AssetIdentityError) as error:
            problems.append(f"{alias}: {error}")
            continue
        size = sum(int(row["size_bytes"]) for row in records)
        count = _expected_count(entry)
        alias_problems: list[str] = []
        if observed != expected:
            alias_problems.append(f"sha256 {observed} != recorded {expected}")
        if count is not None and len(records) != count:
            alias_problems.append(f"file count {len(records)} != recorded {count}")
        expected_size = entry.get("size_bytes")
        if isinstance(expected_size, int) and size != expected_size:
            alias_problems.append(f"size {size} != recorded {expected_size}")
        rows.append(
            {
                "alias": alias,
                "hash_kind": kind,
                "expected_sha256": expected,
                "observed_sha256": observed,
                "files": len(records),
                "size_bytes": size,
                "status": "match" if not alias_problems else "mismatch",
            }
        )
        problems.extend(f"{alias}: {problem}" for problem in alias_problems)
    status = "match" if not problems and len(rows) == len(aliases) else "mismatch"
    return {
        "mode": mode,
        "status": status,
        "algorithm": ALGORITHM,
        # Common across phases; each phase verifies only aliases it actually mounts.
        "digest": contract["digest"],
        "aliases": rows,
        "problems": problems,
    }


def require_assets(receipt: dict[str, Any]) -> None:
    if receipt.get("mode") == "strict" and receipt.get("status") != "match":
        detail = "\n".join(f"  {problem}" for problem in receipt.get("problems", []))
        raise AssetIdentityError(f"runtime asset identity verification failed:\n{detail}")
