#!/usr/bin/env python3
"""Content identity for the host-side benchmark protocol.

The container image receipt covers code copied into the image.  It does not cover the
host files that choose the phase, mounts, command, timeout, artifact selection, and
scoring path.  This module gives those files a release-owned content identity without
depending on a Git branch, commit, tag, or clean working tree.

``host_contract.lock.json`` is deliberately outside its own digest.  Updating it is a
release action: edit the protocol, review the diff, then run this module with
``--refresh-lock``.  Ordinary development can use the existing warn/off policies, but
only a strict match is eligible for an official replay receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

CONTRACT_SCHEMA_VERSION = 1
CONTRACT_ALGORITHM = "sha256-canonical-file-map-v1"
LOCK_NAME = "host_contract.lock.json"

TASK_FILES = ("task.toml", "declaration.py", "instruction.md", "environment/assets.lock.yaml")


class HostContractError(RuntimeError):
    """The checked-out host protocol cannot be identified as the release contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_map(paths: dict[str, Path]) -> dict[str, str]:
    missing = [label for label, path in paths.items() if not path.is_file()]
    if missing:
        raise HostContractError(
            "host benchmark contract is incomplete; missing " + ", ".join(sorted(missing))
        )
    symlinks = [label for label, path in paths.items() if path.is_symlink()]
    if symlinks:
        raise HostContractError(
            "host benchmark contract refuses symlinked source: " + ", ".join(sorted(symlinks))
        )
    return {label: _sha256(path) for label, path in sorted(paths.items())}


def _digest(orchestrator_files: dict[str, str], task_files: dict[str, str]) -> str:
    payload = {
        "algorithm": CONTRACT_ALGORITHM,
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "orchestrator_files": orchestrator_files,
        "task_files": task_files,
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def observed_contract(
    task_dir: Path,
    *,
    instruction: Path | None = None,
    orchestrator_dir: Path | None = None,
) -> dict[str, Any]:
    """Return a checkout-location-independent content identity for one task."""

    task_dir = task_dir.resolve()
    orchestrator_dir = (orchestrator_dir or Path(__file__).resolve().parent).resolve()
    instruction = (instruction or task_dir / "instruction.md").resolve()
    # All executable Python and shell sources below orchestrator participate.  This
    # catches a newly added recovery or helper path automatically instead of relying on
    # an allowlist that can forget the very file it is meant to bind.  Generated caches
    # and the JSON lock are outside the selector.
    orchestrator_paths = {
        path.relative_to(orchestrator_dir).as_posix(): path
        for path in orchestrator_dir.rglob("*")
        if path.is_file() and path.suffix in {".py", ".sh"} and "__pycache__" not in path.parts
    }
    orchestrator_files = _hash_map(
        {f"orchestrator/{relative}": path for relative, path in orchestrator_paths.items()}
    )
    task_paths = {
        "task/task.toml": task_dir / "task.toml",
        "task/declaration.py": task_dir / "declaration.py",
        # A byte-identical explicitly supplied brief is equivalent to the default;
        # its host pathname is not part of the scientific protocol.
        "task/instruction.md": instruction,
        # Asset content hashes are only meaningful if the release owns the lock
        # that declares them. The large bytes stay outside this host contract.
        "task/environment/assets.lock.yaml": task_dir / "environment/assets.lock.yaml",
    }
    task_files = _hash_map(task_paths)
    return {
        "algorithm": CONTRACT_ALGORITHM,
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "task_id": task_dir.name,
        "digest": _digest(orchestrator_files, task_files),
        "orchestrator_digest": hashlib.sha256(
            json.dumps(orchestrator_files, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest(),
        "orchestrator_files": orchestrator_files,
        "task_files": task_files,
    }


def _read_lock(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as error:
        raise HostContractError(f"cannot read host contract lock {path}: {error}") from error
    if not isinstance(value, dict):
        raise HostContractError(f"host contract lock is not a JSON object: {path}")
    if value.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise HostContractError(
            f"unsupported host contract lock schema in {path}: {value.get('schema_version')!r}"
        )
    if value.get("algorithm") != CONTRACT_ALGORITHM:
        raise HostContractError(
            f"unsupported host contract algorithm in {path}: {value.get('algorithm')!r}"
        )
    return value


def verify_contract(
    task_dir: Path,
    *,
    mode: str,
    instruction: Path | None = None,
    orchestrator_dir: Path | None = None,
    lock_path: Path | None = None,
) -> dict[str, Any]:
    """Compare the current protocol with the release lock and return a receipt.

    The caller decides whether a mismatch is fatal.  This keeps the receipt format
    usable by runner, run-config, and tests while preserving warn/off development.
    """

    if mode not in {"strict", "warn", "off"}:
        raise HostContractError(f"unknown host contract mode {mode!r}")
    orchestrator_dir = (orchestrator_dir or Path(__file__).resolve().parent).resolve()
    lock_path = (lock_path or orchestrator_dir / LOCK_NAME).resolve()
    observed = observed_contract(
        task_dir,
        instruction=instruction,
        orchestrator_dir=orchestrator_dir,
    )
    lock_error = None
    try:
        lock = _read_lock(lock_path)
    except HostContractError as error:
        if mode == "strict":
            raise
        lock = {}
        lock_error = str(error)
    task_rows = lock.get("tasks") if isinstance(lock.get("tasks"), dict) else {}
    expected_row = task_rows.get(observed["task_id"])
    expected = expected_row.get("digest") if isinstance(expected_row, dict) else None
    expected_files: dict[str, str] = {}
    if isinstance(lock.get("orchestrator_files"), dict):
        expected_files.update(lock["orchestrator_files"])
    if isinstance(expected_row, dict) and isinstance(expected_row.get("task_files"), dict):
        expected_files.update(expected_row["task_files"])
    observed_files = {**observed["orchestrator_files"], **observed["task_files"]}
    changed_files = sorted(
        label
        for label in set(expected_files) | set(observed_files)
        if expected_files.get(label) != observed_files.get(label)
    )
    if mode == "off":
        status = "not_checked"
    elif not isinstance(expected, str) or not expected:
        status = "unrecorded"
    elif expected == observed["digest"] and not changed_files:
        status = "match"
    else:
        status = "mismatch"
    try:
        lock_sha256 = _sha256(lock_path) if lock_path.is_file() else None
    except OSError as error:
        if mode == "strict":
            raise HostContractError(
                f"cannot hash host contract lock {lock_path}: {error}"
            ) from error
        lock_sha256 = None
        lock_error = lock_error or str(error)
    return {
        "mode": mode,
        "status": status,
        "algorithm": CONTRACT_ALGORITHM,
        "expected": expected,
        "observed": observed["digest"],
        "task_id": observed["task_id"],
        "orchestrator_digest": observed["orchestrator_digest"],
        # Host paths are not protocol identity and should not leak into a public
        # receipt.  The digest identifies the lock; this label is diagnostic only.
        "lock_file": lock_path.name,
        "lock_sha256": lock_sha256,
        "lock_error": lock_error,
        "changed_files": changed_files,
    }


def require_contract(receipt: dict[str, Any]) -> None:
    """Fail closed only for strict verification; warn/off remain development modes."""

    if receipt.get("mode") != "strict" or receipt.get("status") == "match":
        return
    changed = receipt.get("changed_files") or []
    detail = f"\n  changed: {', '.join(changed)}" if changed else ""
    raise HostContractError(
        "host benchmark contract verification failed\n"
        f"  status:   {receipt.get('status')}\n"
        f"  expected: {receipt.get('expected')}\n"
        f"  observed: {receipt.get('observed')}"
        f"{detail}\n"
        "Use warn/off for local development, or refresh the lock only as a reviewed "
        "benchmark release action. Git commit, branch, and worktree cleanliness are "
        "intentionally not checked."
    )


def lock_payload(repo_root: Path) -> dict[str, Any]:
    """Build the reviewed release lock for every task in a repository checkout."""

    repo_root = repo_root.resolve()
    orchestrator_dir = repo_root / "orchestrator"
    tasks_root = repo_root / "tasks"
    task_dirs = sorted(
        path
        for path in tasks_root.iterdir()
        if path.is_dir() and all((path / relative).is_file() for relative in TASK_FILES)
    )
    if not task_dirs:
        raise HostContractError(f"no complete tasks found below {tasks_root}")
    observed = [observed_contract(path, orchestrator_dir=orchestrator_dir) for path in task_dirs]
    common_files = observed[0]["orchestrator_files"]
    if any(row["orchestrator_files"] != common_files for row in observed[1:]):
        raise HostContractError("orchestrator content changed while the lock was generated")
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "algorithm": CONTRACT_ALGORITHM,
        "orchestrator_files": common_files,
        "tasks": {
            row["task_id"]: {
                "digest": row["digest"],
                "task_files": row["task_files"],
            }
            for row in observed
        },
    }


def write_lock(repo_root: Path, path: Path | None = None) -> Path:
    repo_root = repo_root.resolve()
    path = (path or repo_root / "orchestrator" / LOCK_NAME).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(lock_payload(repo_root), indent=2, sort_keys=True) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-lock",
        action="store_true",
        help="write the release lock after a reviewed protocol change",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--task", type=Path)
    parser.add_argument("--mode", choices=("strict", "warn", "off"), default="strict")
    args = parser.parse_args()
    if args.refresh_lock:
        print(write_lock(args.repo_root))
        return 0
    if args.task is None:
        parser.error("--task is required unless --refresh-lock is used")
    receipt = verify_contract(args.task, mode=args.mode)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    try:
        require_contract(receipt)
    except HostContractError as error:
        print(error, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
