#!/usr/bin/env python3
"""Verify the runtime asset aliases used by one or all AI4AI tasks.

Presence, regular-file byte totals and file counts are cheap and run by default.
``--hash`` additionally reads every byte and checks recorded file/tree digests; the
complete suite is roughly 110 GiB, so hashing is intentionally explicit.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError as error:  # pragma: no cover - command-line diagnosis
    raise SystemExit("PyYAML is required; install the project with `pip install -e .`") from error


REPO = Path(__file__).resolve().parents[1]
TASKS = REPO / "tasks"
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class FileRecord:
    path: str
    sha256: str
    size_bytes: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_files(root: Path, *, include_hashes: bool) -> tuple[list[FileRecord], list[str]]:
    """Return the canonical asset manifest and unsupported filesystem entries."""

    if root.is_file() and not root.is_symlink():
        digest = sha256_file(root) if include_hashes else ""
        return [FileRecord(root.name, digest, root.stat().st_size)], []
    records: list[FileRecord] = []
    unsupported: list[str] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if ".cache" in relative.parts:
            continue
        if path.is_symlink():
            unsupported.append(str(relative))
            continue
        if not path.is_file():
            continue
        digest = sha256_file(path) if include_hashes else ""
        records.append(FileRecord(relative.as_posix(), digest, path.stat().st_size))
    return records, unsupported


def tree_manifest_sha256(records: list[FileRecord]) -> str:
    """Hash the exact compact-JSON manifest used by every schema-v2 asset lock."""

    payload = [
        {"path": item.path, "sha256": item.sha256, "size_bytes": item.size_bytes}
        for item in records
    ]
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def required_aliases(task: Path) -> set[str]:
    """Read asset: constants from the declaration without executing task code."""

    tree = ast.parse((task / "declaration.py").read_text(encoding="utf-8"))
    return {
        value.value.removeprefix("asset:")
        for value in ast.walk(tree)
        if isinstance(value, ast.Constant)
        and isinstance(value.value, str)
        and value.value.startswith("asset:")
    }


def lock_entries(task: Path) -> dict[str, dict[str, Any]]:
    payload = yaml.safe_load((task / "environment/assets.lock.yaml").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    if isinstance(payload.get("assets"), dict):
        return {
            str(name): value for name, value in payload["assets"].items() if isinstance(value, dict)
        }
    entries: dict[str, dict[str, Any]] = {}
    for group in ("models", "data"):
        for name, value in (payload.get(group) or {}).items():
            if isinstance(value, dict):
                entries[f"{group}/{name}"] = value
    return entries


def _expected_count(entry: dict[str, Any]) -> int | None:
    for key in ("file_count", "expected_files"):
        value = entry.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _single_file(root: Path, entry: dict[str, Any], records: list[FileRecord]) -> Path | None:
    if root.is_file():
        return root
    named = entry.get("file")
    if isinstance(named, str):
        candidate = root / named
        return candidate if candidate.is_file() else None
    required = entry.get("required_files")
    if isinstance(required, list) and len(required) == 1 and isinstance(required[0], str):
        candidate = root / required[0]
        if not candidate.is_file():
            candidate = root / Path(required[0]).name
        return candidate if candidate.is_file() else None
    if len(records) == 1:
        return root / records[0].path
    return None


def verify_alias(
    alias: str, root: Path, entry: dict[str, Any], *, include_hashes: bool
) -> dict[str, Any]:
    result: dict[str, Any] = {"alias": alias, "path": str(root), "status": "ok"}
    if not root.exists():
        return {**result, "status": "missing", "problems": ["path does not exist"]}
    records, unsupported = regular_files(root, include_hashes=include_hashes)
    size = sum(item.size_bytes for item in records)
    result.update(files=len(records), size_bytes=size)
    problems: list[str] = []
    if unsupported:
        problems.append(f"unsupported symlinks: {', '.join(unsupported[:5])}")
    expected_size = entry.get("size_bytes")
    if isinstance(expected_size, int) and size != expected_size:
        problems.append(f"size {size} != recorded {expected_size}")
    expected_count = _expected_count(entry)
    if expected_count is not None and len(records) != expected_count:
        problems.append(f"file count {len(records)} != recorded {expected_count}")

    digest = entry.get("content_sha256")
    hash_kind = entry.get("hash_kind")
    result["hash_status"] = "not_checked"
    if include_hashes and isinstance(digest, str) and HEX_SHA256.fullmatch(digest):
        if hash_kind == "tree_manifest_sha256":
            observed = tree_manifest_sha256(records)
        elif hash_kind == "file_sha256":
            candidate = _single_file(root, entry, records)
            if candidate is None:
                observed = None
                problems.append("file_sha256 does not identify exactly one regular file")
            else:
                observed = sha256_file(candidate)
        else:
            observed = None
            problems.append(f"unsupported hash_kind {hash_kind!r}")
        if observed is not None:
            result.update(hash_status="match" if observed == digest else "mismatch")
            if observed != digest:
                problems.append(f"sha256 {observed} != recorded {digest}")
    elif include_hashes:
        # Partial legacy identities are still useful diagnostics, but they are not a
        # complete alias identity: extra tokenizer/config/data files could otherwise
        # change while ``--hash`` reported success.  Exact mode therefore always
        # rejects an alias without one aggregate digest.
        file_map = entry.get("files")
        if isinstance(file_map, dict):
            mismatches = []
            for name, expected in file_map.items():
                candidate = root / str(name)
                if not candidate.is_file() or sha256_file(candidate) != expected:
                    mismatches.append(str(name))
            result["hash_status"] = "match" if not mismatches else "mismatch"
            if mismatches:
                problems.append(f"per-file digest mismatch: {', '.join(mismatches)}")
        else:
            weight = entry.get("model_safetensors_sha256")
            candidate = root / "model.safetensors"
            if isinstance(weight, str) and HEX_SHA256.fullmatch(weight):
                observed = sha256_file(candidate) if candidate.is_file() else None
                result["hash_status"] = "match" if observed == weight else "mismatch"
                if observed != weight:
                    problems.append("model.safetensors is missing or has the wrong digest")
            else:
                result["hash_status"] = "unrecorded"
        problems.append("no complete content_sha256 is recorded for this alias")
    if problems:
        result.update(status="invalid", problems=problems)
    return result


def resolve_task(value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        direct = (REPO / candidate).resolve()
        candidate = direct if (direct / "task.toml").is_file() else TASKS / value
    candidate = candidate.resolve()
    if not (candidate / "task.toml").is_file():
        raise SystemExit(f"unknown task: {value}")
    return candidate


def verify_task(task: Path, assets: Path, *, include_hashes: bool) -> dict[str, Any]:
    entries = lock_entries(task)
    rows = []
    for alias in sorted(required_aliases(task)):
        entry = entries.get(alias)
        if entry is None:
            rows.append(
                {
                    "alias": alias,
                    "path": str(assets / alias),
                    "status": "invalid",
                    "problems": ["alias is absent from assets.lock.yaml"],
                }
            )
            continue
        rows.append(verify_alias(alias, assets / alias, entry, include_hashes=include_hashes))
    return {
        "task": task.name,
        "assets": str(assets.resolve()),
        "hashes_checked": include_hashes,
        "status": "ok" if all(row["status"] == "ok" for row in rows) else "invalid",
        "aliases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--task", help="task id or task directory")
    selector.add_argument("--all", action="store_true", help="verify every task")
    parser.add_argument("--assets", type=Path, help="one task's asset root")
    parser.add_argument("--assets-root", type=Path, help="parent containing task-id roots")
    parser.add_argument("--hash", action="store_true", help="read and verify all asset bytes")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.all:
        if args.assets_root is None:
            parser.error("--all requires --assets-root")
        jobs = [
            (task, args.assets_root / task.name)
            for task in sorted(TASKS.iterdir())
            if (task / "task.toml").is_file()
        ]
    else:
        if args.assets is None:
            parser.error("--task requires --assets")
        jobs = [(resolve_task(args.task), args.assets)]
    reports = [verify_task(task, assets, include_hashes=args.hash) for task, assets in jobs]
    if args.json:
        print(json.dumps(reports if args.all else reports[0], indent=2, sort_keys=True))
    else:
        for report in reports:
            print(f"{report['task']}: {report['status']} ({report['assets']})")
            for row in report["aliases"]:
                detail = "; ".join(row.get("problems", []))
                suffix = f" - {detail}" if detail else ""
                print(
                    f"  {row['status']:7} {row['alias']} "
                    f"files={row.get('files', '-')} bytes={row.get('size_bytes', '-')}"
                    f" hash={row.get('hash_status', '-')}" + suffix
                )
    return 0 if all(report["status"] == "ok" for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
