"""Discover, validate and publish formal checkpoint candidates.

v1.5 recipes write ``/out/checkpoints/checkpoint-<numeric-progress>/``.  This
host-side module inventories every candidate, records frozen loadability
receipts, selects the latest valid candidates, and creates the internal
``final-artifacts`` view used by scorers.  The older output_glob API remains as
a compatibility path for v4/v5 manifests and declarations.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

WEIGHT_PATTERNS = (
    "*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt", "*.index.json",
)
CONFIG_MARKERS = ("config.json", "adapter_config.json")
ARTIFACT_KINDS = {"auto", "file", "weights", "model"}
ARTIFACT_RECEIPT_SCHEMA_VERSION = 2
CHECKPOINT_NAME = re.compile(r"^checkpoint-([0-9]+)$")
HARNESS_MARKER = ".harness-owned-v1.5"
EXTERNAL_CHECKPOINT_ALGORITHM = "sha256-external-checkpoint-tree-v1"


class ArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResolvedArtifact:
    progress: int
    candidate_path: str
    path: str
    kind: str


def load_phases(task: Path) -> dict[str, Any]:
    declaration = task / "declaration.py"
    if not declaration.is_file():
        raise ArtifactError(f"missing declaration: {declaration}")
    orchestrator = Path(__file__).resolve().parent
    if str(orchestrator) not in sys.path:
        sys.path.insert(0, str(orchestrator))
    spec = importlib.util.spec_from_file_location(
        f"{task.name}_artifact_declaration", declaration
    )
    if spec is None or spec.loader is None:
        raise ArtifactError(f"cannot import {declaration}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    phases = getattr(module, "PHASES", None)
    if not isinstance(phases, dict):
        raise ArtifactError(f"{declaration} defines no PHASES mapping")
    return phases


def _nonempty_matches(path: Path, patterns: tuple[str, ...]) -> list[Path]:
    return sorted(
        candidate
        for pattern in patterns
        for candidate in path.glob(pattern)
        if candidate.is_file() and candidate.stat().st_size > 0
    )


def has_weight_file(path: Path) -> bool:
    return bool(_nonempty_matches(path, WEIGHT_PATTERNS))


def loadable_model_directories(path: Path) -> list[Path]:
    directories = [path, *(item for item in path.rglob("*") if item.is_dir())]
    return [
        item for item in directories
        if has_weight_file(item)
        and any((item / marker).is_file() for marker in CONFIG_MARKERS)
    ]


def resolve_declared_match(
    path: Path, *, numbered: bool, artifact_kind: str = "auto"
) -> Path | None:
    """Return the exact file or directory a validator/scorer should receive."""

    if artifact_kind not in ARTIFACT_KINDS:
        raise ArtifactError(
            f"unknown artifact kind {artifact_kind!r}; expected {sorted(ARTIFACT_KINDS)}"
        )
    if artifact_kind == "file":
        return path if path.is_file() and path.stat().st_size > 0 else None
    if artifact_kind == "weights":
        return path if path.is_dir() and has_weight_file(path) else None
    if artifact_kind == "model":
        return (
            path if path.is_dir() and has_weight_file(path)
            and any((path / marker).is_file() for marker in CONFIG_MARKERS) else None
        )
    if path.is_file():
        return path if path.stat().st_size > 0 else None
    if not path.is_dir():
        return None
    if not numbered:
        return path if any(item.is_file() for item in path.rglob("*")) else None
    models = loadable_model_directories(path)
    if not models:
        return None
    preferred = path / "actor/huggingface"
    if preferred in models:
        return preferred
    if path in models:
        return path
    models.sort(key=lambda item: (len(item.relative_to(path).parts), str(item)))
    return models[0]


def numeric_suffix(path: Path) -> int | None:
    match = re.search(r"(\d+)$", path.name)
    return int(match.group(1)) if match else None


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _symlink_escape(candidate: Path) -> str | None:
    """Return the first symlink escaping a candidate root, if any."""

    root = candidate.resolve()
    if candidate.is_symlink():
        return str(candidate)
    if not candidate.is_dir():
        return None
    for item in candidate.rglob("*"):
        if item.is_symlink() and not _inside(item.resolve(), root):
            return str(item)
    return None


def resolve_artifact_series(
    out: Path, output_glob: str, *, artifact_limit: int, artifact_kind: str,
) -> dict[str, Any]:
    """Legacy v4/v5 output_glob resolver."""

    if not output_glob:
        raise ArtifactError("retrain phase declares no output_glob")
    if artifact_limit < 1:
        raise ArtifactError("artifact_limit must be at least one")
    root = out.resolve()
    numbered = any(character in output_glob for character in "*?[")
    candidates = sorted(out.glob(output_glob)) if numbered else [out / output_glob]
    valid: list[ResolvedArtifact] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        progress = numeric_suffix(candidate) if numbered else (numeric_suffix(candidate) or 0)
        if progress is None:
            rejected.append({"candidate_path": str(candidate), "reason": "missing numeric progress suffix"})
            continue
        resolved_candidate = candidate.resolve()
        if not _inside(resolved_candidate, root):
            rejected.append({"progress": progress, "candidate_path": str(candidate), "reason": "artifact resolves outside the formal output directory"})
            continue
        artifact = resolve_declared_match(candidate, numbered=numbered, artifact_kind=artifact_kind)
        if artifact is None:
            rejected.append({"progress": progress, "candidate_path": str(candidate), "reason": f"incomplete or not loadable as {artifact_kind}"})
            continue
        resolved = artifact.resolve()
        if not _inside(resolved, root):
            rejected.append({"progress": progress, "candidate_path": str(candidate), "reason": "scoreable artifact resolves outside the formal output directory"})
            continue
        valid.append(ResolvedArtifact(progress, str(resolved_candidate), str(resolved), artifact_kind))
    valid.sort(key=lambda item: (item.progress, item.path))
    accepted = valid[-artifact_limit:]
    return {
        "schema_version": 1, "output_root": str(root), "output_glob": output_glob,
        "artifact_kind": artifact_kind, "artifact_limit": artifact_limit,
        "selection_rule": "latest_complete_by_progress",
        "discovered_count": len(candidates),
        "accepted": [asdict(item) for item in accepted],
        "ignored": [{**asdict(item), "reason": "outside latest artifact limit"} for item in valid[:-artifact_limit]],
        "rejected": rejected,
    }


def discover_checkpoint_series(
    out: Path, checkpoint_glob: str, *, checkpoint_payload: str,
    artifact_limit: int, checkpoint_kind: str,
) -> dict[str, Any]:
    """Inventory standard candidates without deciding loadability or acceptance."""

    if checkpoint_glob != "checkpoints/checkpoint-*":
        raise ArtifactError(
            "v1.5 checkpoint_glob must be exactly 'checkpoints/checkpoint-*'"
        )
    if artifact_limit < 1:
        raise ArtifactError("artifact_limit must be at least one")
    if checkpoint_payload.startswith("/") or ".." in Path(checkpoint_payload).parts:
        raise ArtifactError("checkpoint_payload must stay below each candidate root")
    root = out.resolve()
    checkpoint_root = out / "checkpoints"
    rows: list[dict[str, Any]] = []
    if checkpoint_root.is_dir():
        for candidate in sorted(checkpoint_root.iterdir(), key=lambda item: item.name):
            match = CHECKPOINT_NAME.fullmatch(candidate.name)
            if not match:
                if "checkpoint-" in candidate.name:
                    rows.append({
                        "progress": None, "candidate_path": str(candidate),
                        "payload_path": None, "kind": checkpoint_kind,
                        "discovery_status": "invalid", "validation_status": "not_run",
                        "selection_status": "rejected",
                        "reason": "unfinished or non-canonical checkpoint name",
                    })
                continue
            progress = int(match.group(1))
            base = {
                "progress": progress, "candidate_path": str(candidate),
                "payload_path": None, "kind": checkpoint_kind,
                "discovery_status": "invalid", "validation_status": "not_run",
                "selection_status": "rejected", "reason": None,
            }
            resolved_candidate = candidate.resolve()
            if candidate.is_symlink() or not _inside(resolved_candidate, root):
                rows.append({**base, "reason": "checkpoint root is a symlink or resolves outside /out"})
                continue
            escape = _symlink_escape(candidate)
            if escape:
                rows.append({**base, "reason": f"checkpoint contains an escaping symlink: {escape}"})
                continue
            payload = candidate if checkpoint_payload == "." else candidate / checkpoint_payload
            resolved = payload.resolve()
            if not _inside(resolved, resolved_candidate):
                rows.append({**base, "reason": "checkpoint payload resolves outside its candidate root"})
                continue
            artifact = resolve_declared_match(payload, numbered=True, artifact_kind=checkpoint_kind)
            if artifact is None:
                rows.append({**base, "reason": f"incomplete structural payload for {checkpoint_kind}"})
                continue
            rows.append({
                **base, "payload_path": str(artifact.resolve()),
                "discovery_status": "valid", "validation_status": "pending",
                "selection_status": "pending", "reason": None,
            })
    progress_counts: dict[int, int] = {}
    for row in rows:
        if row["discovery_status"] == "valid":
            progress = int(row["progress"])
            progress_counts[progress] = progress_counts.get(progress, 0) + 1
    duplicates = {value for value, count in progress_counts.items() if count > 1}
    for row in rows:
        if row.get("progress") in duplicates:
            row.update(
                discovery_status="invalid", validation_status="not_run",
                selection_status="rejected",
                reason=f"ambiguous duplicate progress {row['progress']}",
            )
    rows.sort(key=lambda row: (
        row["progress"] is None,
        int(row["progress"]) if row["progress"] is not None else 0,
        row["candidate_path"],
    ))
    return {
        "schema_version": ARTIFACT_RECEIPT_SCHEMA_VERSION,
        "output_root": str(root), "checkpoint_glob": checkpoint_glob,
        "checkpoint_payload": checkpoint_payload, "checkpoint_kind": checkpoint_kind,
        "artifact_limit": artifact_limit,
        "selection_rule": "latest_frozen_loadable_by_numeric_progress",
        "candidates": rows, "accepted": [], "ignored": [],
        "rejected": [row for row in rows if row["discovery_status"] == "invalid"],
        "publication_root": None,
    }


def discover_task_checkpoints(task: Path, out: Path) -> dict[str, Any]:
    retrain = load_phases(task).get("retrain")
    if retrain is None:
        raise ArtifactError(f"{task}/declaration.py defines no retrain phase")
    checkpoint_glob = str(getattr(retrain, "checkpoint_glob", ""))
    if not checkpoint_glob:
        return resolve_artifact_series(
            out, str(retrain.output_glob),
            artifact_limit=int(getattr(retrain, "artifact_limit", 1)),
            artifact_kind=str(getattr(retrain, "artifact_kind", "auto")),
        )
    return discover_checkpoint_series(
        out, checkpoint_glob,
        checkpoint_payload=str(getattr(retrain, "checkpoint_payload", ".")),
        artifact_limit=int(getattr(retrain, "artifact_limit", 1)),
        checkpoint_kind=str(getattr(retrain, "checkpoint_kind", "auto")),
    )


def _external_checkpoint_identity(path: Path) -> dict[str, Any]:
    """Hash the exact external artifact bytes without following escaping links."""

    resolved = path.resolve()
    if not resolved.exists():
        raise ArtifactError(f"external checkpoint does not exist: {path}")
    if not (resolved.is_file() or resolved.is_dir()):
        raise ArtifactError(f"external checkpoint is not a regular file or directory: {path}")
    root_kind = "file" if resolved.is_file() else "directory"
    escape = _symlink_escape(resolved)
    if escape:
        raise ArtifactError(f"external checkpoint contains an escaping symlink: {escape}")

    records: list[dict[str, Any]] = []
    paths = [resolved] if resolved.is_file() else sorted(resolved.rglob("*"))
    for item in paths:
        relative = item.name if resolved.is_file() else item.relative_to(resolved).as_posix()
        if item.is_symlink():
            records.append(
                {
                    "path": relative,
                    "kind": "symlink",
                    "target": os.readlink(item),
                }
            )
            continue
        if item.is_dir():
            continue
        if not item.is_file():
            raise ArtifactError(f"external checkpoint contains a non-regular entry: {item}")
        digest = hashlib.sha256()
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        records.append(
            {
                "path": relative,
                "kind": "file",
                "sha256": digest.hexdigest(),
                "size_bytes": item.stat().st_size,
            }
        )
    if not records:
        raise ArtifactError(f"external checkpoint is empty: {path}")
    # Root kind is part of the digest domain, not just descriptive metadata. Docker
    # bind-mount semantics differ for a file and a directory even when the latter
    # contains one same-named file with identical bytes.
    encoded = json.dumps(
        {"root_kind": root_kind, "records": records},
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return {
        "algorithm": EXTERNAL_CHECKPOINT_ALGORITHM,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "root_kind": root_kind,
        "file_count": sum(row["kind"] == "file" for row in records),
        "symlink_count": sum(row["kind"] == "symlink" for row in records),
        "size_bytes": sum(int(row.get("size_bytes", 0)) for row in records),
    }


def external_checkpoint_series(
    task: Path, out: Path, checkpoints: list[tuple[int, Path]]
) -> dict[str, Any]:
    """Create a schema-v2 artifact receipt for 1--3 operator checkpoints."""

    if not 1 <= len(checkpoints) <= 3:
        raise ArtifactError("checkpoint-only evaluation requires 1 to 3 checkpoints")
    progresses = [progress for progress, _ in checkpoints]
    if any(progress < 0 for progress in progresses):
        raise ArtifactError("checkpoint progress must be a non-negative integer")
    if len(set(progresses)) != len(progresses):
        raise ArtifactError("checkpoint progress values must be unique")
    retrain = load_phases(task).get("retrain")
    if retrain is None:
        raise ArtifactError(f"{task}/declaration.py defines no retrain phase")
    declared_limit = int(getattr(retrain, "artifact_limit", 1))
    if len(checkpoints) > declared_limit:
        raise ArtifactError(
            f"task permits at most {declared_limit} artifacts, got {len(checkpoints)}"
        )
    kind = str(
        getattr(retrain, "checkpoint_kind", "")
        or getattr(retrain, "artifact_kind", "auto")
    )
    candidates = []
    seen_paths: dict[str, int] = {}
    for progress, supplied in sorted(checkpoints, key=lambda row: row[0]):
        resolved = supplied.resolve()
        resolved_text = str(resolved)
        if resolved_text in seen_paths:
            raise ArtifactError(
                f"external checkpoint path is duplicated at progress {seen_paths[resolved_text]} "
                f"and {progress}: {resolved}"
            )
        identity = _external_checkpoint_identity(supplied)
        seen_paths[resolved_text] = progress
        candidates.append(
            {
                "progress": progress,
                "candidate_path": str(resolved),
                "payload_path": str(resolved),
                "kind": kind,
                "discovery_status": "valid",
                "validation_status": "pending",
                "selection_status": "pending",
                "reason": None,
                "provenance": "external_checkpoint",
                "content_identity": identity,
            }
        )
    return {
        "schema_version": ARTIFACT_RECEIPT_SCHEMA_VERSION,
        "output_root": str(out.resolve()),
        "checkpoint_glob": None,
        "checkpoint_payload": ".",
        "checkpoint_kind": kind,
        "artifact_limit": declared_limit,
        "selection_rule": "validated_external_checkpoints",
        "provenance": "external_checkpoint",
        "candidates": candidates,
        "accepted": [],
        "ignored": [],
        "rejected": [],
        "publication_root": None,
    }


def initialize_external_receipt(
    receipt: Path, task: Path, out: Path, checkpoints: list[tuple[int, Path]]
) -> dict[str, Any]:
    requested = external_checkpoint_series(task, out, checkpoints)
    if not receipt.exists():
        write_receipt(receipt, requested)
        return requested
    existing = read_receipt(receipt)

    def identity(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": payload.get("schema_version"),
            "output_root": payload.get("output_root"),
            "checkpoint_kind": payload.get("checkpoint_kind"),
            "artifact_limit": payload.get("artifact_limit"),
            "provenance": payload.get("provenance"),
            "candidates": [
                {
                    "progress": row.get("progress"),
                    "candidate_path": row.get("candidate_path"),
                    "payload_path": row.get("payload_path"),
                    "kind": row.get("kind"),
                    "provenance": row.get("provenance"),
                    "content_identity": row.get("content_identity"),
                }
                for row in payload.get("candidates", [])
                if isinstance(row, dict)
            ],
        }

    if identity(existing) != identity(requested):
        raise ArtifactError(
            "existing evaluation receipt conflicts with supplied checkpoints; "
            "use a new evaluation name"
        )
    return existing


def verify_external_checkpoint(payload: dict[str, Any], progress: int) -> dict[str, Any]:
    """Re-hash one external checkpoint immediately around a container phase."""

    if payload.get("provenance") != "external_checkpoint":
        raise ArtifactError("artifact receipt is not an external-checkpoint receipt")
    matches = [
        row
        for row in payload.get("candidates", [])
        if isinstance(row, dict) and row.get("progress") == progress
    ]
    if len(matches) != 1:
        raise ArtifactError(f"expected one external checkpoint at progress {progress}")
    row = matches[0]
    path = Path(str(row.get("payload_path") or ""))
    observed = _external_checkpoint_identity(path)
    if observed != row.get("content_identity"):
        raise ArtifactError(
            f"external checkpoint at progress {progress} changed after evaluation "
            "initialization; use a new evaluation name"
        )
    return observed


def record_validation(payload: dict[str, Any], progress: int, receipt: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != ARTIFACT_RECEIPT_SCHEMA_VERSION:
        raise ArtifactError("frozen validation requires an artifact receipt at schema v2")
    status = receipt.get("status")
    if status not in {"valid", "invalid"}:
        raise ArtifactError("validation receipt must declare terminal status valid or invalid")
    matches = [row for row in payload["candidates"] if row.get("progress") == progress and row.get("discovery_status") == "valid"]
    if len(matches) != 1:
        raise ArtifactError(f"expected one structurally valid checkpoint at progress {progress}")
    row = matches[0]
    existing = row.get("validation_status")
    if existing in {"valid", "invalid"}:
        if existing != status or row.get("validation_receipt") != receipt:
            raise ArtifactError(
                f"conflicting terminal validation for checkpoint-{progress}"
            )
        return payload
    row["validation_status"] = status
    row["validation_receipt"] = receipt
    row["reason"] = receipt.get("reason") if status == "invalid" else None
    row["selection_status"] = "pending" if status == "valid" else "rejected"
    return payload


def _materialize_publications(out: Path, accepted: list[dict[str, Any]]) -> None:
    publication = out / "final-artifacts"
    marker = publication / HARNESS_MARKER
    if publication.exists() and not marker.is_file():
        raise ArtifactError(
            f"reserved harness directory already exists without ownership marker: {publication}"
        )
    publication.mkdir(parents=True, exist_ok=True)
    marker.write_text("schema=1\n", encoding="utf-8")
    expected = {f"artifact-{row['progress']}" for row in accepted}
    for item in publication.iterdir():
        if item.name == HARNESS_MARKER or item.name in expected:
            continue
        if not item.is_symlink():
            raise ArtifactError(f"unexpected non-symlink in harness publication directory: {item}")
        item.unlink()
    for row in accepted:
        link = publication / f"artifact-{row['progress']}"
        target = Path(row["payload_path"])
        relative = os.path.relpath(target, publication)
        if link.is_symlink() and link.resolve() == target.resolve():
            row["published_path"] = str(link)
            continue
        if link.exists() or link.is_symlink():
            if not link.is_symlink():
                raise ArtifactError(f"refusing to replace non-symlink publication: {link}")
            link.unlink()
        temporary = publication / f".{link.name}.{os.getpid()}.tmp"
        try:
            temporary.symlink_to(relative)
            temporary.replace(link)
        finally:
            if temporary.is_symlink():
                temporary.unlink()
        row["published_path"] = str(link)


def finalize_validated(payload: dict[str, Any], out: Path) -> dict[str, Any]:
    if payload.get("schema_version") != ARTIFACT_RECEIPT_SCHEMA_VERSION:
        return payload
    pending = [row for row in payload["candidates"] if row.get("validation_status") == "pending"]
    if pending:
        raise ArtifactError(
            "cannot select artifacts while frozen validation is pending for "
            + ", ".join(f"checkpoint-{row['progress']}" for row in pending)
        )
    valid = sorted(
        (row for row in payload["candidates"] if row.get("validation_status") == "valid"),
        key=lambda row: (int(row["progress"]), row["payload_path"]),
    )
    limit = int(payload["artifact_limit"])
    accepted = valid[-limit:]
    accepted_ids = {id(row) for row in accepted}
    for row in valid:
        row["selection_status"] = "accepted" if id(row) in accepted_ids else "ignored"
        row["reason"] = None if id(row) in accepted_ids else "outside latest valid artifact limit"
    _materialize_publications(out, accepted)
    payload["accepted"] = accepted
    payload["ignored"] = [row for row in valid if id(row) not in accepted_ids]
    payload["rejected"] = [row for row in payload["candidates"] if row.get("selection_status") == "rejected"]
    payload["publication_root"] = str((out / "final-artifacts").resolve())
    return payload


def write_receipt(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_receipt(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"cannot read artifact receipt {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactError(f"artifact receipt is not an object: {path}")
    return payload


def resolve_output_glob(out: Path, output_glob: str) -> Path:
    payload = resolve_artifact_series(out, output_glob, artifact_limit=1, artifact_kind="auto")
    if not payload["accepted"]:
        raise ArtifactError(f"no accepted artifact matches {output_glob!r} below {out}")
    return Path(payload["accepted"][-1]["path"])


def resolve_task_artifacts(task: Path, out: Path) -> dict[str, Any]:
    return discover_task_checkpoints(task, out)


def resolve_task_artifact(task: Path, out: Path) -> Path:
    payload = discover_task_checkpoints(task, out)
    if not payload["accepted"]:
        raise ArtifactError(f"no accepted artifact declared by {task} below {out}")
    row = payload["accepted"][-1]
    return Path(row.get("published_path") or row.get("path") or row["payload_path"])


def _print_rows(payload: dict[str, Any], kind: str) -> None:
    if kind == "validation-tsv":
        rows = [row for row in payload.get("candidates", []) if row.get("validation_status") == "pending"]
    elif kind == "candidates-tsv":
        rows = [
            row
            for row in payload.get("candidates", [])
            if row.get("discovery_status") == "valid"
        ]
    else:
        rows = payload.get("accepted", [])
    for row in rows:
        if payload.get("provenance") == "external_checkpoint":
            # The content identity binds payload_path. final-artifacts is only a
            # convenience view and its symlink must never become an evaluation input.
            path = row.get("payload_path")
        else:
            path = row.get("published_path") or row.get("path") or row.get("payload_path")
        print(f"{row['progress']}\t{path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        choices=(
            "discover",
            "external",
            "verify-external",
            "record-validation",
            "finalize",
            "list",
        ),
        default="discover",
    )
    parser.add_argument("--task", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--progress", type=int)
    parser.add_argument("--checkpoint", action="append", default=[], metavar="PROGRESS=PATH")
    parser.add_argument(
        "--format",
        choices=(
            "latest",
            "paths",
            "tsv",
            "validation-tsv",
            "candidates-tsv",
            "json",
        ),
        default="latest",
    )
    args = parser.parse_args()
    empty_legacy_discovery = False
    try:
        if args.command == "discover":
            if args.task is None or args.out is None:
                raise ArtifactError("discover requires --task and --out")
            payload = discover_task_checkpoints(args.task, args.out)
            if args.receipt:
                write_receipt(args.receipt, payload)
            empty_legacy_discovery = (
                payload.get("schema_version") == 1 and not payload.get("accepted")
            )
        elif args.command == "external":
            if args.task is None or args.out is None or args.receipt is None:
                raise ArtifactError("external requires --task, --out and --receipt")
            checkpoints: list[tuple[int, Path]] = []
            for item in args.checkpoint:
                raw_progress, separator, raw_path = item.partition("=")
                if not separator or not raw_progress.isdigit() or not raw_path:
                    raise ArtifactError(
                        f"--checkpoint wants non-negative PROGRESS=PATH, got {item!r}"
                    )
                checkpoints.append((int(raw_progress), Path(raw_path)))
            payload = initialize_external_receipt(
                args.receipt, args.task, args.out, checkpoints
            )
        elif args.command == "verify-external":
            if args.receipt is None or args.progress is None:
                raise ArtifactError("verify-external requires --receipt and --progress")
            payload = verify_external_checkpoint(read_receipt(args.receipt), args.progress)
        elif args.command == "record-validation":
            if args.receipt is None or args.validation is None or args.progress is None:
                raise ArtifactError("record-validation requires --receipt, --progress and --validation")
            payload = record_validation(read_receipt(args.receipt), args.progress, read_receipt(args.validation))
            write_receipt(args.receipt, payload)
        elif args.command == "finalize":
            if args.receipt is None or args.out is None:
                raise ArtifactError("finalize requires --receipt and --out")
            payload = finalize_validated(read_receipt(args.receipt), args.out)
            write_receipt(args.receipt, payload)
        else:
            if args.receipt is None:
                raise ArtifactError("list requires --receipt")
            payload = read_receipt(args.receipt)
        if args.format == "json":
            print(json.dumps(payload, sort_keys=True))
        elif args.format in {"tsv", "validation-tsv", "candidates-tsv"}:
            _print_rows(payload, args.format)
        elif args.format == "paths":
            for row in payload.get("accepted", []):
                print(row.get("published_path") or row.get("path") or row.get("payload_path"))
        elif payload.get("accepted"):
            row = payload["accepted"][-1]
            print(row.get("published_path") or row.get("path") or row.get("payload_path"))
    except ArtifactError as exc:
        parser.exit(1, f"artifact: {exc}\n")
    return 1 if empty_legacy_discovery else 0


if __name__ == "__main__":
    raise SystemExit(main())
