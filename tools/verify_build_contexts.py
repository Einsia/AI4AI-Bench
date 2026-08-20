#!/usr/bin/env python3
"""Verify and attest the named build contexts for AI4AI task images.

The verifier is intentionally read-only with respect to the contexts.  It checks
the byte identities that are already pinned by a task lock, rejects unsafe
filesystem entries, and writes a location-independent receipt.  A directory is
*not* release-ready merely because it exists: a context without authoritative
provenance in ``assets.lock.yaml`` (or, for wheelhouses, the task's hashed
requirements lock) is a failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError as error:  # pragma: no cover - command-line diagnosis
    raise SystemExit("PyYAML is required; install the project with `pip install -e .`") from error


REPO = Path(__file__).resolve().parents[1]
TASKS = REPO / "tasks"
COMMON_CONTEXT_LOCK = REPO / "tools" / "build-contexts.lock.yaml"
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
REQUIREMENT = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s;\\]+)")
LOCK_HASH = re.compile(r"--hash=sha256:([0-9a-f]{64})")
DOCKER_CONTEXT = re.compile(r"(?:--mount=[^\n]*?\bfrom=|--from=)([A-Za-z0-9_.-]+)")
FROM_STAGE = re.compile(
    r"^\s*FROM\s+\S+(?:\s+AS\s+([A-Za-z0-9_.-]+))?", re.IGNORECASE | re.MULTILINE
)

# These patterns are deliberately high-confidence.  A source file named
# ``credentials.py`` is not evidence of a credential; a private key or an auth
# token embedded in a URL is.
SECRET_BYTES = re.compile(
    rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|"
    rb"\b(?:ghp_|github_pat_|glpat-|sk-)[A-Za-z0-9_-]{16,}|"
    rb"\bhf_[A-Za-z0-9]{20,}|"
    rb"(?i:https?://[^\s/:]{1,64}:[^\s/@]{6,128}@)|"
    rb"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"
)
INTERNAL_BYTES = re.compile(
    rb"/(?:shared|cluster)/(?:home|scratch|poc)/|"
    rb"\b(?:[a-z]{2,}-gpu[0-9]+|[a-z][0-9]+-instan-[0-9]+)\b"
)
SUSPICIOUS_BASENAMES = {
    ".env",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "99proxy",
    "auth.json",
    "credentials.json",
    "id_ed25519",
    "id_rsa",
}


@dataclass(frozen=True)
class ContextRecord:
    path: str
    type: str
    mode: int
    size_bytes: int = 0
    sha256: str = ""
    target: str = ""


@dataclass(frozen=True)
class RegularRecord:
    path: str
    sha256: str
    size_bytes: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def context_manifest_sha256(records: list[ContextRecord]) -> str:
    """Hash content, layout, executable bits, and internal symlink targets."""

    return canonical_sha256([asdict(record) for record in records])


def tree_manifest_sha256(records: list[RegularRecord]) -> str:
    """Return the schema-v2 regular-tree digest used by asset locks."""

    payload = [
        {"path": item.path, "sha256": item.sha256, "size_bytes": item.size_bytes}
        for item in records
    ]
    return canonical_sha256(payload)


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _suspicious_path(relative: Path) -> bool:
    lowered = tuple(part.lower() for part in relative.parts)
    if relative.name.lower() in SUSPICIOUS_BASENAMES:
        return True
    if len(lowered) >= 2 and lowered[-2:] in {
        (".aws", "credentials"),
        (".docker", "config.json"),
    }:
        return True
    return False


def scan_file(path: Path, *, chunk_size: int = 1024 * 1024) -> tuple[bool, bool]:
    """Stream high-confidence credential and internal-path checks over every file."""

    secret = False
    internal = False
    tail = b""
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            window = tail + chunk
            secret = secret or SECRET_BYTES.search(window) is not None
            internal = internal or INTERNAL_BYTES.search(window) is not None
            if secret and internal:
                break
            tail = window[-512:]
    return secret, internal


def inventory_context(root: Path) -> tuple[list[ContextRecord], list[RegularRecord], list[str]]:
    """Inventory a context without following links, and return security failures."""

    canonical_root = root.resolve()
    context_records: list[ContextRecord] = []
    regular_records: list[RegularRecord] = []
    problems: list[str] = []

    def visit(directory: Path) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name.encode())
        except OSError as error:
            problems.append(f"cannot read {directory.relative_to(root)}: {error}")
            return
        for entry in entries:
            path = Path(entry.path)
            relative = path.relative_to(root)
            relative_text = relative.as_posix()
            try:
                metadata = path.lstat()
            except OSError as error:
                problems.append(f"cannot stat {relative_text}: {error}")
                continue
            mode = stat.S_IMODE(metadata.st_mode)
            if stat.S_ISLNK(metadata.st_mode):
                target = os.readlink(path)
                resolved = (path.parent / target).resolve(strict=False)
                context_records.append(ContextRecord(relative_text, "symlink", mode, target=target))
                if not _is_inside(resolved, canonical_root):
                    problems.append(f"symlink escapes context: {relative_text} -> {target}")
                elif not resolved.exists():
                    problems.append(f"broken symlink: {relative_text} -> {target}")
                if INTERNAL_BYTES.search(target.encode(errors="surrogateescape")):
                    problems.append(f"internal host path in symlink target: {relative_text}")
                continue
            if stat.S_ISDIR(metadata.st_mode):
                context_records.append(ContextRecord(relative_text, "directory", mode))
                visit(path)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                problems.append(f"unsupported filesystem entry: {relative_text}")
                continue

            digest = sha256_file(path)
            context_records.append(
                ContextRecord(relative_text, "file", mode, metadata.st_size, digest)
            )
            if ".git" not in relative.parts:
                regular_records.append(RegularRecord(relative_text, digest, metadata.st_size))
            if _suspicious_path(relative):
                problems.append(f"credential-like path in context: {relative_text}")
            try:
                secret, internal = scan_file(path)
            except OSError as error:
                problems.append(f"cannot scan {relative_text}: {error}")
                continue
            if secret:
                problems.append(f"credential-like bytes in context: {relative_text}")
            if internal:
                problems.append(f"internal host/path bytes in context: {relative_text}")

    visit(root)
    return context_records, regular_records, sorted(set(problems))


def _expected_count(entry: dict[str, Any]) -> int | None:
    for key in ("file_count", "expected_files"):
        value = entry.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _single_regular(root: Path, records: list[RegularRecord], entry: dict[str, Any]) -> Path | None:
    named = entry.get("file") or entry.get("payload_filename")
    if isinstance(named, str) and (root / named).is_file():
        return root / named
    if len(records) == 1:
        return root / records[0].path
    return None


def _normalise_distribution(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def requirements_lock(task: Path) -> dict[tuple[str, str], set[str]]:
    path = task / "environment/runtime-requirements.lock"
    if not path.is_file():
        return {}
    requirements: dict[tuple[str, str], set[str]] = {}
    current: tuple[str, str] | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        match = REQUIREMENT.match(raw.strip())
        if match:
            current = (_normalise_distribution(match.group(1)), match.group(2))
            requirements.setdefault(current, set())
        if current is not None:
            requirements[current].update(LOCK_HASH.findall(raw))
        if raw and not raw[0].isspace() and not match and not raw.lstrip().startswith("#"):
            current = None
    return requirements


def verify_wheelhouse(
    task: Path, root: Path, regular: list[RegularRecord]
) -> tuple[str, list[str], dict[str, Any]]:
    locked = requirements_lock(task)
    if not locked or any(not hashes for hashes in locked.values()):
        return "missing", ["wheelhouse has no complete hashed requirements lock"], {}
    problems: list[str] = []
    matched: set[tuple[str, str]] = set()
    accepted_hashes = {digest for hashes in locked.values() for digest in hashes}
    for record in regular:
        if not record.path.endswith(".whl"):
            problems.append(f"non-wheel file in exact wheelhouse: {record.path}")
            continue
        if record.sha256 not in accepted_hashes:
            problems.append(f"wheel digest is absent from requirements lock: {record.path}")
            continue
        filename = Path(record.path).name
        for key, hashes in locked.items():
            name, version = key
            prefix = f"{name.replace('-', '_')}-{version.replace('-', '_')}".lower()
            alternate = f"{name}-{version}".lower()
            normalised_filename = filename.lower()
            if record.sha256 in hashes and (
                normalised_filename.startswith(prefix) or normalised_filename.startswith(alternate)
            ):
                matched.add(key)
                break
        else:
            problems.append(f"wheel filename does not match its locked package: {record.path}")
    missing = sorted(set(locked) - matched)
    if missing:
        preview = ", ".join(f"{name}=={version}" for name, version in missing[:8])
        problems.append(f"locked wheels missing: {preview}")
    if len(regular) != len(locked):
        problems.append(f"wheel count {len(regular)} != locked requirement count {len(locked)}")
    details = {
        "requirements_lock": "environment/runtime-requirements.lock",
        "locked_requirements": len(locked),
        "matched_requirements": len(matched),
    }
    return "locked-artifacts", problems, details


def verify_locked_files(
    root: Path, regular: list[RegularRecord], entry: dict[str, Any]
) -> tuple[str, list[str], dict[str, Any]]:
    files = entry.get("files")
    if not isinstance(files, dict) or not files:
        return "missing", ["context has no aggregate digest or exact file inventory"], {}
    observed = {record.path: record.sha256 for record in regular}
    expected: dict[str, str] = {}
    for name, value in files.items():
        if isinstance(value, str):
            expected[str(name)] = value
        elif isinstance(value, dict) and isinstance(value.get("sha256"), str):
            expected[str(name)] = str(value["sha256"])
        else:
            return "missing", [f"no sha256 is recorded for {name}"], {}
    problems = []
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        if missing:
            problems.append(f"locked files missing: {', '.join(missing[:8])}")
        if extra:
            problems.append(f"unlocked files present: {', '.join(extra[:8])}")
    for name in sorted(set(observed) & set(expected)):
        if observed[name] != expected[name]:
            problems.append(f"digest mismatch: {name}")
    return "locked-files", problems, {"locked_files": len(expected)}


def verify_entry(
    task: Path, name: str, root: Path, regular: list[RegularRecord], entry: dict[str, Any]
) -> tuple[str, list[str], dict[str, Any]]:
    if name == "wheelhouse":
        return verify_wheelhouse(task, root, regular)

    expected_digest = entry.get("content_sha256")
    hash_kind = entry.get("hash_kind")
    problems: list[str] = []
    details: dict[str, Any] = {}
    if isinstance(expected_digest, str) and HEX_SHA256.fullmatch(expected_digest):
        observed: str | None
        if hash_kind == "tree_manifest_sha256":
            observed = tree_manifest_sha256(regular)
        elif hash_kind == "file_sha256":
            candidate = _single_regular(root, regular, entry)
            observed = sha256_file(candidate) if candidate is not None else None
            if observed is None:
                problems.append("file_sha256 does not identify exactly one regular file")
        else:
            observed = None
            problems.append(f"unsupported hash_kind {hash_kind!r}")
        details.update(expected_sha256=expected_digest, observed_sha256=observed)
        if observed is not None and observed != expected_digest:
            problems.append(f"content sha256 {observed} != recorded {expected_digest}")
        expected_count = _expected_count(entry)
        if expected_count is not None and len(regular) != expected_count:
            problems.append(f"file count {len(regular)} != recorded {expected_count}")
        expected_size = entry.get("size_bytes")
        observed_size = sum(record.size_bytes for record in regular)
        if isinstance(expected_size, int) and observed_size != expected_size:
            problems.append(f"size {observed_size} != recorded {expected_size}")
        return "exact-hash", problems, details

    nested_status, nested_problems, nested_details = verify_locked_files(root, regular, entry)
    if nested_status != "missing":
        return nested_status, nested_problems, nested_details
    return (
        "missing",
        [
            "missing authoritative provenance: record a complete content identity with "
            "content_sha256 or exact file inventory"
        ],
        {},
    )


def load_lock(task: Path) -> dict[str, Any]:
    path = task / "environment/assets.lock.yaml"
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a YAML mapping")
    return payload


def load_common_contexts() -> dict[str, dict[str, Any]]:
    try:
        payload = yaml.safe_load(COMMON_CONTEXT_LOCK.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    contexts = payload.get("contexts") if isinstance(payload, dict) else None
    if not isinstance(contexts, dict):
        return {}
    return {str(name): entry for name, entry in contexts.items() if isinstance(entry, dict)}


def build_context_block(lock: dict[str, Any]) -> dict[str, Any] | None:
    build = lock.get("build_contexts")
    if not isinstance(build, dict):
        image = lock.get("image")
        build = image.get("build_contexts") if isinstance(image, dict) else None
    return build if isinstance(build, dict) else None


def context_declaration(lock: dict[str, Any]) -> tuple[list[str], dict[str, dict[str, Any]]]:
    build = build_context_block(lock)
    if not isinstance(build, dict):
        return [], {}
    names = build.get("names")
    if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
        return [], {}
    explicit = build.get("contexts") or build.get("entries") or {}
    if not isinstance(explicit, dict):
        explicit = {}
    return list(names), {
        str(key): value for key, value in explicit.items() if isinstance(value, dict)
    }


def lock_asset_entries(lock: dict[str, Any]) -> dict[str, dict[str, Any]]:
    assets = lock.get("assets")
    if isinstance(assets, dict):
        return {str(key): value for key, value in assets.items() if isinstance(value, dict)}
    return {}


def context_entry(
    name: str,
    lock: dict[str, Any],
    explicit: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if name in explicit:
        return explicit[name]
    assets = lock_asset_entries(lock)
    if name in assets:
        return assets[name]
    # Schema-v1 OpenUnlearning records its source checkout at the top level.
    if name == "open_unlearning_source" and isinstance(lock.get("source"), dict):
        source = lock["source"]
        return {
            "kind": "git_repository",
            "source": source.get("repository"),
            "revision": source.get("repository_revision"),
            "content_sha256": source.get("content_sha256"),
            "hash_kind": source.get("hash_kind"),
            "file_count": source.get("file_count"),
            "size_bytes": source.get("size_bytes"),
        }
    return load_common_contexts().get(name, {})


def docker_context_names(task: Path) -> set[str]:
    dockerfile = task / "environment/Dockerfile"
    if not dockerfile.is_file():
        return set()
    text = dockerfile.read_text(encoding="utf-8")
    stages = {match for match in FROM_STAGE.findall(text) if match}
    return {name for name in DOCKER_CONTEXT.findall(text) if name not in stages}


def verify_context(task: Path, name: str, root: Path, entry: dict[str, Any]) -> dict[str, Any]:
    # Receipts are intended to travel with release artifacts.  Keep the logical
    # BuildKit name, never a lab-specific host path.
    result: dict[str, Any] = {"name": name, "path": name, "status": "invalid"}
    if not root.exists():
        return {**result, "problems": ["named build context does not exist"]}
    if not root.is_dir() or root.is_symlink():
        return {**result, "problems": ["named build context must be a real directory"]}
    context_records, regular_records, security_problems = inventory_context(root)
    result.update(
        files=sum(record.type == "file" for record in context_records),
        directories=sum(record.type == "directory" for record in context_records),
        symlinks=sum(record.type == "symlink" for record in context_records),
        size_bytes=sum(record.size_bytes for record in context_records if record.type == "file"),
        context_manifest_sha256=context_manifest_sha256(context_records),
        regular_tree_sha256=tree_manifest_sha256(regular_records),
    )
    provenance, provenance_problems, details = verify_entry(
        task, name, root, regular_records, entry
    )
    problems = sorted(set(security_problems + provenance_problems))
    result.update(provenance=provenance, provenance_details=details, problems=problems)
    result["status"] = "ok" if not problems and provenance != "missing" else "invalid"
    return result


def _repository_source_hash(task: Path) -> str:
    orchestrator = str(REPO / "orchestrator")
    sys.path.insert(0, orchestrator)
    try:
        from task import source_hash  # type: ignore[import-not-found]

        return source_hash(task)
    finally:
        sys.path.remove(orchestrator)


def file_sha256_or_none(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def task_build_config(task: Path) -> dict[str, Any]:
    """Return task.toml fields that can change the image build.

    Image tags, observed digests, source receipts and runtime resources are build
    outputs or run settings.  Hashing the raw task.toml into a tag and then writing
    that tag back would be circular.  The raw file digest is still retained in the
    receipt; this projection is what enters ``build_inputs_sha256``.
    """

    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
        import tomli as tomllib  # type: ignore[no-redef]

    path = task / "task.toml"
    with path.open("rb") as stream:
        payload = tomllib.load(stream)
    environment = payload.get("environment")
    if not isinstance(environment, dict):
        return {}
    selected = {
        key: value
        for key, value in environment.items()
        if key in {"build_args", "dockerfile", "platform", "target"} or key.startswith("base_")
    }
    return selected


def resolve_task(value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        direct = (REPO / candidate).resolve()
        candidate = direct if (direct / "task.toml").is_file() else TASKS / value
    candidate = candidate.resolve()
    if not (candidate / "task.toml").is_file():
        raise ValueError(f"unknown task: {value}")
    return candidate


def verify_task(task: Path, context_root: Path) -> dict[str, Any]:
    problems: list[str] = []
    try:
        lock = load_lock(task)
    except ValueError as error:
        return {"task": task.name, "status": "invalid", "problems": [str(error)], "contexts": []}
    names, explicit = context_declaration(lock)
    if not names:
        problems.append("assets.lock.yaml has no valid build_contexts.names declaration")
    if len(names) != len(set(names)):
        problems.append("build_contexts.names contains duplicates")
    build = build_context_block(lock)
    if isinstance(build, dict) and build.get("network") != "none":
        problems.append("build_contexts.network must be 'none'")
    referenced = docker_context_names(task)
    undeclared = sorted(referenced - set(names))
    unused = sorted(set(names) - referenced)
    if undeclared:
        problems.append(f"Dockerfile references undeclared contexts: {', '.join(undeclared)}")
    if unused:
        problems.append(f"declared contexts are unused by Dockerfile: {', '.join(unused)}")

    contexts = [
        verify_context(task, name, context_root / name, context_entry(name, lock, explicit))
        for name in names
    ]
    if any(context["status"] != "ok" for context in contexts):
        problems.append("one or more named contexts failed verification")

    environment = task / "environment"
    build_config = task_build_config(task)
    repository_inputs = {
        "task_source_sha256": _repository_source_hash(task),
        # Audit-only full-file identity.  It intentionally does not enter the
        # canonical build hash because task.toml also stores the tag/digests
        # produced by this build.
        "task_toml_sha256": file_sha256_or_none(task / "task.toml"),
        "task_build_config": build_config,
        "task_build_config_sha256": canonical_sha256(build_config),
        "assets_lock_sha256": file_sha256_or_none(environment / "assets.lock.yaml"),
        "dockerfile_sha256": file_sha256_or_none(environment / "Dockerfile"),
        "dockerfile_base_sha256": file_sha256_or_none(environment / "Dockerfile.base"),
        "common_context_lock_sha256": file_sha256_or_none(COMMON_CONTEXT_LOCK),
    }
    canonical_repository_inputs = {
        key: value for key, value in repository_inputs.items() if key != "task_toml_sha256"
    }
    build_inputs = {
        "schema_version": 1,
        "task": task.name,
        "repository_inputs": canonical_repository_inputs,
        "contexts": {
            context["name"]: context.get("context_manifest_sha256") for context in contexts
        },
    }
    status = "ok" if not problems else "invalid"
    return {
        "task": task.name,
        "context_root": ".",
        "status": status,
        "problems": problems,
        "repository_inputs": repository_inputs,
        "build_inputs_sha256": canonical_sha256(build_inputs),
        "contexts": contexts,
    }


def write_receipt(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        # link(2) is an atomic no-replace publication on the same filesystem.
        # Unlike Path.replace(), it cannot destroy an earlier release receipt.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--task", help="task id or task directory")
    selector.add_argument("--all", action="store_true", help="verify every task")
    parser.add_argument(
        "--context-root",
        type=Path,
        default=os.environ.get("AI4AI_BUILD_CONTEXT_ROOT"),
        help=(
            "for one task: directory containing its named contexts; for --all: "
            "parent containing <task-id>/<context-name> "
            "(env: AI4AI_BUILD_CONTEXT_ROOT)"
        ),
    )
    parser.add_argument("--receipt", type=Path, help="JSON receipt path")
    parser.add_argument("--json", action="store_true", help="also print the receipt JSON")
    args = parser.parse_args()
    if args.context_root is None:
        parser.error("--context-root or AI4AI_BUILD_CONTEXT_ROOT is required")

    try:
        if args.all:
            tasks = [
                task
                for task in sorted(TASKS.iterdir())
                if task.is_dir() and (task / "task.toml").is_file()
            ]
            reports = [verify_task(task, args.context_root / task.name) for task in tasks]
            default_receipt = args.context_root / "build-context-receipt.json"
        else:
            task = resolve_task(args.task)
            reports = [verify_task(task, args.context_root)]
            default_receipt = args.context_root / f"build-context-receipt.{task.name}.json"
    except ValueError as error:
        parser.error(str(error))

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository": ".",
        "status": "ok" if all(report["status"] == "ok" for report in reports) else "invalid",
        "tasks": reports,
    }
    receipt = args.receipt or default_receipt
    try:
        write_receipt(receipt, payload)
    except FileExistsError:
        parser.error(f"refusing to overwrite existing receipt: {receipt}")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"build contexts: {payload['status']} ({receipt})")
        for report in reports:
            print(
                f"  {report['status']:7} {report['task']} "
                f"build_inputs_sha256={report.get('build_inputs_sha256', '-')}"
            )
            for context in report.get("contexts", []):
                detail = "; ".join(context.get("problems", []))
                suffix = f" - {detail}" if detail else ""
                print(
                    f"    {context['status']:7} {context['name']} "
                    f"provenance={context.get('provenance', '-')}" + suffix
                )
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
