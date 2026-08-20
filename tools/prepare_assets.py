#!/usr/bin/env python3
"""Plan or materialize public AI4AI runtime assets into clean alias directories.

Downloads always land in a temporary staging directory.  Only declared files are
copied into a new final alias, so Hugging Face cache metadata and provider-specific
nesting cannot silently change the benchmark identity. Existing aliases are never
overwritten.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile
from datetime import date, datetime
from pathlib import Path
from typing import Any

import verify_assets

HF_REPO = re.compile(r"^[^/\s]+/[^/\s]+$")
AT_FDCWD = -100
RENAME_NOREPLACE = 1
RECEIPT_DIRECTORY = ".ai4ai-materialization"


def _json_default(value: Any) -> str:
    """Encode YAML timestamp scalars deterministically in JSON receipts.

    Asset locks should quote provenance timestamps because they are identifiers,
    not values on which the materializer performs date arithmetic.  This fallback
    keeps a future unquoted YAML timestamp from crashing receipt generation while
    preserving its ISO representation.
    """

    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        default=_json_default,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_digest(path: Path, expected: Any, *, label: str) -> None:
    if not isinstance(expected, str):
        return
    observed = _sha256(path)
    if observed != expected:
        raise RuntimeError(f"{label} sha256 {observed} != recorded {expected}")


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode != 0:
        rendered = " ".join(command)
        raise RuntimeError(f"command failed with status {completed.returncode}: {rendered}")


def _download_spec(alias: str, entry: dict[str, Any]) -> dict[str, Any] | None:
    explicit = entry.get("download")
    if isinstance(explicit, dict):
        return dict(explicit)
    materialize = entry.get("materialize")
    if isinstance(materialize, dict) and isinstance(materialize.get("source"), str):
        source, separator, revision = materialize["source"].partition("@")
        if separator and HF_REPO.fullmatch(source):
            return {
                "tool": "huggingface_hub.snapshot_download",
                "repo_id": source,
                "revision": revision,
                "repo_type": "dataset",
            }
    source = entry.get("source")
    revision = entry.get("revision")
    # A derived dataset needs a declared transformation rather than an apparently
    # convenient download of its upstream corpus. Raw datasets and model snapshots
    # can use the conventional Hub source/revision pair when no explicit block exists.
    if (
        entry.get("kind") != "derived_dataset"
        and isinstance(source, str)
        and HF_REPO.fullmatch(source)
        and isinstance(revision, str)
    ):
        return {
            "tool": "huggingface_hub.snapshot_download",
            "repo_id": source,
            "revision": revision,
            "repo_type": "dataset" if alias.startswith("data/") else "model",
        }
    if isinstance(entry.get("download_url"), str) and isinstance(
        entry.get("payload_filename"), str
    ):
        return {
            "tool": "url",
            "url": entry["download_url"],
            "filename": entry["payload_filename"],
        }
    return None


def _copy_tree(source: Path, target: Path) -> None:
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if ".cache" in relative.parts or path.is_symlink() or not path.is_file():
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)


def _safe_relative_path(value: str, *, field: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise RuntimeError(f"{field} must be a safe relative path: {value!r}")
    return path


def _archive_destination(root: Path, name: str, *, strip_components: int) -> Path | None:
    relative = _safe_relative_path(name, field="archive member")
    parts = relative.parts[strip_components:]
    if not parts:
        return None
    return root.joinpath(*parts)


def _safe_extract_tar(archive: Path, target: Path, *, strip_components: int = 0) -> None:
    """Extract regular files/directories without accepting archive links or escapes."""

    target.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, "r:*") as handle:
        for member in handle.getmembers():
            destination = _archive_destination(
                target, member.name, strip_components=strip_components
            )
            if destination is None:
                continue
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                raise RuntimeError(f"unsafe tar member type: {member.name}")
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise RuntimeError(f"unsupported tar member type: {member.name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = handle.extractfile(member)
            if source is None:
                raise RuntimeError(f"cannot read tar member: {member.name}")
            with source, destination.open("xb") as stream:
                shutil.copyfileobj(source, stream)
            destination.chmod(member.mode & 0o777)


def _safe_extract_zip(archive: Path, target: Path, *, strip_components: int = 0) -> None:
    target.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            destination = _archive_destination(
                target, member.filename, strip_components=strip_components
            )
            if destination is None:
                continue
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise RuntimeError(f"unsafe zip symlink: {member.filename}")
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if mode and not stat.S_ISREG(mode):
                raise RuntimeError(f"unsupported zip member type: {member.filename}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with handle.open(member) as source, destination.open("xb") as stream:
                shutil.copyfileobj(source, stream)
            if mode:
                destination.chmod(mode & 0o777)


def _safe_extract(archive: Path, target: Path, *, strip_components: int = 0) -> None:
    if zipfile.is_zipfile(archive):
        _safe_extract_zip(archive, target, strip_components=strip_components)
        return
    if tarfile.is_tarfile(archive):
        _safe_extract_tar(archive, target, strip_components=strip_components)
        return
    raise RuntimeError(f"unsupported archive format: {archive.name}")


def _download_file(specification: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(str(specification["url"]), destination)  # noqa: S310
    _check_digest(
        destination,
        specification.get("sha256") or specification.get("archive_sha256"),
        label=str(specification.get("url")),
    )


def _git_archive(specification: dict[str, Any], destination: Path) -> None:
    repository = str(specification.get("repository") or specification.get("url") or "")
    revision = str(specification.get("revision") or specification.get("checkout") or "")
    if not repository or not revision:
        raise RuntimeError("git_archive requires repository and revision")
    clone = destination.parent / (destination.name + ".clone")
    archive = destination.parent / (destination.name + ".tar")
    _run(["git", "init", "-q", str(clone)])
    _run(["git", "-C", str(clone), "remote", "add", "origin", repository])
    _run(["git", "-C", str(clone), "fetch", "--depth", "1", "origin", revision])
    _run(
        [
            "git",
            "-C",
            str(clone),
            "archive",
            "--format=tar",
            "--output",
            str(archive),
            "FETCH_HEAD",
        ]
    )
    _safe_extract_tar(archive, destination)


def _prepare_source(specification: dict[str, Any], target: Path, staging: Path) -> None:
    """Download one declared source into a provider-neutral regular-file tree."""

    tool = specification.get("tool")
    if tool == "huggingface_hub.snapshot_download":
        try:
            from huggingface_hub import snapshot_download
        except ModuleNotFoundError as error:
            raise RuntimeError("install huggingface_hub before using --execute") from error
        kwargs: dict[str, Any] = {
            "repo_id": specification["repo_id"],
            "revision": specification["revision"],
            "local_dir": target,
            "cache_dir": staging / "hf-cache",
        }
        if specification.get("repo_type"):
            kwargs["repo_type"] = specification["repo_type"]
        if specification.get("allow_patterns"):
            kwargs["allow_patterns"] = list(specification["allow_patterns"])
        snapshot_download(**kwargs)
        return
    if tool == "url":
        target.mkdir(parents=True, exist_ok=False)
        filename = _safe_relative_path(str(specification["filename"]), field="download filename")
        payload = target / filename
        _download_file(specification, payload)
        if specification.get("extract"):
            extracted = target.parent / (target.name + ".extracted")
            _safe_extract(
                payload,
                extracted,
                strip_components=int(specification.get("strip_components", 0)),
            )
            shutil.rmtree(target)
            extracted.rename(target)
        return
    if tool == "url_set":
        target.mkdir(parents=True, exist_ok=False)
        items = specification.get("items")
        if items is None and isinstance(specification.get("url_template"), str):
            start = int(specification.get("start", 0))
            end = int(specification["end"])
            if start < 0 or end < start or end - start > 10_000:
                raise RuntimeError("url_set range is invalid or unreasonably large")
            filename_template = str(specification.get("filename_template") or "{index}")
            items = [
                {
                    "url": specification["url_template"].format(index=index),
                    "filename": filename_template.format(index=index),
                }
                for index in range(start, end + 1)
            ]
        if not isinstance(items, list) or not items:
            raise RuntimeError("url_set requires items or an inclusive URL-template range")
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("url"), str):
                raise RuntimeError("every url_set item requires a URL")
            filename = item.get("filename") or Path(item["url"].partition("?")[0]).name
            destination = target / _safe_relative_path(str(filename), field="URL-set filename")
            _download_file(item, destination)
        return
    if tool in {"git", "git_archive"}:
        _git_archive(specification, target)
        member = specification.get("member")
        if isinstance(member, str):
            selected = target / _safe_relative_path(member, field="git member")
            if not selected.exists() or selected.is_symlink():
                raise RuntimeError(f"git member is missing: {member}")
            isolated = target.parent / (target.name + ".member")
            isolated.mkdir()
            if selected.is_dir():
                _copy_tree(selected, isolated)
            else:
                destination_name = specification.get("destination") or selected.name
                destination = isolated / _safe_relative_path(
                    str(destination_name), field="git member destination"
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(selected, destination)
            shutil.rmtree(target)
            isolated.rename(target)
        return
    if tool == "pypi_wheel_member":
        target.mkdir(parents=True, exist_ok=False)
        wheel = staging / "payload.whl"
        _download_file(
            {"url": specification["url"], "sha256": specification.get("wheel_sha256")},
            wheel,
        )
        member = _safe_relative_path(str(specification["member"]), field="wheel member")
        destination = target / _safe_relative_path(
            str(specification.get("destination") or member.name),
            field="wheel member destination",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(wheel) as handle:
            info = handle.getinfo(member.as_posix())
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise RuntimeError(f"wheel member is a symlink: {member}")
            with handle.open(info) as source, destination.open("xb") as stream:
                shutil.copyfileobj(source, stream)
        if specification.get("executable", True):
            destination.chmod(0o755)
        return
    raise RuntimeError(f"unsupported download tool {tool!r}")


def _materialize(stage: Path, candidate: Path, entry: dict[str, Any]) -> None:
    """Build the exact alias under ``candidate`` without touching its final path."""

    candidate.mkdir(parents=True, exist_ok=False)
    materialize = entry.get("materialize") or entry.get("materialization") or {}
    if isinstance(materialize, str):
        raise RuntimeError(f"manual materialization is required: {materialize}")
    copy_only = materialize.get("copy_only") if isinstance(materialize, dict) else None
    if isinstance(copy_only, list):
        for name in copy_only:
            relative = _safe_relative_path(str(name), field="copy_only entry")
            exact = stage / relative
            matches = [exact] if exact.is_file() else list(stage.rglob(relative.as_posix()))
            matches = [path for path in matches if path.is_file() and not path.is_symlink()]
            if len(matches) != 1:
                raise RuntimeError(f"expected one staged {name}, found {len(matches)}")
            destination = candidate / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(matches[0], destination)
    else:
        source_path = entry.get("source_path")
        download = entry.get("download")
        download_instruction = download.get("materialize") if isinstance(download, dict) else None
        if isinstance(source_path, str) and download_instruction:
            relative = _safe_relative_path(source_path, field="source_path")
            source = stage / relative
            if not source.is_file():
                raise RuntimeError(f"staged source_path is missing: {source_path}")
            shutil.copyfile(source, candidate / source.name)
        else:
            _copy_tree(stage, candidate)
    if isinstance(materialize, dict) and isinstance(materialize.get("sidecar"), str):
        sidecar = _safe_relative_path(materialize["sidecar"], field="sidecar")
        destination = candidate / sidecar
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(str(materialize.get("sidecar_content", "")), encoding="utf-8")


def _prepare_inline(entry: dict[str, Any], candidate: Path) -> bool:
    if "content" not in entry or entry.get("hash_kind") != "file_sha256":
        return False
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(
        json.dumps(entry["content"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return True


def _task_image(task: Path) -> str:
    match = re.search(
        r'^image\s*=\s*"([^"]+)"\s*$',
        (task / "task.toml").read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    if not match or "@sha256:" not in match.group(1):
        raise RuntimeError(f"task image is not pinned by registry digest: {task}")
    return match.group(1)


def _builder_script(task: Path, builder: dict[str, Any]) -> Path:
    script = _safe_relative_path(str(builder.get("script", "")), field="builder script")
    # Builder paths are repository-relative so one reviewed script can serve a task.
    repo = Path(__file__).resolve().parents[1]
    path = (repo / script).resolve()
    if repo not in path.parents or not path.is_file():
        raise RuntimeError(f"builder script does not exist: {script}")
    expected = builder.get("script_sha256")
    _check_digest(path, expected, label=f"builder {script}")
    return path


def _render_builder_argument(
    value: str,
    *,
    source_mounts: dict[str, str],
    alias_mounts: dict[str, str],
) -> str:
    if value == "{output}":
        return "/publish/candidate"
    if value == "{repo}":
        return "/builder/repo"
    match = re.fullmatch(r"\{source:([A-Za-z0-9_.-]+)\}", value)
    if match:
        try:
            return source_mounts[match.group(1)]
        except KeyError as error:
            raise RuntimeError(f"unknown builder source {match.group(1)!r}") from error
    match = re.fullmatch(r"\{alias:([^{}]+)\}", value)
    if match:
        try:
            return alias_mounts[match.group(1)]
        except KeyError as error:
            raise RuntimeError(f"unknown builder alias {match.group(1)!r}") from error
    if "{" in value or "}" in value:
        raise RuntimeError(f"unsupported builder placeholder: {value}")
    return value


def _run_task_builder(
    task: Path,
    candidate: Path,
    builder: dict[str, Any],
    sources: dict[str, Path],
    alias_sources: dict[str, Path],
) -> None:
    _builder_script(task, builder)
    image = str(builder.get("image") or _task_image(task))
    if "@sha256:" not in image:
        raise RuntimeError("task-image builders require an immutable image digest")
    command = builder.get("command")
    if not isinstance(command, list) or not command or not all(
        isinstance(item, str) for item in command
    ):
        raise RuntimeError("task-image builder requires a string-list command")
    repo = Path(__file__).resolve().parents[1]
    candidate.parent.mkdir(parents=True, exist_ok=True)
    publication_mount = candidate.parent / "builder-publication"
    publication_mount.mkdir()
    docker = shlex.split(os.environ.get("AI4AI_DOCKER", "docker"))
    if not docker:
        raise RuntimeError("AI4AI_DOCKER resolves to an empty command")
    docker.extend(
        [
            "run",
            "--rm",
            "--network",
            "none",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--env",
            "HOME=/tmp",
            "--volume",
            f"{repo}:/builder/repo:ro",
            "--volume",
            f"{publication_mount}:/publish:rw",
        ]
    )
    source_mounts: dict[str, str] = {}
    for index, (name, path) in enumerate(sorted(sources.items())):
        mount = f"/inputs/source-{index}"
        docker.extend(["--volume", f"{path}:{mount}:ro"])
        source_mounts[name] = mount
    alias_mounts: dict[str, str] = {}
    for index, (name, path) in enumerate(sorted(alias_sources.items())):
        mount = f"/inputs/alias-{index}"
        docker.extend(["--volume", f"{path}:{mount}:ro"])
        alias_mounts[name] = mount
    rendered = [
        _render_builder_argument(
            item, source_mounts=source_mounts, alias_mounts=alias_mounts
        )
        for item in command
    ]
    _run([*docker, image, *rendered])
    built = publication_mount / "candidate"
    if not built.exists() or built.is_symlink():
        raise RuntimeError("task-image builder did not create its candidate output")
    built.rename(candidate)


def _prepare_declared_materialization(
    entry: dict[str, Any],
    candidate: Path,
    materialize: dict[str, Any],
    staging: Path,
    *,
    task: Path,
    assets: Path,
) -> bool:
    """Prepare a multi-source, alias-derived, or task-image-built asset."""

    sources_spec = materialize.get("sources") or {}
    if not isinstance(sources_spec, dict):
        raise RuntimeError("materialize.sources must be a mapping")
    sources: dict[str, Path] = {}
    alias_sources: dict[str, Path] = {}
    inputs = staging / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    for name, specification in sorted(sources_spec.items()):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(name)):
            raise RuntimeError(f"unsafe materialization source name: {name!r}")
        if not isinstance(specification, dict):
            raise RuntimeError(f"materialization source {name!r} must be a mapping")
        alias = specification.get("alias")
        if isinstance(alias, str):
            source = assets / _safe_relative_path(alias, field="source alias")
            if not source.exists() or source.is_symlink():
                raise RuntimeError(f"source alias is not materialized: {alias}")
            alias_sources[alias] = source
            continue
        target = inputs / str(name)
        source_work = inputs / (str(name) + ".work")
        source_work.mkdir()
        _prepare_source(specification, target, source_work)
        sources[str(name)] = target

    derive = materialize.get("derive")
    if isinstance(derive, dict):
        source_alias = derive.get("source_alias") or derive.get("source_asset")
        if not isinstance(source_alias, str):
            raise RuntimeError("derive requires source_alias")
        source = assets / _safe_relative_path(source_alias, field="derive source alias")
        if not source.exists() or source.is_symlink():
            raise RuntimeError(f"derive source alias is not materialized: {source_alias}")
        candidate.mkdir(parents=True, exist_ok=False)
        copy = derive.get("copy") or []
        copy_as = derive.get("copy_as") or {}
        if not isinstance(copy, list) or not isinstance(copy_as, dict):
            raise RuntimeError("derive copy/copy_as have invalid types")
        for raw in copy:
            relative = _safe_relative_path(str(raw), field="derive copy")
            origin = source / relative
            if not origin.is_file() or origin.is_symlink():
                raise RuntimeError(f"derive source file is missing: {relative}")
            destination = candidate / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin, destination)
        for destination_raw, source_raw in copy_as.items():
            destination_relative = _safe_relative_path(
                str(destination_raw), field="derive copy_as destination"
            )
            source_relative = _safe_relative_path(str(source_raw), field="derive copy_as source")
            origin = source / source_relative
            if not origin.is_file() or origin.is_symlink():
                raise RuntimeError(f"derive source file is missing: {source_relative}")
            destination = candidate / destination_relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin, destination)
        return True

    combine = materialize.get("combine")
    if isinstance(combine, dict):
        candidate.mkdir(parents=True, exist_ok=False)
        for source_name, destination_name in sorted(combine.items()):
            if source_name not in sources:
                raise RuntimeError(f"combine references unknown source {source_name!r}")
            destination = candidate / _safe_relative_path(
                str(destination_name), field="combine destination"
            )
            destination.mkdir(parents=True, exist_ok=False)
            _copy_tree(sources[source_name], destination)
        return True

    builder = materialize.get("builder")
    if isinstance(builder, dict):
        if builder.get("type") != "task_image":
            raise RuntimeError("only offline task_image builders are supported")
        _run_task_builder(task, candidate, builder, sources, alias_sources)
        return True

    if len(sources) == 1 and not alias_sources:
        _materialize(next(iter(sources.values())), candidate, entry)
        return True
    return False


def _path_exists(path: Path) -> bool:
    """Like ``Path.exists``, but also treats a broken symlink as occupied."""

    return os.path.lexists(path)


def _rename_noreplace(source: Path, target: Path) -> None:
    """Atomically publish ``source`` and refuse to replace any existing target.

    Prefer Linux ``renameat2(RENAME_NOREPLACE)``. Some otherwise suitable shared
    filesystems return ``EINVAL``/``EOPNOTSUPP`` for that flag; on those filesystems,
    serialize publication with a filesystem lock before rechecking the destination
    and issuing the final same-filesystem rename. Competing preparers therefore cannot
    cross the check/rename boundary, while pre-existing content is never replaced.
    """

    if _path_exists(target):
        raise RuntimeError(f"refusing to overwrite existing alias: {target}")
    if source.stat().st_dev != target.parent.stat().st_dev:
        raise RuntimeError(
            "staging and asset roots must be on the same filesystem for atomic publication"
        )
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace publication requires Linux renameat2")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        AT_FDCWD,
        os.fsencode(source),
        AT_FDCWD,
        os.fsencode(target),
        RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise RuntimeError(f"refusing to overwrite existing alias: {target}")
    if error == errno.EXDEV:
        raise RuntimeError(
            "staging and asset roots must be on the same filesystem for atomic publication"
        )
    unsupported = {errno.EINVAL, errno.ENOSYS}
    if hasattr(errno, "EOPNOTSUPP"):
        unsupported.add(errno.EOPNOTSUPP)
    if error in unsupported:
        _publish_with_serialized_noreplace(source, target)
        return
    raise OSError(error, os.strerror(error), str(target))


def _publish_with_serialized_noreplace(source: Path, target: Path) -> None:
    """Serialize no-replace publication when the filesystem lacks ``renameat2``.

    Lock files stay outside the scientific alias and are intentionally persistent:
    unlinking a lock while another process has it open can split future callers over
    two inodes. ``flock`` releases automatically if a preparer exits or crashes.
    """

    if not ((source.is_dir() or source.is_file()) and not source.is_symlink()):
        raise RuntimeError(f"publication source is not a regular file or directory: {source}")
    lock_directory = target.parent / ".ai4ai-publication-locks"
    lock_directory.mkdir(mode=0o700, exist_ok=True)
    if lock_directory.is_symlink() or not lock_directory.is_dir():
        raise RuntimeError(f"publication lock root is unsafe: {lock_directory}")
    lock_name = hashlib.sha256(target.name.encode("utf-8")).hexdigest() + ".lock"
    descriptor = os.open(
        lock_directory / lock_name,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        if _path_exists(target):
            raise RuntimeError(f"refusing to overwrite existing alias: {target}")
        os.rename(source, target)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _receipt_bytes(
    alias: str,
    entry: dict[str, Any],
    result: dict[str, Any],
    *,
    task: Path | None,
) -> bytes:
    materialize = entry.get("materialize") or entry.get("materialization") or {}
    builder_identity = None
    if isinstance(materialize, dict) and isinstance(materialize.get("builder"), dict):
        builder = materialize["builder"]
        builder_identity = {
            "type": builder.get("type"),
            "image": builder.get("image") or (_task_image(task) if task else None),
            "script": builder.get("script"),
            "script_sha256": builder.get("script_sha256"),
        }
    value = {
        "schema_version": 1,
        "task_id": task.name if task is not None else None,
        "alias": alias,
        "content_sha256": entry.get("content_sha256"),
        "hash_kind": entry.get("hash_kind"),
        "files": result.get("files"),
        "size_bytes": result.get("size_bytes"),
        "lock_entry_sha256": hashlib.sha256(_canonical_json(entry).encode("utf-8")).hexdigest(),
        "builder": builder_identity,
    }
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def ensure_materialization_receipt(
    receipt: Path,
    alias: str,
    entry: dict[str, Any],
    result: dict[str, Any],
    *,
    task: Path | None,
) -> str:
    """Publish or validate deterministic provenance without touching alias bytes."""

    expected = _receipt_bytes(alias, entry, result, task=task)
    if _path_exists(receipt):
        if receipt.is_symlink() or not receipt.is_file():
            raise RuntimeError(f"materialization receipt is not a regular file: {receipt}")
        if receipt.read_bytes() != expected:
            raise RuntimeError(f"materialization receipt conflicts with the asset lock: {receipt}")
        return "existing"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=receipt.name + ".", suffix=".tmp", dir=receipt.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(expected)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            _rename_noreplace(temporary, receipt)
        except RuntimeError:
            # A concurrent identical preparer may win the no-replace race.
            if receipt.is_file() and not receipt.is_symlink() and receipt.read_bytes() == expected:
                temporary.unlink(missing_ok=True)
                return "existing"
            raise
    finally:
        temporary.unlink(missing_ok=True)
    return "created"


def staging_root(assets: Path, explicit: Path | None = None) -> Path:
    """Return a large-asset staging root without silently falling back to ``/tmp``."""

    if explicit is not None:
        return explicit.expanduser().resolve()
    configured = os.environ.get("AI4AI_ASSET_STAGING_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    assets = assets.expanduser().resolve()
    return assets.parent / ".ai4ai-asset-staging"


def prepare_alias(
    alias: str,
    entry: dict[str, Any],
    target: Path,
    *,
    staging: Path,
    task: Path | None = None,
    assets: Path | None = None,
    receipt: Path | None = None,
) -> dict[str, Any]:
    """Materialize, verify, and atomically publish one immutable alias."""

    if _path_exists(target):
        raise RuntimeError(f"refusing to overwrite existing alias: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True, exist_ok=True)
    if staging.stat().st_dev != target.parent.stat().st_dev:
        raise RuntimeError(
            "staging and asset roots must be on the same filesystem for atomic publication"
        )
    prefix = "ai4ai-asset-" + re.sub(r"[^A-Za-z0-9_.-]", "-", alias)[:48] + "-"
    with tempfile.TemporaryDirectory(prefix=prefix, dir=staging) as temporary:
        work = Path(temporary)
        candidate = work / "candidate"
        if not _prepare_inline(entry, candidate):
            materialize = entry.get("materialize") or entry.get("materialization")
            prepared = False
            if isinstance(materialize, dict) and any(
                key in materialize for key in ("sources", "derive", "builder")
            ):
                if task is None or assets is None:
                    raise RuntimeError("declared materialization requires task and asset roots")
                prepared = _prepare_declared_materialization(
                    entry,
                    candidate,
                    materialize,
                    work / "declared",
                    task=task,
                    assets=assets,
                )
            if not prepared:
                specification = _download_spec(alias, entry)
                if not specification:
                    raise RuntimeError("exact public materializer is not released")
                downloads = work / "downloads"
                downloads.mkdir()
                if specification["tool"] in {
                    "huggingface_hub.snapshot_download",
                    "url",
                    "url_set",
                    "git",
                    "git_archive",
                    "pypi_wheel_member",
                }:
                    payload = downloads / "payload"
                    provider_work = downloads / "provider-work"
                    provider_work.mkdir()
                    _prepare_source(specification, payload, provider_work)
                    _materialize(payload, candidate, entry)
                else:
                    raise RuntimeError(f"unsupported download tool {specification['tool']}")
        result = verify_assets.verify_alias(alias, candidate, entry, include_hashes=True)
        if result["status"] != "ok":
            raise RuntimeError("; ".join(result.get("problems", [])))
        _rename_noreplace(candidate, target)
        if receipt is not None:
            ensure_materialization_receipt(
                receipt, alias, entry, result, task=task
            )
        return result


def materialization_dependencies(entry: dict[str, Any]) -> set[str]:
    materialize = entry.get("materialize") or entry.get("materialization") or {}
    if not isinstance(materialize, dict):
        return set()
    dependencies: set[str] = set()
    derive = materialize.get("derive")
    if isinstance(derive, dict):
        source = derive.get("source_alias") or derive.get("source_asset")
        if isinstance(source, str):
            dependencies.add(source)
    sources = materialize.get("sources") or {}
    if isinstance(sources, dict):
        for specification in sources.values():
            if isinstance(specification, dict) and isinstance(specification.get("alias"), str):
                dependencies.add(specification["alias"])
    # Compatibility with the original DiGress lock spelling.
    legacy = entry.get("derive")
    if isinstance(legacy, dict) and isinstance(legacy.get("source_asset"), str):
        dependencies.add(legacy["source_asset"])
    return dependencies


def dependency_order(aliases: list[str], entries: dict[str, dict[str, Any]]) -> list[str]:
    visiting: set[str] = set()
    visited: set[str] = set()
    ordered: list[str] = []

    def visit(alias: str) -> None:
        if alias in visited:
            return
        if alias in visiting:
            raise RuntimeError(f"asset materialization dependency cycle at {alias}")
        visiting.add(alias)
        for dependency in sorted(materialization_dependencies(entries.get(alias, {}))):
            if dependency not in entries:
                raise RuntimeError(f"{alias} depends on unknown alias {dependency}")
            visit(dependency)
        visiting.remove(alias)
        visited.add(alias)
        ordered.append(alias)

    for alias in aliases:
        visit(alias)
    return ordered


def plan_row(alias: str, entry: dict[str, Any]) -> dict[str, Any]:
    specification = _download_spec(alias, entry)
    if "content" in entry and entry.get("hash_kind") == "file_sha256":
        return {"alias": alias, "status": "ready", "method": "inline canonical JSON"}
    if specification and specification.get("tool") in {
        "huggingface_hub.snapshot_download",
        "url",
        "url_set",
        "git",
        "git_archive",
        "pypi_wheel_member",
    }:
        return {
            "alias": alias,
            "status": "ready",
            "method": specification.get("tool"),
            "source": specification.get("repo_id") or specification.get("url"),
            "revision": specification.get("revision"),
        }
    materialize = entry.get("materialize") or entry.get("materialization")
    if isinstance(materialize, dict) and any(
        key in materialize for key in ("sources", "derive", "builder")
    ):
        builder = materialize.get("builder")
        method = "alias derivation"
        if isinstance(builder, dict):
            method = f"{builder.get('type', 'declared')} builder"
        elif materialize.get("sources"):
            method = "declared multi-source materialization"
        return {"alias": alias, "status": "ready", "method": method}
    detail = entry.get("derivation") or entry.get("note") or entry.get("kind")
    if isinstance(detail, dict):
        detail = detail.get("tool") or "declared manual derivation"
    if isinstance(detail, str):
        detail = " ".join(detail.split())
        if len(detail) > 120:
            detail = detail[:117] + "..."
    return {
        "alias": alias,
        "status": "manual_or_unreleased",
        "method": detail or "no exact public materializer is declared",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--task", help="task id or task directory")
    selector.add_argument("--all", action="store_true", help="prepare every task")
    parser.add_argument("--assets", type=Path, help="one task's asset root")
    parser.add_argument("--assets-root", type=Path, help="parent containing task-id roots")
    parser.add_argument(
        "--staging-root",
        type=Path,
        help=(
            "large temporary staging root (default: AI4AI_ASSET_STAGING_ROOT, then "
            "a sibling of --assets); it must share a filesystem with --assets"
        ),
    )
    parser.add_argument("--alias", action="append", default=[])
    parser.add_argument("--execute", action="store_true", help="download/materialize ready aliases")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.all:
        if args.assets_root is None:
            parser.error("--all requires --assets-root")
        if args.assets is not None or args.alias:
            parser.error("--all does not accept --assets or --alias")
        jobs = [
            (task, args.assets_root / task.name)
            for task in sorted(verify_assets.TASKS.iterdir())
            if (task / "task.toml").is_file()
        ]
    else:
        if args.assets is None:
            parser.error("--task requires --assets")
        if args.assets_root is not None:
            parser.error("--task does not accept --assets-root")
        task = verify_assets.resolve_task(args.task)
        jobs = [(task, args.assets)]

    plans: list[dict[str, Any]] = []
    for task, assets in jobs:
        entries = verify_assets.lock_entries(task)
        requested = sorted(args.alias or verify_assets.required_aliases(task))
        try:
            aliases = dependency_order(requested, entries)
        except RuntimeError as error:
            parser.error(str(error))
        rows = [plan_row(alias, entries.get(alias, {})) for alias in aliases]
        plans.append(
            {"task": task.name, "assets": str(assets), "aliases": rows, "order": aliases}
        )
    if not args.execute:
        if args.json:
            # Preserve the original single-task JSON interface; --all necessarily
            # adds task grouping because aliases repeat between task roots.
            payload: Any = plans if args.all else plans[0]["aliases"]
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            for plan, (_, assets) in zip(plans, jobs, strict=True):
                print(
                    f"Asset preparation plan for {plan['task']} "
                    f"({len(plan['aliases'])} aliases)"
                )
                for row in plan["aliases"]:
                    source = f" from {row['source']}" if row.get("source") else ""
                    revision = f"@{row['revision']}" if row.get("revision") else ""
                    print(
                        f"  {row['status']:20} {row['alias']} <- "
                        f"{row['method']}{source}{revision}"
                    )
                print(f"Staging root: {staging_root(assets, args.staging_root)}")
            print("No files were created; add --execute to materialize ready aliases.")
        return 0

    failures: list[str] = []
    results: list[dict[str, Any]] = []
    for plan, (task, assets) in zip(plans, jobs, strict=True):
        entries = verify_assets.lock_entries(task)
        assets.mkdir(parents=True, exist_ok=True)
        # A single explicit staging root is suitable for one task only; --all gets
        # task-specific children so all atomic publishes stay on their destination FS.
        explicit_staging = args.staging_root
        if args.staging_root is not None and args.all:
            explicit_staging = args.staging_root / task.name
        temporary_root = staging_root(assets, explicit_staging)
        temporary_root.mkdir(parents=True, exist_ok=True)
        if temporary_root.stat().st_dev != assets.stat().st_dev:
            parser.error("--staging-root must share a filesystem with --assets")
        for alias in plan["order"]:
            entry = entries.get(alias)
            if not entry:
                failures.append(f"{task.name}/{alias}: absent from assets.lock.yaml")
                continue
            target = assets / _safe_relative_path(alias, field="alias")
            receipt = assets / RECEIPT_DIRECTORY / (alias + ".json")
            if _path_exists(target):
                existing = verify_assets.verify_alias(alias, target, entry, include_hashes=True)
                if existing["status"] == "ok":
                    try:
                        receipt_status = ensure_materialization_receipt(
                            receipt, alias, entry, existing, task=task
                        )
                    except Exception as error:
                        failures.append(f"{task.name}/{alias}: {error}")
                        continue
                    if not args.json:
                        print(f"already verified {task.name}/{alias}: {target}")
                    results.append({"task": task.name, "alias": alias, "status": "existing"})
                    results[-1]["receipt"] = receipt_status
                    continue
                failures.append(
                    f"{task.name}/{alias}: existing immutable alias is invalid: "
                    + "; ".join(existing.get("problems", []))
                )
                continue
            try:
                prepare_alias(
                    alias,
                    entry,
                    target,
                    staging=temporary_root,
                    task=task,
                    assets=assets,
                    receipt=receipt,
                )
                if not args.json:
                    print(f"prepared {task.name}/{alias}: {target}")
                results.append({"task": task.name, "alias": alias, "status": "prepared"})
            except Exception as error:  # keep independent aliases moving
                failures.append(f"{task.name}/{alias}: {error}")
    if args.json:
        print(json.dumps({"results": results, "failures": failures}, indent=2, sort_keys=True))
    if not args.json:
        for failure in failures:
            print(f"ERROR {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
