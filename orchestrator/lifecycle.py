"""Exploration lifecycle receipts and schema-v9 reporting.

This module is used in two trust domains.  The task image calls only the
``submit`` and ``no-candidate`` commands.  The host runner owns capture,
termination classification, receipt validation, and queue reporting.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = 9
RECEIPT_SCHEMA_VERSION = 1
LEGACY_UNKNOWN = "legacy_unknown"

AGENT_EXIT_STATES = {
    "running",
    "completed",
    "failed",
    "timed_out",
    "container_lost",
}
TERMINATION_REASONS = {
    "agent_explicit_submit",
    "agent_explicit_no_candidate",
    "agent_early_exit",
    "phase_deadline",
    "runtime_failure",
    "operator_interrupt",
}
SUBMISSION_ORIGINS = {
    "agent",
    "host_early_exit_capture",
    "host_deadline_capture",
    "none",
}
CANDIDATE_STATES = {"nonempty", "empty", "missing", "rejected", "no_candidate"}

_SECRET = re.compile(
    r"(?i)(api[_-]?key|authorization|bearer|password|secret|token)(=|\s+)([^\s]+)"
)
_CONTROL_MARKERS = (
    "/opt/harness/lifecycle.py",
    "/opt/harness/submit.sh",
    "/opt/harness/no_candidate.sh",
    "codex exec",
    "sleep infinity",
)
_WORK_MARKERS = (
    "/workspace/",
    "/opt/harness/fast_eval",
    "torchrun",
    "accelerate launch",
    "ray::",
    "trainer",
    "train.py",
    "final_eval",
)


class LifecycleError(RuntimeError):
    """A lifecycle receipt is missing, contradictory, or corrupt."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def deadline_remaining(deadline_file: Path = Path("/logs/deadline.json")) -> int | None:
    payload = read_json(deadline_file)
    try:
        return max(0, int(float(payload["deadline_unix"]) - time.time())) if payload else None
    except (KeyError, TypeError, ValueError):
        return None


def _redact(command: str) -> str:
    return _SECRET.sub(lambda match: f"{match.group(1)}{match.group(2)}<redacted>", command)


def process_snapshot() -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,ppid=,stat=,etimes=,comm=,args="],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 5)
        if len(fields) != 6:
            continue
        pid, ppid, state, elapsed, executable, command = fields
        try:
            row = {
                "pid": int(pid),
                "ppid": int(ppid),
                "state": state,
                "elapsed_seconds": int(elapsed),
                "executable": executable,
                "command": _redact(command)[:1000],
            }
        except ValueError:
            continue
        rows.append(row)
    return rows


def gpu_process_snapshot() -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory,gpu_uuid",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 3)]
        if len(fields) != 4:
            continue
        try:
            rows.append(
                {
                    "pid": int(fields[0]),
                    "process_name": fields[1],
                    "used_memory_mib": int(fields[2]),
                    "gpu_uuid": fields[3],
                }
            )
        except ValueError:
            continue
    return rows


def _is_control_process(row: dict[str, Any]) -> bool:
    command = str(row.get("command", ""))
    return any(marker in command for marker in _CONTROL_MARKERS)


def active_work(
    processes: Iterable[dict[str, Any]], gpu_processes: Iterable[dict[str, Any]]
) -> bool:
    if any(True for _ in gpu_processes):
        return True
    for row in processes:
        command = str(row.get("command", ""))
        if not _is_control_process(row) and any(marker in command for marker in _WORK_MARKERS):
            return True
    return False


def session_active(processes: Iterable[dict[str, Any]]) -> bool:
    return any("codex exec" in str(row.get("command", "")) for row in processes)


def _clean_workspace(root: Path) -> None:
    for directory in root.rglob("__pycache__"):
        if directory.is_dir():
            shutil.rmtree(directory, ignore_errors=True)
    for suffix in ("*.pyc", "*.pyo"):
        for path in root.rglob(suffix):
            try:
                path.unlink()
            except OSError:
                pass


def _make_temporary_git_writable(git_dir: Path) -> None:
    """Make only the copied Git metadata writable for staging a snapshot."""

    for root, directories, files in os.walk(git_dir):
        root_path = Path(root)
        root_path.chmod(root_path.stat().st_mode | stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        for name in directories:
            path = root_path / name
            path.chmod(path.stat().st_mode | stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        for name in files:
            path = root_path / name
            if not path.is_symlink():
                path.chmod(path.stat().st_mode | stat.S_IRUSR | stat.S_IWUSR)


def snapshot_patch(
    out_dir: Path,
    *,
    workspace: Path = Path("/workspace"),
    git_base: Path = Path("/opt/harness/git-base/.git"),
) -> tuple[Path, list[str]]:
    """Atomically snapshot the workspace without writing a submission receipt."""

    if not git_base.is_dir():
        raise LifecycleError(f"pristine git base is missing: {git_base}")
    if not workspace.is_dir():
        raise LifecycleError(f"workspace is missing: {workspace}")
    out_dir.mkdir(parents=True, exist_ok=True)
    _clean_workspace(workspace)
    with tempfile.TemporaryDirectory(prefix="ai4ai-lifecycle-") as temporary:
        git_dir = Path(temporary) / ".git"
        shutil.copytree(git_base, git_dir)
        _make_temporary_git_writable(git_dir)
        environment = os.environ.copy()
        environment.update({"GIT_DIR": str(git_dir), "GIT_WORK_TREE": str(workspace)})
        add = subprocess.run(
            ["git", "add", "-A", "-f"],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        if add.returncode != 0:
            raise LifecycleError(f"git add failed: {add.stderr.strip()[:400]}")
        diff = subprocess.run(
            ["git", "diff", "--cached", "--binary"],
            check=False,
            capture_output=True,
            env=environment,
        )
        names = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "-z"],
            check=False,
            capture_output=True,
            env=environment,
        )
        if diff.returncode != 0 or names.returncode != 0:
            raise LifecycleError("git diff failed while snapshotting the workspace")
    patch = out_dir / "candidate.patch"
    _atomic_bytes(patch, diff.stdout)
    changed = [
        item.decode("utf-8", "surrogateescape")
        for item in names.stdout.split(b"\0")
        if item
    ]
    return patch, changed


def _observation() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return process_snapshot(), gpu_process_snapshot()


def observation_payload() -> dict[str, Any]:
    processes, gpu_processes = _observation()
    return {
        "observed_at": utc_now(),
        "active_processes": processes,
        "active_gpu_processes": gpu_processes,
        "active_work": active_work(processes, gpu_processes),
        "agent_session_active": session_active(processes),
    }


def write_agent_submission(out_dir: Path) -> dict[str, Any]:
    for receipt in ("submit.json", "no_candidate.json"):
        if (out_dir / receipt).exists():
            raise LifecycleError(f"lifecycle receipt already exists: {receipt}")
    processes, gpu_processes = _observation()
    patch, changed = snapshot_patch(out_dir)
    payload = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "receipt_type": "submission",
        "origin": "agent_explicit",
        "submitted_at": utc_now(),
        "remaining_seconds_at_submission": deadline_remaining(),
        "changed_files": changed,
        "patch_bytes": patch.stat().st_size,
        "patch_sha256": sha256_file(patch),
        "active_processes_at_submission": processes,
        "active_gpu_processes_at_submission": gpu_processes,
        "active_work_at_submit": active_work(processes, gpu_processes),
        "agent_session_active_at_submission": session_active(processes),
    }
    atomic_json(out_dir / "submit.json", payload)
    return payload


def write_no_candidate(out_dir: Path, reason: str) -> dict[str, Any]:
    if not reason.strip():
        raise LifecycleError("no-candidate requires a non-empty reason")
    for receipt in ("submit.json", "no_candidate.json"):
        if (out_dir / receipt).exists():
            raise LifecycleError(f"lifecycle receipt already exists: {receipt}")
    processes, gpu_processes = _observation()
    payload = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "receipt_type": "no_candidate",
        "origin": "agent_explicit",
        "declared_at": utc_now(),
        "reason": reason.strip(),
        "remaining_seconds_at_termination": deadline_remaining(),
        "active_processes_at_termination": processes,
        "active_gpu_processes_at_termination": gpu_processes,
        "active_work_at_submit": active_work(processes, gpu_processes),
        "agent_session_active_at_termination": session_active(processes),
    }
    atomic_json(out_dir / "no_candidate.json", payload)
    return payload


def classify_patch(path: Path) -> str:
    if not path.is_file():
        return "missing"
    return "nonempty" if path.stat().st_size else "empty"


def _valid_receipt(path: Path, receipt_type: str) -> dict[str, Any] | None:
    payload = read_json(path)
    if not payload:
        return None
    if payload.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        return None
    if payload.get("receipt_type") != receipt_type:
        return None
    if payload.get("origin") != "agent_explicit":
        return None
    if receipt_type == "no_candidate" and not str(payload.get("reason", "")).strip():
        return None
    return payload


def resolve_agent_receipts(out_dir: Path) -> tuple[str | None, dict[str, Any] | None, str | None]:
    """Return receipt kind, payload, and a fail-closed conflict reason."""

    submit_path = out_dir / "submit.json"
    no_candidate_path = out_dir / "no_candidate.json"
    submit = _valid_receipt(submit_path, "submission")
    no_candidate = _valid_receipt(no_candidate_path, "no_candidate")
    invalid = []
    if submit_path.exists() and not submit:
        invalid.append(submit_path.name)
    if no_candidate_path.exists() and not no_candidate:
        invalid.append(no_candidate_path.name)
    if invalid:
        return None, None, f"invalid lifecycle receipt: {', '.join(invalid)}"
    if submit and no_candidate:
        return None, None, "both submit.json and no_candidate.json exist"
    if submit:
        patch = out_dir / "candidate.patch"
        actual_bytes = patch.stat().st_size if patch.is_file() else None
        actual_sha = sha256_file(patch)
        if submit.get("patch_bytes") != actual_bytes or submit.get("patch_sha256") != actual_sha:
            return "submit", submit, "submit receipt does not match candidate.patch"
        return "submit", submit, None
    if no_candidate:
        return "no_candidate", no_candidate, None
    return None, None, None


def lifecycle_from_v4(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    for field in (
        "agent_exit_state",
        "termination_reason",
        "submission_origin",
        "candidate_state",
    ):
        normalized[field] = LEGACY_UNKNOWN
    normalized.setdefault("remaining_seconds_at_termination", None)
    normalized.setdefault("candidate_patch_bytes", None)
    normalized.setdefault("active_processes_at_termination", [])
    normalized.setdefault("active_gpu_processes_at_termination", [])
    normalized.setdefault("active_work_at_submit", None)
    normalized.setdefault("agent_session_active_at_termination", None)
    normalized.setdefault("candidate_rejection_reason", None)
    return normalized


def normalize_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        version = int(payload.get("schema_version", 0))
    except (TypeError, ValueError):
        version = 0
    normalized = lifecycle_from_v4(payload) if version < 5 else dict(payload)
    if version < 6:
        normalized.setdefault("formal_artifacts", [])
        normalized.setdefault("final_artifact_results", [])
        normalized.setdefault("final_selection_rule", None)
        normalized.setdefault("selected_artifact", None)
        normalized.setdefault("selected_progress", None)
        normalized.setdefault("retrain_budget_seconds", None)
        normalized.setdefault("retrain_budget_utilization", None)
    return normalized


def retrain_eligible(lifecycle: dict[str, Any]) -> bool:
    return (
        lifecycle.get("termination_reason") == "agent_explicit_submit"
        and lifecycle.get("submission_origin") == "agent"
        and lifecycle.get("candidate_state") in {"nonempty", "empty"}
    )


def write_rejection(path: Path, reason: str, stage: str) -> None:
    atomic_json(
        path,
        {
            "schema_version": 1,
            "status": "rejected",
            "stage": stage,
            "reason": reason,
            "recorded_at": utc_now(),
        },
    )


def report_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        payload = read_json(path)
        if not payload:
            continue
        row = normalize_manifest(payload)
        row["manifest_path"] = str(path)
        rows.append(row)
    return rows


def _manifest_paths(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob("manifest.json"))


def _print_report(rows: list[dict[str, Any]], output_format: str) -> None:
    fields = (
        "status",
        "agent_exit_state",
        "termination_reason",
        "submission_origin",
        "candidate_state",
        "active_work_at_submit",
        "host",
        "gpu",
        "task",
        "model",
        "effort",
        "session_id",
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "reasoning_output_tokens",
        "total_tokens",
        "manifest_path",
    )
    if output_format == "json":
        print(json.dumps(rows, indent=2, sort_keys=True))
        return
    print("\t".join(fields))
    for row in rows:
        print("\t".join(str(row.get(field, "")) for field in fields))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    submit = subparsers.add_parser("submit")
    submit.add_argument("--out", type=Path, default=Path("/out"))
    no_candidate = subparsers.add_parser("no-candidate")
    no_candidate.add_argument("reason")
    no_candidate.add_argument("--out", type=Path, default=Path("/out"))
    patch = subparsers.add_parser("snapshot-patch")
    patch.add_argument("--out", type=Path, default=Path("/out"))
    subparsers.add_parser("observe")
    report = subparsers.add_parser("report")
    report.add_argument("root", type=Path)
    report.add_argument("--format", choices=("tsv", "json"), default="tsv")
    eligible = subparsers.add_parser("retrain-eligible")
    eligible.add_argument("lifecycle", type=Path)
    reject = subparsers.add_parser("reject")
    reject.add_argument("--path", type=Path, required=True)
    reject.add_argument("--stage", required=True)
    reject.add_argument("reason")
    args = parser.parse_args(argv)

    try:
        if args.command == "submit":
            payload = write_agent_submission(args.out)
            print(
                f"submit: {len(payload['changed_files'])} changed file(s), "
                f"{payload['patch_bytes']} bytes"
            )
        elif args.command == "no-candidate":
            write_no_candidate(args.out, args.reason)
            print("no-candidate: declaration recorded")
        elif args.command == "snapshot-patch":
            path, changed = snapshot_patch(args.out)
            print(json.dumps({"patch": str(path), "changed_files": changed}))
        elif args.command == "observe":
            print(json.dumps(observation_payload(), sort_keys=True))
        elif args.command == "report":
            _print_report(report_rows(_manifest_paths(args.root)), args.format)
        elif args.command == "retrain-eligible":
            payload = read_json(args.lifecycle)
            return 0 if payload and retrain_eligible(payload) else 3
        elif args.command == "reject":
            write_rejection(args.path, args.reason, args.stage)
    except LifecycleError as error:
        print(f"lifecycle: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
