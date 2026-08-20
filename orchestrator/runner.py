"""Run any phase of any task from its declaration.

Replaces explore.py, retrain.py and score.py, which were three copies of "read
task.toml, check the image, take the resource limits, build a ContainerSpec, run
it" with the task's own paths and budgets baked into each one. Adding a second task
meant editing generic code, and the nine tasks still waiting do not share this
one's shape.

What stayed in Python is the five hooks -- the logic that genuinely differs between
phases rather than the data. Everything else comes from tasks/<task>/declaration.py.

    runner.py explore --assets A --out O --logs L
    runner.py retrain --assets A --out O --patch P/candidate.patch
    runner.py score   --assets A --out O --logs L --checkpoint C

Mount sources are written as `asset:<relative>` for things under --assets and
`run:<name>` for paths given on the command line, so the declaration never contains
a host path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import agent
import egress
from asset_identity import (
    AssetIdentityError,
    aliases_from_phase,
    aliases_from_phases,
    require_assets,
    verify_assets,
)
from container import (
    DEFAULT_RUNTIME,
    ContainerError,
    ContainerSpec,
    Mount,
    background,
    run,
    runtime_argv,
)
from host_contract import HostContractError, require_contract, verify_contract
from lifecycle import (
    atomic_json,
    classify_patch,
    read_json,
    resolve_agent_receipts,
    sha256_file,
    utc_now,
    write_rejection,
)
from phase_identity import content_digest, phase_payload
from phase_loader import PhaseDeclarationError
from phase_loader import load_phases as _load_phases
from task import (
    describe_patch_rejections,
    ensure_image_available,
    image_identity,
    inspect_gpu_hardware,
    load_task,
    require_free_space,
    require_free_vram,
    require_fresh_image,
    require_image_digest,
    require_image_tools,
    require_no_gpu_reservations,
    require_runnable_kernels,
    resources_from,
    source_hash,
    wait_for_free_vram_after_probe,
)

DEFAULT_TASK = Path("tasks/opd_math_1p5b")


# --------------------------------------------------------------------------- #
# loading the task's declaration
# --------------------------------------------------------------------------- #


def load_phases(task_dir: Path) -> dict[str, Any]:
    """Import tasks/<task>/declaration.py without requiring it to be a package.

    This is an import, so the module body runs: every Phase(...) call in it is
    executed here. The file is data in intent, not in mechanism -- it could run
    anything, and nothing stops it. Worth remembering for a system whose boundary is
    a mount list rather than a checker.
    """

    try:
        return _load_phases(task_dir, namespace="runner")
    except PhaseDeclarationError as error:
        raise ContainerError(str(error)) from error


def resolve_source(source: str, assets: Path, runtime: dict[str, Path]) -> Path:
    """`asset:models/student` -> assets/models/student; `run:patch` -> a CLI path."""

    kind, _, rest = source.partition(":")
    if kind == "asset":
        return assets / rest
    if kind == "run":
        head, _, tail = rest.partition("/")
        if head not in runtime:
            raise ContainerError(
                f"the declaration wants run:{rest}, but this phase was not given --{head}"
            )
        return runtime[head] / tail if tail else runtime[head]
    raise ContainerError(f"mount source must start with asset: or run:, got {source!r}")


# --------------------------------------------------------------------------- #
# hooks: the logic that genuinely differs per phase
# --------------------------------------------------------------------------- #


def write_deadline(context: dict[str, Any]) -> None:
    """Give the container an absolute deadline it can read and cannot move.

    /logs is mounted read-only, so timer.sh computes the remainder from this and
    the Agent cannot extend it.
    """

    logs: Path = context["runtime"]["logs"]
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / "deadline.json"
    if path.exists():
        existing = read_json(path)
        expected_phase = context["phase_name"]
        expected_budget = context["phase"].timeout_sec
        if (
            not existing
            or existing.get("phase") != expected_phase
            or existing.get("budget_seconds") != expected_budget
            or not isinstance(existing.get("deadline_unix"), (int, float))
        ):
            raise ContainerError(
                "existing deadline receipt conflicts with this phase; refusing to reset it"
            )
        return
    started = int(time.time())
    path.write_text(
        json.dumps(
            {
                "phase": context["phase_name"],
                "started_unix": started,
                "deadline_unix": started + context["phase"].timeout_sec,
                "budget_seconds": context["phase"].timeout_sec,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def report_submission(context: dict[str, Any]) -> None:
    """Report the explicit lifecycle outcome without inferring scientific success."""
    out: Path = context["runtime"]["out"]
    lifecycle = read_json(out / "lifecycle.json")
    if not lifecycle:
        print(f"{context['phase_name']}: lifecycle receipt is missing", file=sys.stderr)
        return
    print(
        f"{context['phase_name']}: termination={lifecycle['termination_reason']} "
        f"origin={lifecycle['submission_origin']} "
        f"candidate={lifecycle['candidate_state']}"
    )


def inspect_patch(context: dict[str, Any]) -> None:
    """Refuse a patch carrying data or weights, before a GPU is claimed.

    /assets being read-only stops the fixed inputs being edited, not bypassed: a
    parquet written under /workspace rides in here inside candidate.patch, which
    submit.sh generates with --binary. run.sh also refuses a TRAIN_DATA outside
    /assets, but the Agent owns run.sh and can delete that check. This one it
    cannot reach.
    """

    patch: Path = context["runtime"]["patch"]
    run_config_path = context["args"].run_config
    run_config = read_json(run_config_path) if run_config_path else None
    identity = (run_config or {}).get("candidate_patch_identity")
    if (run_config or {}).get("execution_mode") == "external_patch_replay":
        actual_sha256 = sha256_file(patch)
        actual_bytes = patch.stat().st_size if patch.is_file() else None
        if not (
            isinstance(identity, dict)
            and identity.get("provenance") == "external_patch"
            and identity.get("sha256") == actual_sha256
            and identity.get("size_bytes") == actual_bytes
        ):
            reason = "external patch bytes do not match the immutable run configuration"
            write_rejection(
                context["runtime"]["out"].parent / "rejection.json",
                reason,
                "patch_identity_check",
            )
            raise SystemExit(f"{context['phase_name']}: {reason}")
    reasons = describe_patch_rejections(patch)
    if reasons and not context["args"].allow_data_in_patch:
        detail = "\n".join(f"  {reason}" for reason in reasons)
        write_rejection(
            context["runtime"]["out"].parent / "rejection.json",
            "; ".join(reasons),
            "patch_check",
        )
        raise SystemExit(
            f"{context['phase_name']}: refusing to replay {patch}\n{detail}\n"
            "A candidate patch carries source, not data or weights. The model, the "
            "data and the evaluator are the fixed inputs.\n"
            "Pass --allow-data-in-patch if this is deliberate."
        )
    if reasons:
        print(f"{context['phase_name']}: WARNING replaying a patch that carries data:")
        for reason in reasons:
            print(f"  {reason}")


# Keys in the resolved Hydra config that must name one of the fixed inputs. Named
# explicitly rather than scanned for, because the config carries defaults that are never
# read -- `critic.model.path: ~/models/deepseek-llm-7b-chat` is present in every run and
# would false-positive a scan while nothing loads it.
FIXED_INPUT_KEYS = (
    ("actor_rollout_ref", "model", "path"),
    ("data", "train_files"),
    ("data", "val_files"),
    ("distillation", "teacher_models", "teacher_model", "model_path"),
)


def _scalar(value: str) -> Any:
    """Coerce the subset of YAML scalar types used by fixed-input keys."""
    if value in ("null", "~", ""):
        return None
    if value in ("true", "false"):
        return value == "true"
    stripped = value.strip("'\"")
    try:
        return int(stripped)
    except ValueError:
        return stripped


def _resolved_config(path: Path) -> dict:
    """Read Hydra's resolved config without requiring pyyaml.

    Only what FIXED_INPUT_KEYS needs is parsed -- nested mappings by indentation, scalars,
    and single-level `- item` lists. A general YAML parser here would be more code and more
    surface for no additional check. pyyaml is still preferred when present, being stricter.
    """

    text = path.read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError:
        pass
    else:
        return yaml.safe_load(text) or {}

    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    # The key whose value has not been seen yet: `foo:` is a mapping or a list depending on
    # what the next line is, so the decision waits for it.
    open_key: tuple[dict, str, int] | None = None

    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        stripped = raw.strip()

        if stripped.startswith("- "):
            if open_key is not None:
                parent, key, _ = open_key
                if not isinstance(parent.get(key), list):
                    parent[key] = []
                parent[key].append(stripped[2:].strip().strip("'\""))
            continue

        key, sep, value = stripped.partition(":")
        if not sep:
            continue
        key, value = key.strip(), value.strip()

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if value == "":
            child: dict = {}
            parent[key] = child
            stack.append((indent, child))
            open_key = (parent, key, indent)
        else:
            open_key = None
            parent[key] = _scalar(value)
    return root


def verify_fixed_inputs(context: dict[str, Any]) -> None:
    """Check what the trainer actually resolved, not what the patch looked like.

    inspect_patch reads the diff before a GPU is claimed, which is the cheap end. This
    is the other end: Hydra writes the fully resolved config to
    /out/hydra/.hydra/config.yaml, so after the run there is a record of the paths the
    trainer really used. A candidate that hides a dataset in a Python literal and builds
    a parquet at runtime passes the patch gate and fails here, because train_files then
    names the runtime file instead of the mount.

    Not a boundary, and worth being precise about why: a candidate that edits verl's
    dataset loader to substitute rows leaves train_files pointing at the mount, and this
    check reads clean. The boundary is that the container mounts one read-only data file
    and has no network, so the patch is the only way data can enter -- inspect_patch is
    the lock and this is the cross-check. Two independent readings of the same claim.

    A missing key is a failure rather than a skip. If verl's schema moves, that should
    surface as an error and not as a check that quietly stopped checking.
    """

    out: Path = context["runtime"]["out"]
    config_path = out / "hydra/.hydra/config.yaml"
    if not config_path.is_file():
        problem = f"resolved config is absent at {config_path}"
        print(f"{context['phase_name']}: FIXED INPUTS VIOLATED")
        print(f"  {problem}")
        marker = out / "FIXED_INPUTS_VIOLATED"
        marker.write_text(problem + "\n", encoding="utf-8")
        print(f"  wrote {marker} -- scoring will refuse this run")
        return

    config = _resolved_config(config_path)
    problems: list[str] = []
    for key in FIXED_INPUT_KEYS:
        node: Any = config
        for part in key:
            if not isinstance(node, dict) or part not in node:
                problems.append(f"{'.'.join(key)} is absent from the resolved config")
                node = None
                break
            node = node[part]
        if node is None:
            continue
        values = node if isinstance(node, list) else [node]
        for value in values:
            if not str(value).startswith("/assets/"):
                problems.append(f"{'.'.join(key)} = {value} is not under /assets")

    if problems:
        print(f"{context['phase_name']}: FIXED INPUTS VIOLATED")
        for problem in problems:
            print(f"  {problem}")
        print(
            "  the model and the data are two of the three fixed inputs; this run "
            "trained on something else and its score is not comparable"
        )
        # The lifecycle reads this marker and refuses to score the violating run.
        marker = out / "FIXED_INPUTS_VIOLATED"
        marker.write_text("\n".join(problems) + "\n", encoding="utf-8")
        print(f"  wrote {marker} -- scoring will refuse this run")
        return
    print(f"{context['phase_name']}: fixed inputs verified -- model and data both /assets")


def collect_checkpoints(context: dict[str, Any]) -> None:
    """Collect checkpoints in numeric step order."""

    out: Path = context["runtime"]["out"]
    glob = context["phase"].checkpoint_glob or context["phase"].output_glob
    if not glob:
        return
    root = out / glob.split("/")[0]
    pattern = glob.split("/", 1)[1] if "/" in glob else "*"
    found = []
    for path in root.glob(pattern):
        match = re.search(r"(\d+)$", path.name)
        if match:
            found.append((int(match.group(1)), path))
    found.sort()
    print(f"{context['phase_name']}: {len(found)} checkpoint(s)")
    if found:
        step, path = found[-1]
        print(f"{context['phase_name']}: latest step {step} at {path}")


def report_reward(context: dict[str, Any]) -> None:
    logs: Path = context["runtime"]["logs"]
    out: Path = context["runtime"]["out"]
    rewards = [logs / "verifier/reward", logs / "verifier/reward.txt"]
    present = [
        (path, path.read_text(encoding="utf-8").strip()) for path in rewards if path.is_file()
    ]
    if len(present) == 2 and present[0][1] != present[1][1]:
        raise ContainerError(
            "conflicting verifier rewards: "
            + ", ".join(f"{path.name}={value!r}" for path, value in present)
        )
    if present:
        print(f"{context['phase_name']}: reward {present[0][1]}")
    summary = out / "summary.json"
    if not summary.is_file():
        return
    payload = json.loads(summary.read_text(encoding="utf-8"))
    # Every key checked before use, because this hook runs AFTER the phase and a failed
    # phase leaves whatever was there before. retrain and score share the `run:out` mount,
    # so when score dies before writing its own summary this reads RETRAIN's -- whose keys
    # are a different set entirely. The unconditional payload['metric'] then raised
    # KeyError, and that KeyError became the last line of the log, burying score's actual
    # error (a FileNotFoundError naming the real problem) twelve lines further up.
    #
    # A reporting hook must not be able to fail. Its job is to print a number; if the
    # number is not there, the interesting event is whatever happened instead, and this
    # should get out of the way rather than replace that error with its own.
    missing = [key for key in ("metric", "score") if key not in payload]
    if missing:
        print(
            f"{context['phase_name']}: {summary} has no {', '.join(missing)} -- "
            "not this phase's summary, or the phase did not finish"
        )
        return
    line = f"{context['phase_name']}: {payload['metric']} = {payload['score']:.6f}"
    # correct/n is an accuracy shape. A perplexity or NLL summary has neither, and OWL's
    # declaration drops this hook entirely for that reason -- but a task that keeps the
    # hook and reports a non-accuracy metric should still get its number printed.
    if "correct" in payload and "n" in payload:
        line += f" ({payload['correct']}/{payload['n']})"
    if isinstance(payload.get("stderr"), (int, float)):
        line += f", stderr {payload['stderr']:.6f}"
    print(line)


BEFORE = {
    "write_deadline": write_deadline,
    "inspect_patch": inspect_patch,
}
AFTER = {
    "report_submission": report_submission,
    "collect_checkpoints": collect_checkpoints,
    "report_reward": report_reward,
    "verify_fixed_inputs": verify_fixed_inputs,
}


# --------------------------------------------------------------------------- #
# building the container from the declaration
# --------------------------------------------------------------------------- #


def wrap_command(phase: Any, patch_target: str) -> tuple[str, ...]:
    """Fold patch application and fixed-input exports into a shell wrapper."""

    if not phase.apply_patch and not phase.exports:
        return phase.command

    exports = list(phase.exports_with_wall_clock)
    script = "\n"
    if phase.apply_patch:
        script += (
            "set -euo pipefail\n"
            f"if [ -s {patch_target} ]; then\n"
            "  git -C /workspace apply --verbose "
            "--exclude='**/__pycache__/**' --exclude='*.pyc' --exclude='*.pyo' "
            f"{patch_target}\n"
            f'  echo "{phase.name}: candidate patch applied"\n'
            "else\n"
            f'  echo "{phase.name}: empty patch, training the pristine baseline"\n'
            "fi\n"
        )
    else:
        script += "set -euo pipefail\n"
    if exports:
        script += "export " + " ".join(shlex.quote(item) for item in exports) + "\n"
    script += "exec " + " ".join(phase.command) + "\n"
    return ("bash", "-lc", script)


def build(
    *,
    phase: Any,
    phase_name: str,
    image: str,
    assets: Path,
    runtime: dict[str, Path],
    gpus: tuple[int, ...],
    resources: dict,
    digest: str = "",
    config_digest: str = "",
    agent_binary: Path | None = None,
    agent_container_binary: str | None = None,
    deferred_agent_finish: Path | None = None,
    network: str | None = None,
    extra_environment: dict[str, str] | None = None,
) -> ContainerSpec:
    mounts = []
    patch_target = ""
    for source, target, read_only in phase.mounts:
        mounts.append(Mount(resolve_source(source, assets, runtime), target, read_only))
        if source.startswith("run:patch"):
            patch_target = target
    if agent_binary is not None and agent_container_binary is not None:
        # Read-only, and outside /workspace so it cannot land in candidate.patch.
        mounts.append(Mount(agent_binary, agent_container_binary))
    if deferred_agent_finish is not None:
        # Claude's terminal result carries authoritative usage/model provenance.
        # The frozen image wrappers kill PID 1 immediately after writing a receipt,
        # which truncates that result. These read-only mounts preserve the exact
        # receipt commands but let the CLI return before background() removes PID 1.
        mounts.append(Mount(deferred_agent_finish, "/opt/harness/submit.sh"))
        mounts.append(Mount(deferred_agent_finish, "/opt/harness/no_candidate.sh"))

    command = wrap_command(phase, patch_target)
    command = tuple(part.replace("$gpus", str(len(gpus))) for part in command)

    environment: dict[str, str] = {}
    if phase.pass_image_digest and digest:
        environment["IMAGE_DIGEST"] = digest
    if phase.pass_image_digest and config_digest:
        environment["IMAGE_CONFIG_DIGEST"] = config_digest
    # Set on the container rather than folded into the command, which is the only
    # form the agent path can use: `docker exec` inherits the container's
    # environment and sees nothing a bash wrapper around PID 1 exported. See the
    # call in run_with_agent.
    if extra_environment:
        environment.update(extra_environment)

    return ContainerSpec(
        # Seconds alone collide. Three seeded fast_eval runs launched together for a
        # noise-floor measurement are started within the same second, and the second and
        # third died on "container name is already in use" -- docker exit 125, which is
        # the daemon refusing rather than anything about the task. The pid disambiguates
        # concurrent runs on one host; the timestamp still orders them for a human.
        name=f"ai4ai-{phase_name}-{int(time.time())}-{os.getpid()}",
        image=image,
        command=command,
        **resources_from(resources),
        mounts=tuple(mounts),
        gpu_devices=gpus,
        environment=environment,
        timeout_seconds=phase.timeout_sec,
        read_only_root=phase.read_only_root,
        interactive=phase.interactive,
        network=network,
    )


NETWORK_NAME = "ai4ai-egress"


def network_name_for_run(out: Path) -> str:
    """Give every concurrent agent its own internal bridge and proxy endpoint."""

    suffix = hashlib.sha256(str(out.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"{NETWORK_NAME}-{suffix}"


def _remove_host_secrets(out: Path) -> None:
    """Delete anything key-shaped the agent phase may have left on host disk.

    The secret now lives on the container's tmpfs, so this should find nothing. It runs
    anyway, from the host, because the in-container teardown cannot be relied on: it goes
    through docker exec and submit.sh ends with `kill -TERM 1`, so on a successful
    submission the container is gone before cleanup runs. That is how an earlier layout,
    which wrote the key under /out, left it on disk in three runs out of four -- the one
    exception being the run that failed without submitting.

    Runs from the host precisely so it does not depend on the container existing.
    """

    for path in (out / "tmp/agent-secrets", out / "tmp/agent/auth.json"):
        if not path.exists():
            continue
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            print(f"agent: removed {path}")
        except OSError as error:
            # Worth a loud line rather than an exception: the phase's result is already
            # on disk and a failed unlink should not discard it.
            print(f"agent: WARNING could not remove {path}: {error}")


def _container_alive(container: str, *, runtime: str) -> bool:
    result = subprocess.run(
        [*runtime_argv(runtime), "inspect", "--format", "{{.State.Running}}", container],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def _observe_container(container: str, *, runtime: str) -> dict[str, Any]:
    result = subprocess.run(
        [
            *runtime_argv(runtime),
            "exec",
            container,
            "python3",
            "/opt/harness/lifecycle.py",
            "observe",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        return {
            "observed_at": utc_now(),
            "active_processes": [],
            "active_gpu_processes": [],
            "active_work": None,
            "agent_session_active": None,
            "observation_error": (result.stderr or result.stdout).strip()[:400],
        }
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {
            "observed_at": utc_now(),
            "active_processes": [],
            "active_gpu_processes": [],
            "active_work": None,
            "agent_session_active": None,
            "observation_error": "container observation was not valid JSON",
        }
    return payload if isinstance(payload, dict) else {}


def _freeze_observed_processes(
    container: str, observation: dict[str, Any], *, runtime: str
) -> None:
    pids = sorted(
        {
            int(row["pid"])
            for row in observation.get("active_processes", [])
            if isinstance(row, dict) and str(row.get("pid", "")).isdigit() and int(row["pid"]) > 1
        }
    )
    if not pids:
        return
    subprocess.run(
        [
            *runtime_argv(runtime),
            "exec",
            container,
            "bash",
            "-lc",
            "kill -STOP " + " ".join(map(str, pids)) + " 2>/dev/null || true",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _host_snapshot_patch(
    container: str, *, out: Path, runtime: str
) -> tuple[list[str], str | None]:
    result = subprocess.run(
        [
            *runtime_argv(runtime),
            "exec",
            container,
            "python3",
            "/opt/harness/lifecycle.py",
            "snapshot-patch",
            "--out",
            "/out",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        return [], (result.stderr or result.stdout).strip()[:400] or "snapshot command failed"
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return [], "snapshot command did not return valid JSON"
    changed = payload.get("changed_files", [])
    return [str(item) for item in changed] if isinstance(changed, list) else [], None


def _remaining_seconds(logs: Path) -> int | None:
    deadline = read_json(logs / "deadline.json")
    try:
        return max(0, int(float(deadline["deadline_unix"]) - time.time())) if deadline else None
    except (KeyError, TypeError, ValueError):
        return None


def _patch_rejection(patch: Path) -> str | None:
    if not patch.is_file() or not patch.stat().st_size:
        return None
    reasons = describe_patch_rejections(patch)
    return "; ".join(reasons) if reasons else None


def _write_exploration_lifecycle(out: Path, payload: dict[str, Any]) -> None:
    payload = {
        "schema_version": 1,
        "record_type": "exploration_lifecycle",
        "recorded_at": utc_now(),
        **payload,
    }
    atomic_json(out / "lifecycle.json", payload)


def _finalize_exploration(
    container: str,
    *,
    out: Path,
    logs: Path,
    raw_status: int,
    runtime: str,
    operator_interrupt: bool = False,
) -> int:
    """Resolve explicit receipts first; otherwise preserve a host-owned capture."""

    receipt_kind, receipt, receipt_error = resolve_agent_receipts(out)
    patch = out / "candidate.patch"
    if receipt_kind == "submit" and receipt and not receipt_error:
        rejection = _patch_rejection(patch)
        candidate_state = "rejected" if rejection else classify_patch(patch)
        _write_exploration_lifecycle(
            out,
            {
                "agent_exit_state": "completed",
                "termination_reason": "agent_explicit_submit",
                "submission_origin": "agent",
                "candidate_state": candidate_state,
                "remaining_seconds_at_termination": receipt.get("remaining_seconds_at_submission"),
                "candidate_patch_bytes": patch.stat().st_size if patch.is_file() else None,
                "candidate_patch_sha256": sha256_file(patch),
                "active_processes_at_termination": receipt.get(
                    "active_processes_at_submission", []
                ),
                "active_gpu_processes_at_termination": receipt.get(
                    "active_gpu_processes_at_submission", []
                ),
                "active_work_at_submit": receipt.get("active_work_at_submit"),
                "active_work_at_termination": receipt.get("active_work_at_submit"),
                "agent_session_active_at_termination": receipt.get(
                    "agent_session_active_at_submission"
                ),
                "candidate_rejection_reason": rejection,
                "raw_agent_exit_status": raw_status,
            },
        )
        return 0

    if receipt_kind == "no_candidate" and receipt and not receipt_error:
        _write_exploration_lifecycle(
            out,
            {
                "agent_exit_state": "completed",
                "termination_reason": "agent_explicit_no_candidate",
                "submission_origin": "none",
                "candidate_state": "no_candidate",
                "remaining_seconds_at_termination": receipt.get("remaining_seconds_at_termination"),
                "candidate_patch_bytes": None,
                "candidate_patch_sha256": None,
                "active_processes_at_termination": receipt.get(
                    "active_processes_at_termination", []
                ),
                "active_gpu_processes_at_termination": receipt.get(
                    "active_gpu_processes_at_termination", []
                ),
                "active_work_at_submit": receipt.get("active_work_at_submit"),
                "active_work_at_termination": receipt.get("active_work_at_submit"),
                "agent_session_active_at_termination": receipt.get(
                    "agent_session_active_at_termination"
                ),
                "candidate_rejection_reason": None,
                "no_candidate_reason": receipt.get("reason"),
                "raw_agent_exit_status": raw_status,
            },
        )
        return 0

    if receipt_error:
        _write_exploration_lifecycle(
            out,
            {
                "agent_exit_state": "failed",
                "termination_reason": "runtime_failure",
                "submission_origin": "none",
                "candidate_state": "rejected",
                "remaining_seconds_at_termination": _remaining_seconds(logs),
                "candidate_patch_bytes": patch.stat().st_size if patch.is_file() else None,
                "candidate_patch_sha256": sha256_file(patch),
                "active_processes_at_termination": [],
                "active_gpu_processes_at_termination": [],
                "active_work_at_submit": None,
                "active_work_at_termination": None,
                "agent_session_active_at_termination": None,
                "candidate_rejection_reason": receipt_error,
                "raw_agent_exit_status": raw_status,
            },
        )
        return raw_status if raw_status else 1

    state = read_json(logs / "agent/state.json") or {}
    deadline = raw_status == 124 or state.get("status") == "timed_out"
    if operator_interrupt:
        exit_state = "failed"
        termination = "operator_interrupt"
    elif deadline:
        exit_state = "timed_out"
        termination = "phase_deadline"
    elif raw_status == 0 and state.get("status") == "completed":
        exit_state = "completed"
        termination = "agent_early_exit"
    else:
        exit_state = "failed"
        termination = "runtime_failure"

    if not _container_alive(container, runtime=runtime):
        if termination not in {"phase_deadline", "operator_interrupt"}:
            exit_state = "container_lost"
            termination = "runtime_failure"
        _write_exploration_lifecycle(
            out,
            {
                "agent_exit_state": exit_state,
                "termination_reason": termination,
                "submission_origin": "none",
                "candidate_state": "missing",
                "remaining_seconds_at_termination": _remaining_seconds(logs),
                "candidate_patch_bytes": None,
                "candidate_patch_sha256": None,
                "active_processes_at_termination": [],
                "active_gpu_processes_at_termination": [],
                "active_work_at_submit": None,
                "active_work_at_termination": None,
                "agent_session_active_at_termination": None,
                "candidate_rejection_reason": None,
                "raw_agent_exit_status": raw_status,
            },
        )
        return raw_status if raw_status else 1

    observation = _observe_container(container, runtime=runtime)
    _freeze_observed_processes(container, observation, runtime=runtime)
    changed, capture_error = _host_snapshot_patch(container, out=out, runtime=runtime)
    origin = "host_deadline_capture" if deadline else "host_early_exit_capture"
    capture = {
        "schema_version": 1,
        "receipt_type": "capture",
        "origin": origin,
        "captured_at": utc_now(),
        "remaining_seconds_at_termination": _remaining_seconds(logs),
        "changed_files": changed,
        "patch_bytes": patch.stat().st_size if patch.is_file() else None,
        "patch_sha256": sha256_file(patch),
        "active_processes_at_termination": observation.get("active_processes", []),
        "active_gpu_processes_at_termination": observation.get("active_gpu_processes", []),
        "active_work_at_termination": observation.get("active_work"),
        "agent_session_active_at_termination": observation.get("agent_session_active"),
        "capture_error": capture_error,
    }
    atomic_json(out / "capture.json", capture)
    rejection = capture_error or _patch_rejection(patch)
    candidate_state = "rejected" if rejection else classify_patch(patch)
    _write_exploration_lifecycle(
        out,
        {
            "agent_exit_state": exit_state,
            "termination_reason": termination,
            "submission_origin": origin,
            "candidate_state": candidate_state,
            "remaining_seconds_at_termination": capture["remaining_seconds_at_termination"],
            "candidate_patch_bytes": capture["patch_bytes"],
            "candidate_patch_sha256": capture["patch_sha256"],
            "active_processes_at_termination": capture["active_processes_at_termination"],
            "active_gpu_processes_at_termination": capture["active_gpu_processes_at_termination"],
            "active_work_at_submit": None,
            "active_work_at_termination": capture["active_work_at_termination"],
            "agent_session_active_at_termination": capture["agent_session_active_at_termination"],
            "candidate_rejection_reason": rejection,
            "raw_agent_exit_status": raw_status,
        },
    )
    if termination in {"runtime_failure", "operator_interrupt"}:
        return raw_status if raw_status else (130 if operator_interrupt else 1)
    return 0


def run_with_agent(context: dict[str, Any], *, parser) -> int:
    """Start the container, run the agent inside it, then take the workspace.

    Ordering matters. The container comes up first with no agent running, so the
    egress proxy and the allowlist are in place before anything can make a
    request; the agent is then started through `docker exec`.
    """

    args = context["args"]
    phase = context["phase"]
    runtime = context["runtime"]
    image = context["image"]
    # The whole task config, not just [environment]: resources_from needs the
    # top-level x-ai4ai table for the two limits Harbor has no name for.
    config = context["environment"]
    network_name = network_name_for_run(args.out)

    spec = agent.resolve(args.agent)
    deferred_agent_finish = (
        Path(__file__).with_name("deferred_agent_finish.sh") if spec.name == "claude" else None
    )
    try:
        api_key = spec.api_key()
    except agent.AgentError as error:
        parser.error(str(error))
    if not api_key and not args.dry_run:
        if spec.name == "claude":
            parser.error("ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN must be set to run claude")
        parser.error(f"{spec.api_key_env} must be set to run {spec.name}")

    instruction_path = args.instruction or (args.task / "instruction.md")
    if not instruction_path.is_file():
        parser.error(f"no instruction at {instruction_path}")
    instruction = instruction_path.read_text(encoding="utf-8")
    model = args.model or spec.default_model
    if not model:
        parser.error(
            f"--model is required for {spec.name}: it has no default, because a "
            "name this CLI does not recognise runs anyway with fallback metadata "
            "and quietly is not the model you meant"
        )
    try:
        model = spec.validate_model(model)
    except agent.AgentError as error:
        parser.error(str(error))
    if args.reasoning_effort is not None:
        try:
            effort = spec.validate_reasoning_effort(args.reasoning_effort)
        except agent.AgentError as error:
            parser.error(str(error))
        spec = replace(spec, reasoning_effort=effort)

    # with_command() clears exports for operator overrides. The Agent also replaces the
    # command, but still needs the phase's fixed model and data paths in its environment.
    # Phases without exports remain unaffected.
    agent_environment = dict(phase.exports)

    if args.dry_run:
        container = build(
            # PID 1 just waits. The agent arrives via exec; the host removes PID 1
            # after the agent emits its terminal result or the deadline expires.
            phase=phase.with_command(("sleep", "infinity")),
            phase_name=context["phase_name"],
            image=image,
            assets=args.assets,
            runtime=runtime,
            gpus=(args.gpu,),
            resources=config,
            digest=str(context.get("observed_image_digest") or ""),
            config_digest=str(context.get("observed_image_config_digest") or ""),
            agent_binary=spec.binary.resolve(),
            agent_container_binary=spec.container_binary,
            deferred_agent_finish=deferred_agent_finish,
            network=network_name,
            extra_environment=agent_environment,
        )
        print(" ".join(container.argv()))
        print(f"# then: {spec.container_binary} ... allowlist={list(spec.outbound_targets())}")
        return 0

    gateway = egress.create_network(network_name)
    proxy = egress.EgressProxy(
        allowed_hosts=frozenset(spec.outbound_targets()),
        bind_host=gateway,
        log_path=runtime["logs"] / "agent/egress.log",
    )
    proxy_url = proxy.start()
    print(f"egress: {proxy_url} allowing {sorted(spec.outbound_targets())}")

    container = build(
        phase=phase.with_command(("sleep", "infinity")),
        phase_name=context["phase_name"],
        image=image,
        assets=args.assets,
        runtime=runtime,
        gpus=(args.gpu,),
        resources=config,
        digest=str(context.get("observed_image_digest") or ""),
        config_digest=str(context.get("observed_image_config_digest") or ""),
        agent_binary=spec.binary.resolve(),
        agent_container_binary=spec.container_binary,
        deferred_agent_finish=deferred_agent_finish,
        network=network_name,
        extra_environment=agent_environment,
    )

    raw_status = 1
    lifecycle_status = 1
    operator_interrupt = False
    try:
        with background(container) as name:
            try:
                deadline = read_json(runtime["logs"] / "deadline.json") or {}
                raw_status = agent.run_agent(
                    spec,
                    name,
                    instruction=instruction,
                    model=model,
                    proxy_url=proxy_url,
                    api_key=api_key,
                    agent_log_dir=runtime["logs"] / "agent",
                    # Not a literal "docker": on a host where the account cannot reach the
                    # socket the invocation is wrapped, and this was the one call that did
                    # not honour it -- the agent path failed with "permission denied while
                    # trying to connect to the Docker daemon socket" after the container it
                    # was exec-ing into had already started.
                    runtime=DEFAULT_RUNTIME,
                    deadline_unix=float(deadline.get("deadline_unix", 0)) or None,
                    max_attempts=args.agent_max_attempts,
                    api_concurrency=args.agent_api_concurrency,
                    api_concurrency_root=args.agent_api_concurrency_root,
                )
            except KeyboardInterrupt:
                operator_interrupt = True
                raw_status = 130
                print("agent: operator interrupt received; preserving lifecycle evidence")
            except Exception as error:
                raw_status = 1
                print(f"agent: runtime failure: {error}", file=sys.stderr)
            finally:
                lifecycle_status = _finalize_exploration(
                    name,
                    out=runtime["out"],
                    logs=runtime["logs"],
                    raw_status=raw_status,
                    runtime=DEFAULT_RUNTIME,
                    operator_interrupt=operator_interrupt,
                )
    finally:
        _remove_host_secrets(context["args"].out)
        proxy.stop()
        egress.remove_network(network_name)
        if proxy.denied:
            print(f"egress: allowed {proxy.allowed_count}, denied {proxy.denied[:8]}")
        else:
            print(f"egress: allowed {proxy.allowed_count}, denied nothing")

    return lifecycle_status


def gpu_occupancy_evidence(
    devices: tuple[int, ...], *, mode: str, stage: str
) -> dict[str, object]:
    """Fail closed on occupancy independently of the hardware-match policy.

    ``hardware-check`` controls whether a different accelerator family/capacity is a
    local warning or an official-run failure.  It must never authorize launching on a
    device already reserved by another container or used by another process.
    """

    problems: list[str] = []
    checks = [lambda: require_no_gpu_reservations(devices)]
    if stage == "before_kernel_probe":
        checks.append(lambda: require_free_vram(devices))
    else:
        checks.append(lambda: wait_for_free_vram_after_probe(devices))
    for check in checks:
        try:
            check()
        except ContainerError as error:
            problems.append(str(error))
    if problems:
        detail = "\n".join(problems)
        raise ContainerError(f"GPU occupancy verification failed:\n{detail}")
    return {
        "mode": mode,
        "stage": stage,
        "status": "match" if not problems else "mismatch",
        "problems": problems,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", help="a key of PHASES in the task's declaration.py")
    parser.add_argument("--task", type=Path, default=DEFAULT_TASK)
    parser.add_argument("--image", default=None, help="default: [environment].image")
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--logs", type=Path, default=None)
    parser.add_argument("--patch", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--run-config",
        type=Path,
        default=None,
        help="immutable lifecycle receipt; its content hash is bound into preflight",
    )
    parser.add_argument("--gpu", type=int, default=0, help="device index")
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="*",
        default=None,
        help="several device indices for phases whose protocol explicitly permits them",
    )
    parser.add_argument("--timeout", type=int, default=None, help="override the declared one")
    parser.add_argument("-it", "--interactive", action="store_true")
    parser.add_argument(
        "--source-check",
        choices=("warn", "strict", "off"),
        default=os.environ.get("AI4AI_SOURCE_CHECK", "warn"),
        help="working-tree/image-source check (default: warn; official replay: strict)",
    )
    parser.add_argument(
        "--image-check",
        choices=("strict", "warn"),
        default=os.environ.get("AI4AI_IMAGE_CHECK", "warn"),
        help="declared image digest check (default: warn; official replay: strict)",
    )
    parser.add_argument(
        "--hardware-check",
        choices=("strict", "warn", "off"),
        default=os.environ.get("AI4AI_HARDWARE_CHECK", "warn"),
        help="GPU type/memory check (default: warn; official replay: strict)",
    )
    parser.add_argument(
        "--image-pull-policy",
        choices=("missing", "never"),
        default=os.environ.get("AI4AI_IMAGE_PULL_POLICY", "missing"),
        help="pull a missing image once, or require a preloaded image",
    )
    parser.add_argument(
        "--skip-digest-check",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--allow-data-in-patch", action="store_true")
    parser.add_argument(
        "--export",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="add an environment variable to the phase's exports. Generic on "
        "purpose: retrain.py had a --steps flag, it was passed on every run I ever "
        "made, and that is precisely why the step-count bug stayed hidden -- the "
        "path where nothing overrides the default was never taken. An override now "
        "shows up verbatim in the argv.",
    )
    parser.add_argument(
        "--agent",
        default=None,
        help="run a coding agent inside the container instead of the declared "
        "command. Only meaningful for a phase that gives the agent something to do.",
    )
    parser.add_argument("--model", default=None, help="model for --agent")
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help="explicit agent-specific reasoning effort",
    )
    parser.add_argument(
        "--instruction",
        type=Path,
        default=None,
        help="brief for --agent; defaults to the task's instruction.md",
    )
    parser.add_argument(
        "--agent-max-attempts",
        type=int,
        default=0,
        help="total agent attempts; 0 means transient resumes are bounded only by "
        "the original phase deadline",
    )
    parser.add_argument(
        "--agent-api-concurrency",
        type=int,
        default=0,
        help="independent agent API concurrency limit; 0 disables the semaphore",
    )
    parser.add_argument(
        "--agent-api-concurrency-root",
        type=Path,
        default=Path(os.environ.get("AI4AI_AGENT_API_CONCURRENCY_ROOT", "/tmp/ai4ai-agent-api")),
        help="lock-pool directory; use shared storage for a cross-host API limit",
    )
    parser.add_argument("--dry-run", action="store_true")
    # Split `--` explicitly because adjacent variable-length arguments are parsed
    # differently across supported Python versions.
    argv = sys.argv[1:]
    override: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, override = argv[:cut], argv[cut + 1 :]
    args = parser.parse_args(argv)
    args.extra = override
    if args.skip_digest_check:
        print(
            "warning: --skip-digest-check is deprecated; using --source-check off "
            "--image-check warn. Capability probes still run.",
            file=sys.stderr,
        )
        args.source_check = "off"
        args.image_check = "warn"

    # Reject invalid batch selectors before loading declarations, inspecting images,
    # or checking free GPU memory. This keeps a typo from claiming a card first.
    if args.agent:
        try:
            spec = agent.resolve(args.agent)
            if args.model is not None:
                spec.validate_model(args.model)
            if args.reasoning_effort is not None:
                spec.validate_reasoning_effort(args.reasoning_effort)
            agent.validate_agent_max_attempts(args.agent_max_attempts)
            agent.validate_agent_api_concurrency(args.agent_api_concurrency)
            agent.validate_agent_api_concurrency_root(
                args.agent_api_concurrency_root, args.agent_api_concurrency
            )
        except agent.AgentError as error:
            parser.error(str(error))

    # Verify the host protocol before importing declaration.py.  The declaration is
    # executable Python, so importing first would let a modified contract run before a
    # strict mismatch was noticed.
    try:
        host_contract = verify_contract(
            args.task,
            mode=args.source_check,
            instruction=args.instruction,
        )
        require_contract(host_contract)
    except HostContractError as error:
        raise ContainerError(str(error)) from error
    if args.source_check == "warn" and host_contract["status"] != "match":
        changed = ", ".join(host_contract.get("changed_files") or []) or "unknown files"
        print(
            "warning: host benchmark contract is not release-identical; continuing "
            f"in non-official mode ({host_contract['status']}: {changed})",
            file=sys.stderr,
        )

    phases = load_phases(args.task)
    asset_aliases = aliases_from_phases(phases)
    if args.phase not in phases:
        parser.error(f"unknown phase {args.phase!r}; have {sorted(phases)}")
    phase = phases[args.phase]

    config = load_task(args.task)
    environment = config.get("environment", {})
    image = args.image or environment.get("image")
    if not image:
        parser.error("no --image and no [environment].image in task.toml")

    runtime: dict[str, Path] = {"out": args.out}
    for name in ("logs", "patch", "checkpoint"):
        value = getattr(args, name)
        if value is not None:
            runtime[name] = value
    needed = {
        source.partition(":")[2].partition("/")[0]
        for source, _, _ in phase.mounts
        if source.startswith("run:")
    }
    for name in sorted(needed - set(runtime)):
        parser.error(f"phase {args.phase!r} mounts run:{name}, so --{name} is required")

    if args.timeout is not None:
        phase = phase.with_timeout(args.timeout)
    if args.extra:
        phase = phase.with_command(tuple(args.extra))
    if args.interactive:
        phase = phase.with_interactive()
    if args.export:
        extra: dict[str, str] = {}
        for item in args.export:
            key, sep, value = item.partition("=")
            if not sep:
                parser.error(f"--export wants KEY=VALUE, got {item!r}")
            extra[key] = value
        phase = phase.with_extra_exports(extra)

    # Every writable bind source has to exist before docker runs -- it refuses to
    # create one, and the error names only the path. score.py created
    # /logs/verifier explicitly; deriving it from the declaration means a new phase
    # cannot forget.
    args.out.mkdir(parents=True, exist_ok=True)
    for source, _, read_only in phase.mounts:
        if read_only or not source.startswith("run:"):
            continue
        resolve_source(source, args.assets, runtime).mkdir(parents=True, exist_ok=True)

    devices = tuple(args.gpus) if args.gpus else (args.gpu,)
    context: dict[str, Any] = {
        "phase": phase,
        "phase_name": args.phase,
        "runtime": runtime,
        "assets": args.assets,
        "args": args,
        "image": image,
        "environment": config,
    }

    if not args.dry_run:
        ensure_image_available(image, pull_policy=args.image_pull_policy)
        actual_source = source_hash(args.task)
        source_status = "not_checked" if args.source_check == "off" else "match"
        if args.source_check != "off":
            try:
                require_fresh_image(args.task, environment)
            except ContainerError as error:
                source_status = (
                    "unrecorded" if not environment.get("source_sha256") else "mismatch"
                )
                if args.source_check == "strict":
                    raise
                print(
                    "warning: task source differs from the source receipt recorded for "
                    f"the image; continuing in non-official mode:\n{error}",
                    file=sys.stderr,
                )
        identity = image_identity(image)
        image_status = "match"
        try:
            require_image_digest(
                image,
                environment.get("digest", ""),
                environment.get("config_digest", ""),
                identity=identity,
            )
        except ContainerError as error:
            image_status = "unrecorded" if not environment.get("digest") else "mismatch"
            if environment.get("digest") and not environment.get("config_digest"):
                image_status = "config_unrecorded"
            if args.image_check == "strict":
                raise
            print(
                f"warning: image identity is unverified; results are non-official:\n{error}",
                file=sys.stderr,
            )
        context["observed_image_digest"] = identity["layer_fingerprint"]
        context["observed_image_config_digest"] = identity["config_fingerprint"]
        try:
            asset_identity = verify_assets(
                args.task,
                args.assets.resolve(),
                aliases_from_phase(phase),
                mode=args.source_check,
                contract_aliases=asset_aliases,
            )
            require_assets(asset_identity)
        except AssetIdentityError as error:
            raise ContainerError(str(error)) from error
        if args.run_config and args.source_check == "strict":
            run_config_value = read_json(args.run_config)
            configured_assets = (run_config_value or {}).get("asset_identity") or {}
            if not (
                configured_assets.get("mode") == "strict"
                and configured_assets.get("status") == "locked"
                and configured_assets.get("algorithm") == asset_identity.get("algorithm")
                and configured_assets.get("digest") == asset_identity.get("digest")
            ):
                raise ContainerError(
                    "runtime asset identity differs from the immutable run configuration"
                )
        # Identity checks and capability checks are deliberately independent. A local
        # rebuild may be allowed, but an image without the lifecycle tools still cannot
        # submit and an incompatible CUDA stack still cannot train.
        require_image_tools(image)
        if phase.free_gib:
            require_free_space(args.out, phase.free_gib)
        # GPU visibility, occupancy and kernel execution are runtime safety/capability
        # gates, not official-hardware identity checks.  They remain fail-closed even
        # when local development relaxes the accelerator family/capacity contract.
        occupancy_before = gpu_occupancy_evidence(
            devices, mode=args.hardware_check, stage="before_kernel_probe"
        )
        # And before it, whether the image can compute at all. Three images passed
        # the identity/tool checks and then died on their first matmul, because their
        # torch had no code for this card. This probe remains mandatory even for a
        # locally rebuilt, identity-unverified image.
        require_runnable_kernels(
            image,
            devices[0],
            backend=environment.get("collective_backend", "nccl"),
        )
        occupancy_after = gpu_occupancy_evidence(
            devices, mode=args.hardware_check, stage="after_kernel_probe"
        )
        hardware: dict[str, object]
        if args.hardware_check == "off":
            hardware = {
                "mode": "off",
                "status": "not_checked",
                "allowed_types": environment.get("gpu_types", []),
                "required_peak_memory_mib": config.get("x-ai4ai", {})
                .get("gpu", {})
                .get("peak_memory_mib"),
                "required_free_memory_mib": None,
                "devices": [],
                "occupancy_checks": [occupancy_before, occupancy_after],
                "problems": [],
            }
        else:
            try:
                hardware = {
                    "mode": args.hardware_check,
                    **inspect_gpu_hardware(config, devices),
                }
            except ContainerError as error:
                if args.hardware_check == "strict":
                    raise
                hardware = {
                    "mode": args.hardware_check,
                    "status": "unavailable",
                    "allowed_types": environment.get("gpu_types", []),
                    "required_peak_memory_mib": config.get("x-ai4ai", {})
                    .get("gpu", {})
                    .get("peak_memory_mib"),
                    "required_free_memory_mib": None,
                    "devices": [],
                    "occupancy_checks": [occupancy_before, occupancy_after],
                    "problems": [str(error)],
                }
            occupancy_problems = [
                str(problem)
                for receipt in (occupancy_before, occupancy_after)
                for problem in receipt.get("problems", [])
            ]
            hardware["occupancy_checks"] = [occupancy_before, occupancy_after]
            if occupancy_problems:
                hardware["status"] = "mismatch"
                hardware["problems"] = [
                    *[str(problem) for problem in hardware.get("problems", [])],
                    *occupancy_problems,
                ]
            if hardware["status"] != "match":
                detail = "\n".join(str(value) for value in hardware.get("problems", []))
                if args.hardware_check == "strict":
                    raise ContainerError(f"official hardware verification failed:\n{detail}")
                print(
                    "warning: selected hardware does not satisfy the official task "
                    f"contract; continuing in non-official mode:\n{detail}",
                    file=sys.stderr,
                )
        run_config_path = args.run_config.resolve() if args.run_config else None
        run_config_sha256 = sha256_file(run_config_path) if run_config_path else None
        effective_phase = phase_payload(phase)
        atomic_json(
            args.out.parent / "preflight.json",
            {
                "schema_version": 3,
                "phase": args.phase,
                "verified_at": utc_now(),
                "image": image,
                "source": {
                    "mode": args.source_check,
                    "status": source_status,
                    "expected": environment.get("source_sha256"),
                    "observed": actual_source,
                },
                "host_contract": host_contract,
                "asset_identity": asset_identity,
                # Provenance only; official identity is the path-independent digest above.
                "assets_root": str(args.assets.resolve()),
                "run_config": {
                    "status": "present" if run_config_sha256 else "missing",
                    "file": run_config_path.name if run_config_path else None,
                    "sha256": run_config_sha256,
                },
                "effective_phase": {
                    **effective_phase,
                    "sha256": content_digest(effective_phase),
                },
                "image_identity": {
                    "mode": args.image_check,
                    "status": image_status,
                    "expected_layers": environment.get("digest"),
                    "observed_layers": identity["layer_fingerprint"],
                    "expected_config": environment.get("config_digest"),
                    "observed_config": identity["config_fingerprint"],
                    "image_id": identity["image_id"],
                    "repo_digests": identity["repo_digests"],
                    "platform": identity["platform"],
                },
                "hardware": hardware,
                "capability_checks": "passed",
            },
        )
        for name in phase.hooks:
            if name in BEFORE:
                BEFORE[name](context)
    elif "logs" in runtime:
        runtime["logs"].mkdir(parents=True, exist_ok=True)

    if args.agent:
        # The agent path starts the container in the background and execs into it,
        # so it cannot share the single run() call below.
        status = run_with_agent(context, parser=parser)
        if args.dry_run:
            return status
        for name in phase.hooks:
            if name in AFTER:
                AFTER[name](context)
        return status

    status = run(
        build(
            phase=phase,
            phase_name=args.phase,
            image=image,
            assets=args.assets,
            runtime=runtime,
            gpus=devices,
            resources=config,
            digest=str(context.get("observed_image_digest") or ""),
            config_digest=str(context.get("observed_image_config_digest") or ""),
        ),
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return 0

    if status == 124:
        print(f"{args.phase}: exceeded its {phase.timeout_sec}s budget")
    else:
        print(f"{args.phase}: exit {status}")
    for name in phase.hooks:
        if name in AFTER:
            AFTER[name](context)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
