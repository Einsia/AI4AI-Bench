"""The one rollout-and-scoring path, shared by fast_eval and the hidden final.

Both stages roll trajectories out through the frozen RAGEN tree baked into the
image at /opt/harness/ragen and score them from the rows it writes, so a fast_eval
number and a final number are produced by the same environment, the same reward and
the same aggregation. Only the environment seed and the board count differ.

Nothing here reads a recipe or a checkpoint's own metadata. Keep it that way: this
file is the reason "the environment does not change" can be checked by reading one
thing.

Three notes on what this file does differently from the evaluator it replaces.

**One frozen tree instead of two runtime copies.** The reference protocol mounted the RAGEN
source as an asset, copied it, and applied the same set of source transforms twice:
once as a patch file in the trainer, once as eight hardcoded string replacements in
the evaluator. Two spellings of one transform is the shape behind most of this
project's bugs, so the transform is applied once at build time and both copies are
in the image -- editable at /workspace/ragen, frozen here.

**The score is a mean over independent boards, so it gets a binomial standard
error.** The old evaluator reported `population_std` across bank means, n=4. Each
board is one Bernoulli draw with one trajectory, so the error on the mean is
sqrt(p(1-p)/n) over all boards -- about 0.021 at 256 boards and p=0.14, against
0.041 for a single 64-board bank. The bank spread is still reported, because a
large gap between the two says the banks differ in difficulty rather than that the
estimate is noisy.

**A same-seed repeat is not a variance estimate.** The old `public` profile ran the
same environment seed three times with the same engine seed, and its own summary
asserts the board hashes are identical. Its 0.0074 spread is residual engine
nondeterminism -- one board flipping out of 64 -- and it was reported next to a mean
as `population_std`, roughly 6x below the real error bar. Nothing here repeats a
seed; more boards is the only thing that buys resolution.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

# The frozen tree. Read-only in the image, and the score phase runs with
# --read-only on top of that.
FROZEN_RAGEN = Path("/opt/harness/ragen")
RAGEN_COMMIT = "20daedc47558e000f7de912b060646bf2e8026bd"
VERL_COMMIT = "d62da4950573d7a4b7ef2362337952e7ab59e78d"
# The success key RAGEN's Sokoban environment writes into each trajectory's
# metadata, and the aggregate it prints on its own line. Both are read, and checked
# against each other -- one is the environment's arithmetic and the other is ours.
SUCCESS_KEY = "CoordSokoban/success"
SUCCESS_RE = re.compile(
    rf"^{re.escape(SUCCESS_KEY)}:\s*([0-9]+(?:\.[0-9]+)?)\s*$", re.MULTILINE
)
# Sampling settings are part of the fixed evaluation protocol.
EVAL_DO_SAMPLE = True
EVAL_TEMPERATURE = 0.5
# The per-request seed scheme the build-time patch installs: engine_seed + env_id.
# It is what makes one board's trajectory reproducible independently of how many
# boards are in the batch.
ENGINE_SEED = 0

# The frozen tree's content hash, recorded at the first build. environment/
# check_image.py computes it and prints it; paste it here and the build then fails
# whenever the two disagree, which is what makes this a pin rather than a comment.
#
# Empty means "not recorded yet", and the checks below warn instead of refusing --
# the same choice orchestrator/task.py makes for an unrecorded image digest, so a
# task can be brought up before it can be attested.
#
# Recorded from the first build that ran check_image.py to completion, on the B300 host,
# against RAGEN@20daedc4 with its verl submodule at d62da495 and
# environment/ragen_runtime_compat.patch applied. Both are hashes of
# /opt/harness/ragen, so they are independent of this file and of the base image: the
# same two values appeared on the build before this one, which differed in grade.py and
# in the whole wheelhouse install. What changes them is the RAGEN tree or the patch.
FROZEN_RAGEN_SHA256 = "aab609b10f680aa6ad954828c44710bac456dcfdbe887c0970a75f214a2dd3a7"
# The environment subtree specifically, hashed separately so a reader can see the
# board generator and the solve reward pinned on their own rather than only inside
# the whole-tree number.
FROZEN_ENVIRONMENT_SHA256 = "3d00e7fd6f004861fa5b163250672df991f9a9da8b68056a75efd09468a688be"
# Modules whose import must resolve inside FROZEN_RAGEN. These three are certain:
# the first is the rollout entry point the evaluators invoke as `python3 -m`, and
# the other two are files the build-time compatibility patch edits, so a tree
# missing them is not the tree these numbers were measured on.
REQUIRED_FROZEN_MODULES = (
    "ragen",
    "ragen.llm_agent.agent_proxy",
    "ragen.llm_agent.ctx_manager",
)


def _tree_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )


def tree_sha256(root: Path) -> str:
    """Hash a directory's contents: relative path, then bytes, in sorted order.

    Refuses a symlink outright rather than following it. A symlink out of the frozen
    tree is the cheapest way to make a hash of "the frozen tree" describe something
    else, and it does not have to be malicious -- an rsync with the wrong flags
    produces the same thing.
    """

    if not root.is_dir():
        raise FileNotFoundError(f"frozen tree is missing: {root}")
    digest = hashlib.sha256()
    for path in _tree_files(root):
        if path.is_symlink():
            raise RuntimeError(f"symlink in the frozen tree: {path}")
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def environment_subtree(root: Path = FROZEN_RAGEN) -> Path:
    """Locate the board generator and reward code inside the frozen tree.

    Discovered rather than hardcoded. The tree arrives through a build context at a
    pinned commit, so its layout is fixed but this file should not encode a path it
    cannot verify from here; a wrong literal would turn into a check that passes by
    hashing nothing.
    """

    for relative in ("ragen/env", "ragen/envs", "ragen/environment"):
        candidate = root / relative
        if candidate.is_dir() and _tree_files(candidate):
            return candidate
    raise FileNotFoundError(
        f"no non-empty environment package under {root}; looked for ragen/env, "
        "ragen/envs, ragen/environment. The board generator and the solve reward "
        "live there, and a frozen tree without them is not the tree the recorded "
        "baselines were measured on."
    )


# Printed by the resolution probe below and parsed back. A prefix rather than plain
# JSON on stdout, because the child imports RAGEN and RAGEN's imports print.
RESOLUTION_MARKER = "ai4ai-frozen-resolution "

# The dict literal is written with dict() rather than {} because this string is
# formatted, and a brace here silently becomes a format field.
_RESOLUTION_PROBE = """
import importlib
import json
import sys

resolved = dict()
for name in sys.argv[1:]:
    module = importlib.import_module(name)
    resolved[name] = getattr(module, "__file__", None)
print({marker!r} + json.dumps(resolved, sort_keys=True))
"""


def resolve_frozen_modules(
    environment: dict[str, str], *, root: Path = FROZEN_RAGEN
) -> dict[str, str]:
    """Import the frozen modules the way the rollout will, and report where they land.

    Run as a subprocess with the rollout's own cwd and PYTHONPATH, because that is
    the resolution being checked: a hash of the files under /opt/harness proves those
    files are right and says nothing about whether the process that rolls
    trajectories out will import them. A stale install in site-packages, a .pth that
    inserts ahead of cwd, or a tree that landed in the wrong place would all leave
    the frozen files untouched and unused.

    One thing this established rather than assumed, because it is what makes the
    frozen tree win: cwd is sys.path[0] for both `python -c` and `python -m`, and the
    rollout's cwd IS the frozen root. So a copy on PYTHONPATH cannot outrank it --
    measured both ways in the same interpreter this probe uses. That is a property of
    how the rollout is invoked, not a guarantee, which is why it is checked here
    rather than trusted.

    Raises on anything that resolves outside `root`.
    """

    probe = _RESOLUTION_PROBE.format(marker=RESOLUTION_MARKER)
    result = subprocess.run(
        [sys.executable, "-c", probe, *REQUIRED_FROZEN_MODULES],
        cwd=str(root),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "could not import the frozen RAGEN modules:\n"
            + (result.stderr or result.stdout)[-2000:]
        )
    line = next(
        (
            item[len(RESOLUTION_MARKER) :]
            for item in reversed(result.stdout.splitlines())
            if item.startswith(RESOLUTION_MARKER)
        ),
        None,
    )
    if line is None:
        raise RuntimeError("the frozen-module probe printed no resolution line")
    resolved: dict[str, str] = json.loads(line)

    root_resolved = root.resolve()
    outside: dict[str, str | None] = {}
    for name in REQUIRED_FROZEN_MODULES:
        path = resolved.get(name)
        if path is None:
            outside[name] = None
            continue
        real = Path(path).resolve()
        if not real.is_relative_to(root_resolved):
            outside[name] = str(real)
    if outside:
        raise RuntimeError(
            f"the rollout would not import the frozen environment: {outside}. "
            f"Every module must resolve under {root_resolved}; the scoring phase "
            "runs the image's copy, not the candidate's."
        )
    return {name: str(Path(path).resolve()) for name, path in resolved.items() if path}


def frozen_environment_report(environment: dict[str, str]) -> dict[str, Any]:
    """Everything this harness can prove about the environment it is about to score.

    Computed from the image tree and from the interpreter that will run the rollout.
    Nothing here reads a file the candidate wrote. A mismatch against a recorded pin
    raises; an unrecorded pin warns, so the first build can record one.
    """

    tree_hash = tree_sha256(FROZEN_RAGEN)
    subtree = environment_subtree()
    subtree_hash = tree_sha256(subtree)
    resolved = resolve_frozen_modules(environment)

    for label, actual, pinned in (
        ("frozen RAGEN tree", tree_hash, FROZEN_RAGEN_SHA256),
        ("frozen environment subtree", subtree_hash, FROZEN_ENVIRONMENT_SHA256),
    ):
        if not pinned:
            print(
                f"warning: no pin recorded for the {label}; it hashes to "
                f"{actual}. Record it in harness/grade.py at the next build."
            )
        elif actual != pinned:
            raise RuntimeError(
                f"{label} does not match its pin\n  expected {pinned}\n"
                f"  actual   {actual}\n"
                "The environment transitions and solve reward are frozen; this run "
                "would measure a different task, so the trial is invalid."
            )
    return {
        "frozen_ragen_root": str(FROZEN_RAGEN),
        "frozen_ragen_sha256": tree_hash,
        "frozen_ragen_sha256_pinned": FROZEN_RAGEN_SHA256 or None,
        "frozen_environment_subtree": str(subtree),
        "frozen_environment_sha256": subtree_hash,
        "frozen_environment_sha256_pinned": FROZEN_ENVIRONMENT_SHA256 or None,
        "frozen_module_paths": resolved,
        "ragen_commit": RAGEN_COMMIT,
        "verl_commit": VERL_COMMIT,
    }


def rollout_environment(output: Path, *, root: Path = FROZEN_RAGEN) -> dict[str, str]:
    """Environment for a rollout out of the frozen tree, with every cache under /out.

    /tmp is a 256 MiB tmpfs and the score phase adds --read-only on top, so a
    compiler cache left at its default has two ways to fail: it fills the tmpfs, or
    it cannot be created at all. Triton's failure mode is the confusing one -- it
    writes a cuda_utils .so and then mmaps it, and a too-small tmpfs surfaces as
    "failed to map segment from shared object", which reads like a packaging problem
    rather than a full filesystem.
    """

    environment = os.environ.copy()
    # Never attach to a Ray service inherited from the parent shell or a previous
    # bank. Each rollout starts and tears down its own local service state.
    for name in ("RAY_ADDRESS", "RAY_NAMESPACE", "RAY_JOB_ID", "RAY_RUNTIME_ENV_HOOK"):
        environment.pop(name, None)
    pythonpath = [str(root), str(root / "verl")]
    if environment.get("PYTHONPATH"):
        pythonpath.append(environment["PYTHONPATH"])
    cache = output / "cache"
    for path in (
        cache / "huggingface",
        cache / "torch",
        cache / "triton",
        cache / "inductor",
        cache / "vllm",
        cache / "ray",
        cache / "tmp",
    ):
        path.mkdir(parents=True, exist_ok=True)
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(pythonpath),
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "disabled",
            "HF_HOME": str(cache / "huggingface"),
            "TORCH_HOME": str(cache / "torch"),
            "TRITON_CACHE_DIR": str(cache / "triton"),
            "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
            "VLLM_CACHE_ROOT": str(cache / "vllm"),
            "RAY_TMPDIR": str(cache / "ray"),
            "XDG_CACHE_HOME": str(cache),
            "TMPDIR": str(cache / "tmp"),
            "HYDRA_FULL_ERROR": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    return environment


def run_process(
    command: list[str],
    cwd: Path,
    log_path: Path,
    environment: dict[str, str],
    *,
    finished: Callable[[], bool] | None = None,
    quiet_seconds: float = 30.0,
    poll_seconds: float = 5.0,
) -> int:
    """Run a child in its own process group; return its status rather than raising.

    Group forwarding: the rollout starts a vLLM engine with worker processes, and
    without it a wall-clock stop kills the parent and leaves the engine holding the
    device -- which the container teardown then has to clean up, after the
    orchestrator has already moved on.

    `finished` is what makes this survive the rollout's exit. Measured on the B300
    host, `python -m ragen.llm_agent.agent_proxy --config-name eval` completes every
    board, prints its aggregate, writes trajectories.jsonl -- and then does not exit:

      VLLM_ENABLE_V1_MULTIPROCESSING=1 (the default): the parent sits in do_wait and
        the VLLM::EngineCore child sits in futex_wait_queue. Neither moves; CPU time
        stops advancing. The engine keeps the whole device.
      VLLM_ENABLE_V1_MULTIPROCESSING=0: the same work finishes and the process then
        dies with SIGSEGV at interpreter exit.

    Neither is recoverable from outside and neither can be configured away: agent_proxy
    builds its LLM and never shuts it down, and main() overwrites
    VLLM_WORKER_MULTIPROC_METHOD itself, so the environment cannot redirect it. That
    file is in the frozen tree, which this harness exists to keep unmodified.

    So the read loop cannot be `for line in process.stdout` -- the EngineCore inherits
    stdout, so the pipe stays open even when the rollout is gone. A reader thread pumps
    the log instead, and the wait ends when either the child exits or `finished()` says
    the artifacts are all present after `quiet_seconds` without a line.

    What keeps that from turning a failure into a pass: `finished` is not "it looked
    done". rollout() passes a predicate that requires the full row count, every row
    parseable, and the aggregate line printed -- the same three conditions it then
    checks again itself. A rollout that died early satisfies none of them, and this
    waits for it, hits the phase timeout, and fails.
    """

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        previous_term = signal.getsignal(signal.SIGTERM)

        def forward_term(_signum: int, _frame: object) -> None:
            os.killpg(process.pid, signal.SIGTERM)

        signal.signal(signal.SIGTERM, forward_term)
        assert process.stdout is not None
        # Flushed per line, unlike the version this replaces. That one buffered, so a
        # rollout killed at the wall left an empty rollout.log -- and the log is the
        # only place the aggregate is printed, so losing it loses the run.
        last_line_at = [time.monotonic()]

        def pump() -> None:
            for line in process.stdout:  # type: ignore[union-attr]
                sys.stdout.write(line)
                log.write(line)
                log.flush()
                last_line_at[0] = time.monotonic()

        reader = threading.Thread(target=pump, name="rollout-log", daemon=True)
        reader.start()
        try:
            while True:
                status = process.poll()
                if status is not None:
                    reader.join(timeout=30.0)
                    break
                quiet_for = time.monotonic() - last_line_at[0]
                if finished is not None and quiet_for >= quiet_seconds and finished():
                    print(
                        f"grade: the rollout produced every artifact and then stopped "
                        f"exiting ({quiet_for:.0f}s without output). Taking the device "
                        f"back; see the note in run_process.",
                        flush=True,
                    )
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    reader.join(timeout=30.0)
                    status = 0
                    break
                time.sleep(poll_seconds)
        finally:
            if process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            reader.join(timeout=30.0)
            signal.signal(signal.SIGTERM, previous_term)
    return status


def stop_local_services(environment: dict[str, str], log_path: Path) -> None:
    """Best-effort cleanup for Ray children that escaped the rollout process group.

    The containing train/eval phase owns the GPU lock, so a container-wide `ray
    stop` cannot interrupt another legitimate phase. Its stdout is retained beside
    the rollout instead of discarded.
    """

    ray = shutil.which("ray", path=environment.get("PATH"))
    if ray is None:
        log_path.write_text(
            "ray executable not found; process-group cleanup only\n", encoding="utf-8"
        )
        return
    try:
        result = subprocess.run(
            [ray, "stop", "--force"],
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=45.0,
        )
        text = (
            f"command: {ray} stop --force\nstatus: {result.returncode}\n"
            + result.stdout
            + result.stderr
        )
    except subprocess.TimeoutExpired as error:
        text = f"ray stop --force timed out after {error.timeout}s\n"
    log_path.write_text(text, encoding="utf-8")


def validation_overrides(boards: int, environment_seed: int, engine_seed: int) -> list[str]:
    """The Hydra overrides that pin one bank of boards.

    `seed.val` is what the board generator consumes, so it selects which boards
    exist. `sampling_seed` rides into vLLM twice through the build-time patch: once
    at engine construction and once per request as `sampling_seed + env_id`, which is
    what makes a board's trajectory independent of the batch it was submitted in.
    group_size=1 is one trajectory per board -- more trajectories per board re-measure
    something close to deterministic, and boards are free.
    """

    if boards < 1:
        raise ValueError("board count must be positive")
    return [
        f"seed.val={environment_seed}",
        f"+sampling_seed={engine_seed}",
        f"es_manager.val.env_groups={boards}",
        "es_manager.val.group_size=1",
        f"es_manager.val.env_configs.n_groups=[{boards}]",
    ]


def rollout(
    *,
    checkpoint: Path,
    boards: int,
    environment_seed: int,
    engine_seed: int,
    output: Path,
    environment: dict[str, str],
    root: Path = FROZEN_RAGEN,
) -> tuple[list[dict[str, Any]], float, float]:
    """Roll `boards` trajectories out of the frozen tree. Returns rows, its own
    aggregate, and wall seconds."""

    output.mkdir(parents=True, exist_ok=True)
    trajectories = output / "trajectories.jsonl"
    log_path = output / "rollout.log"
    existing = [path for path in (trajectories, log_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to mix a rollout with existing artifacts: "
            + ", ".join(str(path) for path in existing)
        )

    rollout_env = environment.copy()
    ray_tmp = output / "ray"
    ray_tmp.mkdir(parents=True, exist_ok=True)
    rollout_env["RAY_TMPDIR"] = str(ray_tmp)
    for name in ("RAY_ADDRESS", "RAY_NAMESPACE", "RAY_JOB_ID", "RAY_RUNTIME_ENV_HOOK"):
        rollout_env.pop(name, None)
    command = [
        sys.executable,
        "-m",
        "ragen.llm_agent.agent_proxy",
        "--config-name",
        "eval",
        f"system.CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '0')}",
        f"model_path={checkpoint}",
        *validation_overrides(boards, environment_seed, engine_seed),
        f"actor_rollout_ref.rollout.val_kwargs.do_sample={str(EVAL_DO_SAMPLE).lower()}",
        f"actor_rollout_ref.rollout.val_kwargs.temperature={EVAL_TEMPERATURE}",
        f"output.dir={output}",
        f"output.filename={trajectories.name}",
        "output.format=jsonl",
        "output.append_timestamp=false",
        # The old evaluator left this at Hydra's default, which is a directory under
        # the process cwd. That worked because it copied the tree somewhere
        # writable; the frozen tree is read-only, so an unset run dir turns into a
        # permission error before the first board.
        f"hydra.run.dir={output / 'hydra'}",
    ]
    def reported_values() -> list[float]:
        """The aggregates the rollout printed, if the log exists yet.

        Anchored, and kept anchored deliberately: the value must be the whole of the
        line. A looser match would also read the same key out of a per-step training
        metrics line, and take the wrong number without failing.
        """

        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        return [float(match.group(1)) for match in SUCCESS_RE.finditer(text)]

    def finished() -> bool:
        """Every artifact present: the aggregate printed, and one parseable row per
        board. This is what lets run_process stop waiting on a rollout that has done
        its work and will not exit -- so it has to be the strict version of the
        question, not a heuristic. It is the same check performed again below."""

        if not reported_values():
            return False
        try:
            read_complete_jsonl(trajectories, expected_rows=boards, attempts=1)
        except (RuntimeError, OSError, json.JSONDecodeError):
            return False
        return True

    started = time.monotonic()
    try:
        status = run_process(command, root, log_path, rollout_env, finished=finished)
    finally:
        stop_local_services(rollout_env, output / "service-cleanup.log")
    elapsed = time.monotonic() - started

    reported = reported_values()
    if not reported:
        raise RuntimeError(
            f"the rollout printed no {SUCCESS_KEY} aggregate (child status {status}); "
            f"see {log_path}"
        )
    # Validated before the status is judged, and that order is the point. The child
    # cannot exit cleanly on this platform -- see run_process -- so the status alone
    # would fail every run. A complete, parseable set of rows plus the printed
    # aggregate is the evidence that the rollout happened; a bad status with the
    # artifacts intact is a teardown fault, and a bad status without them fails here.
    try:
        rows = read_complete_jsonl(trajectories, expected_rows=boards)
    except Exception as error:
        raise RuntimeError(
            f"the rollout did not leave {boards} complete trajectories "
            f"(child status {status}): {error}"
        ) from error
    if status:
        print(
            f"grade: the rollout exited with status {status} after writing all "
            f"{boards} trajectories and its aggregate; scoring the artifacts.",
            flush=True,
        )
    return rows, reported[-1], elapsed


def read_complete_jsonl(
    path: Path, *, expected_rows: int, attempts: int = 20, delay_seconds: float = 0.1
) -> list[dict[str, Any]]:
    """Read a JSONL only once every row is present and parseable.

    Kept from the old evaluator. On an NFS-backed output directory the file becomes
    visible before it is coherent, and the failure is a JSONDecodeError on the last
    line -- which looks like a corrupt run rather than a slow filesystem.
    """

    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            lines = [
                line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
            ]
            if len(lines) != expected_rows:
                raise RuntimeError(f"expected {expected_rows} rows, found {len(lines)}")
            return [json.loads(line) for line in lines]
        except (json.JSONDecodeError, OSError, RuntimeError) as error:
            last_error = error
            time.sleep(delay_seconds)
    raise RuntimeError(f"{path} did not become coherent after {attempts} attempts") from last_error


def initial_board(row: dict[str, Any]) -> str | None:
    """The first user message, which contains the rendered starting board.

    This is the board's identity for uniqueness purposes. It is the observation the
    policy actually saw, so two rows with the same first message are the same problem
    however their ids differ.
    """

    messages = row.get("messages") or []
    if not isinstance(messages, list) or not messages:
        return None
    first = messages[0]
    return first.get("content") if isinstance(first, dict) else None


def trajectory_success(row: dict[str, Any]) -> float:
    value = (row.get("metadata") or {}).get(SUCCESS_KEY)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"trajectory has no numeric {SUCCESS_KEY}: {row.get('custom_id')!r}")
    converted = float(value)
    if not math.isfinite(converted) or converted not in {0.0, 1.0}:
        raise RuntimeError(f"Sokoban success must be 0 or 1, got {converted}")
    return converted


def select_unique_boards(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """First `count` rows with distinct starting boards, in generation order.

    The generator can repeat a board, so a run asking for N boards is not guaranteed
    N distinct ones -- which is why the final over-generates and takes a prefix. Order
    is the stream's order, so the selection is a function of the seed alone and does
    not depend on which boards the policy happened to solve.
    """

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        board = initial_board(row)
        if board is None or board in seen:
            continue
        seen.add(board)
        selected.append(row)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(
            f"expected {count} distinct starting boards, the stream yielded "
            f"{len(selected)} from {len(rows)} trajectories"
        )
    return selected


def check_rows_unique(rows: list[dict[str, Any]], *, expected_rows: int) -> None:
    """Refuse a row set with a duplicate request key or the wrong count.

    A rollout writes one file and a merge can write a shard twice; the row count
    stays right while the content is wrong, so counting alone does not catch it.
    """

    if len(rows) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} trajectories, got {len(rows)}")
    keys = [row.get("custom_id") for row in rows]
    if len(set(keys)) != expected_rows:
        raise RuntimeError(
            f"{expected_rows - len(set(keys))} duplicate request key(s) among "
            f"{expected_rows} trajectories; rows were merged twice or a seed repeated"
        )


def board_bank_sha256(rows: list[dict[str, Any]]) -> str:
    """Hash the starting boards, in order. Identifies which problems were scored."""

    boards = [initial_board(row) for row in rows]
    if any(board is None for board in boards):
        raise RuntimeError("a scored trajectory has no starting board")
    return hashlib.sha256("\0".join(boards).encode()).hexdigest()  # type: ignore[arg-type]


def summarize(banks: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one or more banks of scored boards into the reported number.

    `score` is the mean over every board, not the mean of the bank means. They agree
    while banks are equal-sized and would silently diverge if one were ever short.

    `stderr` treats each board as one Bernoulli draw, which is what it is: one
    trajectory per board, boards independent by construction. That is the error bar
    to compare two candidates with. `bank_spread` is the same information at n=banks
    and is reported for a different question -- a spread much larger than stderr says
    the banks differ in difficulty, so a candidate scored on one bank is not
    comparable to a candidate scored on another.
    """

    if not banks:
        raise ValueError("no banks to summarize")
    successes = sum(int(bank["successes"]) for bank in banks)
    boards = sum(int(bank["boards"]) for bank in banks)
    if boards <= 0:
        raise ValueError("no boards were scored")
    score = successes / boards
    stderr = math.sqrt(max(score * (1.0 - score), 0.0) / boards)
    bank_scores = [float(bank["score"]) for bank in banks]
    spread = statistics.pstdev(bank_scores) if len(bank_scores) > 1 else 0.0
    hashes = [bank["initial_board_sha256"] for bank in banks]
    if len(set(hashes)) != len(hashes):
        raise RuntimeError(
            "two banks scored the same boards. Banks exist to add boards; a repeated "
            "bank measures engine nondeterminism and reports it as resolution."
        )
    return {
        "score": score,
        "stderr": stderr,
        "successes": successes,
        "boards": boards,
        "step_size": 1.0 / boards,
        "bank_count": len(banks),
        "bank_scores": bank_scores,
        "bank_spread": spread,
        "initial_board_sha256s": hashes,
        "initial_board_bank_sha256": hashlib.sha256("\0".join(hashes).encode()).hexdigest(),
    }


def score_bank(
    rows: list[dict[str, Any]],
    *,
    expected_rows: int,
    scored_boards: int,
    environment_seed: int,
    engine_seed: int,
    reported: float | None = None,
    wall_seconds: float = 0.0,
) -> dict[str, Any]:
    """Turn one rollout's rows into a bank result, checking it against RAGEN's own
    aggregate when there is one.

    The cross-check is worth keeping: it caught nothing on the reference protocol, and it is
    the only thing standing between "the rows say 0.14" and "the environment said
    0.14". It only applies when every generated row is scored -- the final scores a
    prefix, so its own mean is over a different set and the comparison is skipped.
    """

    check_rows_unique(rows, expected_rows=expected_rows)
    selected = select_unique_boards(rows, scored_boards)
    successes = sum(int(trajectory_success(row)) for row in selected)
    score = successes / len(selected)
    if reported is not None and len(selected) == len(rows):
        if not math.isclose(score, reported, abs_tol=1e-12):
            raise RuntimeError(
                f"row-derived score {score} disagrees with the environment's own "
                f"aggregate {reported}. The rows and the reward are out of step."
            )
    return {
        "score": score,
        "successes": successes,
        "boards": len(selected),
        "generated": len(rows),
        "unique_request_keys": len({row.get("custom_id") for row in rows}),
        "initial_board_sha256": board_bank_sha256(selected),
        "environment_seed": environment_seed,
        "engine_seed": engine_seed,
        "wall_seconds": wall_seconds,
    }


def synthetic_bank(
    *, environment_seed: int, boards: int, generated: int | None = None
) -> list[dict[str, Any]]:
    """Rows shaped like a real rollout's, for --mock and --smoke.

    Deterministic in the seed so a mock run is reproducible, and every board string
    distinct so the uniqueness checks are exercised rather than bypassed.
    """

    total = generated if generated is not None else boards
    return [
        {
            "custom_id": f"mock-{environment_seed}-{index}",
            "messages": [
                {"role": "user", "content": f"mock-board-{environment_seed}-{index % boards}"}
            ],
            "metadata": {SUCCESS_KEY: float((index * 7 + environment_seed) % 8 == 0)},
        }
        for index in range(total)
    ]


def smoke() -> None:
    overrides = validation_overrides(64, 4242, 0)
    assert overrides == [
        "seed.val=4242",
        "+sampling_seed=0",
        "es_manager.val.env_groups=64",
        "es_manager.val.group_size=1",
        "es_manager.val.env_configs.n_groups=[64]",
    ], overrides

    banks = []
    for seed in (4242, 4243):
        rows = synthetic_bank(environment_seed=seed, boards=64)
        banks.append(
            score_bank(
                rows,
                expected_rows=64,
                scored_boards=64,
                environment_seed=seed,
                engine_seed=ENGINE_SEED,
                reported=sum(trajectory_success(row) for row in rows) / 64,
            )
        )
    summary = summarize(banks)
    assert summary["boards"] == 128, summary
    assert summary["bank_count"] == 2, summary
    # 128 independent draws, so the error bar is the binomial one over all of them
    # and not the two-point spread between the banks.
    assert summary["stderr"] > 0.0, summary
    assert abs(summary["score"] - summary["successes"] / 128) < 1e-12, summary

    # A repeated bank is refused rather than averaged: identical boards make the
    # spread a determinism probe, which the old `public` profile reported as though
    # it were an error bar.
    repeated = [banks[0], dict(banks[0])]
    try:
        summarize(repeated)
    except RuntimeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("summarize accepted two banks with identical boards")

    # Over-generation: 96 rows carrying 64 distinct boards must score exactly the
    # 64, which is the shape the final runs in (640 generated, first 512 unique scored).
    #
    # The seed here is 4244, one of the proxy's four banks, and NOT the final's. It used
    # to be the final's actual environment seed, which put that integer in
    # /opt/harness/grade.py inside the image, one grep away from any Agent, next to a
    # comment naming it as the final's. Boards are computed from the seed rather than
    # mounted, so the seed is the entire withheld quantity for this task -- there is no
    # second thing an Agent would still be missing. environment/check_image.py refuses a
    # build whose /workspace or /opt/harness contains it, which is what caught this.
    # Nothing in this assertion depends on which seed it is: synthetic_bank only
    # interpolates the integer into mock board strings.
    rows = synthetic_bank(environment_seed=4244, boards=64, generated=96)
    bank = score_bank(
        rows, expected_rows=96, scored_boards=64, environment_seed=4244, engine_seed=0
    )
    assert bank["boards"] == 64 and bank["generated"] == 96, bank
    print(json.dumps({"grade_smoke": "passed", "boards": summary["boards"]}, sort_keys=True))


if __name__ == "__main__":
    smoke()
