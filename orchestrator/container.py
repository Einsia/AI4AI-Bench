"""Construct and supervise task containers.

A timeout must stop both the client process and the daemon-owned container. The
cidfile identifies the latter after the client stops waiting.
"""

from __future__ import annotations

import contextlib
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path

CIDFILE_MAX_BYTES = 66
SECRET_ENV_SUFFIXES = ("_API_KEY", "_TOKEN", "_PASSWORD", "_SECRET")

class ContainerError(RuntimeError):
    """A container could not be constructed, started, or stopped."""

# Shell-split so deployments may provide a runtime wrapper with arguments.
DEFAULT_RUNTIME = os.environ.get("AI4AI_DOCKER", "docker")

def runtime_argv(runtime: str) -> list[str]:
    """Split a runtime string into argv, so it may be a command and not just a name."""

    parts = shlex.split(runtime)
    if not parts:
        raise ContainerError("the container runtime resolved to an empty command")
    return parts


def environment_argv(environment: dict[str, str]) -> list[str]:
    """Return Docker environment flags without putting secrets in process argv."""

    argv: list[str] = []
    for key in sorted(environment):
        value = environment[key]
        argv.extend(("--env", key if key.endswith(SECRET_ENV_SUFFIXES) else f"{key}={value}"))
    return argv


def subprocess_environment(environment: dict[str, str]) -> dict[str, str]:
    """Expose secret values to Docker through its inherited environment only."""

    process_environment = os.environ.copy()
    process_environment.update(
        {key: value for key, value in environment.items() if key.endswith(SECRET_ENV_SUFFIXES)}
    )
    return process_environment



@dataclass(frozen=True)
class Phase:
    """One task-declared container phase, resolved by runner.py."""

    name: str
    timeout_sec: int
    command: tuple[str, ...]
    mounts: tuple[tuple[str, str, bool], ...] = ()
    exports: dict[str, str] = field(default_factory=dict)
    hooks: tuple[str, ...] = ()
    read_only_root: bool = False
    apply_patch: bool = False
    interactive: bool = False
    reserve_sec: int = 0
    free_gib: int = 0
    output_glob: str = ""
    # Formal retrains may publish a bounded series of independently scoreable
    # artifacts. v1.4 declarations use the defaults and therefore keep their
    # single-artifact behaviour; v1.5 tasks opt into the plural contract.
    artifact_limit: int = 1
    # auto preserves the v1.4 resolver. The explicit v1.5 kinds make the host's
    # completeness check match the object the frozen scorer receives.
    artifact_kind: str = "auto"
    # v1.5 candidates are produced by the recipe below /out/checkpoints.  The
    # harness, not candidate-controlled run.sh, validates and publishes them.
    # output_glob/artifact_kind remain above for v4/v5 declarations.
    checkpoint_glob: str = ""
    checkpoint_payload: str = "."
    checkpoint_kind: str = "auto"
    # Only the scoring phase records the image it ran under, in resolved_config.json.
    pass_image_digest: bool = False

    @property
    def exports_with_wall_clock(self) -> tuple[str, ...]:
        """The wall clock comes first, then the declared exports.

        MAX_WALL_TIME_SECONDS is the real brake: run.sh stops one step before it so
        the last checkpoint is whole. 600 s is taken off the container timeout so the
        trainer stops before docker does.

        TOTAL_TRAINING_STEPS stays under candidate control in run.sh; the host
        exports only the wall clock and fixed inputs.
        """

        if not self.reserve_sec:
            return tuple(f"{key}={value}" for key, value in self.exports.items())
        train_seconds = max(self.timeout_sec - 600, 600)
        return (
            f"MAX_WALL_TIME_SECONDS={train_seconds}",
            f"DEADLINE_RESERVE_SECONDS={self.reserve_sec}",
            *(f"{key}={value}" for key, value in self.exports.items()),
        )

    def with_timeout(self, seconds: int) -> Phase:
        return replace(self, timeout_sec=seconds)

    def with_command(self, command: tuple[str, ...]) -> Phase:
        return replace(self, command=command, apply_patch=False, exports={})

    def with_interactive(self) -> Phase:
        return replace(self, interactive=True)

    # The wall clock is not in `exports` -- exports_with_wall_clock prepends it -- so
    # refusing declared keys alone would still let an override through for it. In shell,
    # `export A=1 A=2` leaves A=2, so an appended duplicate wins.
    UNOVERRIDABLE = ("MAX_WALL_TIME_SECONDS", "DEADLINE_RESERVE_SECONDS")

    def with_extra_exports(self, extra: dict[str, str]) -> Phase:
        """Add undeclared operator exports without overriding fixed inputs or time."""

        forbidden = sorted(
            key for key in extra if key in self.exports or key in self.UNOVERRIDABLE
        )
        if forbidden:
            raise ContainerError(
                f"{self.name} declares {', '.join(forbidden)}; they are the fixed inputs "
                "and the wall clock, and are not overridable with --export"
            )
        merged = {**self.exports, **extra}
        if self.apply_patch or self.exports:
            return replace(self, exports=merged)
        return replace(self, exports=merged, apply_patch=False)


def checkpoint_validation_phase(
    score: Phase, task_id: str, *, timeout_sec: int = 3600
) -> Phase:
    """Use the frozen score image and mounts for a loadability-only gate."""

    return replace(
        score,
        name="checkpoint-validate",
        timeout_sec=timeout_sec,
        command=(
            "python3", "/opt/harness/validate_checkpoint.py",
            "--task", task_id,
            "--checkpoint", "/ckpt",
            "--assets", "/assets",
            "--output", "/out/validation.json",
        ),
        hooks=(),
        read_only_root=False,
        pass_image_digest=False,
    )

@dataclass(frozen=True)
class Mount:
    source: Path
    target: str
    read_only: bool = True

    def docker_value(self) -> str:
        parts = [
            "type=bind",
            f"source={self.source.resolve()}",
            f"target={self.target}",
        ]
        if self.read_only:
            parts.append("readonly")
        return ",".join(parts)

@dataclass(frozen=True)
class ContainerSpec:
    """One phase's container. Immutable, and printable without running anything."""

    name: str
    image: str
    command: tuple[str, ...]
    workdir: str = "/workspace"
    mounts: tuple[Mount, ...] = ()
    environment: dict[str, str] = field(default_factory=dict)
    gpu_devices: tuple[int, ...] = (0,)
    # No defaults for the four resource limits. They are declared in task.toml's
    # [environment] and read by resources_from(); a default here would be a second
    # source of truth that silently wins whenever a key is missing, which is how
    # the memory ceiling came to be 64 g in one place and 128 Gi in another.
    cpus: float = field(kw_only=True)
    memory: str = field(kw_only=True)
    shm_size: str = field(kw_only=True)
    pids_limit: int = field(kw_only=True)
    # /tmp is a tmpfs so a container cannot fill the host disk. 256 MiB was enough
    # when all device kernels were prebuilt. On Blackwell (sm103), flashinfer
    # JIT-compiles its prefill kernels into
    # /tmp/.cache/flashinfer and ninja died with "No space left on device".
    #
    # Fourth instance of one shape. The first three were Triton, fixed with
    # TRITON_CACHE_DIR in run.sh, fast_eval.sh and final_eval.py; flashinfer was missed
    # because it only compiles on an architecture the image was not built for, so an
    # earlier-architecture run could not expose it. Redirecting the cache is the
    # tidier fix, but those three files are in source_hash and this field is not -- so
    # this needs no rebuild and no new digest.
    tmpfs_tmp_size: str = "8g"
    timeout_seconds: int = 14400
    # Container A must not set this: the Agent writes /workspace in the image
    # layer, so the root filesystem has to be writable. The final container can.
    read_only_root: bool = False
    interactive: bool = False
    # None means --network=none, which is the default and the only setting under
    # which the container provably cannot fetch a different model. An agent needs
    # its API, so it gets a named --internal network with an allowlisting proxy on
    # the host gateway; see egress.py. Nothing else should set this.
    network: str | None = None

    def argv(
        self, *, cidfile: Path | None = None, runtime: str = DEFAULT_RUNTIME
    ) -> list[str]:
        argv = [
            *runtime_argv(runtime),
            "run",
            "--rm",
            "--pull=never",
            f"--network={self.network or 'none'}",
            "--init",
            "--name",
            self.name,
            "--pids-limit",
            str(self.pids_limit),
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--gpus",
            # The quotes are part of the value, not shell quoting. This daemon
            # rejects a bare `device=1,2,3` with "cannot set both Count and
            # DeviceIDs on device request"; the quoted form is accepted and gives
            # exactly those devices. A single device works either way, so quote
            # always rather than branching.
            '"device=' + ",".join(str(device) for device in self.gpu_devices) + '"',
            # Docker picks the physical devices above and then presents them to
            # CUDA as a compact container-local range. Say so explicitly, or Ray
            # can rediscover a host index such as 6 and hand an invalid ordinal
            # to a worker that can only address device 0.
            "--env",
            "CUDA_VISIBLE_DEVICES=" + ",".join(str(i) for i in range(len(self.gpu_devices))),
            "--cpus",
            format(self.cpus, ".15g"),
            "--memory",
            self.memory,
            "--shm-size",
            self.shm_size,
            "--tmpfs",
            f"/tmp:rw,nosuid,nodev,size={self.tmpfs_tmp_size},mode=1777",
            # Docker adds noexec to every --tmpfs, so a .so written under /tmp can be
            # compiled and never loaded: dlopen fails with "failed to map segment from
            # shared object". Any library that JIT-compiles a kernel has to put it on a
            # real mount, and /out is the writable bind every phase has.
            #
            # This is the fourth library to hit it -- Triton three times, now flashinfer
            # on sm103, where the image has no prebuilt kernels and so compiles at run
            # time. It belongs here rather than in each task's run.sh because the cause
            # is how this file mounts /tmp, not anything a task chose. flashinfer derives
            # its cache from FLASHINFER_WORKSPACE_BASE, defaulting to home() -- which is
            # /tmp here, since the container runs as a uid with no passwd entry.
            "--env",
            "FLASHINFER_WORKSPACE_BASE=/out",
            "--workdir",
            self.workdir,
        ]
        if self.read_only_root:
            argv.append("--read-only")
        if cidfile is not None:
            argv.extend(("--cidfile", str(cidfile)))
        if self.interactive:
            argv.extend(("-it",))
        for mount in self.mounts:
            argv.extend(("--mount", mount.docker_value()))
        argv.extend(environment_argv(self.environment))
        argv.append(self.image)
        argv.extend(self.command)
        return argv

def _read_container_id(cidfile: Path) -> str | None:
    """Read the id docker wrote, refusing anything that is not a plain file.

    A symlink or an oversized file here would turn the forced cleanup below into
    an arbitrary-target `docker rm -f`.
    """

    try:
        stat = cidfile.lstat()
    except FileNotFoundError:
        return None
    if not os.path.isfile(cidfile) or stat.st_nlink != 1 or stat.st_size > CIDFILE_MAX_BYTES:
        raise ContainerError(f"refusing to trust an unusual cidfile: {cidfile}")
    value = cidfile.read_text(encoding="utf-8").strip()
    if not value or not all(character in "0123456789abcdef" for character in value):
        return None
    return value

def _force_remove(cidfile: Path, runtime: str) -> None:
    container_id = _read_container_id(cidfile)
    if container_id is None:
        return
    subprocess.run(
        [*runtime_argv(runtime), "rm", "-f", container_id],
        check=False,
        capture_output=True,
        timeout=120,
    )

@contextlib.contextmanager
def background(
    spec: ContainerSpec, *, runtime: str = DEFAULT_RUNTIME, ready_timeout: float = 120.0
) -> Iterator[str]:
    """Start the container detached, yield its name, and remove it on the way out.

    For the agent phase, where PID 1 waits and the work arrives over `exec`. The
    The caller enforces the Agent's absolute deadline while driving ``docker
    exec``. The container is force-removed on any exit path, including a crash or
    Ctrl-C, so a GPU is never left held.
    """

    if shutil.which(runtime_argv(runtime)[0]) is None:
        raise ContainerError(f"{runtime} is not on PATH")
    argv = spec.argv(runtime=runtime)
    # Insert -d right after `run`.
    argv.insert(argv.index("run") + 1, "-d")
    start = subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
        env=subprocess_environment(spec.environment),
    )
    if start.returncode != 0:
        raise ContainerError(f"could not start {spec.name}: {start.stderr.strip()[:400]}")

    deadline = time.monotonic() + ready_timeout
    while time.monotonic() < deadline:
        state = subprocess.run(
            [*runtime_argv(runtime), "inspect", spec.name, "--format", "{{.State.Running}}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if state.stdout.strip() == "true":
            break
        time.sleep(1)
    else:
        subprocess.run(
            [*runtime_argv(runtime), "rm", "-f", spec.name], check=False, capture_output=True
        )
        raise ContainerError(f"{spec.name} did not reach running state in {ready_timeout:.0f}s")

    try:
        yield spec.name
    finally:
        subprocess.run(
            [*runtime_argv(runtime), "rm", "-f", spec.name],
            check=False,
            capture_output=True,
            timeout=180,
        )

def run(spec: ContainerSpec, *, runtime: str = DEFAULT_RUNTIME, dry_run: bool = False) -> int:
    """Start the container and enforce its wall clock. Returns the exit status.

    A timeout returns 124, matching coreutils `timeout`, so a caller can tell a
    budget overrun from a crash.
    """

    if dry_run:
        print(" ".join(spec.argv(runtime=runtime)))
        return 0
    if shutil.which(runtime_argv(runtime)[0]) is None:
        raise ContainerError(f"{runtime} is not on PATH")

    directory = Path(tempfile.mkdtemp(prefix="orchestrator-"))
    # docker refuses to start if the cidfile already exists, so name it without
    # creating it.
    cidfile = directory / "cid"
    try:
        result = subprocess.run(
            spec.argv(cidfile=cidfile, runtime=runtime),
            check=False,
            timeout=spec.timeout_seconds,
            env=subprocess_environment(spec.environment),
        )
        return result.returncode
    except subprocess.TimeoutExpired:
        print(f"{spec.name}: hit its {spec.timeout_seconds}s budget; removing the container")
        return 124
    finally:
        # Runs on the timeout path, on a crash, and on Ctrl-C. This is the half
        # that actually frees the GPU.
        _force_remove(cidfile, runtime)
        shutil.rmtree(directory, ignore_errors=True)
