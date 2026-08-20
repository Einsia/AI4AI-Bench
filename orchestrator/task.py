"""Resolve task declarations and reject inconsistent runtime state."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from container import DEFAULT_RUNTIME, ContainerError, runtime_argv

REQUIRED_RESOURCE_KEYS = ("cpus", "memory_mb")


def resources_from(config: dict) -> dict:
    """Turn declared resources into container arguments, with no hidden defaults."""

    environment = config.get("environment", config)
    # Harbor names cpus, memory_mb, storage_mb, gpus and the network fields. It has
    # no shm_size and no pids_limit, so those are marked as ours rather than passed
    # off as spec -- and they sit in a top-level x-ai4ai table, which is why this
    # takes the whole config rather than just [environment].
    container = config.get("x-ai4ai", {}).get("container", {})
    missing = [key for key in REQUIRED_RESOURCE_KEYS if key not in environment]
    missing += [key for key in ("shm_size", "pids_limit") if key not in container]
    if missing:
        raise ContainerError(
            f"task.toml is missing {missing}. These set the container limits and "
            "have no default: cpus and memory_mb go in [environment], shm_size and "
            "pids_limit in [x-ai4ai.container]."
        )
    return {
        "cpus": float(environment["cpus"]),
        # Harbor states memory in MB; docker wants a suffixed string.
        "memory": f"{int(environment['memory_mb'])}m",
        "shm_size": str(container["shm_size"]),
        "pids_limit": int(container["pids_limit"]),
    }


def load_task(task_dir: Path) -> dict:
    """Read the task declaration used for budgets, resources and image identity."""

    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10 has no tomllib
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError as error:
            raise ContainerError(
                f"reading task.toml needs Python 3.11+, or `pip install tomli` on "
                f"{sys.version_info.major}.{sys.version_info.minor}"
            ) from error

    path = task_dir / "task.toml"
    if not path.is_file():
        raise ContainerError(f"no task.toml at {path}")
    with path.open("rb") as handle:
        return tomllib.load(handle)


def source_hash(task_dir: Path) -> str:
    """Hash the files that go into the image, so a stale image is detectable.

    Pinned submodules arrive through named build contexts. Asset declarations,
    wheelhouse resolvers, derived-asset tools and separate base-image recipes stay
    outside the task image. They do not belong in its source identity.
    """

    # Files under environment/ that task Dockerfiles do not copy. Keep this explicit
    # so a new maintenance input requires a deliberate ownership decision.
    NOT_IN_IMAGE = {
        "assets.lock.yaml",
        "build_proxy_asset.py",
        "Dockerfile.base",
        # A scoring-only companion image and its inputs. It builds FROM the task
        # image rather than into it, so editing its recipe must not mark the task
        # image stale.
        "Dockerfile.eval",
        "eval-requirements.in",
        "eval-requirements.lock",
        "README-eval-image.md",
        "resolve_wheelhouse.sh",
        "runtime-requirements.in",
    }

    parts: list[str] = []
    for relative in ("solution", "harness", "environment"):
        root = task_dir / relative
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if "verl" in path.relative_to(task_dir).parts:
                continue
            if path.suffix in {".pyc"} or "__pycache__" in path.parts:
                continue
            if path.name in NOT_IN_IMAGE:
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            parts.append(f"{path.relative_to(task_dir)}:{digest}")
    repo_root = task_dir.parents[1]
    shared_inputs = [
        Path(".dockerignore"),
        Path("LICENSE"),
        Path("THIRD_PARTY_NOTICES.md"),
        Path("orchestrator/lifecycle.py"),
    ]
    harness_root = repo_root / "orchestrator/container_harness"
    shared_inputs.extend(
        path.relative_to(repo_root)
        for path in sorted(harness_root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )
    for relative in shared_inputs:
        path = repo_root / relative
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            parts.append(f"repo:{relative}:{digest}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def require_fresh_image(task_dir: Path, environment: dict) -> str:
    """Refuse to run an image built from different source than what is on disk."""

    recorded = str(environment.get("source_sha256", ""))
    actual = source_hash(task_dir)
    if not recorded:
        raise ContainerError(
            f"task.toml records no source_sha256 for {environment.get('image')}; "
            f"the current source is {actual}. Strict source verification needs a "
            "release receipt."
        )
    if recorded != actual:
        raise ContainerError(
            "the image was built from different source than the working tree\n"
            f"  task.toml source_sha256  {recorded[:16]}\n"
            f"  working tree             {actual[:16]}\n"
            "Rebuild, then record the new digest and source_sha256 in task.toml.\n"
            "Use --source-check warn (the public default) or off for a local run; "
            "official replay uses --source-check strict."
        )
    return actual


def image_layers(image: str, *, runtime: str = DEFAULT_RUNTIME) -> list[str]:
    """The image's filesystem layer digests, which identify its content.

    Separate from the image ID on purpose. The ID is the sha256 of the config
    blob, and a registry round trip rewrites that blob -- media types and layer
    compression differ between what buildx wrote and what the registry serves --
    so `docker push` followed by `docker pull` yields the same bytes under a
    different ID. A measured transport changed the config ID while leaving every
    filesystem layer digest identical.
    """

    if shutil.which(runtime_argv(runtime)[0]) is None:
        raise ContainerError(
            f"{runtime_argv(runtime)[0]} is not on PATH, so the image cannot be checked"
        )
    result = subprocess.run(
        [
            *runtime_argv(runtime),
            "image",
            "inspect",
            image,
            "--format",
            "{{range .RootFS.Layers}}{{.}} {{end}}",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise ContainerError(f"{image} is not present locally: {result.stderr.strip()}")
    return result.stdout.split()


CONFIG_IDENTITY_FIELDS = (
    "User",
    "Env",
    "Entrypoint",
    "Cmd",
    "WorkingDir",
    "Shell",
    "Healthcheck",
    "StopSignal",
    "ExposedPorts",
    "Volumes",
    "Labels",
    "OnBuild",
)


def image_identity(image: str, *, runtime: str = DEFAULT_RUNTIME) -> dict[str, object]:
    """Return portable filesystem and execution-config identities for an image.

    RootFS diff IDs survive registry compression, but they do not cover ENV, CMD,
    ENTRYPOINT, USER, labels, or the working directory.  Official verification records
    both identities so a config-only retag cannot pass as the released runtime.
    """

    command = runtime_argv(runtime)
    if shutil.which(command[0]) is None:
        raise ContainerError(f"{command[0]} is not on PATH, so the image cannot be checked")
    result = subprocess.run(
        [*command, "image", "inspect", image],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise ContainerError(f"{image} is not present locally: {result.stderr.strip()}")
    try:
        metadata = json.loads(result.stdout)[0]
        layers = metadata["RootFS"]["Layers"]
    except (IndexError, KeyError, TypeError, ValueError) as error:
        raise ContainerError(f"docker returned invalid image metadata for {image}") from error
    if not isinstance(layers, list) or not all(isinstance(value, str) for value in layers):
        raise ContainerError(f"docker returned invalid RootFS layers for {image}")
    layer_fingerprint = "layers:" + hashlib.sha256("\n".join(layers).encode()).hexdigest()
    raw_config = metadata.get("Config") or {}
    canonical_config = {
        "os": metadata.get("Os"),
        "architecture": metadata.get("Architecture"),
        "variant": metadata.get("Variant"),
        "config": {field: raw_config.get(field) for field in CONFIG_IDENTITY_FIELDS},
    }
    config_fingerprint = "config:" + hashlib.sha256(
        json.dumps(canonical_config, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    repo_digests = [
        value.rsplit("@", 1)[-1]
        for value in metadata.get("RepoDigests") or []
        if isinstance(value, str) and "@" in value
    ]
    return {
        "image": image,
        "image_id": metadata.get("Id"),
        "repo_digests": repo_digests,
        "layer_fingerprint": layer_fingerprint,
        "config_fingerprint": config_fingerprint,
        "platform": f"{metadata.get('Os')}/{metadata.get('Architecture')}",
    }


def ensure_image_available(
    image: str, *, pull_policy: str = "missing", runtime: str = DEFAULT_RUNTIME
) -> None:
    """Make a declared image available without allowing it to drift mid-run.

    Public users normally start from an empty Docker cache.  The phase containers
    themselves still use ``--pull=never``; this one preflight is the only place where
    the tag may be resolved from a registry.  Once present, the digest gate below
    decides whether the resolved bytes are the declared release artifact.
    """

    if pull_policy not in {"missing", "never"}:
        raise ContainerError(
            f"unknown image pull policy {pull_policy!r}; expected 'missing' or 'never'"
        )
    command = runtime_argv(runtime)
    if shutil.which(command[0]) is None:
        raise ContainerError(f"{command[0]} is not on PATH, so the image cannot be used")
    present = subprocess.run(
        [*command, "image", "inspect", image],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if present.returncode == 0:
        return
    detail = (present.stderr or present.stdout).strip() or "image inspect failed"
    if pull_policy == "never":
        raise ContainerError(
            f"{image} is not present locally and image pulling is disabled: {detail}\n"
            "Pull it explicitly, or use --image-pull-policy missing."
        )
    pulled = subprocess.run(
        [*command, "pull", image],
        check=False,
        capture_output=True,
        text=True,
        timeout=7200,
    )
    if pulled.returncode != 0:
        pull_detail = (pulled.stderr or pulled.stdout).strip() or "image pull failed"
        raise ContainerError(f"could not pull {image}: {pull_detail}")


def require_image_digest(
    image: str,
    expected: str,
    expected_config: str = "",
    *,
    runtime: str = DEFAULT_RUNTIME,
    identity: dict[str, object] | None = None,
) -> dict[str, object]:
    """Check the image about to run is the one that was attested.

    `expected` may be either an image ID/registry digest or a portable
    `layers:<sha256>` fingerprint.  A layer fingerprint must be paired with the
    canonical `config:<sha256>` identity because filesystem layers alone do not cover
    execution settings. Missing identities are failures here; warning-vs-strict policy
    is handled by the caller.
    """

    observed = identity or image_identity(image, runtime=runtime)
    fingerprint = str(observed["layer_fingerprint"])
    actual = str(observed.get("image_id") or "unknown")
    candidates = {actual} | {str(value) for value in observed.get("repo_digests", [])}
    if not expected:
        raise ContainerError(
            f"task.toml records no image digest for {image}; strict verification "
            "requires a published image receipt"
        )
    if expected.startswith("layers:"):
        if fingerprint != expected:
            raise ContainerError(
                f"image content mismatch for {image}\n  expected {expected}\n"
                f"  actual   {fingerprint}\n"
                "The layers differ, so this is a different build rather than a transported copy."
            )
    elif expected not in candidates:
        listed = "\n  ".join(sorted(candidates))
        raise ContainerError(
            f"image digest mismatch for {image}\n  expected {expected}\n  found\n  {listed}\n"
            f"  content  {fingerprint}\n"
            "If the image was moved between hosts, a push/pull rewrites the ID while the layers\n"
            "stay identical -- record the content fingerprint above instead. Otherwise rebuild\n"
            "from environment/Dockerfile, or update task.toml if the change is intended."
        )
    if not expected_config:
        raise ContainerError(
            f"task.toml records no config_digest for {image}; filesystem layers alone "
            "do not cover ENV, ENTRYPOINT, CMD, USER, or WorkingDir"
        )
    observed_config = str(observed["config_fingerprint"])
    if observed_config != expected_config:
        raise ContainerError(
            f"image execution config mismatch for {image}\n"
            f"  expected {expected_config}\n  actual   {observed_config}"
        )
    return observed


def require_free_space(path: Path, gib: int) -> None:
    """Refuse to start when the output volume cannot meet the declared reserve."""

    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    free_gib = usage.free / 2**30
    if free_gib < gib:
        raise ContainerError(
            f"{path} has {free_gib:.0f} GiB free, need at least {gib}. "
            "A truncated checkpoint is worse than a refusal."
        )


def required_gpu_memory_mib(config: dict) -> int:
    """Return the task's explicit GPU-memory requirement.

    Runtime resource checks must not infer requirements from experiment history. The
    extension is intentionally small and names the accelerator family that produced the
    value; tasks without a measurement keep the compatibility behavior of returning zero.
    """

    extension = config.get("x-ai4ai", {})
    gpu = extension.get("gpu", {}) if isinstance(extension, dict) else {}
    if not gpu:
        return 0
    if not isinstance(gpu, dict):
        raise ContainerError("[x-ai4ai.gpu] must be a table")
    value = gpu.get("peak_memory_mib")
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ContainerError("[x-ai4ai.gpu].peak_memory_mib must be a positive integer")
    measured_type = gpu.get("type")
    allowed_types = config.get("environment", {}).get("gpu_types", [])
    if measured_type and measured_type not in allowed_types:
        raise ContainerError(
            f"GPU memory was measured on {measured_type!r}, not one of {allowed_types!r}"
        )
    return value


def inspect_gpu_hardware(config: dict, devices: tuple[int, ...]) -> dict[str, object]:
    """Inspect whether selected devices satisfy the declared official hardware.

    ``gpu_types`` is an official-replay constraint, not an unconditional local
    development gate.  This function only produces evidence; runner policy decides
    whether a mismatch is fatal, a non-official warning, or intentionally unchecked.
    The measured peak-memory reserve follows the same policy for the same reason.
    """

    environment = config.get("environment", {})
    allowed = environment.get("gpu_types", []) if isinstance(environment, dict) else []
    if not isinstance(allowed, list) or any(
        not isinstance(value, str) or not value.strip() for value in allowed
    ):
        raise ContainerError("[environment].gpu_types must be a list of non-empty strings")
    if not devices:
        raise ContainerError("at least one GPU device is required")
    need = required_gpu_memory_mib(config)
    required_free = int(need * 1.1) if need else 0
    if shutil.which("nvidia-smi") is None:
        raise ContainerError("nvidia-smi is unavailable; cannot verify the GPU hardware type")
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip() or "nvidia-smi failed"
        raise ContainerError(f"cannot inspect GPU hardware: {detail}")

    observed: dict[int, dict[str, object]] = {}
    for line in result.stdout.splitlines():
        parts = [value.strip() for value in line.split(",", 4)]
        if len(parts) != 5 or not parts[0].isdigit():
            continue
        try:
            total = int(parts[3])
            free = int(parts[4])
        except ValueError:
            continue
        observed[int(parts[0])] = {
            "index": int(parts[0]),
            "name": parts[1],
            "uuid": parts[2],
            "memory_total_mib": total,
            "memory_free_mib": free,
        }

    def normalise(value: str) -> str:
        return re.sub(r"[^A-Z0-9]+", "", value.upper())

    rows: list[dict[str, object]] = []
    problems: list[str] = []
    for device in devices:
        row = observed.get(device)
        if row is None:
            problems.append(f"nvidia-smi returned no hardware record for GPU {device}")
            continue
        name = str(row["name"])
        normalised_name = normalise(name)
        type_match = not allowed or any(
            normalise(expected) in normalised_name for expected in allowed
        )
        memory_match = not required_free or int(row["memory_free_mib"]) >= required_free
        row = {**row, "type_match": type_match, "free_memory_match": memory_match}
        rows.append(row)
        if not type_match:
            problems.append(f"GPU {device} is {name!r}, official task hardware is {allowed!r}")
        if not memory_match:
            problems.append(
                f"GPU {device} has {row['memory_free_mib']} MiB free; official recipe "
                f"reserve is {required_free} MiB"
            )
    return {
        "status": "match" if not problems else "mismatch",
        "allowed_types": allowed,
        "required_peak_memory_mib": need or None,
        "required_free_memory_mib": required_free or None,
        "devices": rows,
        "problems": problems,
    }


REQUIRED_IMAGE_TOOLS = {
    "bash": "every Agent command runs through bash -lc",
    "git": "submit.sh is a git diff; without it nothing can be submitted",
    "python3": "every harness entry point is python3, invoked by name",
    "sed": "coreutils floor",
    "head": "coreutils floor",
}
REQUIRED_HARNESS_PATHS = {
    "/opt/harness/lifecycle.py": "shared lifecycle implementation",
    "/opt/harness/submit.sh": "explicit Agent submission tool",
    "/opt/harness/no_candidate.sh": "explicit no-candidate tool",
    "/opt/harness/save_checkpoint.py": "atomic standard-checkpoint helper",
    "/opt/harness/validate_checkpoint.py": "frozen checkpoint loadability gate",
}


def require_image_tools(image: str, *, runtime: str = DEFAULT_RUNTIME) -> None:
    """Refuse to start an image missing tools required by the agent lifecycle."""

    tool_probe = [
        f"command -v {tool} >/dev/null 2>&1 || echo MISSING:{tool}"
        for tool in sorted(REQUIRED_IMAGE_TOOLS)
    ]
    path_probe = [
        f"test -r {path} || echo MISSING_PATH:{path}" for path in sorted(REQUIRED_HARNESS_PATHS)
    ]
    probe = "; ".join(tool_probe + path_probe)
    result = subprocess.run(
        [
            *runtime_argv(runtime),
            "run",
            "--rm",
            "--pull=never",
            "--network=none",
            "--entrypoint",
            "bash",
            image,
            "-lc",
            probe,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode != 0 and "MISSING:" not in result.stdout:
        # bash itself absent, or the image will not start. Either way, say so plainly
        # rather than reporting every tool as missing.
        raise ContainerError(
            f"{image} could not be probed for its tool floor: "
            f"{(result.stderr or result.stdout).strip()[:200]}"
        )
    absent = [
        line.split(":", 1)[1] for line in result.stdout.splitlines() if line.startswith("MISSING:")
    ]
    if absent:
        detail = "\n".join(f"  {tool} ({REQUIRED_IMAGE_TOOLS[tool]})" for tool in absent)
        raise ContainerError(
            f"{image} is missing tools the Agent needs:\n{detail}\n"
            "Install them in the task's environment/Dockerfile. A conda image commonly "
            "ships python3.11 with no `python3` name, which is one symlink."
        )
    absent_paths = [
        line.split(":", 1)[1]
        for line in result.stdout.splitlines()
        if line.startswith("MISSING_PATH:")
    ]
    if absent_paths:
        detail = "\n".join(f"  {path} ({REQUIRED_HARNESS_PATHS[path]})" for path in absent_paths)
        raise ContainerError(f"{image} is missing lifecycle harness files:\n{detail}")


# Launch real kernels rather than comparing version strings. A matmul exercises torch
# kernels while a single-rank collective exercises a separately compiled communication
# library. The collective backend is task-declared because not every valid image can use
# NCCL on its target GPU. The matmul runs first, so a task-specific gloo probe cannot hide
# an image whose basic CUDA kernels are incompatible.
_MATMUL_PROBE = (
    "import os, torch, torch.distributed as dist;"
    "x = torch.randn(64, 64, device='cuda');"
    "assert torch.isfinite(x @ x).all();"
)
_COLLECTIVE_PROBE = (
    "os.environ.setdefault('MASTER_ADDR', '127.0.0.1');"
    "os.environ.setdefault('MASTER_PORT', '29555');"
    "dist.init_process_group({backend!r}, rank=0, world_size=1);"
    "t = torch.ones(8, device='cuda');"
    "dist.all_reduce(t);"
    "torch.cuda.synchronize();"
    "assert t.sum().item() == 8;"
    "dist.destroy_process_group();"
)
_REPORT = "print('KERNELS OK', torch.version.cuda, torch.cuda.get_arch_list())"


def require_runnable_kernels(
    image: str, device: int = 0, *, backend: str = "nccl", runtime: str = DEFAULT_RUNTIME
) -> None:
    """Refuse an image whose torch or declared collective backend cannot run on the GPU.

    Image identity and package versions do not prove kernel compatibility. The probe uses
    an actual matmul followed by a one-process collective, because torch and the collective
    library can carry different architecture coverage. Runtime launches are authoritative;
    host-reported product labels and version ordering are not.
    """

    if backend not in ("nccl", "gloo"):
        raise ContainerError(
            f"unknown collective backend {backend!r}; expected nccl or gloo. This comes "
            "from [environment].collective_backend in the task's task.toml."
        )
    probe = _MATMUL_PROBE + _COLLECTIVE_PROBE.format(backend=backend) + _REPORT
    result = subprocess.run(
        [
            *runtime_argv(runtime),
            "run",
            "--rm",
            "--pull=never",
            "--network=none",
            "--gpus",
            f"device={device}",
            "--entrypoint",
            "python3",
            image,
            "-c",
            probe,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if "KERNELS OK" in result.stdout:
        return

    output = (result.stderr or result.stdout).strip()
    # The driver's own words are the most useful line here: torch names the capability it
    # sees and the arch list it was built for, which is the whole diagnosis.
    verdict = next(
        (
            line
            for line in output.splitlines()
            if "not compatible" in line or "no kernel image" in line
        ),
        output.splitlines()[-1] if output.splitlines() else "no output",
    )
    raise ContainerError(
        f"{image} cannot execute a CUDA kernel on device {device}:\n"
        f"  {verdict}\n"
        "The image identity and tools are valid, but its torch cannot run on this card. "
        "Choose a base by a successful kernel probe, not by version ordering."
    )


def require_free_gpu_memory(devices: tuple[int, ...], need_mib: int) -> None:
    """Refuse to start on a device somebody else is already using.

    --gpu defaults to 0, and on a shared host device 0 is often already occupied by
    a resident inference server. A run started blindly would OOM during engine init
    rather than at submission; the late failure looks like a model problem.

    Absent nvidia-smi is not fatal -- a --dry-run or a CPU-only checkout has no
    reason to require it.
    """

    if shutil.which("nvidia-smi") is None:
        return
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        return
    free: dict[int, int] = {}
    for line in result.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            free[int(parts[0])] = int(parts[1])
    short = {index: free[index] for index in devices if index in free and free[index] < need_mib}
    if short:
        detail = ", ".join(f"device {i} has {m} MiB free" for i, m in sorted(short.items()))
        spare = sorted(i for i, m in free.items() if m >= need_mib and i not in devices)
        hint = f" Free devices: {spare}." if spare else ""
        raise ContainerError(
            f"{detail}; this task needs about {need_mib} MiB.{hint} "
            "Pass --gpu/--gpus explicitly, or wait for the device to clear."
        )


def require_no_gpu_reservations(
    devices: tuple[int, ...], *, runtime: str = DEFAULT_RUNTIME
) -> None:
    """Refuse a GPU already assigned to a running container on this daemon.

    A container can reserve ``--gpus device=N`` before it creates an NVML compute
    process.  At that point memory and utilisation both read zero, but starting a
    second lifecycle on the same device is still a race.  Inspect the runtime before
    sampling NVML so that dormant trainers count as occupied too.
    """

    command = runtime_argv(runtime)
    # Kernel probes use ``docker run --rm``. In a concurrent batch one can disappear
    # between ps and inspect, which means it no longer holds a reservation, not that the
    # daemon is unreadable. Re-list only for that exact race; every other runtime error
    # still fails closed. A bounded retry also refuses a daemon whose view never settles.
    inspected = None
    for snapshot in range(5):
        listed = subprocess.run(
            [*command, "ps", "-q"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if listed.returncode != 0:
            detail = (listed.stderr or listed.stdout).strip() or f"exit {listed.returncode}"
            raise ContainerError(f"cannot inspect running container GPU reservations: {detail}")
        container_ids = listed.stdout.split()
        if not container_ids:
            return
        inspected = subprocess.run(
            [*command, "inspect", *container_ids],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if inspected.returncode == 0:
            break
        detail = (inspected.stderr or inspected.stdout).strip() or f"exit {inspected.returncode}"
        if "No such object:" not in detail or snapshot == 4:
            raise ContainerError(f"cannot read running container GPU reservations: {detail}")
        time.sleep(0.1)
    assert inspected is not None
    try:
        containers = json.loads(inspected.stdout)
    except (TypeError, ValueError) as error:
        raise ContainerError("container runtime returned invalid inspect JSON") from error

    requested = set(devices)
    conflicts: list[str] = []
    for container in containers:
        if not container.get("State", {}).get("Running", False):
            continue
        name = str(container.get("Name", "unknown")).lstrip("/")
        for request in container.get("HostConfig", {}).get("DeviceRequests") or []:
            capabilities = request.get("Capabilities") or []
            if not any("gpu" in group for group in capabilities):
                continue
            raw_ids = request.get("DeviceIDs") or []
            reserved: set[int] = set()
            for raw in raw_ids:
                try:
                    reserved.add(int(raw))
                except (TypeError, ValueError):
                    # UUID and MIG spellings cannot be mapped safely to host indices.
                    reserved = set(requested)
                    break
            count = int(request.get("Count", 0) or 0)
            if not raw_ids and count != 0:
                # Docker selected the device dynamically; inspect does not expose which
                # host index won, so fail closed for every requested device.
                reserved = set(requested)
            overlap = requested & reserved
            if overlap:
                conflicts.append(f"{name} reserves GPU(s) {sorted(overlap)}")
    if conflicts:
        raise ContainerError(
            "a running container already reserves the requested device(s):\n  "
            + "\n  ".join(conflicts)
        )


def require_free_vram(
    devices: tuple[int, ...],
    *,
    fraction: float = 0.85,
    max_idle_memory_mib: float = 64.0,
    samples: int = 2,
    sample_interval_sec: float = 2.0,
) -> None:
    """Require repeated zero-use telemetry for every requested device.

    Scheduler ownership alone does not establish idleness. Missing telemetry,
    compute processes, utilization or memory above the driver floor fail closed.
    """

    if shutil.which("nvidia-smi") is None:
        raise ContainerError("nvidia-smi is unavailable; cannot verify requested GPU occupancy")
    if samples < 1:
        raise ValueError("samples must be at least one")
    for sample in range(samples):
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip() or f"exit {result.returncode}"
            raise ContainerError(f"cannot read requested GPU occupancy: {detail}")
        readings: dict[int, tuple[str, float, float, float]] = {}
        for line in result.stdout.strip().splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 5:
                continue
            try:
                readings[int(fields[0])] = (
                    fields[1],
                    float(fields[2]),
                    float(fields[3]),
                    float(fields[4]),
                )
            except ValueError:
                continue

        processes = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if processes.returncode != 0:
            detail = (
                processes.stderr or processes.stdout
            ).strip() or f"exit {processes.returncode}"
            raise ContainerError(f"cannot read requested GPU compute processes: {detail}")
        by_uuid: dict[str, list[str]] = {}
        for line in processes.stdout.strip().splitlines():
            fields = [field.strip() for field in line.split(",", 3)]
            if len(fields) >= 3:
                by_uuid.setdefault(fields[0], []).append(f"pid {fields[1]} ({fields[2]})")

        busy: list[str] = []
        for device in devices:
            if device not in readings:
                raise ContainerError(
                    f"nvidia-smi returned no occupancy reading for requested GPU {device}"
                )
            uuid, used, total, utilization = readings[device]
            if by_uuid.get(uuid):
                busy.append(f"GPU {device}: compute {'; '.join(by_uuid[uuid])}")
            allowed_memory = min(max_idle_memory_mib, total * (1.0 - fraction))
            if used > allowed_memory:
                busy.append(
                    f"GPU {device}: {used:.0f} MiB in use (idle ceiling {allowed_memory:.0f} MiB)"
                )
            if utilization != 0:
                busy.append(f"GPU {device}: {utilization:.0f}% compute utilisation")
        if busy:
            raise ContainerError(
                "another process is using the requested device(s):\n  "
                + "\n  ".join(busy)
                + "\nFormal runs require a completely idle device. Pick another with --gpu, "
                "or wait. A Slurm hold alone does not make a GPU idle."
            )
        if sample + 1 < samples:
            time.sleep(sample_interval_sec)


def wait_for_free_vram_after_probe(
    devices: tuple[int, ...],
    *,
    timeout_sec: float = 30.0,
    poll_interval_sec: float = 1.0,
) -> None:
    """Wait briefly for the runner's own CUDA probe telemetry to settle.

    The kernel container has already exited, but an ``nvidia-smi`` utilization
    sample can remain nonzero for one polling interval. Every attempt still uses
    the full fail-closed occupancy check; persistent or external work times out.
    """

    deadline = time.monotonic() + timeout_sec
    last_error: ContainerError | None = None
    while True:
        try:
            require_free_vram(devices, samples=2, sample_interval_sec=1.0)
            return
        except ContainerError as error:
            last_error = error
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ContainerError(
                f"requested GPU did not return to idle after the CUDA kernel probe: {last_error}"
            ) from last_error
        time.sleep(min(poll_interval_sec, remaining))


# A candidate patch carries source. It must not carry data or weights.
#
# /assets is read-only, which stops the fixed inputs being edited but not being
# bypassed: a parquet written under /workspace rides into this container inside
# candidate.patch, which submit.sh generates with --binary. run.sh also refuses a
# TRAIN_DATA outside /assets, but the Agent owns run.sh and can delete that check.
# This one it cannot reach.
DATA_SUFFIXES = frozenset(
    {
        ".parquet",
        ".arrow",
        ".feather",
        ".orc",
        ".safetensors",
        ".bin",
        ".pt",
        ".pth",
        ".ckpt",
        ".gguf",
        ".npy",
        ".npz",
        ".h5",
        ".hdf5",
        ".pkl",
        ".pickle",
        ".tar",
        ".gz",
        ".tgz",
        ".zip",
        ".bz2",
        ".xz",
        ".zst",
        ".7z",
        ".db",
        ".sqlite",
        ".sqlite3",
        ".lmdb",
    }
)

# Source files are small. A 4 MiB addition is either generated data or something
# that should not be in a diff.
MAX_ADDED_FILE_BYTES = 4 * 1024 * 1024

# Candidate patches carry authored source only. Compiled bytecode is regenerated from
# that source, so submit and deadline capture delete it before staging. Reject it here as
# defense in depth in case an older image or a custom submit path skips that cleanup.
BYTECODE_SUFFIXES = frozenset({".pyc", ".pyo"})
BYTECODE_PARTS = frozenset({"__pycache__"})

# A per-file ceiling cannot catch a dataset whose rows are short: 500 rows of
# `{"q":7,"a":"5"}` are 7 KB and pass. Count rows in data-shaped files and JSON-record
# rows under any extension. Do not cap total authored source: candidates may replace the
# training algorithm, and source size is not evidence that data or weights were embedded.
#
# 16 is above any configuration file worth hand-writing and far below a question set
# (AIME is 60, MATH-500 is 500). This applies to added lines whether the file is new or
# already existed, because appending rows to one of verl's test fixtures is the same
# move as creating a file.
TEXT_DATA_SUFFIXES = frozenset({".jsonl", ".ndjson", ".json", ".csv", ".tsv"})
MAX_DATA_ROWS = 16


def describe_patch_rejections(patch: Path) -> list[str]:
    """Return reasons this patch must not be replayed, empty if it is fine.

    Reads the diff headers rather than applying anything, so a rejection costs
    nothing and happens before a GPU is claimed.
    """

    text = patch.read_text(encoding="utf-8", errors="replace")
    reasons: list[str] = []
    current: str | None = None
    added_new_file = False
    deleted_file = False
    binary_patch = False
    bytes_in_current = 0
    rows_in_current = 0
    data_like_rows = 0

    def close() -> None:
        nonlocal current, added_new_file, deleted_file, binary_patch
        nonlocal bytes_in_current, rows_in_current, data_like_rows
        if current and not deleted_file:
            suffix = Path(current).suffix.lower()
            if suffix in BYTECODE_SUFFIXES or any(
                part in Path(current).parts for part in BYTECODE_PARTS
            ):
                reasons.append(f"{current} is generated Python bytecode")
            elif suffix in DATA_SUFFIXES:
                reasons.append(f"{current} has a data or weights extension ({suffix})")
            if binary_patch:
                reasons.append(f"{current} is a binary blob")
        if current and added_new_file and bytes_in_current > MAX_ADDED_FILE_BYTES:
            reasons.append(
                f"{current} adds {bytes_in_current / 2**20:.1f} MiB, over the "
                f"{MAX_ADDED_FILE_BYTES / 2**20:.0f} MiB limit for a new file"
            )
        if (
            current
            and Path(current).suffix.lower() in TEXT_DATA_SUFFIXES
            and rows_in_current > MAX_DATA_ROWS
        ):
            reasons.append(
                f"{current} adds {rows_in_current} rows, over the {MAX_DATA_ROWS}-row "
                "limit for a data-shaped file -- the data is one of the three fixed "
                "inputs, so a dataset in the patch is out of scope"
            )
        if current and data_like_rows > MAX_DATA_ROWS:
            reasons.append(
                f"{current} adds {data_like_rows} JSON-record-like rows, over the "
                f"{MAX_DATA_ROWS}-row limit -- embedding data under a source or text "
                "extension does not make it authored method code"
            )
        current = None
        added_new_file = deleted_file = binary_patch = False
        bytes_in_current = rows_in_current = data_like_rows = 0

    for line in text.splitlines():
        if line.startswith("diff --git "):
            close()
            current = line.split(" b/", 1)[-1] if " b/" in line else line
        elif line.startswith("new file mode"):
            added_new_file = True
        elif line.startswith("deleted file mode"):
            deleted_file = True
        elif line.startswith("GIT binary patch"):
            binary_patch = True
        elif line.startswith("+") and not line.startswith("+++"):
            bytes_in_current += len(line)
            rows_in_current += 1
            value = line[1:].strip()
            if value.startswith("{") and value.endswith("}"):
                try:
                    data_like_rows += int(isinstance(json.loads(value), dict))
                except json.JSONDecodeError:
                    pass
    close()
    return reasons
