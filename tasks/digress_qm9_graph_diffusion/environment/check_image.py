"""Build-time self-check. Fails the build rather than leaving a broken image.

Three halves, and the first is specific to this task being the only conda-based
image in the portfolio.

**The runtime tools the Agent path needs.** git in particular: /opt/harness/submit.sh
shells out to it, and without git there is no candidate.patch, so the 12 h retrain
phase would train the pristine baseline while every log looked healthy. A missing git
has to fail the build, not the trial. bash, coreutils and the rest are what
orchestrator/agent.py's `docker exec ... bash -lc` calls need.

**The layout.** Every path in instruction.md is a promise to the Agent, and a promise
that only shows up as a runtime error four hours in is worse than a build failure.

**That the two DiGress copies are byte-identical.** The image carries the tree twice
-- editable at /workspace/digress, read-only at /opt/harness/digress -- and the whole
point is that the second is what the evaluators import. If a build ever produced two
different trees, fast_eval and the final would score against something other than
what the Agent trained with, silently.

Python 3.9: this image pins it.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

failures = []

# --- runtime tools -----------------------------------------------------------
#
# git is the one that matters most and the one most likely to be dropped by
# somebody trimming an apt line on the grounds that nothing compiles at run time.
REQUIRED_TOOLS = {
    "git": "submit.sh produces candidate.patch with it; without git there is no candidate",
    "bash": "every harness entry point, and the orchestrator's `bash -lc` wrapper",
    "tee": "orchestrator/agent.py pipes the agent's output through it",
    "stdbuf": "same, for line buffering",
    "grep": "agent.py's turn-failure detection",
    "cat": "agent.py writes the instruction copy with a heredoc",
    "python3": "the harness and the solution",
}
for tool, why in REQUIRED_TOOLS.items():
    if shutil.which(tool) is None:
        failures.append(f"missing runtime tool {tool} -- {why}")

# git has to actually run, not merely exist on PATH.
if shutil.which("git") is not None:
    result = subprocess.run(["git", "--version"], capture_output=True, text=True)
    if result.returncode != 0:
        failures.append(f"git is present but not runnable: {result.stderr.strip()}")
    else:
        print(f"  git         {result.stdout.strip()}")

# --- python and its packages -------------------------------------------------

if sys.version_info[:2] != (3, 9):
    failures.append(
        "expected python 3.9 (the pinned conda environment), got "
        f"{sys.version_info[0]}.{sys.version_info[1]}. The harness is written to 3.9 -- "
        "no tomllib, no zip(strict=), no match statement."
    )

for module, pin in (("torch", "2.0.1"), ("rdkit", None), ("torch_geometric", "2.3.1")):
    try:
        imported = __import__(module)
    except ImportError as error:
        failures.append(f"cannot import {module}: {error}")
        continue
    version = getattr(imported, "__version__", "unknown")
    print(f"  {module:<11} {version}")
    if pin and not str(version).startswith(pin):
        failures.append(f"expected {module} {pin}, got {version}")

try:
    import pytorch_lightning

    print(f"  lightning   {pytorch_lightning.__version__}")
except ImportError as error:
    failures.append(f"cannot import pytorch_lightning: {error}")

# --- the gloo fallback is still in place --------------------------------------
#
# environment/lightning_gloo_fallback.py patches lightning_fabric earlier in the build.
# Asserted again here, against the finished image, because the patch and the thing it
# patches are installed by two different steps: a base rebuild that moved
# pytorch-lightning would silently restore upstream's unconditional "nccl" and the only
# evidence would be a retrain dying 40 minutes in on "named symbol not found".
#
# This checks the DECISION, not that a string is in a file. On a card absent from torch's
# arch list -- capability (10, 3) against a cu118 arch list ending at sm_90 -- the answer
# has to be gloo, because NCCL here is cubin-only with no PTX and
# cannot launch at all. The build host has no GPU, so the two accessors are stubbed; that
# is the same thing the patch's own self-test does and it is the only way to test the
# branch that matters from inside a build.
try:
    import torch
    from lightning_fabric.utilities.distributed import (
        _get_default_process_group_backend_for_device as _choose_backend,
    )

    _real_capability = torch.cuda.get_device_capability
    _real_arch_list = torch.cuda.get_arch_list
    try:
        torch.cuda.get_device_capability = lambda _device=None: (10, 3)
        torch.cuda.get_arch_list = lambda: ["sm_37", "sm_80", "sm_90", "compute_37"]
        _absent = _choose_backend(torch.device("cuda", 0))
        # And the other direction: a card the wheel does have a cubin for must still get
        # NCCL, or the patch is a blanket override rather than a fallback and every other
        # host silently loses its transport.
        torch.cuda.get_device_capability = lambda _device=None: (9, 0)
        _present = _choose_backend(torch.device("cuda", 0))
    finally:
        torch.cuda.get_device_capability = _real_capability
        torch.cuda.get_arch_list = _real_arch_list

    if _absent != "gloo":
        failures.append(
            "lightning still chooses "
            f"{_absent!r} for a card absent from torch's arch list; expected 'gloo'. "
            "environment/lightning_gloo_fallback.py did not survive into this image, and "
            "NCCL cannot run on the target GPU (cubin-only, no PTX)."
        )
    elif _present != "nccl":
        failures.append(
            f"lightning chooses {_present!r} for sm_90, which IS in the arch list; expected "
            "'nccl'. The patch must be a fallback for cards with no cubin, not an "
            "unconditional override."
        )
    else:
        print("  backend     gloo when no cubin matches, nccl when one does")
except Exception as error:  # noqa: BLE001
    failures.append(f"cannot verify the lightning backend fallback: {error!r}")

# --- layout ------------------------------------------------------------------

REQUIRED = {
    "/workspace/run.sh": "launcher",
    "/workspace/train.py": "driver",
    "/workspace/digress/src/main.py": "the editable pinned tree",
    "/opt/harness/digress/src/main.py": "the grader's copy of the pinned tree",
    "/opt/harness/fast_eval.sh": "Agent tool",
    "/opt/harness/submit.sh": "Agent tool",
    "/opt/harness/timer.sh": "Agent tool",
    "/opt/harness/fast_eval.py": "evaluator",
    "/opt/harness/final_eval.py": "evaluator",
    "/opt/harness/grade.py": "shared metric and artifact check",
    "/opt/harness/gpu_phase_lock.py": "train/eval mutual exclusion",
    "/opt/harness/git-base/.git": "pristine baseline for submit.sh",
}
for path, role in REQUIRED.items():
    if not Path(path).exists():
        failures.append(f"missing {path} ({role})")


def tree_hash(root: Path) -> str:
    """Content hash of a directory, ignoring mtimes and __pycache__."""

    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        digest.update(str(path.relative_to(root)).encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


editable = Path("/workspace/digress")
grader = Path("/opt/harness/digress")
if editable.is_dir() and grader.is_dir():
    left, right = tree_hash(editable), tree_hash(grader)
    if left != right:
        failures.append(
            f"the editable tree and the grader's copy differ ({left[:16]} vs {right[:16]}). "
            "They are copied from one build context and must start identical -- otherwise "
            "the evaluator scores against code the Agent never ran."
        )
    else:
        print(f"  digress     {left[:16]} (both copies)")

    guard = "if files_exist(self.processed_paths):\n            return"
    for root in (editable, grader):
        loader = (root / "src/datasets/qm9_dataset.py").read_text(encoding="utf-8")
        if loader.count(guard) != 1:
            failures.append(f"processed-only QM9 guard missing from {root}")

# The grader copy must not be writable, or the second copy buys nothing.
if grader.is_dir():
    # Mode bits, not os.access. This build step runs as root, and root bypasses file
    # permissions entirely (CAP_DAC_OVERRIDE), so os.access(path, os.W_OK) returns True
    # for every file no matter what chmod did -- the check could never pass, whatever the
    # Dockerfile set. It reported LICENSE, README.md and requirements.txt as writable
    # immediately after `chmod -R a-w /opt/harness` had run on them.
    #
    # What the boundary actually rests on is the mode bits, since the container runs as a
    # non-root host uid at trial time and for that uid the bits are the whole story. Those
    # are readable as root, so this tests the thing the requirement is about.
    writable = [
        str(path)
        for path in list(grader.rglob("*"))[:5000]
        if path.is_file() and (path.stat().st_mode & 0o222)
    ]
    if writable:
        failures.append(
            f"the grader's DiGress copy has write bits set ({writable[:3]} ...). "
            "chmod -R a-w /opt/harness is what makes it a boundary."
        )

# The harness self-checks, which is cheaper here than in a trial.
for script in (
    "/opt/harness/grade.py",
    "/opt/harness/final_eval.py",
    "/opt/harness/fast_eval.py",
    "/opt/harness/gpu_phase_lock.py",
):
    if not Path(script).is_file():
        continue
    argv = [sys.executable, script] if script.endswith("grade.py") else [
        sys.executable, script, "--smoke"
    ]
    result = subprocess.run(argv, capture_output=True, text=True)
    if result.returncode != 0:
        failures.append(
            f"{script} self-check failed: {(result.stderr or result.stdout).strip()[:400]}"
        )

# The dataset is a mount, never baked in. If it were in the image, the test split
# would ship with the container rather than being mounted, and there would be no
# record of which phases can see it.
#
# Data, not names. `Path("/opt").glob("**/qm9*")` covers /opt/conda, so it matched
# torch_geometric/datasets/qm9.py and its .pyc -- the library module that knows how to
# READ QM9, which every install of torch_geometric has and which contains no data at
# all. The build failed on those two files. Any check that treats the reader as the
# dataset will fail on a correct image forever, since the reader is a dependency.
#
# So: skip installed packages, and only count extensions the export actually uses.
# .py and .pyc are code by definition and can never be the export.
DATASET_SUFFIXES = {".pt", ".pth", ".npz", ".npy", ".csv", ".sdf", ".zip", ".gz", ".pickle"}
searched = list(Path("/opt").glob("**/qm9*")) + list(Path("/workspace").glob("**/qm9_no_h*"))
for leaked in searched:
    # site-packages and dist-packages are the dependency tree, not our layout.
    if any(part in ("site-packages", "dist-packages", "__pycache__") for part in leaked.parts):
        continue
    if leaked.is_file() and leaked.suffix not in DATASET_SUFFIXES:
        continue
    if leaked.is_dir() or leaked.is_file():
        failures.append(f"the QM9 export must be mounted, not baked in: {leaked}")

if failures:
    raise SystemExit("Image check failed:\n" + "\n".join("  " + item for item in failures))

print("image check passed")
sys.stdout.flush()
