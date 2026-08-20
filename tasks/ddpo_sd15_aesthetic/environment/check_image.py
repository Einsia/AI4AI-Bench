"""Build-time self-check. Fails the build rather than leaving a broken image.

Three halves: the dependency set is what we pinned, the layout is what the task
documents, and -- specific to this task -- the two copies of the upstream tree are
both present, both patched, and mutually independent.

That last one is the one worth having. The aesthetic scorer is simultaneously the
training reward and the evaluator, and the only thing separating them is that
/opt/harness holds its own copy. If a future edit collapsed them into one tree, or put
the candidate's tree on PYTHONPATH, nothing at runtime would complain: training would
work, fast_eval would work, and the reward would quietly be the candidate's.
"""

from __future__ import annotations

import subprocess
import sys
from importlib import resources
from pathlib import Path

WORKSPACE_DDPO = Path("/workspace/ddpo-pytorch")
HARNESS_DDPO = Path("/opt/harness/ddpo-pytorch")
REWARD_MLP = "sac+logos+ava1-l14-linearMSE.pth"

# Judged by exit status, not by whether stdout was empty. `pip check` prints
# "No broken requirements found." on SUCCESS, so the earlier
#
#     issues = [line for line in result.stdout.splitlines() if line.strip()]
#     if issues: raise SystemExit(...)
#
# raised on a clean image every time -- measured: the first build of this tree failed
# here with "Unexpected dependency issues: No broken requirements found.". It went
# unnoticed because the Dockerfile's final RUN ended in `|| true`, which the shell
# applies to the whole && chain, so the build reported success and produced an image
# with none of the checks below actually run. Both halves are fixed; this is the half
# that has to fail loudly when the dependency set really is broken.
result = subprocess.run(
    [sys.executable, "-m", "pip", "check"], check=False, capture_output=True, text=True
)
if result.returncode != 0:
    detail = (result.stdout + result.stderr).strip()
    raise SystemExit(f"pip check failed:\n{detail}")

# numpy BEFORE diffusers, because on the wrong numpy the diffusers import is what fails
# and it fails as a five-frame traceback ending in `np.float_`, which does not read as a
# version problem. This base ships numpy 2.3.2; the lock pins 1.26.4 over it because
# wandb 0.15.12 touches np.float_ at import time and diffusers reaches wandb through
# accelerate.tracking unconditionally. Checked as a pin and not as `< 2` because 1.26.4
# is pinned by the task runtime -- see runtime-requirements.in.
import numpy  # noqa: E402

if numpy.__version__ != "1.26.4":
    raise SystemExit(
        f"Expected numpy 1.26.4, got {numpy.__version__}. The base image ships numpy 2.x "
        "and the lock must downgrade it: wandb 0.15.12 runs np.float_ at import time, "
        "numpy 2.0 removed np.float_, and `import diffusers` pulls wandb in through "
        "accelerate.tracking -- so on numpy 2 nothing in this task can import at all."
    )

import accelerate  # noqa: E402
import diffusers  # noqa: E402
import transformers  # noqa: E402

for module, expected in (
    (accelerate, "0.20.3"),
    (diffusers, "0.17.1"),
    (transformers, "4.30.2"),
):
    if module.__version__ != expected:
        raise SystemExit(f"Expected {module.__name__} {expected}, got {module.__version__}")

# --- can the installed torch emit a kernel this hardware will run? ---------------
#
# The build has no GPU, so this cannot be require_runnable_kernels -- that check runs on
# the host and really does launch a matmul. What IS knowable at build time is the arch
# list torch was compiled with, which is a static property of the wheel, and it is the
# thing that was actually wrong: the previous base was cu121, its list stopped at sm_90,
# and every task on this card died on a bare matmul with "no kernel image is available
# for execution on the device". A cu121 wheel reintroduced by an edit to the lock or the
# base would pass every other check in this file and produce an image that cannot
# multiply two matrices.
#
# Satisfied by either route onto a newer device: an sm_XX cubin at or above the target
# (forward-compatible inside a major version) or a compute_XX PTX entry the driver JITs.
# The card here reports capability (10, 3).
#
# NOT torch.cuda.get_arch_list(): it opens with `if not is_available(): return []`, and
# there is no driver inside a build, so it returns an empty list and this check reads a
# correct cu128 wheel as having no kernels at all. Measured -- the first build with this
# check in it failed with `torch 2.8.0+cu128 was built for []`. The underlying binding
# reads the compile-time string and needs no driver: verified in this base image that
# with no GPU it returns "sm_70 ... sm_100 sm_120" while get_arch_list() returns [], and
# that with a GPU the two agree.
import torch  # noqa: E402

try:
    arch_list = torch._C._cuda_getArchFlags().split()
except AttributeError as exc:  # pragma: no cover - torch built without CUDA at all
    raise SystemExit(
        "torch exposes no _C._cuda_getArchFlags(), which means this is a CPU-only build. "
        f"This task needs CUDA. ({exc})"
    ) from exc
if not arch_list:
    raise SystemExit(
        "torch reports an empty CUDA arch list at build time. _C._cuda_getArchFlags() "
        "reads a compile-time constant and should never be empty for a CUDA build."
    )
TARGET_MAJOR = 10
cubins = {int(a[3:].rstrip("a"))
          for a in arch_list if a.startswith("sm_") and a[3:].rstrip("a").isdigit()}
ptx = [a for a in arch_list if a.startswith("compute_")]
if not any(c // 10 >= TARGET_MAJOR for c in cubins) and not ptx:
    raise SystemExit(
        f"torch {torch.__version__} was built for {arch_list}, which has no cubin at "
        f"sm_{TARGET_MAJOR}0 or above and no compute_XX PTX entry to JIT forward. This "
        "image would raise 'no kernel image is available for execution on the device' on "
        "the first CUDA op. The measurement device reports capability (10, 3)."
    )

# PTX alone is NOT sufficient in practice and this is the note that says so, because the
# check above accepts it. DiGress passed a matmul on compute_37 PTX and still died in
# real training with "no kernel image is available for engine execution": third-party
# compiled extensions ship their own cubins and do not participate in PTX JIT. This task
# has no such extension -- its whole stack is torch, torchvision and pure-Python
# packages -- but a future edit that adds one cannot rely on this check.

# --- the two trees --------------------------------------------------------------

for tree in (WORKSPACE_DDPO, HARNESS_DDPO):
    if not tree.is_dir():
        raise SystemExit(f"upstream tree missing: {tree}")
    scorer = tree / "ddpo_pytorch/aesthetic_scorer.py"
    text = scorer.read_text(encoding="utf-8")
    # The patch has to be present in each resolved tree, not merely applied somewhere
    # during the build.
    if "DDPO_CLIP_PATH" not in text:
        raise SystemExit(f"the offline-CLIP patch is missing from {scorer}")
    if '"openai/clip-vit-large-patch14"' in text:
        raise SystemExit(f"{scorer} still names the hub model; the build would need network")
    ddim = tree / "ddpo_pytorch/diffusers_patch/ddim_with_logprob.py"
    if "from diffusers.utils.torch_utils import randn_tensor" not in ddim.read_text():
        raise SystemExit(f"the randn_tensor import patch is missing from {ddim}")
    train_py = (tree / "tools/train.py").read_text(encoding="utf-8")
    if "if epoch != 0 and epoch % config.save_freq" in train_py:
        raise SystemExit(
            f"{tree}/tools/train.py still skips the epoch-0 save. With save_freq=1 "
            "that means a run killed by the wall clock exports nothing."
        )
    if "if epoch % config.save_freq == 0 and accelerator.is_main_process:" not in train_py:
        raise SystemExit(f"{tree}/tools/train.py has no recognisable save condition")

    # The reward is a CLIP embedding plus this MLP. Both trees need it: the candidate's
    # to train against, the harness's to score with.
    mlp = tree / "ddpo_pytorch/assets" / REWARD_MLP
    if not mlp.is_file():
        raise SystemExit(f"the aesthetic reward MLP is missing: {mlp}")
    prompts = (tree / "ddpo_pytorch/assets/simple_animals.txt").read_text(encoding="utf-8")
    lines = [line for line in prompts.splitlines() if line.strip()]
    # The whole prompt distribution, not a slice of it. 45 is what the pinned commit
    # ships; a shorter list would mean the container holds a subset of the
    # distribution the task claims to sample.
    if len(lines) != 45:
        raise SystemExit(f"{tree} simple_animals.txt has {len(lines)} prompts, expected 45")

# The harness's tree must resolve on its own, with nothing merged in from the
# candidate's. This is the check that the namespace-package hazard is closed.
sys.path.insert(0, str(HARNESS_DDPO))
import ddpo_pytorch  # noqa: E402

resolved = [str(entry) for entry in ddpo_pytorch.__path__]
if resolved != [str(HARNESS_DDPO / "ddpo_pytorch")]:
    raise SystemExit(
        f"ddpo_pytorch resolves to {resolved}, expected only the harness copy. "
        "Something put a second tree on the import path at build time."
    )
if not resources.files("ddpo_pytorch.assets").joinpath(REWARD_MLP).is_file():
    raise SystemExit("the harness tree's reward MLP is not reachable through the package")

import ddpo_pytorch.prompts  # noqa: E402

if len(set(ddpo_pytorch.prompts.simple_animals()[0] for _ in range(500))) < 20:
    raise SystemExit("simple_animals() is not drawing from the full prompt list")

# --- layout ---------------------------------------------------------------------

required = {
    "/workspace/run.sh": "launcher",
    "/workspace/train.py": "driver",
    "/workspace/ddpo_config.py": "the config upstream loads",
    "/workspace/ddpo-pytorch/tools/train.py": "the framework, the Agent's copy",
    "/opt/harness/fast_eval.sh": "Agent tool",
    "/opt/harness/submit.sh": "Agent tool",
    "/opt/harness/timer.sh": "Agent tool",
    "/opt/harness/fast_eval.py": "evaluator",
    "/opt/harness/final_eval.py": "evaluator",
    "/opt/harness/grade.py": "evaluator",
    "/opt/harness/ddpo-pytorch/ddpo_pytorch/aesthetic_scorer.py": "the reward, read-only",
    "/opt/harness/git-base/.git": "pristine baseline for submit.sh",
}
missing = {path: role for path, role in required.items() if not Path(path).exists()}
if missing:
    detail = "\n".join(f"  {path} ({role})" for path, role in missing.items())
    raise SystemExit(f"Image layout is incomplete:\n{detail}")

# The final's protocol is a mounted asset. If it were ever baked in, the exploration
# container would carry the final's generation seed, since both phases use this image --
# and on a generated dataset the seed is the held-out set.
for leaked in list(Path("/").glob("opt/**/final_reference*")) + list(
    Path("/").glob("workspace/**/final_reference*")
):
    raise SystemExit(f"the final's protocol must not be baked into the image: {leaked}")

# A stray .git under either tree would ride into candidate.patch, or shadow the
# pristine baseline submit.sh restores.
for stray in (WORKSPACE_DDPO / ".git", HARNESS_DDPO / ".git"):
    if stray.exists():
        raise SystemExit(f"{stray} should have been removed after the patch was applied")

print("image check passed")
print(f"  python        {sys.version.split()[0]}")
print(f"  torch         {torch.__version__}  cuda {torch.version.cuda}")
print(f"  arch list     {' '.join(arch_list)}")
print(f"  numpy         {numpy.__version__}")
print(f"  accelerate    {accelerate.__version__}")
print(f"  diffusers     {diffusers.__version__}")
print(f"  transformers  {transformers.__version__}")
print(f"  reward tree   {HARNESS_DDPO} (read-only)")
print(f"  agent tree    {WORKSPACE_DDPO} (editable)")
sys.stdout.flush()
