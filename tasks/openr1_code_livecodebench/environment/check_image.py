"""Build-time self-check. Fails the build rather than leaving a broken image.

Two halves: the dependency set is what we pinned, and the layout is what the task
documents. The layout half exists because every path in instruction.md is a
promise to the Agent, and a promise that only shows up as a runtime error four
hours in is worse than a build failure.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

result = subprocess.run(
    ["python", "-m", "pip", "check"], check=False, capture_output=True, text=True
)
# Keyed on the exit status, not on whether anything was printed. `pip check` writes
# "No broken requirements found." to stdout and exits 0 when the set is clean, so
# treating any output as a problem made this raise on a healthy image -- with pip's
# own success line quoted back as the "issue". The trailing `|| true` on the
# Dockerfile's RUN chain then swallowed the failure and the build reported success.
if result.returncode != 0:
    # Both streams: pip writes some failures to stderr, so reading only stdout can raise
    # with an empty message -- a correct verdict with the evidence missing.
    issues = [
        line.strip()
        for line in (result.stdout + "\n" + result.stderr).splitlines()
        if line.strip()
    ]
    raise SystemExit("Unexpected dependency issues:\n" + "\n".join(issues))

import torch  # noqa: E402
import transformers  # noqa: E402

# From the base image, not the lock. A second torch installed over this one is how
# a CUDA build becomes a CPU wheel without anything failing until the first
# .to("cuda"). Measured on the sibling DPO task: a vacuum-resolved wheelhouse put
# torch 2.13.0 over the base's 2.8.0 and the build still reported success.
#
# 2.8.x, not 2.3.x: the base moved to cu128 because cu121's kernels stop at sm_90
# and this device is sm_103. See the ARG PYTORCH_IMAGE comment in the Dockerfile.
if not torch.__version__.startswith("2.8."):
    raise SystemExit(f"expected torch 2.8.x from the base image, got {torch.__version__}")
# The build cannot see a GPU, so this cannot run a kernel. What it can do is refuse
# a base whose compiled archs could not possibly cover the deployment device --
# which is the failure that made this base change necessary, and it was silent
# until the first matmul. require_runnable_kernels does the real test on the host.
#
# torch._C._cuda_getArchFlags(), not torch.cuda.get_arch_list(): the public wrapper
# returns [] when no device is visible, which is every docker build, so a check
# built on it would pass vacuously here and catch nothing.
if torch.version.cuda is None:
    raise SystemExit("the base image's torch is a CPU build")
try:
    _flags = torch._C._cuda_getArchFlags() or ""
except Exception as _exc:  # noqa: BLE001
    raise SystemExit(f"cannot read this torch's compiled arch flags: {_exc}") from _exc
_sm = {int(name[3:]) for name in _flags.split() if name.startswith("sm_") and name[3:].isdigit()}
_ptx = {
    int(name[8:]) for name in _flags.split()
    if name.startswith("compute_") and name[8:].isdigit()
}
# sm_10x is forward-compatible inside its major, so an sm_100 cubin covers the
# sm_103 device; a compute_XX PTX entry would let the driver JIT to anything newer.
# Neither route existed on the cu121 base, which is why nothing ran.
if not (any(value >= 100 for value in _sm) or _ptx):
    raise SystemExit(
        "this torch has no sm_100+ cubin and no compute_XX PTX, so it cannot run on "
        f"the sm_103 deployment device. arch flags: {_flags!r}"
    )
if transformers.__version__ != "4.46.1":
    raise SystemExit(f"expected transformers 4.46.1, got {transformers.__version__}")

# The pinned evaluator has to be importable from where grade.py looks for it, and
# it has to be the tree with the modules the protocol names. Checked by import
# rather than by file existence: the reference protocol loaded
# lcb_runner/prompts/code_generation.py by path precisely because importing
# lcb_runner.prompts pulls in provider SDKs that are not installed here, and that
# distinction is easy to lose in a later edit.
sys.path.insert(0, "/opt/harness/livecodebench")
for module in (
    "lcb_runner.benchmarks.code_generation",
    "lcb_runner.evaluation.compute_code_generation_metrics",
    "lcb_runner.lm_styles",
    "lcb_runner.utils.extraction_utils",
):
    __import__(module)

from lcb_runner.lm_styles import LMStyle  # noqa: E402

# grade.py selects the prompt wrapper and the code-fence convention by this name.
# If upstream ever renames it, generation and extraction stop agreeing and every
# row scores zero -- a failure that looks like a broken model.
if not hasattr(LMStyle, "CodeQwenInstruct"):
    raise SystemExit("the pinned LiveCodeBench LMStyle has no CodeQwenInstruct entry")

required = {
    "/workspace/run.sh": "launcher",
    "/workspace/train.py": "method",
    "/opt/harness/fast_eval.sh": "Agent tool",
    "/opt/harness/submit.sh": "Agent tool",
    "/opt/harness/timer.sh": "Agent tool",
    "/opt/harness/fast_eval.py": "evaluator",
    "/opt/harness/final_eval.py": "evaluator",
    "/opt/harness/grade.py": "evaluator",
    "/opt/harness/livecodebench/lcb_runner": "pinned official evaluator",
    "/opt/harness/livecodebench/lcb_runner/prompts/code_generation.py": "pinned prompt module",
    "/opt/harness/git-base/.git": "pristine baseline for submit.sh",
}
missing = {path: role for path, role in required.items() if not Path(path).exists()}
if missing:
    detail = "\n".join(f"  {path} ({role})" for path, role in missing.items())
    raise SystemExit(f"Image layout is incomplete:\n{detail}")

# The evaluation rows are mounts, never image content. If any release file were
# baked in, the exploration container would carry them too -- both phases run this
# image -- and test6.jsonl in particular would put the final's questions in front
# of the Agent.
for pattern in ("test*.jsonl", "**/test*.jsonl"):
    for leaked in Path("/opt").glob(pattern):
        raise SystemExit(f"LiveCodeBench rows must not be baked into the image: {leaked}")

# The evaluators must be self-checkable without a GPU, which is also what
# score-mock relies on.
for script in ("/opt/harness/grade.py", "/opt/harness/fast_eval.py", "/opt/harness/final_eval.py"):
    argv = ["python", script] + ([] if script.endswith("grade.py") else ["--smoke"])
    check = subprocess.run(argv, check=False, capture_output=True, text=True)
    if check.returncode != 0:
        raise SystemExit(f"{script} smoke failed:\n{check.stdout}\n{check.stderr}")

print("image check passed")
print(f"  torch         {torch.__version__}")
print(f"  transformers  {transformers.__version__}")
print("  livecodebench /opt/harness/livecodebench")
sys.stdout.flush()
