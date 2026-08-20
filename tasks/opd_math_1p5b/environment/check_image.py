"""Build-time self-check. Fails the build rather than leaving a broken image.

Two halves: the dependency set is what we pinned, and the layout is what the
task documents. The layout half exists because every path in instruction.md is a
promise to the Agent, and a promise that only shows up as a runtime error four
hours in is worse than a build failure.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

# Inherited from the immutable upstream base image. All optional DeepEP/GObject/
# OpenCV paths that the single-GPU FSDP+vLLM runtime never imports.
ALLOWED_BASE_IMAGE_ISSUES = (
    "deep-ep 1.2.1+7febc6e requires pynvml, which is not installed.",
    "pygobject 3.42.1 requires pycairo, which is not installed.",
    "cupy-cuda12x 14.0.1 has requirement numpy<2.6,>=2.0, but you have numpy 1.26.4.",
    (
        "opencv-python-headless 4.13.0.92 has requirement numpy>=2; "
        'python_version >= "3.9", but you have numpy 1.26.4.'
    ),
)

result = subprocess.run(
    ["python", "-m", "pip", "check"], check=False, capture_output=True, text=True
)
# `pip check` exits 0 and prints "No broken requirements found." when the environment is
# healthy. Judging by stdout being non-empty therefore raised on a CLEAN image -- and the
# unbraced `|| true` on the Dockerfile's final RUN chain swallowed the raise, so the build
# reported success with this file never having run to completion. The tell was /opt/build
# surviving in the finished image, since it is removed by a later step on the same chain.
# Three images had digests recorded in that state.
#
# Judge by the exit status. Only when it is nonzero are the stdout lines real issues, and
# only then does the allowlist mean anything.
# Both streams, because pip writes some failures to stderr -- reading only stdout can
# produce a correct verdict with the evidence missing.
issues = (
    [
        line.strip()
        for line in (result.stdout + "\n" + result.stderr).splitlines()
        if line.strip()
    ]
    if result.returncode != 0
    else []
)
unexpected = [line for line in issues if line not in ALLOWED_BASE_IMAGE_ISSUES]
if unexpected:
    raise SystemExit("Unexpected dependency issues:\n" + "\n".join(unexpected))

import pyarrow  # noqa: E402

if pyarrow.__version__ != "23.0.1":
    raise SystemExit(f"Expected pyarrow 23.0.1, got {pyarrow.__version__}")

# verl must resolve through PYTHONPATH to the live tree under /workspace, not to the copy the
# base image preinstalled at /opt/verl. If this ever points back at /opt, an
# Agent's edits would be silently ignored -- the failure mode is a run that
# looks fine and trains the wrong code.
import verl  # noqa: E402

verl_root = Path(verl.__file__).resolve().parent
if verl_root != Path("/workspace/verl/verl"):
    raise SystemExit(f"verl resolves to {verl_root}, expected /workspace/verl/verl")

# The wall-clock patch has to be present in the tree that resolves, not merely
# applied somewhere during the build.
main_ppo = (verl_root / "trainer/main_ppo.py").read_text(encoding="utf-8")
if "colocate_teacher" not in main_ppo:
    raise SystemExit("single_gpu_wall_clock.patch is missing from the resolved verl tree")

# Import every module the patch touches, not just verl's top level.
#
# This check exists because of a real failure: a stripped blank context line
# shortened one hunk body below its declared count, git apply dropped the tail
# without complaining, and teacher_model.py was left ending inside an unclosed
# list comprehension. The build passed -- `import verl` does not reach
# teacher_loop -- and training died at init_workers instead, several minutes in,
# behind a Ray error that reported no detail. A truncated patch has to fail the
# build, not the run.
import importlib  # noqa: E402

for module in (
    "verl.trainer.main_ppo",
    "verl.trainer.ppo.ray_trainer",
    "verl.experimental.teacher_loop.teacher_model",
):
    try:
        importlib.import_module(module)
    except SyntaxError as error:
        raise SystemExit(
            f"{module} does not parse -- the patch applied only partially: {error}"
        ) from error
    except Exception as error:  # noqa: BLE001
        # A missing GPU or optional dependency at build time is fine; a syntax
        # error is not, and that is the only thing being screened for here.
        print(f"  note        {module} imported with {type(error).__name__}: {error}")

required = {
    "/workspace/run.sh": "launcher",
    "/workspace/train.py": "driver",
    "/opt/harness/fast_eval.sh": "Agent tool",
    "/opt/harness/submit.sh": "Agent tool",
    "/opt/harness/timer.sh": "Agent tool",
    "/opt/harness/fast_eval.py": "evaluator",
    "/opt/harness/final_eval.py": "evaluator",
    "/opt/harness/grade.py": "evaluator",
    "/opt/harness/justrl/evals/utils.py": "pinned rule grader",
    "/opt/harness/justrl/evals/grade.py": "pinned CV_PROMPT source",
    "/opt/harness/git-base/.git": "pristine baseline for submit.sh",
}
missing = {path: role for path, role in required.items() if not Path(path).exists()}
if missing:
    detail = "\n".join(f"  {path} ({role})" for path, role in missing.items())
    raise SystemExit(f"Image layout is incomplete:\n{detail}")

if shutil.which("flock") is None:
    raise SystemExit("Image is missing flock, required for training/evaluation isolation")
if shutil.which("rg") is None:
    raise SystemExit("Image is missing rg, required for Agent source exploration")

# The base image's own copies must be gone, or `import verl` becomes ambiguous
# and an Agent has two trees to wonder about.
for stale in ("/opt/verl", "/opt/opd", "/opt/llm-algobench"):
    if Path(stale).exists():
        raise SystemExit(f"{stale} should have been removed from the image")

# AIME is a final-only input. If it were ever baked in, the exploration
# container would carry the final questions, since both use this image.
for leaked in Path("/").glob("opt/**/aime*"):
    raise SystemExit(f"final data must not be baked into the image: {leaked}")

print("image check passed")
print(f"  verl        {verl_root}")
print(f"  pyarrow     {pyarrow.__version__}")
for issue in issues:
    print(f"  upstream    {issue}")
sys.stdout.flush()
