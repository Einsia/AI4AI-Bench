"""Build-time self-check. Fails the build rather than leaving a broken image.

Three halves: the dependency set is what we pinned, the layout is what the task
documents, and the two evaluators' own smokes pass. The layout half exists because
every path in instruction.md is a promise to the Agent, and a promise that only
shows up as a runtime error four hours in is worse than a build failure.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

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
# Both streams. pip writes some failures to stderr, so reading only stdout can raise
# with an empty message -- a correct verdict with the evidence missing, which sends
# whoever reads the build log looking in the wrong place.
if result.returncode != 0:
    issues = [
        line.strip()
        for line in (result.stdout + "\n" + result.stderr).splitlines()
        if line.strip()
    ]
    raise SystemExit("Unexpected dependency issues:\n" + "\n".join(issues))

import peft  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402

for module, expected in (
    (torch, "2.8.0"),
    (transformers, "4.57.6"),
    (peft, "0.19.1"),
):
    if not module.__version__.startswith(expected):
        raise SystemExit(f"Expected {module.__name__} {expected}, got {module.__version__}")

# paged_adamw_32bit is run.sh's default optimizer and what the recorded 2646 s /
# 2616 s baselines used. Its wheel is the one line of the lock with no hash carried
# over from the reference protocol, so it is the most likely thing to be missing from a
# first build -- and without this check the failure would surface twelve hours into a
# retrain run as an unknown-optimizer error.
import bitsandbytes  # noqa: E402

if not bitsandbytes.__version__.startswith("0.49"):
    raise SystemExit(f"Expected bitsandbytes 0.49.0, got {bitsandbytes.__version__}")

# final_eval.py and build_proxy_asset.py read the RewardBench parquet directly;
# artifact.py reads shapes out of the adapter header without loading the weights.
import pyarrow.parquet  # noqa: E402, F401
from safetensors import safe_open  # noqa: E402, F401

required = {
    "/workspace/run.sh": "launcher",
    "/workspace/train.py": "trainer",
    "/opt/harness/fast_eval.sh": "Agent tool",
    "/opt/harness/submit.sh": "Agent tool",
    "/opt/harness/timer.sh": "Agent tool",
    "/opt/harness/fast_eval.py": "proxy evaluator",
    "/opt/harness/final_eval.py": "hidden final",
    "/opt/harness/grade.py": "shared scoring path",
    "/opt/harness/artifact.py": "artifact-side check",
    "/opt/harness/git-base/.git": "pristine baseline for submit.sh",
}
missing = {path: role for path, role in required.items() if not Path(path).exists()}
if missing:
    detail = "\n".join(f"  {path} ({role})" for path, role in missing.items())
    raise SystemExit(f"Image layout is incomplete:\n{detail}")

# RewardBench is a mounted asset, mounted into the scoring phase only. If any of it
# were ever baked in, the exploration container would carry the final's rows, since
# both phases use this image -- which is the one thing the two-tier split exists to
# prevent.
for pattern in ("opt/**/*reward*bench*", "opt/**/filtered-*.parquet", "workspace/**/*.parquet"):
    for leaked in Path("/").glob(pattern):
        raise SystemExit(f"final data must not be baked into the image: {leaked}")

# The evaluators' own self-checks, so a build cannot ship a broken weight table or a
# proxy allocation that does not sum to 512.
for script in (
    "/opt/harness/grade.py",
    "/opt/harness/artifact.py",
):
    check = subprocess.run([sys.executable, script], check=False, capture_output=True, text=True)
    if check.returncode != 0:
        raise SystemExit(f"{script} smoke failed:\n{check.stdout}\n{check.stderr}")
    print(f"  smoke       {check.stdout.strip()}")

for script in ("/opt/harness/fast_eval.py", "/opt/harness/final_eval.py"):
    check = subprocess.run(
        [sys.executable, script, "--smoke"], check=False, capture_output=True, text=True
    )
    if check.returncode != 0:
        raise SystemExit(f"{script} smoke failed:\n{check.stdout}\n{check.stderr}")
    print(f"  smoke       {check.stdout.strip()}")

print("image check passed")
print(f"  torch       {torch.__version__}")
print(f"  transformers {transformers.__version__}")
print(f"  peft        {peft.__version__}")
print(f"  bitsandbytes {bitsandbytes.__version__}")
sys.stdout.flush()
