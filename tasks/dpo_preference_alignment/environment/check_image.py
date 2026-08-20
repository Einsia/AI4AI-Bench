"""Build-time self-check. Fails the build rather than leaving a broken image.

Four halves: the dependency set is what we pinned, the IFEval scorer actually
resolves, the layout is what the task documents, and the evaluators' own smokes pass.

The scorer half is the one that earns its place. `harness/grade.py` patches five
import paths inside lighteval -- `instructions_registry`, `instructions_utils`,
`instructions.LetterFrequencyChecker`, `main.IFEvalMetrics`, `main.ifeval_prompt` --
and none of them is exercised by any smoke that runs without lighteval installed. So
without this check, a lighteval version whose module layout moved would build a clean
image and fail in the *final*, after a 12 h retrain, with an ImportError. The previous
branch had the same check under a different name (`evaluate_ifeval.py
--dependency-smoke`) and ran it from check.sh; it belongs in the build.

The layout half exists because every path in instruction.md is a promise to the Agent,
and a promise that only shows up as a runtime error four hours in is worse than a
build failure.
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

# paged_adamw_32bit is run.sh's default optimizer and what every recorded formal run
# used (2435 s, 2630 s, 2842 s). Its wheel is one of the lock's unresolved lines, so
# it is the most likely thing to be missing from a first build -- and without this the
# failure would surface twelve hours into a retrain as an unknown-optimizer error.
import bitsandbytes  # noqa: E402

if not bitsandbytes.__version__.startswith("0.49"):
    raise SystemExit(f"Expected bitsandbytes 0.49.0, got {bitsandbytes.__version__}")

# DPOTrainer and DPOConfig, which solution/train.py builds on.
from trl import DPOConfig, DPOTrainer  # noqa: E402, F401

sys.path.insert(0, "/opt/harness")

import grade  # noqa: E402

# The IFEval scorer, end to end: resolve every lighteval import path grade.py
# patches, install the literal-symbol checker, and score one completion against one
# synthetic row. A row whose only instruction is "no comma" is scored strictly, so a
# comma-free completion must pass and a comma-carrying one must fail -- which
# exercises the registry lookup rather than merely importing it.
scorer_state = grade.install_ifeval_scorer()
if len(scorer_state) != 3:
    raise SystemExit("grade.install_ifeval_scorer did not return its three handles")

from lighteval.tasks.tasks.ifeval import instructions_registry  # noqa: E402

if "keywords:letter_frequency" not in instructions_registry.INSTRUCTION_DICT:
    raise SystemExit("the literal-symbol frequency checker was not registered")

probe = {
    "key": 1,
    "prompt": "Write one sentence with no commas.",
    "instruction_id_list": ["punctuation:no_comma"],
    "kwargs": [{}],
}
clean = grade.score_completion(probe, "A short sentence without any comma at all.")
dirty = grade.score_completion(probe, "A short sentence, with a comma.")
if float(clean["prompt_level_strict_acc"]) != 1.0:
    raise SystemExit(f"a compliant completion scored {clean}")
if float(dirty["prompt_level_strict_acc"]) != 0.0:
    raise SystemExit(f"a non-compliant completion scored {dirty}")
print(f"  scorer      lighteval IFEval resolved, {grade.TOKENIZATION_PROTOCOL}")

required = {
    "/workspace/run.sh": "launcher",
    "/workspace/train.py": "trainer",
    "/opt/harness/fast_eval.sh": "Agent tool",
    "/opt/harness/submit.sh": "Agent tool",
    "/opt/harness/timer.sh": "Agent tool",
    "/opt/harness/fast_eval.py": "proxy evaluator",
    "/opt/harness/final_eval.py": "hidden final",
    "/opt/harness/grade.py": "shared scoring path",
    "/opt/harness/generate.py": "shared generation path",
    "/opt/harness/checkpoint.py": "artifact resolver",
    "/opt/harness/git-base/.git": "pristine baseline for submit.sh",
}
missing = {path: role for path, role in required.items() if not Path(path).exists()}
if missing:
    detail = "\n".join(f"  {path} ({role})" for path, role in missing.items())
    raise SystemExit(f"Image layout is incomplete:\n{detail}")

# IFEval is a mounted asset, and the 413 sealed rows are mounted into the scoring
# phase only. If any of it were ever baked in, the exploration container would carry
# the final's rows, since every phase uses this image -- which is the one thing the
# two-tier split exists to prevent.
#
# Installed packages are not in scope, and saying so is load-bearing rather than a
# convenience: `opt/**/*.parquet` matches the four parquet fixtures pyarrow ships under
# its own tests/data, so the unqualified sweep fails every build that installs pyarrow --
# which is this one. That never showed up while the self-check was dying at `import
# bitsandbytes` several checks earlier. What the check is actually for is data arriving
# through this Dockerfile's COPY steps, and those write /workspace and /opt/harness.
for pattern in ("opt/**/*ifeval*", "workspace/**/*ifeval*", "opt/**/*.parquet"):
    for leaked in Path("/").glob(pattern):
        if {"site-packages", "dist-packages"} & set(leaked.parts):
            continue
        # grade.py and the two evaluators legitimately have ifeval in their text, not
        # their names; this catches data files.
        if leaked.suffix in {".jsonl", ".parquet", ".json", ".csv"}:
            raise SystemExit(f"final data must not be baked into the image: {leaked}")

# The harness's own self-checks, so a build cannot ship a broken split or a
# three-number report that does not add up.
for script in ("/opt/harness/grade.py", "/opt/harness/checkpoint.py", "/opt/harness/generate.py"):
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
print(f"  torch        {torch.__version__}")
print(f"  transformers {transformers.__version__}")
print(f"  peft         {peft.__version__}")
print(f"  bitsandbytes {bitsandbytes.__version__}")
print(f"  rows         {grade.SOURCE_ROWS} source, {grade.FINAL_ROWS} final, "
      f"{grade.PROXY_ROWS} proxy, {grade.HELD_OUT_ROWS} held out")
sys.stdout.flush()
