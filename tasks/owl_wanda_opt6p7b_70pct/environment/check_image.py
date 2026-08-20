"""Build-time self-check. Fails the build rather than leaving a broken image.

Three halves: the dependency set is what we pinned, the layout is what the task
documents, and neither evaluation split is baked in. The layout half exists because
every path in instruction.md is a promise to the Agent, and a promise that only shows
up as a runtime error four hours in is worse than a build failure.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# transformers 4.28.1 against a torch 2.8.0 base is intentional -- OWL dddb7a4 uses
# the 2023 HF API surface -- so pip check reports the mismatch every time. Anything
# NOT on this list is a real problem. accelerate is installed with --no-deps for the
# same reason.
ALLOWED_BASE_IMAGE_ISSUES = (
    "accelerate 0.18.0 requires torch, which is not installed.",
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

import torch  # noqa: E402
import transformers  # noqa: E402

if not torch.__version__.startswith("2.8.0"):
    raise SystemExit(f"Expected torch 2.8.0, got {torch.__version__}")
# Pinned exactly. A newer transformers renames OPT's decoder modules and
# lib/prune_all.py's find_layers stops finding them -- the failure is a model that
# prunes nothing and reports 0% sparsity, which the artifact check would catch at
# score time instead of here.
if transformers.__version__ != "4.28.1":
    raise SystemExit(f"Expected transformers 4.28.1, got {transformers.__version__}")

# pandas is what the evaluator reads its parquet with; the reference protocol used `datasets`
# and made the score depend on an arrow cache.
import pandas  # noqa: E402

if not pandas.__version__.startswith("1.5."):
    raise SystemExit(f"Expected pandas 1.5.x, got {pandas.__version__}")

# The OWL tree must resolve under /workspace, editable, and must carry the two
# modules the adapter monkeypatches into. If prune_all.py ever moves, owl_opt.py's
# imports fail several minutes into a run behind a traceback that names lib.data.
required = {
    "/workspace/run.sh": "launcher",
    "/workspace/prune.py": "driver",
    "/workspace/owl_opt.py": "OPT adapter",
    "/workspace/owl/main.py": "pinned OWL entry point",
    "/workspace/owl/lib/prune_all.py": "Wanda metric, OWL allocation, return_given_alpha",
    "/workspace/owl/lib/data.py": "the get_loaders the adapter replaces",
    "/opt/harness/fast_eval.sh": "Agent tool",
    "/opt/harness/submit.sh": "Agent tool",
    "/opt/harness/timer.sh": "Agent tool",
    "/opt/harness/fast_eval.py": "evaluator",
    "/opt/harness/final_eval.py": "evaluator",
    "/opt/harness/grade.py": "evaluator",
    "/opt/harness/git-base/.git": "pristine baseline for submit.sh",
}
missing = {path: role for path, role in required.items() if not Path(path).exists()}
if missing:
    detail = "\n".join(f"  {path} ({role})" for path, role in missing.items())
    raise SystemExit(f"Image layout is incomplete:\n{detail}")

# The upstream helpers the adapter imports by name. Checked as source text rather than
# by importing, because importing lib.prune_all pulls in the whole OWL dependency
# chain at build time on a machine with no GPU.
#
# Check names available from the imported module, including re-exports.
prune_all = Path("/workspace/owl/lib/prune_all.py").read_text(encoding="utf-8")
for symbol in ("WrappedGPT", "find_layers", "prepare_calibration_input_opt",
               "return_given_alpha"):
    # Defined here, or imported into this module's namespace from elsewhere in the
    # pinned tree. Either satisfies the import in owl_opt.py.
    defined = f"class {symbol}" in prune_all or f"def {symbol}" in prune_all
    imported = f"import {symbol}" in prune_all or f"{symbol}," in prune_all
    if not (defined or imported):
        raise SystemExit(
            f"pinned OWL lib/prune_all.py neither defines nor imports {symbol!r}; "
            "solution/owl_opt.py does `from lib.prune_all import "
            f"{symbol}` and the pruning pass would fail at layer 0"
        )

# The harness must be self-contained: /opt/harness cannot import from /workspace,
# because the scoring phase runs the original image layer while /workspace holds the
# candidate's edits. A harness that reached into /workspace would let a candidate
# rewrite its own evaluator.
for name in ("grade.py", "fast_eval.py", "final_eval.py"):
    source = Path(f"/opt/harness/{name}").read_text(encoding="utf-8")
    if "/workspace" in source:
        raise SystemExit(f"/opt/harness/{name} references /workspace")

validator = Path("/opt/harness/validate_checkpoint.py").read_text(encoding="utf-8")
if 'torch_dtype=torch.bfloat16, device_map={"": 0},' not in validator:
    raise SystemExit("OWL frozen validator lacks its Transformers 4.28.1 device-map adaptation")

sys.path.insert(0, "/opt/harness")
import grade  # noqa: E402

grade.smoke()
subprocess.run(
    [sys.executable, "/opt/harness/final_eval.py", "--smoke"], check=True
)
subprocess.run(
    [sys.executable, "/opt/harness/fast_eval.py", "--smoke"], check=True
)
subprocess.run(
    [sys.executable, "/workspace/prune.py", "--smoke"], check=True
)

# WikiText2 is a mounted input in both tiers. If either split were ever baked in, the
# exploration container would carry the final's text, since all phases use this image.
for leaked in Path("/").glob("opt/**/*wikitext*"):
    raise SystemExit(f"evaluation text must not be baked into the image: {leaked}")
for leaked in Path("/").glob("workspace/**/*wikitext*"):
    raise SystemExit(f"evaluation text must not be baked into the image: {leaked}")

print("image check passed")
print(f"  torch         {torch.__version__}")
print(f"  transformers  {transformers.__version__}")
print(f"  pandas        {pandas.__version__}")
for issue in issues:
    print(f"  upstream      {issue}")
sys.stdout.flush()
