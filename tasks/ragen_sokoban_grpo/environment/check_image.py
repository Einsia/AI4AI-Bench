"""Build-time self-check. Fails the build rather than leaving a broken image.

Three halves. The layout is what the task documents, because every path in
instruction.md is a promise to the Agent and a promise that only shows up as a
runtime error four hours in is worse than a build failure. The two RAGEN trees
resolve where they are supposed to. And the frozen tree's content hash is computed
and compared with the pin in harness/grade.py -- which is the check that makes that
pin real rather than a comment.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/opt/harness")

# Inherited from the immutable upstream base image, which is not ours to fix. Extend
# this list only with something you have established the single-B300 FSDP+vLLM
# runtime never imports.
ALLOWED_BASE_IMAGE_ISSUES: tuple[str, ...] = ()

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
    raise SystemExit(
        "Unexpected dependency issues:\n"
        + "\n".join(unexpected)
        + "\n\nIf these come from the base image and the runtime never imports them, "
        "add them verbatim to ALLOWED_BASE_IMAGE_ISSUES with a note saying why."
    )

required = {
    "/workspace/run.sh": "launcher",
    "/workspace/finalize.py": "merge and metadata driver",
    "/workspace/ragen/train.py": "editable RAGEN entry point",
    "/workspace/ragen/verl": "editable verl submodule",
    "/opt/harness/fast_eval.sh": "Agent tool",
    "/opt/harness/submit.sh": "Agent tool",
    "/opt/harness/timer.sh": "Agent tool",
    "/opt/harness/fast_eval.py": "evaluator",
    "/opt/harness/final_eval.py": "evaluator",
    "/opt/harness/grade.py": "evaluator",
    "/opt/harness/gpu_phase_lock.py": "train/eval mutual exclusion",
    "/opt/harness/ragen/train.py": "frozen RAGEN tree",
    "/opt/harness/ragen/ragen/llm_agent/agent_proxy.py": "frozen rollout entry point",
    "/opt/harness/ragen/verl": "frozen verl submodule",
    "/opt/harness/git-base/.git": "pristine baseline for submit.sh",
}
missing = {path: role for path, role in required.items() if not Path(path).exists()}
if missing:
    detail = "\n".join(f"  {path} ({role})" for path, role in missing.items())
    raise SystemExit(f"Image layout is incomplete:\n{detail}")

lock_smoke = subprocess.run(
    [sys.executable, "/opt/harness/gpu_phase_lock.py", "--smoke"],
    check=False,
    capture_output=True,
    text=True,
)
if lock_smoke.returncode != 0:
    raise SystemExit(
        "GPU phase lock self-check failed:\n"
        + (lock_smoke.stdout + lock_smoke.stderr)[-2000:]
    )

# The base image's own copies must be gone, or `import ragen` becomes ambiguous and
# an Agent has two trees to wonder about besides the two this file put there.
for stale in ("/opt/ragen", "/opt/RAGEN", "/opt/llm-algobench"):
    if Path(stale).exists():
        raise SystemExit(f"{stale} should have been removed from the image")

# The patch has to be present in both trees, not merely applied somewhere during the
# build. A partially applied hunk leaves a file that imports and misbehaves.
for root in ("/workspace/ragen", "/opt/harness/ragen"):
    proxy = Path(root) / "ragen/llm_agent/agent_proxy.py"
    text = proxy.read_text(encoding="utf-8")
    for fragment in (
        "seed=int(config.sampling_seed)",
        "int(self.config.sampling_seed) + int(env_id)",
        'lm_inputs.non_tensor_batch["env_ids"]',
    ):
        if fragment not in text:
            raise SystemExit(
                f"{proxy} is missing {fragment!r}: the per-request seed patch did not "
                "apply. Without it a board's trajectory depends on its batch."
            )
    workers = Path(root) / "ragen/workers/fsdp_workers.py"
    if 'attn_implementation="sdpa"' not in workers.read_text(encoding="utf-8"):
        raise SystemExit(f"{workers} still asks for flash_attention_2")

# Both trees must parse. `import ragen` does not reach the environment or the
# workers, so a truncated patch there would pass an import check and fail at
# init_workers instead, minutes into training and behind a Ray error with no detail.
for root in ("/workspace/ragen", "/opt/harness/ragen"):
    compile_result = subprocess.run(
        ["python", "-m", "compileall", "-q", "-x", r"(^|/)(\.git|__pycache__)/", root],
        check=False,
        capture_output=True,
        text=True,
    )
    if compile_result.returncode != 0:
        raise SystemExit(
            f"{root} does not compile -- the patch applied only partially:\n"
            + (compile_result.stdout + compile_result.stderr)[-4000:]
        )

# The frozen tree's content hash, and the pin. grade.py computes both halves; this
# only decides whether a mismatch fails the build.
import grade  # noqa: E402

frozen_hash = grade.tree_sha256(grade.FROZEN_RAGEN)
subtree = grade.environment_subtree()
subtree_hash = grade.tree_sha256(subtree)

for label, actual, pinned, constant in (
    ("frozen RAGEN tree", frozen_hash, grade.FROZEN_RAGEN_SHA256, "FROZEN_RAGEN_SHA256"),
    (
        "frozen environment subtree",
        subtree_hash,
        grade.FROZEN_ENVIRONMENT_SHA256,
        "FROZEN_ENVIRONMENT_SHA256",
    ),
):
    if not pinned:
        print(f"  RECORD ME  {constant} = \"{actual}\"   ({label})")
    elif actual != pinned:
        raise SystemExit(
            f"{label} does not match harness/grade.py's {constant}\n"
            f"  pinned {pinned}\n  actual {actual}\n"
            "Either the tree changed or the pin is stale. Update both together, the "
            "same way the image digest and source_sha256 are updated together."
        )

# The final's environment seed must not be reachable from the image. It lives in
# declaration.py, which is host-side, and the whole reason a candidate cannot train
# on the final boards is that this stays true. Boards are generated from a seed, so
# unlike a mounted question set there is nothing else to withhold.
for tree in (Path("/workspace"), Path("/opt/harness")):
    for path in tree.rglob("*.py"):
        if "__pycache__" in path.parts or "/ragen/" in path.as_posix():
            continue
        if "environment_seed" in path.read_text(encoding="utf-8", errors="replace"):
            text = path.read_text(encoding="utf-8", errors="replace")
            # A parameter named environment_seed is fine; a literal default is not.
            for marker in ("environment_seed = 1", "environment_seed=1", "SEED = 123"):
                if marker in text:
                    raise SystemExit(
                        f"{path} looks like it hardcodes a final environment seed. "
                        "The seed is supplied by the score phase, not baked in."
                    )

# The PTX assembler, checked at build time because the alternative is finding out
# during a rollout. The image's own CUDA is 12.8, whose ptxas has no `sm_103a` -- the
# card's target -- and every torch.compile in a run died on
# `ptxas fatal : Value 'sm_103a' is not defined for option 'gpu-name'`.
#
# Two things have to hold, and they are independent. The binary must accept the target,
# and the version it REPORTS must be one triton 3.4.0 accepts and maps to a PTX ISA
# that supports sm_103: triton refuses CUDA major 13 outright, and it writes the PTX
# `.version` directive from this string, so a 12.8 binary yields `.version 8.7` and
# then `PTX .version 8.7 does not support .target sm_103a`. Only 12.9 satisfies both.
#
# No GPU is needed for either check, which matters: the build has no device.
ptxas_path = Path(os.environ.get("TRITON_PTXAS_PATH", ""))
if not ptxas_path.is_file() or not os.access(ptxas_path, os.X_OK):
    raise SystemExit(
        f"TRITON_PTXAS_PATH={ptxas_path} is not an executable file. The Dockerfile "
        "sets it to the CUDA 12.9 ptxas copied from the cuda_ptxas build context."
    )
ptxas_version = subprocess.run(
    [str(ptxas_path), "--version"], check=False, capture_output=True, text=True
)
release = re.search(r"release (\d+)\.(\d+)", ptxas_version.stdout + ptxas_version.stderr)
if release is None:
    raise SystemExit(
        f"{ptxas_path} --version printed no `release X.Y`, which is the pattern "
        "triton's knobs.NvidiaTool.from_path parses. It would be rejected at runtime.\n"
        + (ptxas_version.stdout + ptxas_version.stderr)[-2000:]
    )
major, minor = int(release.group(1)), int(release.group(2))
if (major, minor) < (12, 9) or major > 12:
    raise SystemExit(
        f"{ptxas_path} reports CUDA {major}.{minor}. sm_103 needs 12.9 or later, and "
        "triton 3.4.0's ptx_get_version raises for any major above 12, so 12.9 through "
        "12.x is the whole of the usable range."
    )
targets = subprocess.run(
    [str(ptxas_path), "--help"], check=False, capture_output=True, text=True
)
if "sm_103a" not in (targets.stdout + targets.stderr):
    raise SystemExit(
        f"{ptxas_path} does not list sm_103a among its --gpu-name values, which is the "
        "target triton asks for on a compute-capability (10, 3) device."
    )

print("image check passed")
print(f"  frozen ragen   {grade.FROZEN_RAGEN} {frozen_hash[:16]}")
print(f"  environment    {subtree} {subtree_hash[:16]}")
print(f"  ptxas          {ptxas_path} CUDA {major}.{minor}, knows sm_103a")
for issue in issues:
    print(f"  upstream       {issue}")
sys.stdout.flush()
