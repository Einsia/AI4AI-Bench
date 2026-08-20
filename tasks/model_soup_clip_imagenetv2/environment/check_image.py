"""Build-time self-check. Fails the build rather than leaving a broken image.

Two halves: the pieces the task imports are importable, and the layout is what the
task documents. The layout half exists because every path in instruction.md is a
promise to the Agent, and a promise that only shows up as a runtime error four
hours in is worse than a build failure.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# `pip check` reports consistency through its return code; healthy output is allowed.
result = subprocess.run(
    [sys.executable, "-m", "pip", "check"], check=False, capture_output=True, text=True
)
if result.returncode != 0:
    issues = [
        line.strip()
        for line in (result.stdout + "\n" + result.stderr).splitlines()
        if line.strip()
    ]
    raise SystemExit("Unexpected dependency issues:\n" + "\n".join(issues))

# The pinned model-soups tree, baked in read-only. This is the file that decides
# what a state dict means -- see harness/forward.py -- so the whole task rests on
# it resolving from /opt/harness and nowhere else.
sys.path.insert(0, "/opt/harness/model_soups")
import utils  # noqa: E402

if Path(utils.__file__).resolve().parent != Path("/opt/harness/model_soups"):
    raise SystemExit(f"model-soups utils resolves to {utils.__file__}, expected /opt/harness")
if not hasattr(utils, "get_model_from_sd"):
    raise SystemExit("the pinned model-soups utils.py has no get_model_from_sd")

import clip  # noqa: E402

if not hasattr(clip, "load"):
    raise SystemExit("OpenAI CLIP is installed but has no load()")

# The harness has to be importable from the solution's point of view, because
# solution/soup.py imports forward.py and grade.py from /opt/harness. That coupling
# is deliberate -- the search and the final must agree about the architecture -- and
# it is the kind of thing that breaks silently when a file is renamed.
sys.path.insert(0, "/opt/harness")
import forward  # noqa: E402
import grade  # noqa: E402
import soup_check  # noqa: E402

if forward.UPSTREAM != Path("/opt/harness/model_soups"):
    raise SystemExit(f"forward.UPSTREAM is {forward.UPSTREAM}, which is not where the tree is")
if grade.EXPECTED_CLASSES != 1000 or grade.EXPECTED_PER_CLASS != 10:
    raise SystemExit("grade.py no longer describes ImageNetV2 matched-frequency")
if set(grade.PROXY_OFFSETS) - set(grade.FINAL_OFFSETS):
    raise SystemExit("the proxy offsets are not a subset of the final's; the split is broken")

# The artifact check is the task's only boundary, so its self-test runs at build
# time. It builds a synthetic ingredient basis and proves the check accepts honest
# soups and refuses perturbed ones -- see harness/soup_check.py. It needs no GPU and
# no assets, which is why it can run here.
soup_check.smoke()

# Layout: every path instruction.md names.
for path in (
    Path("/workspace/run.sh"),
    Path("/workspace/soup.py"),
    Path("/opt/harness/fast_eval.sh"),
    Path("/opt/harness/submit.sh"),
    Path("/opt/harness/timer.sh"),
    Path("/opt/harness/fast_eval.py"),
    Path("/opt/harness/final_eval.py"),
    Path("/opt/harness/soup_check.py"),
    Path("/opt/harness/git-base/.git"),
):
    if not path.exists():
        raise SystemExit(f"instruction.md promises {path} and it is not in the image")

# The evaluators' own self-checks, on synthetic rows.
for module in ("fast_eval", "final_eval"):
    check = subprocess.run(
        [sys.executable, f"/opt/harness/{module}.py", "--smoke"],
        check=False,
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        raise SystemExit(f"{module} --smoke failed:\n{check.stdout}\n{check.stderr}")

print('{"schema_version":1,"task_id":"model_soup_clip_imagenetv2","image_check":"passed"}')
