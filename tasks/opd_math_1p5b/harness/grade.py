"""The one grading path, shared by fast_eval and the hidden final.

Both stages call `grade_answer_verl` from the pinned JustRL tree baked into the
image at /opt/harness/justrl/evals, so a fast_eval score and a final score are
produced by the same rule grader. The CompassVerifier fallback is lifted from
the same tree's CV_PROMPT.

Nothing here reads a checkpoint or a model. Keep it that way: this file is the
reason "the evaluator does not change" can be checked by reading one thing.
"""

from __future__ import annotations

import ast
import importlib
import math
import statistics
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

JUSTRL_EVALS = Path("/opt/harness/justrl/evals")
JUSTRL_COMMIT = "cf444e9920b599c865ee6742c88dc162dcabcec2"


def load_rule_grader() -> Callable[[str, str], bool]:
    """Return JustRL's `grade_answer_verl` from the pinned tree."""

    if not JUSTRL_EVALS.is_dir():
        raise FileNotFoundError(f"pinned JustRL evals missing: {JUSTRL_EVALS}")
    path = str(JUSTRL_EVALS)
    if path not in sys.path:
        sys.path.insert(0, path)
    utils = importlib.import_module("utils")
    grader = getattr(utils, "grade_answer_verl", None)
    if grader is None:
        raise RuntimeError("pinned JustRL utils.py has no grade_answer_verl")
    return grader


def load_cv_prompt() -> str:
    """Extract CV_PROMPT without importing grade.py, which pulls in vLLM."""

    source = (JUSTRL_EVALS / "grade.py").read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "CV_PROMPT" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError("CV_PROMPT not found in pinned JustRL grade.py")


def grade_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach `rule_score` and `final_score` to each row in place."""

    grader = load_rule_grader()
    for row in rows:
        score = bool(grader(row["response"], row["answer"]))
        row["rule_score"] = score
        row["final_score"] = score
    return rows


def check_keys_unique(rows: list[dict[str, Any]]) -> None:
    """Refuse a row set where a (question, seed) pair appears twice.

    Sharded generation writes one file per shard and merges them. A shard written
    twice while another is missing leaves the row count correct and the content
    wrong, so counting rows does not catch it. The judge-side evaluator checked
    this and the integration dropped it; measured on six real runs, the real rate is
    zero, which is the point -- it should stay zero rather than be assumed.
    """

    seen: dict[tuple[Any, ...], int] = {}
    for row in rows:
        key = (row.get("task"), row.get("example_id"), row.get("seed"))
        seen[key] = seen.get(key, 0) + 1
    repeated = {key: count for key, count in seen.items() if count > 1}
    if repeated:
        sample = list(repeated.items())[:3]
        raise ValueError(
            f"{len(repeated)} (task, question, seed) key(s) appear more than once, "
            f"e.g. {sample}. Rows are duplicated or a shard was merged twice."
        )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate graded rows, clustering the standard error by question.

    The K samples drawn for one question are correlated, so treating all N=Q*K
    rows as independent understates the error. Score each question first, then
    take the standard error across questions. `stderr_naive_binomial` is kept
    only to show how much that correlation costs -- do not compare against it.
    """

    if not rows:
        raise ValueError("no rows to summarize")
    check_keys_unique(rows)
    by_question: dict[Any, list[bool]] = defaultdict(list)
    for row in rows:
        by_question[(row["task"], row["example_id"])].append(bool(row["final_score"]))

    per_question = [sum(values) / len(values) for values in by_question.values()]
    questions = len(per_question)
    score = sum(per_question) / questions
    if questions > 1:
        stderr = statistics.stdev(per_question) / math.sqrt(questions)
    else:
        stderr = float("nan")

    total = len(rows)
    correct = sum(bool(row["final_score"]) for row in rows)
    naive = math.sqrt(max(score * (1.0 - score), 0.0) / total)
    if not math.isfinite(score):
        raise RuntimeError("score is non-finite")
    return {
        "score": score,
        "stderr": stderr,
        "stderr_naive_binomial": naive,
        "questions": questions,
        "samples_per_question": total // questions,
        "n": total,
        "correct": correct,
        "step_size": 1.0 / total,
        "rule_passes": sum(bool(row.get("rule_score")) for row in rows),
        "length_clipped": sum(bool(row.get("length_clipped")) for row in rows),
    }


def smoke() -> None:
    rows = [
        {
            "task": "math500",
            "example_id": question,
            "seed": seed,
            "final_score": (question + seed) % 2 == 0,
            "rule_score": (question + seed) % 2 == 0,
        }
        for question in range(200)
        for seed in range(4)
    ]
    summary = summarize(rows)
    assert summary["questions"] == 200, summary
    assert summary["n"] == 800, summary
    assert summary["step_size"] == 1.0 / 800, summary
    assert abs(summary["score"] - 0.5) < 1e-9, summary
    # every question is exactly 2/4 here, so the clustered stderr is 0 while the
    # naive one is not -- the clearest possible demonstration that they differ
    assert summary["stderr"] == 0.0, summary
    assert summary["stderr_naive_binomial"] > 0.0, summary
    print("grade.py smoke passed")


if __name__ == "__main__":
    smoke()
