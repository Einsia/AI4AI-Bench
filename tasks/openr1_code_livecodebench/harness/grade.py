"""The one evaluation path, shared by fast_eval and the hidden final.

Everything the LiveCodeBench protocol consists of lives here: which prompt
template, greedy decoding with n=1, the 2048-token cap, code extraction, and
execution against the official test cases through the pinned
`lcb_runner.evaluation.compute_code_generation_metrics.codegen_metrics`. The two
callers decide only *which rows* and *which checkpoint*; they cannot vary the
protocol, because none of it is theirs to pass.

That split is the point. "The proxy and the final are scored by the same
evaluator" is a claim you can check by reading one file, and the only difference
between a proxy score and a final score is the row list handed to `load_rows`.

The LiveCodeBench tree is baked into the image at /opt/harness/livecodebench
rather than mounted, for the same reason the sibling task bakes in its grader:
the Agent owns /workspace and nothing else, so an evaluator inside /opt/harness
cannot be edited by a candidate patch, and the score phase runs the image's copy
regardless.

Nothing here chooses a checkpoint or a row set. Keep it that way.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

# The pinned official evaluator, copied into the image by environment/Dockerfile.
LCB_ROOT = Path(os.environ.get("LCB_ROOT", "/opt/harness/livecodebench"))
LCB_COMMIT = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
# LiveCodeBench's own style enum entry for this model family. It selects the
# prompt wrapper and the code-fence convention `extract_code` expects, so it has
# to match between generation and extraction.
LM_STYLE_NAME = "CodeQwenInstruct"
# Protocol constants. These are evaluator settings, not tuning knobs.
MAX_NEW_TOKENS = 2048
# Per-program execution ceiling handed to the official evaluator, in seconds.
EXECUTION_TIMEOUT = 6
# The official evaluator's process pool for running generated programs.
EVALUATION_PROCESSES = 16


def configure_ephemeral_multiprocessing_runtime(root: Path = Path("/dev/shm")) -> Path:
    """Put the evaluator's temporaries in container-private tmpfs before it forks.

    LiveCodeBench runs each generated program in its own process and opens a
    multiprocessing Manager socket to collect results. /tmp here is a 256 MiB
    tmpfs and the score phase runs with a read-only root filesystem, so the
    default location is both too small and, on some paths, unwritable. /dev/shm
    is a separate docker-managed tmpfs sized by [x-ai4ai.container].shm_size and
    stays writable under --read-only.

    Both sources of truth are updated: `tempfile` caches its chosen directory on
    first use, so setting only TMPDIR leaves already-imported code pointing at
    /tmp. Carried over from the old evaluator, where fixing one and not the other
    is exactly the bug that was hit.
    """

    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"multiprocessing tmpfs root is invalid: {root}")
    runtime = root / f"openr1-lcb-{os.getpid()}"
    runtime.mkdir(mode=0o700, exist_ok=True)
    os.environ["TMPDIR"] = str(runtime)
    tempfile.tempdir = str(runtime)
    return runtime


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def row_order_key(row: dict[str, Any]) -> str:
    """sha256 of the question id, which is the frozen row order.

    Not a shuffle with a seed: the order has to be reproducible from the data
    alone, because the 64-row proxy and the 204-row confirmation slice are
    defined as positions in it. Changing this function silently redefines both.
    """

    return hashlib.sha256(str(row["question_id"]).encode()).hexdigest()


def load_rows(
    data: Path, files: tuple[str, ...], *, offset: int = 0, count: int = 0
) -> list[dict[str, Any]]:
    """Read the named release files, order them by hash, and take a slice.

    `count == 0` means "everything from offset on". The caller names the files
    because that is the one thing a proxy and the final differ in.
    """

    missing = [name for name in files if not (data / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"LiveCodeBench release file(s) {missing} are not under {data}. "
            f"Present: {sorted(path.name for path in data.glob('*.jsonl')) or 'nothing'}"
        )
    rows: list[dict[str, Any]] = []
    for name in files:
        with (data / name).open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    if not rows:
        raise ValueError(f"no rows read from {files} under {data}")

    # Two release files could in principle carry the same question, which would
    # make the hash order ambiguous and quietly move both slices. Measured on the
    # real export the rate is zero; this keeps it zero rather than assuming it.
    seen: dict[Any, int] = {}
    for row in rows:
        seen[row["question_id"]] = seen.get(row["question_id"], 0) + 1
    repeated = sorted(key for key, total in seen.items() if total > 1)
    if repeated:
        raise ValueError(
            f"{len(repeated)} question_id(s) appear in more than one of {files}, "
            f"e.g. {repeated[:3]}. The hash order, and therefore every row slice, "
            "is not well defined on a set with duplicates."
        )

    rows.sort(key=row_order_key)
    if offset < 0 or count < 0:
        raise ValueError(f"offset and count must not be negative, got {offset}, {count}")
    if offset >= len(rows):
        raise ValueError(f"offset {offset} is past the {len(rows)} available rows")
    selected = rows[offset:] if count == 0 else rows[offset : offset + count]
    if count and len(selected) != count:
        raise ValueError(
            f"asked for {count} rows from offset {offset}, got {len(selected)}; "
            f"{files} holds {len(rows)}"
        )
    return selected


def load_evaluator() -> dict[str, Any]:
    """Import the pinned LiveCodeBench modules, and nothing else it ships.

    `lcb_runner.prompts` executes provider-specific imports -- Anthropic and
    friends -- that an offline single-model run has no use for and that are not
    installed. The pinned code-generation prompt module is loaded directly by
    path to keep the dependency set to what this protocol needs.

    That module also opens its few-shot examples relative to the process working
    directory, so the working directory is set here rather than left to whatever
    launched the evaluator. Every path a caller passes must already be absolute;
    fast_eval.py and final_eval.py resolve theirs before calling in.
    """

    if not (LCB_ROOT / "lcb_runner").is_dir():
        raise FileNotFoundError(
            f"the pinned LiveCodeBench tree is missing from {LCB_ROOT}. It is baked "
            "into the image by environment/Dockerfile, not mounted, so an image "
            "built without the lcb_source build context will fail here."
        )
    root = str(LCB_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
    from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
    from lcb_runner.lm_styles import LMStyle
    from lcb_runner.utils.extraction_utils import extract_code

    prompt_path = LCB_ROOT / "lcb_runner/prompts/code_generation.py"
    os.chdir(LCB_ROOT)
    spec = importlib.util.spec_from_file_location("lcb_codegen_prompt", prompt_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the pinned LiveCodeBench prompt module: {prompt_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return {
        "problem_class": CodeGenerationProblem,
        "codegen_metrics": codegen_metrics,
        "extract_code": extract_code,
        "lm_style": getattr(LMStyle, LM_STYLE_NAME),
        "format_prompt": module.format_prompt_generation,
    }


def build_prompts(
    evaluator: dict[str, Any], rows: list[dict[str, Any]]
) -> tuple[list[Any], list[str]]:
    problems = [evaluator["problem_class"](**row) for row in rows]
    prompts = [evaluator["format_prompt"](problem, evaluator["lm_style"]) for problem in problems]
    return problems, prompts


def generate(
    model_path: Path, prompts: list[str], max_new_tokens: int = MAX_NEW_TOKENS
) -> tuple[list[str], list[int]]:
    """Greedy, n=1, one prompt per forward pass.

    One prompt at a time avoids padding/batch-neighbour changes to greedy decoding.

    `do_sample=False` with temperature/top_p/top_k explicitly None: transformers
    warns and, on some versions, silently keeps a sampling default if they are
    left set.
    """

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, use_fast=True, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda",
    )
    model.eval()

    generations: list[str] = []
    generated_tokens: list[int] = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        generated_tokens.append(int(new_tokens.shape[0]))
        generations.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return generations, generated_tokens


def row_passed(tests: Any) -> bool:
    """LiveCodeBench's own pass rule: every test case strictly greater than zero.

    `> 0`, not truthiness, and the difference is not cosmetic. The upstream
    evaluator writes a numeric code per test case and encodes failures as
    *negative* numbers -- -1 for a runtime error, -2 for a timeout -- so
    `bool(-1)` is True and a truthiness test scores a crashing program as a pass.
    Upstream's own aggregation is `np.all(np.array(generation) > 0)`; this mirrors
    it so the per-row records and the reported pass@1 cannot disagree.

    `summarize` cross-checks the two anyway. That check is what surfaced this.
    """

    # results[i] is a list over generations; n=1 here, so entry 0 is this row's
    # list of per-test-case codes.
    outcomes = tests[0] if isinstance(tests, list) and tests else tests
    if not isinstance(outcomes, list):
        outcomes = [outcomes]
    if not outcomes:
        return False
    for value in outcomes:
        if isinstance(value, bool):
            if not value:
                return False
        elif isinstance(value, (int, float)):
            if not value > 0:
                return False
        else:
            # A string or None here means the upstream result shape changed, and
            # guessing at it would silently move the score.
            raise TypeError(f"unexpected LiveCodeBench test result {value!r} in {outcomes!r}")
    return True


def execute(
    evaluator: dict[str, Any], problems: list[Any], generations: list[str]
) -> tuple[list[dict[str, Any]], float]:
    """Extract code, run it against the official tests, return per-row records.

    The returned pass@1 is the official evaluator's own aggregate rather than a
    recount of the rows, so the reported number is the one LiveCodeBench computes.
    `summarize` re-derives a count from the records and refuses a disagreement.
    """

    extracted = [evaluator["extract_code"](text, evaluator["lm_style"]) for text in generations]
    samples = [problem.get_evaluation_sample() for problem in problems]
    configure_ephemeral_multiprocessing_runtime()
    metrics, raw_results, metadata = evaluator["codegen_metrics"](
        samples,
        [[code] for code in extracted],
        k_list=[1],
        num_process_evaluate=min(EVALUATION_PROCESSES, len(samples)),
        timeout=EXECUTION_TIMEOUT,
        debug=False,
    )
    records: list[dict[str, Any]] = []
    for index, (problem, generation, code) in enumerate(
        zip(problems, generations, extracted, strict=True)
    ):
        tests = raw_results[index]
        passed = row_passed(tests)
        records.append(
            {
                "question_id": getattr(problem, "question_id", None),
                "example_id": index,
                "extracted_code": code,
                "extracted": bool(code.strip()),
                "completion": generation,
                "tests": tests,
                "metadata": metadata[index],
                "final_score": passed,
            }
        )
    return records, float(metrics["pass@1"])


def summarize(records: list[dict[str, Any]], reported: float | None = None) -> dict[str, Any]:
    """Aggregate graded rows.

    The standard error is a descriptive binomial quantity over problem outcomes.
    It is not a paired interval and does not include training-seed or replay
    variance.
    """

    if not records:
        raise ValueError("no rows to summarize")
    total = len(records)
    correct = sum(bool(row["final_score"]) for row in records)
    score = correct / total
    stderr = math.sqrt(max(score * (1.0 - score), 0.0) / total)
    if not math.isfinite(score):
        raise RuntimeError("score is non-finite")
    if reported is not None and abs(reported - score) > 1e-9:
        raise ValueError(
            f"the official evaluator reported pass@1 {reported!r} while the rows it "
            f"returned count {correct}/{total} = {score!r}. One of the two is being "
            "read wrong; the score is not trustworthy until they agree."
        )
    token_counts = [int(row.get("generated_tokens", 0)) for row in records]
    return {
        "score": score,
        "stderr": stderr,
        "stderr_kind": "binomial_descriptive_not_seed_or_paired_uncertainty",
        "n": total,
        "correct": correct,
        "step_size": 1.0 / total,
        # A row with no fenced code block cannot pass, so this separates "wrote a
        # wrong program" from "did not write a program". A collapse shows up here
        # first, well before the score moves.
        "extracted": sum(bool(row.get("extracted")) for row in records),
        "extracted_rate": sum(bool(row.get("extracted")) for row in records) / total,
        "mean_generated_tokens": sum(token_counts) / total,
        "length_clipped": sum(
            bool(row.get("length_clipped", token_count >= MAX_NEW_TOKENS))
            for row, token_count in zip(records, token_counts, strict=True)
        ),
    }


def smoke() -> None:
    # The pass rule first, because a truthiness test here scores errors as passes.
    assert row_passed([[1, 1, 1]]) is True
    assert row_passed([[True, True]]) is True
    assert row_passed([[1, 0, 1]]) is False
    assert row_passed([[1, -1]]) is False, "a runtime error (-1) must not count as a pass"
    assert row_passed([[-2]]) is False, "a timeout (-2) must not count as a pass"
    assert row_passed([[True, False]]) is False
    assert row_passed([[]]) is False
    try:
        row_passed([["passed"]])
    except TypeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("an unrecognized result shape was silently graded")

    records = [
        {"question_id": f"q{index}", "final_score": index % 4 == 0, "extracted": index % 8 != 7}
        for index in range(128)
    ]
    summary = summarize(records)
    assert summary["n"] == 128, summary
    assert summary["correct"] == 32, summary
    assert abs(summary["score"] - 0.25) < 1e-12, summary
    assert summary["step_size"] == 1.0 / 128, summary
    assert summary["stderr"] > 0.0, summary
    assert summary["extracted"] == 112, summary
    # The count and the evaluator's own aggregate have to agree, or the score is
    # being read out of the wrong place.
    try:
        summarize(records, reported=0.5)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("a disagreeing official aggregate was accepted")
    print("grade.py smoke passed")


if __name__ == "__main__":
    smoke()
