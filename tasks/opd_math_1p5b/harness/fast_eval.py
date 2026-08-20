"""End-to-end MATH-500 pass@1 on one GPU, for use inside the 4 h phase.

Same family as the hidden final, same grader, same prompt shape -- only the
question set and the sample count differ. It is not a surrogate metric: it
generates solutions and grades the answers, exactly as the final does.

The generation cap is fixed at 12288 because truncation is graded as wrong; changing
the cap would change the metric rather than merely its cost. The default evaluates all
500 MATH-500 questions at four samples each, which also removes question selection from
the seed so repeated runs differ only in sampling.

On B300 the full default evaluation measured 10.3 minutes. Three B300 runs on one
unchanged two-step checkpoint scored 0.810 / 0.812 / 0.824, with a seed-to-seed sd of
0.0076 and mean reported stderr of 0.0145. Output always carries `stderr`, clustered by
question rather than treating all generated rows as independent.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from grade import JUSTRL_COMMIT, grade_rows, summarize  # noqa: E402

TASK_ID = "opd_math_1p5b"
METRIC = "math500_pass_at_1"
# JustRL's own template, so a fast_eval prompt and a final prompt differ only in
# the question text.
PROMPT_TEMPLATE = (
    "{problem} Please reason step by step, and put your final answer within \\boxed{{}}."
)
# 0 means every question in the source. All 500 x 4 samples measured 10.3 minutes
# on B300 and takes question selection out of the seed.
DEFAULT_QUESTIONS = 0
DEFAULT_SAMPLES = 4
DEFAULT_SEED = 42
# Fixed as part of the metric. A truncated response is graded wrong, so changing
# this value changes the score definition.
DEFAULT_MAX_NEW_TOKENS = 12288
DEFAULT_MAX_MODEL_LEN = 13312
TEMPERATURE = 0.7
TOP_P = 0.9


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_questions(data: Path, questions: int, seed: int) -> list[dict[str, Any]]:
    """Read MATH-500 in JustRL's parquet schema.

    `questions == 0` means every question, which is the default: it costs 2.4
    minutes more than 200 and removes question selection from the seed, so two
    runs differ only in sampling. A smaller count draws a subset that is a
    deterministic function of `seed` and independent of the checkpoint, so two
    calls at the same seed remain comparable -- but two calls at *different*
    seeds then differ in both questions and samples.
    """

    import pandas as pd

    candidates = [data] if data.is_file() else sorted(data.glob("*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"no MATH-500 parquet under {data}")
    frame = pd.read_parquet(candidates[0])
    required = {"prompt", "reward_model"}
    if not required <= set(frame.columns):
        raise ValueError(f"unexpected MATH-500 schema: {sorted(frame.columns)}")
    if questions > len(frame):
        raise ValueError(f"asked for {questions} questions, source has {len(frame)}")

    if questions in (0, len(frame)):
        selected = list(range(len(frame)))
    else:
        import random

        order = list(range(len(frame)))
        random.Random(seed).shuffle(order)
        selected = sorted(order[:questions])

    rows: list[dict[str, Any]] = []
    for example_id, index in enumerate(selected):
        record = frame.iloc[index]
        problem = str(record["prompt"][0]["content"]).strip()
        answer = str(record["reward_model"]["ground_truth"]).strip()
        if not problem or not answer:
            raise ValueError(f"MATH-500 row {index} has an empty problem or answer")
        rows.append(
            {
                "task": "math500",
                "example_id": example_id,
                "source_index": index,
                "problem": problem,
                "answer": answer,
                "prompt": PROMPT_TEMPLATE.format(problem=problem),
            }
        )
    return rows


def load_stop_token_ids(model_path: Path) -> list[int]:
    for name in ("generation_config.json", "config.json"):
        path = model_path / name
        if not path.is_file():
            continue
        eos = json.loads(path.read_text(encoding="utf-8")).get("eos_token_id")
        if eos is None:
            continue
        values = eos if isinstance(eos, list) else [eos]
        result = list(dict.fromkeys(int(value) for value in values))
        if result:
            return result
    raise ValueError(f"no eos_token_id found under {model_path}")


def has_weights(directory: Path) -> bool:
    return any(directory.glob("*.safetensors")) or any(directory.glob("pytorch_model*.bin"))


def resolve_checkpoint(checkpoint: Path) -> Path:
    """Find the weights, given any of the three paths someone would plausibly pass.

    An HF directory, a `global_step_N` directory, or `/out/checkpoints` itself --
    the last takes the highest step, because that is what `Checkpoints land in
    /out/checkpoints` invites you to type.
    """

    if has_weights(checkpoint):
        return checkpoint
    for candidate in ("huggingface", "hf_model", "actor/huggingface"):
        nested = checkpoint / candidate
        if nested.is_dir() and has_weights(nested):
            return nested

    steps = sorted(
        (int(path.name.removeprefix("global_step_")), path)
        for path in checkpoint.glob("global_step_*")
        if path.name.removeprefix("global_step_").isdigit()
    )
    for _, directory in reversed(steps):
        for candidate in ("actor/huggingface", "huggingface", "hf_model"):
            nested = directory / candidate
            if nested.is_dir() and has_weights(nested):
                return nested

    raise FileNotFoundError(
        f"no model weights under {checkpoint}. Looked for *.safetensors here, in "
        "huggingface/ hf_model/ actor/huggingface/, and in the highest "
        "global_step_* below."
    )


def generate(
    model: Path,
    rows: list[dict[str, Any]],
    samples: int,
    seed: int,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
) -> list[dict[str, Any]]:
    from vllm import LLM, SamplingParams

    stop_ids = load_stop_token_ids(model)
    llm = LLM(
        model=str(model),
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=float(os.environ.get("FAST_EVAL_GPU_MEM_UTIL", "0.85")),
    )
    messages = [[{"role": "user", "content": row["prompt"]}] for row in rows]
    graded: list[dict[str, Any]] = []
    # One call per sample index rather than n=samples, so each pass has its own
    # seed and the same (question, seed) pair reproduces byte for byte.
    for sample_index in range(samples):
        sampling = SamplingParams(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=max_new_tokens,
            seed=seed + sample_index,
            stop_token_ids=stop_ids,
        )
        outputs = llm.chat(messages, sampling, use_tqdm=True)
        for row, output in zip(rows, outputs, strict=True):
            completion = output.outputs[0]
            graded.append(
                {
                    **row,
                    "seed": seed + sample_index,
                    "response": completion.text,
                    "completion_tokens": len(completion.token_ids),
                    "finish_reason": completion.finish_reason,
                    "length_clipped": completion.finish_reason == "length",
                }
            )
    return graded


def evaluate(
    checkpoint: Path,
    data: Path,
    out: Path,
    questions: int,
    samples: int,
    seed: int,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
) -> dict[str, Any]:
    model = resolve_checkpoint(checkpoint)
    rows = load_questions(data, questions, seed)
    started = time.monotonic()
    generated = generate(model, rows, samples, seed, max_new_tokens, max_model_len)
    grade_rows(generated)
    summary = summarize(generated)
    elapsed = time.monotonic() - started
    payload = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "metric": METRIC,
        "direction": "maximize",
        "checkpoint": str(checkpoint),
        "model": str(model),
        "justrl_commit": JUSTRL_COMMIT,
        "prompt_template": PROMPT_TEMPLATE,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_new_tokens": max_new_tokens,
        "clip_rate": summary["length_clipped"] / summary["n"],
        "seed": seed,
        "seconds": elapsed,
        **summary,
    }
    atomic_json(out, payload)
    rows_path = out.with_name(out.stem + "-rows.jsonl")
    rows_path.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in generated),
        encoding="utf-8",
    )
    return payload


def mock(out: Path, questions: int, samples: int, seed: int) -> dict[str, Any]:
    rows = [
        {
            "task": "math500",
            "example_id": question,
            "seed": seed + sample,
            "response": "",
            "answer": "",
            "rule_score": (question * 7 + sample) % 3 == 0,
            "final_score": (question * 7 + sample) % 3 == 0,
            "length_clipped": False,
        }
        for question in range(questions)
        for sample in range(samples)
    ]
    payload = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "metric": METRIC,
        "direction": "maximize",
        "mock": True,
        "seed": seed,
        "seconds": 0.0,
        **summarize(rows),
    }
    atomic_json(out, payload)
    return payload


def smoke() -> None:
    # 500 rather than DEFAULT_QUESTIONS, which is 0 for "all" and has no meaning
    # without a parquet to count.
    questions = 500
    payload = mock(Path("/tmp/fast_eval-smoke.json"), questions, DEFAULT_SAMPLES, DEFAULT_SEED)
    if payload["n"] != questions * DEFAULT_SAMPLES:
        raise RuntimeError(f"unexpected row count: {payload}")
    if payload["stderr"] <= 0.0:
        raise RuntimeError(f"stderr must be positive on varied data: {payload}")
    print(json.dumps({"fast_eval_smoke": "passed", "n": payload["n"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data", type=Path, default=Path("/assets/data/math500"))
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--questions",
        type=int,
        default=DEFAULT_QUESTIONS,
        help="0 (default) means all 500. A subset is cheaper but not by much, "
        "and it makes the seed choose questions as well as samples.",
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="generation cap. Watch clip_rate in the output: a high rate means the "
        "cap is deciding the score, since a truncated answer is graded wrong.",
    )
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--mock", action="store_true", help="grade synthetic rows, no GPU")
    parser.add_argument("--smoke", action="store_true", help="self-check and exit")
    args = parser.parse_args()
    if args.max_model_len is None:
        args.max_model_len = args.max_new_tokens + 1024

    if args.smoke:
        smoke()
        return
    if args.out is None:
        parser.error("--out is required")
    if args.mock:
        print(json.dumps(mock(args.out, args.questions, args.samples, args.seed), sort_keys=True))
        return
    if args.checkpoint is None:
        parser.error("--checkpoint is required outside --mock/--smoke")
    payload = evaluate(
        args.checkpoint.resolve(),
        args.data.resolve(),
        args.out.resolve(),
        args.questions,
        args.samples,
        args.seed,
        args.max_new_tokens,
        args.max_model_len,
    )
    print(
        json.dumps(
            {
                "score": payload["score"],
                "stderr": payload["stderr"],
                "n": payload["n"],
                "clip_rate": payload["clip_rate"],
                "seconds": payload["seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"fast_eval failed: {exc}", file=sys.stderr)
        raise
