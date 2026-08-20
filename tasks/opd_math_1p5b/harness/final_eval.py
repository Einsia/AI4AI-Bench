"""Hidden final: JustRL AIME24/25 @32, checkpoint only.

Carried over from the trusted judge-side evaluator. Three changes:

1. The 13 `training_metadata.json` field checks are gone. They enforced that the
   candidate was still the frozen k1 recipe -- a semantic constraint that v1
   drops. What is still checked is that a checkpoint has weights.
2. **One B300 by default**, the same device count the training phases use. Each
   generation call carries four seeds x all 60 questions, so the formal batch width
   is 240. B300 final evaluations at this fixed width have taken 41-54 minutes.

   Batch composition is held fixed because changing it can perturb sampled tokens.
   Multi-device execution remains an operator diagnostic, not a score-comparable
   substitute for the declared one-device protocol.
3. Grading goes through harness/grade.py, the same entry point fast_eval uses.

Question sources are pinned public snapshots with hash checks -- AI-MO's AIME
2024 parquet and opencompass's AIME 2025 jsonl -- not the JustRL tar. Switching
sources would silently move the score; the B300 baselines are recorded outside
the evaluator in the task metadata and Agent brief.

The Agent never sees this file's inputs at runtime, and the reason is the mount
list rather than any file permission: /assets/data/aime is mounted into this
container and not into the exploration one. Nothing here relies on the asset tree
being unreadable to the account running the benchmark -- a boundary made of chmod
bits would only hold on a host configured a particular way, and would not survive
someone cloning this repository.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path
from typing import Any


def _redirect_caches(root: str = "/out") -> None:
    """Move the compiler caches off /tmp before anything imports torch or vllm.

    /tmp is a 256 MiB tmpfs. Triton writes a compiled cuda_utils .so into
    /tmp/.triton and then mmaps it; when the tmpfs is too small the mmap fails
    with "failed to map segment from shared object" and the vLLM engine dies
    during KV-cache init. The message names an ImportError, so it reads like a
    packaging problem rather than a full filesystem.

    run.sh and fast_eval.sh already do this. This file needs it too, because the
    orchestrator invokes it directly rather than through a wrapper -- which is exactly
    how it got missed.

    HOME is the one that generalises, and it was missing until a fourth library hit the
    same wall. HOME is /tmp in this image, so anything defaulting to ~/.cache/<name>
    lands in the tmpfs; flashinfer, which must JIT-compile on Blackwell because no
    prebuilt cubin ships for sm103a, died in nvcc with "No space left on device" writing
    to /tmp/.cache/flashinfer. Triton, inductor and vLLM were each given their own
    variable while HOME stayed pointed at the tmpfs, so every further library was the
    next outage. The named variables below are still worth setting: some libraries read
    them and ignore HOME.
    """

    for name, relative in (
        ("HOME", "tmp/home"),
        ("XDG_CACHE_HOME", "tmp/cache"),
        ("TMPDIR", "tmp"),
        ("TRITON_CACHE_DIR", "tmp/triton"),
        ("TORCHINDUCTOR_CACHE_DIR", "tmp/inductor"),
        ("VLLM_CACHE_ROOT", "tmp/vllm"),
        ("FLASHINFER_CACHE_DIR", "tmp/flashinfer"),
        ("FLASHINFER_WORKSPACE_BASE", "tmp/flashinfer"),
    ):
        path = Path(os.environ.get(name, f"{root}/{relative}"))
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            # A read-only or absent /out means this is a --smoke or --mock run,
            # which never reaches a compiler.
            continue
        os.environ[name] = str(path)


_redirect_caches()

sys.path.insert(0, str(Path(__file__).resolve().parent))

from grade import JUSTRL_COMMIT, load_cv_prompt, load_rule_grader, summarize  # noqa: E402

TASK_ID = "opd_math_1p5b"
METRIC = "aime24_25_at32"
AIME24_REVISION = "13f9e12f613e720c2a2b2f345dd04b998a29494d"
AIME24_DATA_SHA256 = "025484a99fea498e7d0c3b0ee42afcbec0176405c19c5dbf557b9f6ca6445675"
AIME25_REVISION = "a6ad95f611d72cf628a80b58bd0432ef6638f958"
AIME25_I_SHA256 = "b91b3c96f05d9635d2a0692b124ebe023c1ff59cb19c074275e6c4b349d0659e"
AIME25_II_SHA256 = "16a2dcfbbf9db1b11f8a69a3ba5e4cac73e3641b19a37e2307e9c12240bbed5e"
JUSTRL_MAX_MODEL_LEN = 32768
JUSTRL_MAX_NEW_TOKENS = 31744
PROMPT_TEMPLATE = (
    "{problem} Please reason step by step, and put your final answer within \\boxed{{}}."
)
EXPECTED_QUESTIONS = 60
EXPECTED_SAMPLES_PER_QUESTION = 32
EXPECTED_ROWS = EXPECTED_QUESTIONS * EXPECTED_SAMPLES_PER_QUESTION
GENERATION_SEED = 42
TEMPERATURE = 0.7
TOP_P = 0.9
# Seeds per llm.chat call. 4 x 60 questions = 240 prompts, against vLLM's reported
# capacity of about 75 concurrent full-length requests -- enough queued work to
# keep the batch saturated. Going wider adds queued-request memory without adding
# throughput, since the limit is concurrency, not submission size.
SEEDS_PER_CALL = 4
REWARD_PATH = Path("/logs/verifier/reward.txt")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def weight_sha256(checkpoint: Path) -> str:
    files = sorted(
        {
            path
            for pattern in ("*.safetensors", "pytorch_model*.bin")
            for path in checkpoint.glob(pattern)
        }
    )
    if not files:
        raise RuntimeError(f"checkpoint has no model weights: {checkpoint}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def has_weights(directory: Path) -> bool:
    return any(directory.glob("*.safetensors")) or any(directory.glob("pytorch_model*.bin"))


def resolve_checkpoint(checkpoint: Path) -> Path:
    """Same resolution as fast_eval, so a path that scores there scores here.

    Mounting a `checkpoints/` root and letting this pick the highest step would be
    a mistake for the final -- the operator should name the checkpoint being
    scored -- but accepting `global_step_N` matters, since that is what the retrain phasepy
    reports.
    """

    if has_weights(checkpoint):
        return checkpoint
    for candidate in ("actor/huggingface", "huggingface", "hf_model"):
        nested = checkpoint / candidate
        if nested.is_dir() and has_weights(nested):
            return nested
    raise FileNotFoundError(
        f"no model weights under {checkpoint}; expected either an HF directory or "
        "a global_step_N holding actor/huggingface"
    )


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


def load_tasks(aime_root: Path) -> dict[str, list[dict[str, Any]]]:
    """Read the two pinned AIME snapshots, refusing any content drift."""

    import pandas as pd

    aime24_data = aime_root / "aime2024/data/train-00000-of-00001.parquet"
    if not aime24_data.is_file():
        raise FileNotFoundError(aime24_data)
    if file_sha256(aime24_data) != AIME24_DATA_SHA256:
        raise ValueError("pinned AIME 2024 data hash mismatch")
    frame = pd.read_parquet(aime24_data)
    if len(frame) != 90 or not {"problem", "answer", "url"} <= set(frame.columns):
        raise ValueError("pinned AIME 2024 schema or row count mismatch")
    selected = frame[frame["url"].astype(str).str.contains("2024", regex=False)]
    if len(selected) != 30:
        raise ValueError("pinned AIME 2024 source does not select exactly 30 rows")

    tasks: dict[str, list[dict[str, Any]]] = {"aime24": [], "aime25": []}
    for example_id, row in enumerate(selected.itertuples(index=False)):
        problem = str(row.problem).strip()
        answer = str(row.answer).strip()
        if not problem or not answer:
            raise ValueError("pinned AIME 2024 contains an empty problem or answer")
        tasks["aime24"].append(
            {
                "task": "aime24",
                "example_id": example_id,
                "problem": problem,
                "answer": answer,
                "prompt": PROMPT_TEMPLATE.format(problem=problem),
            }
        )

    for name, expected_hash in (
        ("aime2025-I.jsonl", AIME25_I_SHA256),
        ("aime2025-II.jsonl", AIME25_II_SHA256),
    ):
        path = aime_root / "aime2025" / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if file_sha256(path) != expected_hash:
            raise ValueError(f"pinned AIME 2025 data hash mismatch: {name}")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if len(rows) != 15:
            raise ValueError(f"pinned AIME 2025 row count mismatch: {name}")
        for row in rows:
            problem = str(row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip()
            if not problem or not answer:
                raise ValueError(f"pinned AIME 2025 contains an empty field: {name}")
            tasks["aime25"].append(
                {
                    "task": "aime25",
                    "example_id": len(tasks["aime25"]),
                    "problem": problem,
                    "answer": answer,
                    "prompt": PROMPT_TEMPLATE.format(problem=problem),
                }
            )

    counts = {name: len(rows) for name, rows in tasks.items()}
    if counts != {"aime24": 30, "aime25": 30}:
        raise ValueError(f"pinned AIME sources do not hold 60 questions: {counts}")
    return tasks


def generation_worker(job: tuple[Any, ...]) -> None:
    """Submit all 60 questions for several seeds at once, one SamplingParams each.

    The previous shape submitted only 30 prompts at a time. The B300 protocol instead
    queues four seeds x all 60 questions, a fixed width of 240.

    Widening the batch does perturb individual samples: batched matmuls reduce in a
    different order and temperature sampling turns a last-bit difference into a
    different token. Measured on fast_eval, a 6x width change gave 1/40
    byte-identical responses -- and 39/40 identical *grades*. Scaled to 1920 rows
    that is on the order of +/-0.025 on the score, which sits inside the metric's
    own error bar.

    The task therefore fixes the B300 comparison protocol: the same 60 questions,
    32 samples, prompt template, grader, temperature, top_p, token ceiling and batch
    width. A differently batched diagnostic is not directly score-comparable.
    """

    model_path, tasks, seeds, gpu_id, shard_index, shard_dir, stop_ids = job
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=JUSTRL_MAX_MODEL_LEN,
        gpu_memory_utilization=float(os.environ.get("FINAL_GPU_MEM_UTIL", "0.9")),
    )

    # One flat list of all 60 questions, tagged with the task they came from.
    flat = [(task_name, row) for task_name, rows in tasks.items() for row in rows]
    per_call = int(os.environ.get("FINAL_SEEDS_PER_CALL", str(SEEDS_PER_CALL)))
    output_path = Path(shard_dir) / f"shard-{shard_index:02d}.jsonl"

    with output_path.open("w", encoding="utf-8") as output:
        for start in range(0, len(seeds), per_call):
            chunk = seeds[start : start + per_call]
            messages: list[Any] = []
            params: list[Any] = []
            meta: list[tuple[str, dict[str, Any], int]] = []
            for seed in chunk:
                for task_name, row in flat:
                    messages.append([{"role": "user", "content": row["prompt"]}])
                    params.append(
                        SamplingParams(
                            temperature=TEMPERATURE,
                            top_p=TOP_P,
                            max_tokens=JUSTRL_MAX_NEW_TOKENS,
                            seed=seed,
                            stop_token_ids=stop_ids,
                        )
                    )
                    meta.append((task_name, row, seed))

            generations = llm.chat(messages, params, use_tqdm=True)
            for (task_name, row, seed), generation in zip(meta, generations, strict=True):
                completion = generation.outputs[0]
                output.write(
                    json.dumps(
                        {
                            **row,
                            "task": task_name,
                            "seed": seed,
                            "response": completion.text,
                            "completion_tokens": len(completion.token_ids),
                            "finish_reason": completion.finish_reason,
                            "length_clipped": completion.finish_reason == "length",
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
            output.flush()


def generate(
    model: Path,
    tasks: dict[str, list[dict[str, Any]]],
    output: Path,
    gpu_ids: list[str],
    seeds: list[int],
) -> Path:
    """Shard by seed across whatever devices exist.

    Each `llm.chat` call carries SEEDS_PER_CALL seeds x 60 questions, and the seed
    rides on the per-request SamplingParams. Changing the device count also changes
    seed grouping, so only runs with the same device count and batch width are
    score-comparable; the formal task uses one device.
    """

    seed_shards: list[list[int]] = [[] for _ in gpu_ids]
    for index, seed in enumerate(seeds):
        seed_shards[index % len(gpu_ids)].append(seed)
    shard_dir = output / "generation_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    stop_ids = load_stop_token_ids(model)
    jobs = [
        (str(model), tasks, shard_seeds, gpu_id, shard_index, str(shard_dir), stop_ids)
        for shard_index, (gpu_id, shard_seeds) in enumerate(zip(gpu_ids, seed_shards, strict=True))
        if shard_seeds
    ]
    if len(jobs) == 1:
        generation_worker(jobs[0])
    else:
        context = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(jobs), mp_context=context
        ) as executor:
            for future in concurrent.futures.as_completed(
                [executor.submit(generation_worker, job) for job in jobs]
            ):
                future.result()
    merged = output / "generations.jsonl"
    with merged.open("w", encoding="utf-8") as destination:
        for path in sorted(shard_dir.glob("*.jsonl")):
            destination.write(path.read_text(encoding="utf-8"))
    return merged


def split_indices(indices: list[int], gpu_ids: list[str]) -> list[list[int]]:
    if not gpu_ids:
        raise ValueError("at least one GPU is required")
    shards: list[list[int]] = [[] for _ in gpu_ids]
    for position, index in enumerate(indices):
        shards[position % len(gpu_ids)].append(index)
    return shards


def fallback_worker(job: tuple[str, str, list[int], str]) -> list[tuple[int, str]]:
    generations_path, verifier_path, indices, gpu_id = job
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    index_set = set(indices)
    selected: dict[int, dict[str, Any]] = {}
    with Path(generations_path).open(encoding="utf-8") as source:
        for index, line in enumerate(source):
            if index in index_set:
                selected[index] = json.loads(line)
    records = [selected[index] for index in indices]

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(verifier_path, local_files_only=True)
    verifier = LLM(model=verifier_path, tensor_parallel_size=1)
    cv_prompt = load_cv_prompt()
    prompts = [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": cv_prompt.format(
                        question="",
                        gold_answer=record["answer"],
                        llm_response=record["response"],
                    ),
                }
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        for record in records
    ]
    outputs = verifier.generate(
        prompts, SamplingParams(temperature=0.0, max_tokens=2048), use_tqdm=True
    )
    return [
        (index, result.outputs[0].text.strip())
        for index, result in zip(indices, outputs, strict=True)
    ]


def grade(generations: Path, verifier: Path, output: Path, gpu_ids: list[str]) -> dict[str, Any]:
    """Rule grader first, CompassVerifier only on the rows it rejects."""

    rule_grader = load_rule_grader()
    records = [
        json.loads(line)
        for line in generations.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    fallback_indices: list[int] = []
    for index, record in enumerate(records):
        rule_score = bool(rule_grader(record["response"], record["answer"]))
        record["rule_score"] = rule_score
        record["final_score"] = rule_score
        if not rule_score:
            fallback_indices.append(index)

    if fallback_indices:
        jobs = [
            (str(generations), str(verifier), indices, gpu_id)
            for gpu_id, indices in zip(
                gpu_ids, split_indices(fallback_indices, gpu_ids), strict=True
            )
            if indices
        ]
        if len(jobs) == 1:
            judgments = fallback_worker(jobs[0])
        else:
            context = mp.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=len(jobs), mp_context=context
            ) as executor:
                judgments = [
                    item for result in executor.map(fallback_worker, jobs) for item in result
                ]
        for index, judgment in judgments:
            records[index]["verifier_judgment"] = judgment
            records[index]["final_score"] = judgment == "A"

    (output / "graded_generations.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=True) + "\n" for record in records),
        encoding="utf-8",
    )
    aggregate = summarize(records)
    if aggregate["n"] != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, graded {aggregate['n']}")
    by_task: dict[str, dict[str, Any]] = {}
    for task in ("aime24", "aime25"):
        rows = [row for row in records if row["task"] == task]
        correct = sum(bool(row["final_score"]) for row in rows)
        by_task[task] = {
            "correct": correct,
            "total": len(rows),
            "avg_at_32": correct / len(rows) if rows else 0.0,
        }
    aggregate["datasets"] = by_task
    aggregate["compass_fallbacks"] = len(fallback_indices)
    return aggregate


def write_reward(score: float, reward_path: Path) -> None:
    """Harbor's verifier contract: the scalar lands in /logs/verifier/reward."""

    reward_path.parent.mkdir(parents=True, exist_ok=True)
    reward_path.write_text(f"{score:.10f}\n", encoding="utf-8")


def evaluate(
    checkpoint: Path,
    assets: Path,
    output: Path,
    gpu_count: int | None,
    reward_path: Path,
) -> dict[str, Any]:
    model = resolve_checkpoint(checkpoint)
    verifier = assets / "models/verifier"
    if not (verifier / "config.json").is_file():
        raise FileNotFoundError(verifier / "config.json")
    output.mkdir(parents=True, exist_ok=True)

    import torch

    available = torch.cuda.device_count()
    if available == 0:
        raise RuntimeError("the final needs at least one GPU")
    # One device by default, the same as the 4 h and 12 h phases. Not "all
    # visible": a judge host with eight free devices should not silently change
    # what the task costs.
    count = 1 if gpu_count is None else min(gpu_count, available)
    gpu_ids = [str(index) for index in range(count)]

    tasks = load_tasks(assets / "data/aime")
    seeds = random.Random(GENERATION_SEED).sample(range(2**31 - 1), EXPECTED_SAMPLES_PER_QUESTION)
    weights_hash = weight_sha256(model)
    atomic_json(
        output / "resolved_config.json",
        {
            "schema_version": 1,
            "task_id": TASK_ID,
            "checkpoint": str(checkpoint),
            "model": str(model),
            "checkpoint_weight_sha256": weights_hash,
            "justrl_commit": JUSTRL_COMMIT,
            "aime24_source": "AI-MO/aimo-validation-aime",
            "aime24_revision": AIME24_REVISION,
            "aime25_source": "opencompass/AIME2025",
            "aime25_revision": AIME25_REVISION,
            "prompt_template": PROMPT_TEMPLATE,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_model_len": JUSTRL_MAX_MODEL_LEN,
            "max_new_tokens": JUSTRL_MAX_NEW_TOKENS,
            "samples_per_prompt": len(seeds),
            "seeds": seeds,
            "generation_seed": GENERATION_SEED,
            "gpu_ids": gpu_ids,
            "image_digest": os.environ.get("IMAGE_DIGEST"),
        },
    )

    started = time.monotonic()
    generations = generate(model, tasks, output, gpu_ids, seeds)
    aggregate = grade(generations, verifier, output, gpu_ids)
    summary = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "passed",
        "benchmark": "JustRL AIME24/25 @32",
        "metric": METRIC,
        "direction": "maximize",
        "metrics": {METRIC: aggregate["score"]},
        **aggregate,
        "wall_seconds": time.monotonic() - started,
        "offline": True,
        "checkpoint_weight_sha256": weights_hash,
    }
    atomic_json(output / "summary.json", summary)
    write_reward(aggregate["score"], reward_path)
    return summary


def mock(output: Path, reward_path: Path) -> dict[str, Any]:
    records = [
        {
            "task": task,
            "example_id": example_id,
            "seed": seed,
            "rule_score": (example_id + seed) % 3 == 0,
            "final_score": (example_id + seed) % 3 == 0,
            "length_clipped": False,
        }
        for task in ("aime24", "aime25")
        for example_id in range(30)
        for seed in range(EXPECTED_SAMPLES_PER_QUESTION)
    ]
    output.mkdir(parents=True, exist_ok=True)
    aggregate = summarize(records)
    if aggregate["n"] != EXPECTED_ROWS:
        raise RuntimeError(f"mock produced {aggregate['n']} rows, expected {EXPECTED_ROWS}")
    # Same per-task breakdown the real path adds in grade(), so --mock exercises
    # the output shape a consumer will actually parse.
    aggregate["datasets"] = {
        task: {
            "correct": sum(bool(row["final_score"]) for row in records if row["task"] == task),
            "total": sum(1 for row in records if row["task"] == task),
            "avg_at_32": sum(bool(row["final_score"]) for row in records if row["task"] == task)
            / sum(1 for row in records if row["task"] == task),
        }
        for task in ("aime24", "aime25")
    }
    aggregate["compass_fallbacks"] = 0
    (output / "graded_generations.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "passed",
        "metric": METRIC,
        "direction": "maximize",
        "metrics": {METRIC: aggregate["score"]},
        **aggregate,
        "mock": True,
    }
    atomic_json(output / "summary.json", summary)
    write_reward(aggregate["score"], reward_path)
    return summary


def smoke() -> None:
    records = [
        {
            "task": task,
            "example_id": example_id,
            "seed": seed,
            "rule_score": True,
            "final_score": True,
            "length_clipped": False,
        }
        for task in ("aime24", "aime25")
        for example_id in range(30)
        for seed in range(EXPECTED_SAMPLES_PER_QUESTION)
    ]
    summary = summarize(records)
    if summary["score"] != 1.0 or summary["n"] != EXPECTED_ROWS:
        raise RuntimeError(f"unexpected final smoke result: {summary}")
    print(json.dumps({"final_eval_smoke": "passed", "n": summary["n"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--assets", type=Path, default=Path("/assets"))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="devices to shard the 32 seeds over. Default 1, matching the training phases. "
        "More is only faster -- each seed is its own batch and vLLM seeds per request.",
    )
    parser.add_argument("--reward-path", type=Path, default=REWARD_PATH)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        smoke()
        return
    if args.output is None:
        parser.error("--output is required")
    if args.mock:
        print(json.dumps(mock(args.output.resolve(), args.reward_path), sort_keys=True))
        return
    if args.checkpoint is None:
        parser.error("--checkpoint is required outside --mock/--smoke")
    print(
        json.dumps(
            evaluate(
                args.checkpoint.resolve(),
                args.assets.resolve(),
                args.output.resolve(),
                args.gpus,
                args.reward_path,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"OPD final failed: {exc}", file=sys.stderr)
        raise
