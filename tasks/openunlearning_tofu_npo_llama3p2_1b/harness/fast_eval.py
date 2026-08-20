"""Visible answer-NLL proxy over training rows; not the native TOFU final."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from runtime_guard import GpuTelemetry, checkpoint_read_lock

TASK_ID = "openunlearning_tofu_npo_llama3p2_1b"
ROWS_PER_SPLIT = 24


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_rows(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        payload = json.loads(text)
    else:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(payload, list):
        raise ValueError(f"expected a JSON array or JSONL rows in {path}")
    rows = []
    for row in payload[:ROWS_PER_SPLIT]:
        rows.append({"question": str(row["question"]), "answer": str(row["answer"])})
    if len(rows) != ROWS_PER_SPLIT:
        raise ValueError(f"expected at least {ROWS_PER_SPLIT} rows in {path}")
    return rows


def answer_nll(
    model_path: Path,
    rows: list[dict[str, str]],
    telemetry: GpuTelemetry,
) -> tuple[float, float]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.monotonic()
    telemetry.observe()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to("cuda")
    model.eval()
    losses: list[float] = []
    with torch.inference_mode():
        for row in rows:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["question"]}],
                tokenize=True, add_generation_prompt=True,
            )
            answer = tokenizer.encode(row["answer"], add_special_tokens=False)
            ids = (prompt + answer + [tokenizer.eos_token_id])[:512]
            labels = [-100] * min(len(prompt), len(ids)) + ids[len(prompt):]
            input_ids = torch.tensor([ids], device="cuda")
            label_ids = torch.tensor([labels], device="cuda")
            value = float(model(input_ids=input_ids, labels=label_ids).loss.float().cpu())
            if not math.isfinite(value):
                raise ValueError("non-finite proxy answer NLL")
            losses.append(value)
            telemetry.observe()
    del model
    torch.cuda.empty_cache()
    return sum(losses) / len(losses), time.monotonic() - started


def evaluate(checkpoint: Path, assets: Path, output: Path) -> dict[str, Any]:
    started = time.monotonic()
    telemetry = GpuTelemetry()
    start = assets / "models/training_start"
    data = assets / "data/train"
    forget = load_rows(data / "forget10.json")
    retain = load_rows(data / "retain90.json")
    try:
        with checkpoint_read_lock(checkpoint):
            start_forget, start_forget_sec = answer_nll(start, forget, telemetry)
            start_retain, start_retain_sec = answer_nll(start, retain, telemetry)
            candidate_forget, candidate_forget_sec = answer_nll(
                checkpoint,
                forget,
                telemetry,
            )
            candidate_retain, candidate_retain_sec = answer_nll(
                checkpoint,
                retain,
                telemetry,
            )
        result = {
            "schema_version": 1,
            "task_id": TASK_ID,
            "status": "passed",
            "metric": "proxy_answer_nll_tradeoff",
            "direction": "maximize",
            "proxy_only": True,
            "not_comparable_to": [
                "balanced_unlearning_score",
                "extraction_strength",
                "model_utility",
                "forget_quality",
            ],
            "contract": {
                "rows_per_split": ROWS_PER_SPLIT,
                "model_row_evaluations": 4 * ROWS_PER_SPLIT,
                "sampling": False,
                "samples_per_row": 1,
                "sequence_length_cap": 512,
                "stderr": None,
                "stderr_reason": "one deterministic answer-NLL pass; no repeated estimate",
                "clip_rate": None,
                "clip_rate_reason": "the proxy does not clip scores",
            },
            "rows_per_split": ROWS_PER_SPLIT,
            "training_start": {
                "forget_answer_nll": start_forget,
                "retain_answer_nll": start_retain,
            },
            "candidate": {
                "forget_answer_nll": candidate_forget,
                "retain_answer_nll": candidate_retain,
            },
            "forget_answer_nll_delta": candidate_forget - start_forget,
            "retain_answer_nll_delta": candidate_retain - start_retain,
            "proxy_answer_nll_tradeoff": (candidate_forget - start_forget)
            - max(0.0, candidate_retain - start_retain),
            "phase_runtime_seconds": {
                "training_start_forget": start_forget_sec,
                "training_start_retain": start_retain_sec,
                "candidate_forget": candidate_forget_sec,
                "candidate_retain": candidate_retain_sec,
            },
            "runtime_seconds": time.monotonic() - started,
            "gpu_telemetry": telemetry.summary(),
        }
        atomic_json(output, result)
        return result
    except Exception as exc:
        atomic_json(
            output,
            {
                "schema_version": 1,
                "task_id": TASK_ID,
                "status": "failed",
                "failure": f"{type(exc).__name__}: {exc}",
                "runtime_seconds": time.monotonic() - started,
            },
        )
        raise


def mock(output: Path) -> dict[str, Any]:
    result = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "passed",
        "metric": "proxy_answer_nll_tradeoff",
        "direction": "maximize",
        "proxy_only": True,
        "mock": True,
        "rows_per_split": ROWS_PER_SPLIT,
        "not_comparable_to": [
            "balanced_unlearning_score",
            "extraction_strength",
            "model_utility",
            "forget_quality",
        ],
        "proxy_answer_nll_tradeoff": 0.25,
    }
    atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--assets", type=Path, default=Path("/assets"))
    parser.add_argument("--output", type=Path, default=Path("/out/fast-eval.json"))
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    if args.mock:
        result = mock(args.output)
    else:
        if args.checkpoint is None:
            parser.error("--checkpoint is required unless --mock is used")
        result = evaluate(args.checkpoint, args.assets, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
