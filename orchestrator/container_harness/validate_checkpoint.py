#!/usr/bin/env python3
"""Frozen task-specific checkpoint loadability gate used before top-three selection."""

from __future__ import annotations

import argparse
import dataclasses
import errno
import importlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any


INFRASTRUCTURE_MARKERS = (
    "cuda out of memory", "cuda driver", "no cuda gpus", "no gpu",
    "nccl", "connection reset", "connection refused", "docker",
)


def infrastructure_error(exc: Exception) -> bool:
    """Keep harness/runtime failures retryable instead of blaming a checkpoint."""

    if isinstance(
        exc,
        (
            ImportError,
            AttributeError,
            NameError,
            MemoryError,
            TimeoutError,
            ConnectionError,
            PermissionError,
        ),
    ):
        return True
    if isinstance(exc, OSError) and exc.errno in {
        errno.EIO,
        errno.ENOSPC,
        errno.EMFILE,
        errno.ENFILE,
        errno.EROFS,
    }:
        return True
    message = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in message for marker in INFRASTRUCTURE_MARKERS)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _hf_load(checkpoint: Path, *, tokenizer: bool = True) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, device_map="cuda",
    )
    if tokenizer:
        AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    del model
    torch.cuda.empty_cache()
    return {"loader": "AutoModelForCausalLM.from_pretrained", "parameters": parameters}


def validate(task: str, checkpoint: Path, assets: Path, work: Path) -> dict[str, Any]:
    final_eval = importlib.import_module("final_eval")
    if task == "ddpo_sd15_aesthetic":
        from grade import generate, resolve_generation_artifact

        kind, artifact = resolve_generation_artifact(checkpoint)
        # One real frozen generation exercises both full-pipeline and LoRA loading.
        generate(
            assets / "models/stable-diffusion-v1-5", checkpoint,
            ["a simple animal"], 1729, work / "images",
        )
        return {"loader": "StableDiffusionPipeline", "artifact_kind": kind, "artifact": str(artifact)}
    if task == "digress_qm9_graph_diffusion":
        model = final_eval.resolve_checkpoint(checkpoint)
        structure = final_eval.inspect_artifact(model)
        upstream = final_eval.prepare_upstream(work)
        runtime = work / "checkpoint" / "last.ckpt"
        runtime.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model, runtime)
        final_eval.run_upstream(
            upstream, runtime, assets / "data/qm9_no_h", work,
            1, 1729, "loadability", cache_root=work,
        )
        return {"loader": "DiGress checkpoint-only sampler", "structure": structure}
    if task == "dpo_preference_alignment":
        checkpoint_module = importlib.import_module("checkpoint")
        model, tokenizer = checkpoint_module.load_model(
            checkpoint, assets / "models/policy_start"
        )
        parameters = sum(parameter.numel() for parameter in model.parameters())
        del model, tokenizer
        import torch
        torch.cuda.empty_cache()
        return {"loader": "DPO checkpoint.load_model", "parameters": parameters}
    if task == "model_soup_clip_imagenetv2":
        soup_check = importlib.import_module("soup_check")
        model = final_eval.resolve_checkpoint(checkpoint)
        report = soup_check.check_checkpoint(model, assets / "models/ingredients", log=lambda *_: None)
        return {
            "loader": "model-soup state-dict validator",
            "report": dataclasses.asdict(report),
        }
    if task == "openunlearning_tofu_npo_llama3p2_1b":
        runtime_guard = importlib.import_module("runtime_guard")
        model = runtime_guard.resolve_checkpoint(checkpoint)
        return {**_hf_load(model), "resolved_checkpoint": str(model)}
    if task in {
        "opd_math_1p5b", "openr1_code_livecodebench",
        "owl_wanda_opt6p7b_70pct",
    }:
        model = final_eval.resolve_checkpoint(checkpoint)
        return {**_hf_load(model), "resolved_checkpoint": str(model)}
    if task == "ragen_sokoban_grpo":
        from fast_eval import resolve_checkpoint

        model = resolve_checkpoint(checkpoint)
        final_eval.check_checkpoint_carries_no_code(checkpoint, model)
        return {**_hf_load(model), "resolved_checkpoint": str(model)}
    if task == "ultrafeedback_bt_rm_rewardbench":
        artifact = importlib.import_module("artifact")
        base = assets / "models/base"
        report = artifact.check(checkpoint, base)
        model, tokenizer = artifact.load_model(checkpoint, base)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        del model, tokenizer
        import torch
        torch.cuda.empty_cache()
        return {"loader": "reward artifact.load_model", "parameters": parameters, "report": report}
    raise ValueError(f"no frozen checkpoint validator for task {task!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=Path("/assets"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    try:
        details = validate(
            args.task, args.checkpoint.resolve(), args.assets.resolve(),
            args.output.parent / "work",
        )
    except Exception as exc:  # noqa: BLE001 - invalid loaders have heterogeneous errors
        message = f"{type(exc).__name__}: {exc}"
        if infrastructure_error(exc):
            raise
        atomic_json(args.output, {
            "schema_version": 1, "status": "invalid", "task": args.task,
            "checkpoint": str(args.checkpoint), "reason": message,
            "elapsed_seconds": time.monotonic() - started,
        })
        return 0
    atomic_json(args.output, {
        "schema_version": 1, "status": "valid", "task": args.task,
        "checkpoint": str(args.checkpoint), "details": details,
        "elapsed_seconds": time.monotonic() - started,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
