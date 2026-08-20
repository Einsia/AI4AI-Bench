"""Pairwise DPO-family training from the fixed merged Zephyr SFT start.

With ``ref_model=None`` and a PEFT configuration, TRL evaluates the reference with
the newly created DPO adapter disabled. The policy initialization, frozen reference
and adapter-scoring backbone are therefore the same mounted merged checkpoint.

The output is either a loadable LoRA adapter or a merged model at
``/out/checkpoint``. Training metadata is provenance only; the evaluator resolves
and loads the artifact from its files.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from runtime_guard import cuda_telemetry, peak_memory_bytes

TASK_ID = "dpo_preference_alignment"
# Recorded for provenance only. Nothing in the harness reads this value.
UPSTREAM_REVISION = "1de1fc996972aa76b7d40c64c07b66dec8b6976a"
POLICY_START_REVISION = (
    "27d67f1b5f57dc0953326b2601d68371d40ea8da+156bec577ff12a65236cfc90860dcc61e96c6fd6"
)
DATA_REVISION = "3949bf5f8c17c394422ccfab0c31ea9c20bdeb85"


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def find_split(root: Path, name: str) -> list[Path]:
    """Locate every parquet shard for one split in the mounted snapshot."""

    matches = sorted(root.glob(f"**/{name}-*.parquet")) or sorted(root.glob(f"**/{name}.parquet"))
    if not matches:
        raise FileNotFoundError(f"no {name} parquet under {root}")
    return matches


def reward_accuracy(metrics: dict[str, Any]) -> float | None:
    """Pull the one `rewards/accuracies` value out of a metrics dict.

    TRL prefixes it with the metric_key_prefix, so the key is not stable across
    calls. Returns None rather than raising if it is absent -- a candidate that
    changed the objective may not emit one, and that is allowed now.
    """

    found = [
        float(value)
        for key, value in metrics.items()
        if key.endswith("rewards/accuracies") and isinstance(value, (int, float))
    ]
    return found[0] if len(found) == 1 else None


def main() -> None:
    policy_start = Path(env_str("POLICY_START", "/assets/models/policy_start"))
    train_data = Path(env_str("TRAIN_DATA", "/assets/data/ultrafeedback"))
    output = Path(env_str("OUTPUT_DIR", "/out"))
    checkpoint = output / env_str("CHECKPOINT_NAME", "checkpoint")

    seed = env_int("SEED", 42)
    data_order_seed = env_int("DATA_ORDER_SEED", 42)
    train_samples = env_int("TRAIN_SAMPLES", 61135)
    eval_samples = env_int("EVAL_SAMPLES", 128)
    max_length = env_int("MAX_LENGTH", 1024)
    micro_batch = env_int("PER_DEVICE_BATCH_SIZE", 4)
    accumulation = env_int("GRADIENT_ACCUMULATION_STEPS", 4)
    max_steps = env_int("MAX_STEPS", 772)
    learning_rate = env_float("LEARNING_RATE", 5.0e-6)
    weight_decay = env_float("WEIGHT_DECAY", 0.0)
    warmup_ratio = env_float("WARMUP_RATIO", 0.10)
    clip_norm = env_float("GRADIENT_CLIP_NORM", 1.0)
    optimizer = env_str("OPTIMIZER", "paged_adamw_32bit")
    scheduler = env_str("SCHEDULER", "cosine")
    beta = env_float("DPO_BETA", 0.01)
    loss_type = env_str("DPO_LOSS_TYPE", "sigmoid")
    lora_r = env_int("LORA_R", 128)
    lora_alpha = env_int("LORA_ALPHA", lora_r)
    lora_dropout = env_float("LORA_DROPOUT", 0.05)
    target_modules = [
        name.strip()
        for name in env_str(
            "LORA_TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
        ).split(",")
        if name.strip()
    ]
    select_best = env_bool("SELECT_BEST", False)
    eval_steps = env_int("EVAL_STEPS", 32)
    save_unit = env_str("SAVE_UNIT", "step").strip().lower()
    save_interval = env_int("SAVE_INTERVAL", env_int("SAVE_STEPS", 386))
    save_total_limit_raw = env_int("SAVE_TOTAL_LIMIT", 3)
    save_total_limit = None if save_total_limit_raw == 0 else save_total_limit_raw
    export_mode = env_str("EXPORT_MODE", "adapter").strip().lower()
    export_shard_size = env_str("EXPORT_SHARD_SIZE", "4GB")
    logging_steps = env_int("LOGGING_STEPS", 1)
    attention = env_str("ATTN_IMPLEMENTATION", "sdpa")
    wall_clock = env_int("MAX_WALL_TIME_SECONDS", 0)
    reserve = env_int("DEADLINE_RESERVE_SECONDS", 1200)

    if export_mode not in {"merged", "adapter"}:
        raise ValueError(f"EXPORT_MODE must be 'merged' or 'adapter', got {export_mode!r}")
    if save_unit not in {"step", "epoch"} or save_interval <= 0:
        raise ValueError("SAVE_UNIT must be step or epoch and SAVE_INTERVAL must be positive")
    if os.environ.get("AI4AI_OUTPUT_LOCK_HELD") != "1":
        raise RuntimeError("launch training through /workspace/run.sh so the output lock is held")
    if checkpoint.exists() and any(checkpoint.iterdir()):
        raise FileExistsError(f"checkpoint directory is not empty: {checkpoint}")
    occupied = [
        path
        for path in (output / "train_summary.json", output / "dynamics.jsonl")
        if path.exists()
    ]
    if occupied:
        raise FileExistsError(f"training output already contains receipt files: {occupied}")

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainerCallback,
        set_seed,
    )
    from trl import DPOConfig, DPOTrainer

    gpu = cuda_telemetry(torch)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    set_seed(seed)

    train_paths = find_split(train_data, "train_prefs")
    eval_paths = find_split(train_data, "test_prefs")
    train = load_dataset("parquet", data_files=[str(path) for path in train_paths], split="train")
    valid = load_dataset("parquet", data_files=[str(path) for path in eval_paths], split="train")
    pool_rows = len(train)
    if train_samples > pool_rows or eval_samples > len(valid):
        raise ValueError(
            f"asked for {train_samples} train and {eval_samples} eval rows; the mounted "
            f"pool holds {pool_rows} and {len(valid)}"
        )
    # Shuffle then take, so raising TRAIN_SAMPLES adds rows rather than reshuffling
    # the ones already in use -- the whole split is mounted.
    train = train.shuffle(seed=data_order_seed).select(range(train_samples))
    valid = valid.shuffle(seed=data_order_seed).select(range(eval_samples))
    # Conversational pairs with the prompt implicit in both sides, which is the
    # format TRL extracts a shared prefix from.
    train = train.select_columns(["chosen", "rejected"])
    valid = valid.select_columns(["chosen", "rejected"])
    print(f"train: {len(train)} pairs of {pool_rows} available, {len(valid)} held out", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(policy_start, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    work = output / "work"
    arguments = DPOConfig(
        output_dir=str(work),
        model_init_kwargs={"attn_implementation": attention, "torch_dtype": torch.bfloat16},
        beta=beta,
        loss_type=loss_type,
        max_length=max_length,
        per_device_train_batch_size=micro_batch,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=accumulation,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=learning_rate,
        lr_scheduler_type=scheduler,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        max_grad_norm=clip_norm,
        max_steps=max_steps,
        bf16=True,
        optim=optimizer,
        logging_steps=logging_steps,
        eval_strategy="steps" if select_best else "no",
        eval_steps=min(eval_steps, max_steps),
        save_strategy="steps" if save_unit == "step" else "no",
        save_steps=min(eval_steps if select_best else save_interval, max_steps),
        save_total_limit=save_total_limit,
        # Preference accuracy is a trainer diagnostic, not the IFEval task metric.
        load_best_model_at_end=select_best,
        metric_for_best_model="eval_rewards/accuracies" if select_best else None,
        greater_is_better=True if select_best else None,
        report_to=[],
        seed=seed,
        data_seed=data_order_seed,
        remove_unused_columns=False,
    )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    class WallClock(TrainerCallback):
        """Stop before the container is killed, so the export finishes.

        The retrain phase reserves time for this; see declaration.py. The reserve is
        large here because `EXPORT_MODE=merged` reloads the backbone in bf16, merges
        the adapter and writes ~14.5 GiB. Without it an overrunning run hands back a
        directory that exists and cannot be loaded, which is worse than a shorter run.
        """

        def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001, ANN003
            if wall_clock and time.monotonic() - started > wall_clock - reserve:
                print(
                    f"train: wall clock reached at step {state.global_step}; stopping to "
                    "write a complete checkpoint",
                    flush=True,
                )
                control.should_training_stop = True
            return control

    class PeriodicEpochSave(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):  # noqa: ANN001, ANN003
            completed = int(state.epoch or 0)
            if save_unit == "epoch" and completed > 0 and completed % save_interval == 0:
                control.should_save = True
            return control

    trainer = DPOTrainer(
        model=str(policy_start),
        # None, not a second copy: with a peft_config this makes the reference the
        # same weights with the adapter disabled.
        ref_model=None,
        args=arguments,
        train_dataset=train,
        # Required by the pinned TRL even when step selection is disabled.
        eval_dataset=valid,
        processing_class=tokenizer,
        quantization_config=quantization,
        peft_config=peft_config,
        callbacks=[WallClock(), PeriodicEpochSave()],
    )
    initial = trainer.evaluate(metric_key_prefix="initial") if select_best else {}
    result = trainer.train()
    final = trainer.evaluate(metric_key_prefix="final") if select_best else {}

    adapter = output / "selected_adapter"
    trainer.save_model(adapter)
    tokenizer.save_pretrained(adapter)
    best_step = trainer.state.best_model_checkpoint
    best_metric = trainer.state.best_metric
    steps_completed = int(trainer.state.global_step)
    history = [dict(row) for row in trainer.state.log_history if isinstance(row, dict)]
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    if export_mode == "adapter":
        shutil.copytree(adapter, checkpoint)
    else:
        from peft import PeftModel

        # The backbone is reloaded in bf16 rather than merged into the 4-bit copy:
        # Merging into NF4 weights would bake quantization error into the export.
        base = AutoModelForCausalLM.from_pretrained(
            policy_start,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation=attention,
            device_map="cuda",
        )
        merged = PeftModel.from_pretrained(base, adapter).merge_and_unload()
        checkpoint.mkdir(parents=True)
        merged.save_pretrained(
            checkpoint, safe_serialization=True, max_shard_size=export_shard_size
        )
        tokenizer.save_pretrained(checkpoint)
        del merged, base
        gc.collect()
        torch.cuda.empty_cache()

    (output / "dynamics.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in history),
        encoding="utf-8",
    )
    train_summary = {
            "schema_version": 1,
            "task_id": TASK_ID,
            "checkpoint": str(checkpoint),
            "export_mode": export_mode,
            "train_rows": len(train),
            "train_pool_rows": pool_rows,
            "train_source_shards": [path.name for path in train_paths],
            "eval_source_shards": [path.name for path in eval_paths],
            "eval_rows": len(valid),
            "steps_requested": max_steps,
            "steps_completed": steps_completed,
            "save_unit": save_unit,
            "save_interval": save_interval,
            "save_total_limit": save_total_limit_raw,
            "global_pair_batch": micro_batch * accumulation,
            "selected_checkpoint": best_step,
            "initial_reward_accuracy": reward_accuracy(initial),
            "final_reward_accuracy": reward_accuracy(final),
            "best_reward_accuracy": float(best_metric) if best_metric is not None else None,
            "train_loss": float(result.metrics.get("train_loss", 0.0)),
            "train_runtime_seconds": float(result.metrics.get("train_runtime", 0.0)),
            "wall_seconds": time.monotonic() - started,
            "optimizer_steps_per_second": (
                steps_completed / float(result.metrics.get("train_runtime", 0.0))
                if float(result.metrics.get("train_runtime", 0.0)) > 0
                else None
            ),
            "gpu": gpu,
            "gpu_memory_peak_bytes": peak_memory_bytes(torch),
            "attention_implementation": attention,
            "retained_trainer_checkpoints": sorted(
                path.name for path in work.glob("checkpoint-*") if path.is_dir()
            ),
            "offline": True,
        }
    atomic_json(output / "train_summary.json", train_summary)
    atomic_json(output / "summary.json", train_summary)
    atomic_json(
        output / "training_metadata.json",
        {
            "schema_version": 1,
            "task_id": TASK_ID,
            "note": "provenance only -- no evaluator reads this file",
            "policy_start_revision": POLICY_START_REVISION,
            "data_revision": DATA_REVISION,
            "upstream_revision": UPSTREAM_REVISION,
            "beta": beta,
            "loss_type": loss_type,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "target_modules": target_modules,
            "data_order_seed": data_order_seed,
            "seed": seed,
        },
    )
    # The launcher publishes this final export together with up to two retained
    # trainer checkpoints under the numbered formal-artifact protocol.
    shutil.rmtree(adapter, ignore_errors=True)
    print(f"train: wrote {checkpoint} ({export_mode})", flush=True)


if __name__ == "__main__":
    main()
