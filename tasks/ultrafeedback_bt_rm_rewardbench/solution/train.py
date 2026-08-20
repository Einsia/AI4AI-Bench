"""Train a scalar Bradley-Terry LoRA reward model on the fixed pair file.

The method source is editable. The harness requires the exported artifact to be
a loadable LoRA delta with a scalar head on the pinned base model.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from runtime_guard import cuda_telemetry, peak_memory_bytes

TASK_ID = "ultrafeedback_bt_rm_rewardbench"


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_pairs(path: Path, limit: int) -> list[dict[str, Any]]:
    """Read the preference pool.

    Each line is `{"chosen": [...], "rejected": [...]}`, both a two-turn
    conversation sharing one user prompt. `limit == 0` means the whole pool.

    The mounted file is the complete 8192-row task asset. A nonzero limit selects
    a deterministic prefix and cannot exceed the file.
    """

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"no preference pairs in {path}")
    if limit and limit > len(rows):
        raise ValueError(f"asked for {limit} pairs, the pool holds {len(rows)}")
    return rows[:limit] if limit else rows


def main() -> None:
    base_model = Path(env_str("BASE_MODEL", "/assets/models/base"))
    train_data = Path(env_str("TRAIN_DATA", "/assets/data/pairs.jsonl"))
    output = Path(env_str("OUTPUT_DIR", "/out"))
    checkpoint = output / env_str("CHECKPOINT_NAME", "checkpoint")

    seed = env_int("SEED", 42)
    train_pairs = env_int("TRAIN_PAIRS", 8192)
    max_steps = env_int("MAX_STEPS", 252)
    max_length = env_int("MAX_LENGTH", 4096)
    micro_batch = env_int("MICRO_BATCH_SIZE", 1)
    accumulation = env_int("GRADIENT_ACCUMULATION_STEPS", 64)
    learning_rate = env_float("LEARNING_RATE", 5.0e-6)
    weight_decay = env_float("WEIGHT_DECAY", 0.001)
    warmup_steps = env_int("WARMUP_STEPS", 4)
    clip_norm = env_float("GRADIENT_CLIP_NORM", 1.0)
    optimizer = env_str("OPTIMIZER", "paged_adamw_32bit")
    scheduler = env_str("SCHEDULER", "cosine")
    lora_r = env_int("LORA_R", 128)
    lora_alpha = env_int("LORA_ALPHA", int(2 * lora_r))
    lora_dropout = env_float("LORA_DROPOUT", 0.05)
    target_modules = env_str(
        "LORA_TARGET_MODULES",
        "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    ).split(",")
    attention = env_str("ATTN_IMPLEMENTATION", "sdpa")
    save_unit = env_str("SAVE_UNIT", "step").strip().lower()
    save_interval = env_int("SAVE_INTERVAL", env_int("SAVE_STEPS", 126))
    save_total_limit_raw = env_int("SAVE_TOTAL_LIMIT", 3)
    save_total_limit = None if save_total_limit_raw == 0 else save_total_limit_raw
    temperature = env_float("BT_TEMPERATURE", 1.0)
    margin = env_float("BT_MARGIN", 0.0)
    centering = env_float("REWARD_CENTERING_WEIGHT", 0.0)
    wall_clock = env_int("MAX_WALL_TIME_SECONDS", 0)
    reserve = env_int("DEADLINE_RESERVE_SECONDS", 600)

    if save_unit not in {"step", "epoch"} or save_interval <= 0:
        raise ValueError("SAVE_UNIT must be step or epoch and SAVE_INTERVAL must be positive")

    if os.environ.get("AI4AI_OUTPUT_LOCK_HELD") != "1":
        raise RuntimeError("train.py must be launched through run.sh with the output lock held")
    occupied = [
        path
        for path in (
            checkpoint,
            output / "trainer",
            output / "dynamics.jsonl",
            output / "train_summary.json",
            output / "summary.json",
        )
        if path.exists()
    ]
    if occupied:
        raise FileExistsError(f"refusing to reuse a training output: {occupied}")

    import torch
    import torch.nn.functional as functional
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )

    gpu = cuda_telemetry(torch, require_single=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    rows = load_pairs(train_data, train_pairs)
    print(f"train: {len(rows)} pairs from {train_data}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    # Left, so truncating a long answer keeps its end. The scalar head reads the
    # last non-pad position, so truncating on the right would score padding.
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(row: dict[str, Any]) -> dict[str, Any]:
        encoded = {}
        for side in ("chosen", "rejected"):
            token_ids = tokenizer.apply_chat_template(
                row[side], tokenize=True, add_generation_prompt=False
            )
            token_ids = [int(token) for token in token_ids]
            original_tokens = len(token_ids)
            token_ids = token_ids[-max_length:]
            encoded[f"input_ids_{side}"] = token_ids
            encoded[f"attention_mask_{side}"] = [1] * len(token_ids)
            encoded[f"original_tokens_{side}"] = original_tokens
            encoded[f"was_truncated_{side}"] = original_tokens > max_length
        return encoded

    dataset = Dataset.from_list(rows)
    dataset = dataset.map(
        tokenize,
        num_proc=min(8, max(1, os.cpu_count() or 1)),
        remove_columns=dataset.column_names,
    )

    class PairCollator:
        """Interleave chosen and rejected so one forward pass scores both.

        Order matters and the loss depends on it: even rows are chosen, odd rows
        rejected, which is what `rewards[0::2]` and `rewards[1::2]` below unpack.
        """

        def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
            merged: list[dict[str, Any]] = []
            for feature in features:
                for side in ("chosen", "rejected"):
                    merged.append(
                        {
                            "input_ids": feature[f"input_ids_{side}"],
                            "attention_mask": feature[f"attention_mask_{side}"],
                        }
                    )
            return tokenizer.pad(merged, padding=True, return_tensors="pt")

    class RewardTrainer(Trainer):
        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, torch.Tensor],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            del num_items_in_batch
            rewards = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            ).logits
            chosen = rewards[0::2]
            rejected = rewards[1::2]
            scaled = (chosen - rejected - margin) / temperature
            # Bradley-Terry: -log sigmoid(r_chosen - r_rejected). The only thing
            # the loss knows about a pair is the difference, which is why the
            # reward scale is unconstrained -- hence the optional centering term.
            pair_loss = -functional.logsigmoid(scaled).mean()
            loss = pair_loss + centering * torch.square(chosen + rejected).mean()
            self.pair_accuracy = float((chosen > rejected).float().mean().detach().item())
            self.reward_margin = float((chosen - rejected).float().mean().detach().item())
            self.pair_loss = float(pair_loss.detach().item())
            if return_outputs:
                return loss, {"chosen": chosen, "rejected": rejected, "pair_loss": pair_loss}
            return loss

        def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
            enriched = dict(logs)
            for name in ("pair_accuracy", "reward_margin", "pair_loss"):
                if hasattr(self, name):
                    enriched[name] = float(getattr(self, name))
            super().log(enriched, start_time)

    class WallClock(TrainerCallback):
        """Stop before the container is killed, so the adapter is written whole.

        The retrain phase reserves time for this; see declaration.py. Without it a
        run that overshoots hands back a directory that exists and cannot be
        loaded, which is worse than a shorter run.
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

    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        # One logit per sequence. This is the reward, and final_eval.py checks the
        # produced head is still shaped this way.
        num_labels=1,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation=attention,
        local_files_only=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    # TaskType.SEQ_CLS puts the freshly initialized score head in the adapter's
    # modules_to_save, so the head trains and is exported with the LoRA factors.
    # The backbone stays frozen and is never written -- the scoring container
    # loads it from the same pinned mount this run did.
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=[name.strip() for name in target_modules if name.strip()],
        ),
    )
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    arguments = TrainingArguments(
        output_dir=str(output / "trainer"),
        per_device_train_batch_size=micro_batch,
        gradient_accumulation_steps=accumulation,
        gradient_checkpointing=True,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        lr_scheduler_type=scheduler,
        max_grad_norm=clip_norm,
        bf16=torch.cuda.is_available(),
        optim=optimizer,
        logging_steps=env_int("LOGGING_STEPS", 10),
        eval_strategy="no",
        save_strategy="steps" if save_unit == "step" else "no",
        save_steps=min(save_interval, max_steps),
        save_total_limit=save_total_limit,
        remove_unused_columns=False,
        label_names=[],
        report_to=[],
        seed=seed,
        data_seed=seed,
    )
    class PeriodicEpochSave(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):  # noqa: ANN001, ANN003
            completed = int(state.epoch or 0)
            if save_unit == "epoch" and completed > 0 and completed % save_interval == 0:
                control.should_save = True
            return control

    trainer = RewardTrainer(
        model=model,
        args=arguments,
        train_dataset=dataset,
        data_collator=PairCollator(),
        callbacks=[WallClock(), PeriodicEpochSave()],
    )
    result = trainer.train()

    trainer.save_model(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    (output / "dynamics.jsonl").write_text(
        "".join(
            json.dumps(record, sort_keys=True) + "\n"
            for record in trainer.state.log_history
            if isinstance(record, dict)
        ),
        encoding="utf-8",
    )
    train_runtime = float(result.metrics.get("train_runtime", 0.0))
    steps_completed = int(trainer.state.global_step)
    train_summary = {
            "schema_version": 1,
            "task_id": TASK_ID,
            "pairs": len(rows),
            "steps_requested": max_steps,
            "steps_completed": steps_completed,
            "save_unit": save_unit,
            "save_interval": save_interval,
            "save_total_limit": save_total_limit_raw,
            "global_pair_batch": micro_batch * accumulation,
            "train_runtime_seconds": train_runtime,
            "optimizer_steps_per_second": (
                steps_completed / train_runtime if train_runtime > 0.0 else 0.0
            ),
            "trainer_metrics": {
                key: value
                for key, value in result.metrics.items()
                if isinstance(value, (int, float, str, bool)) or value is None
            },
            "tokenization": {
                "protocol": "apply_chat_template_tokenize_true_no_special_token_readdition",
                "truncation_side": "left",
                "max_length": max_length,
                "chosen_truncated": sum(bool(value) for value in dataset["was_truncated_chosen"]),
                "rejected_truncated": sum(
                    bool(value) for value in dataset["was_truncated_rejected"]
                ),
                "max_chosen_tokens_before_truncation": max(dataset["original_tokens_chosen"]),
                "max_rejected_tokens_before_truncation": max(
                    dataset["original_tokens_rejected"]
                ),
            },
            "wall_seconds": time.monotonic() - started,
            "gpu": gpu,
            "gpu_memory_peak_bytes": peak_memory_bytes(torch),
            "checkpoint": str(checkpoint),
            "attention_implementation": attention,
            "retained_trainer_checkpoints": sorted(
                path.name for path in (output / "trainer").glob("checkpoint-*")
                if path.is_dir()
            ),
            "offline": True,
        }
    atomic_json(output / "train_summary.json", train_summary)
    atomic_json(output / "summary.json", train_summary)
    print(f"train: wrote {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
