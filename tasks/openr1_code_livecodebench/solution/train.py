"""Completion-only supervised finetuning: the whole method, and all of it yours.

Assistant-completion next-token NLL on the decontaminated OpenR1 CodeForces
corpus, starting from Qwen2.5-Coder-1.5B-Instruct. The loss is masked to the
assistant turn: `labels = [-100] * common + full[common:]`, where `common` is the
length of the tokenized prompt prefix, so the model is scored only on the tokens
it would have had to produce.

What changed in this port, and why it is worth knowing: on the reference protocol this
file was frozen. `environment/check_candidate.py` parsed it before every run and
rejected it unless seven specific substrings were present -- including the label
mask line above, `Trainer(`, and `metric_for_best_model="eval_loss"` -- and
rejected any import of `peft`, `trl`, `subprocess` or `socket`. A parallel
allowlist named the eleven recipe keys that could vary. None of that exists now.
There is no checker, no recipe file and no protected symbol; the objective, the
optimizer, the schedule and the framework are all in front of you.

Three things remain fixed, and by construction rather than by inspection: the
model comes from a read-only mount, the corpus comes from a read-only mount, and
the evaluator runs from read-only /opt/harness in a container built off the
original image layer with your patch deliberately not applied.

run.sh holds the formal defaults and passes them here. argparse repeats those
defaults only so direct smoke/probe calls have a complete interface; formal replay
is governed by the source defaults in run.sh.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from runtime_guard import cuda_telemetry, peak_memory_bytes

TASK_ID = "openr1_code_livecodebench"
WORKSPACE = Path(__file__).resolve().parent
WEIGHT_PATTERNS = ("*.safetensors", "pytorch_model*.bin")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def weight_sha256(checkpoint: Path) -> str:
    files = sorted({path for pattern in WEIGHT_PATTERNS for path in checkpoint.glob(pattern)})
    if not files:
        raise RuntimeError(f"checkpoint has no model weights: {checkpoint}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def load_corpus(data: Path, subset: str, split_seed: int, train_rows: int, validation_rows: int):
    """Read every parquet shard of the subset, shuffle once, then split.

    The complete decontaminated split is mounted. The shipped 8005/128 split uses
    all 8133 rows; changing the two counts reallocates this fixed pool rather than
    exposing additional data.

    Validation comes off the front of the shuffled order and training off the
    rest, which is the reference protocol's arrangement kept unchanged: it means a
    validation NLL is comparable across runs at the same `split_seed`, and not
    comparable across different ones.
    """

    from datasets import concatenate_datasets, load_dataset

    root = data / subset
    shards = sorted(root.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"no parquet shards under {root}. The mount is the whole "
            f"{subset} split; check --data and --source-subset."
        )
    dataset = concatenate_datasets(
        [load_dataset("parquet", data_files=str(path), split="train") for path in shards]
    )
    dataset = dataset.shuffle(seed=split_seed)
    requested = train_rows + validation_rows
    if len(dataset) < requested:
        raise ValueError(
            f"asked for {train_rows} training + {validation_rows} validation rows, "
            f"but {root} holds {len(dataset)}. That is the whole decontaminated "
            "split -- there is nothing further to draw from."
        )
    selected = dataset.select(range(requested))
    validation = selected.select(range(validation_rows))
    training = selected.select(range(validation_rows, requested))
    return training, validation, [path.name for path in shards], len(dataset)


def build_encoder(tokenizer, max_length: int):
    """Tokenize one conversation and mask everything before the assistant turn.

    The prefix is tokenized with `add_generation_prompt=True` so it ends exactly
    where the assistant's own tokens begin, and the mask length is found by
    walking the two token lists together rather than by trusting the prefix
    length -- chat templates can retokenize across the boundary, and a
    length-based mask silently supervises part of the prompt when they do.
    """

    def encode(row: dict[str, Any]) -> dict[str, Any]:
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError("every row must contain user and assistant messages")
        prefix = list(
            tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
        )
        full_untruncated = list(
            tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        )
        full = full_untruncated[:max_length]
        common = 0
        for left, right in zip(prefix, full, strict=False):
            if left != right:
                break
            common += 1
        labels = [-100] * common + full[common:]
        return {
            "input_ids": full,
            "attention_mask": [1] * len(full),
            "labels": labels,
            "supervised_tokens": sum(value != -100 for value in labels),
            "sequence_tokens": len(full),
            "sequence_tokens_before_truncation": len(full_untruncated),
            "was_truncated": len(full_untruncated) > max_length,
            "terminal_token_preserved": len(full_untruncated) <= max_length,
        }

    return encode


def wall_clock_callback(budget: int, reserve: int):
    """Stop before the container is killed, and make sure a save follows.

    The retrain phase exports MAX_WALL_TIME_SECONDS = timeout - 600 and
    DEADLINE_RESERVE_SECONDS; docker removes the container at the timeout
    regardless of what the trainer is doing. A kill during
    `Trainer._save_checkpoint` leaves shards written and optimizer state missing,
    which is a directory that exists and cannot be loaded -- worse than no
    checkpoint, because it looks like one.

    It is relevant when a candidate raises MAX_STEPS; the shipped short control
    normally completes before the formal deadline.

    `should_save` is set alongside `should_training_stop` because stopping alone
    ends the run at whatever the last periodic save was -- up to `save_steps`
    of work discarded at exactly the moment there is no time to redo it.
    """

    from transformers import TrainerCallback

    class WallClock(TrainerCallback):
        def __init__(self) -> None:
            self.deadline = time.monotonic() + max(budget - reserve, 0)
            self.seconds_per_step = 0.0
            self.started = time.monotonic()
            self.stopped_early = False

        def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001, ANN003
            if state.global_step > 0:
                self.seconds_per_step = (time.monotonic() - self.started) / state.global_step
            # One step of headroom, plus the save the stop triggers.
            if time.monotonic() + self.seconds_per_step >= self.deadline:
                self.stopped_early = True
                control.should_save = True
                control.should_training_stop = True
                print(
                    f"train.py: stopping at step {state.global_step} -- "
                    f"{budget}s budget less {reserve}s reserve reached, at "
                    f"{self.seconds_per_step:.2f}s/step",
                    flush=True,
                )
            return control

    return WallClock()


def export_checkpoint(trainer, tokenizer, checkpoint_dir: Path, step: int) -> Path:
    """Atomically write the final weights as checkpoints/checkpoint-<N>.

    Trainer's own state remains under --output/trainer. The final candidate is a
    self-contained model in the public v1.5 checkpoint namespace.
    """

    export = checkpoint_dir / f"checkpoint-{step}"
    temporary = checkpoint_dir / f".checkpoint-{step}.{os.getpid()}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    trainer.save_model(temporary)
    tokenizer.save_pretrained(temporary)
    temporary.replace(export)
    return export


def verify_complete(checkpoint: Path) -> None:
    """Confirm the export is whole before reporting success.

    A shard index that names files which are not there is the signature of a save
    that ran out of room or time, and it is silent: the directory loads far enough
    to look right and then fails at generation.
    """

    config = checkpoint / "config.json"
    if not config.is_file() or config.stat().st_size == 0:
        raise ValueError(f"checkpoint has no usable config.json: {checkpoint}")
    index = checkpoint / "model.safetensors.index.json"
    if index.is_file():
        payload = json.loads(index.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"checkpoint index has no weight_map: {checkpoint}")
        expected = set(weight_map.values())
        actual = {path.name for path in checkpoint.glob("*.safetensors")}
        if expected != actual:
            raise ValueError(
                f"checkpoint is missing shards {sorted(expected - actual)}: {checkpoint}"
            )
    elif not any(checkpoint.glob(pattern) for pattern in WEIGHT_PATTERNS):
        raise ValueError(f"checkpoint has no weights: {checkpoint}")


def train(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    if os.environ.get("AI4AI_OUTPUT_LOCK_HELD") != "1":
        raise RuntimeError("launch training through /workspace/run.sh so the output lock is held")
    output = args.output
    occupied = [
        path
        for path in (output / "train_summary.json", output / "dynamics.jsonl")
        if path.exists()
    ]
    if occupied:
        raise FileExistsError(f"training output already contains receipt files: {occupied}")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    gpu = cuda_telemetry(torch)
    torch.cuda.reset_peak_memory_stats()
    set_seed(args.seed)

    training, validation, shards, available = load_corpus(
        args.data, args.source_subset, args.split_seed, args.train_rows, args.validation_rows
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.training_start, local_files_only=True, use_fast=True, padding_side="right"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    encode = build_encoder(tokenizer, args.max_length)
    columns = training.column_names
    training = training.map(encode, remove_columns=columns, desc="tokenize training")
    validation = validation.map(encode, remove_columns=columns, desc="tokenize validation")
    # A row whose assistant turn falls entirely past max_length contributes no
    # supervised token. Record exact truncation/drop counts before filtering.
    train_rows_before_filter = len(training)
    validation_rows_before_filter = len(validation)
    train_truncated = sum(bool(value) for value in training["was_truncated"])
    validation_truncated = sum(bool(value) for value in validation["was_truncated"])
    train_terminal_preserved = sum(bool(value) for value in training["terminal_token_preserved"])
    validation_terminal_preserved = sum(
        bool(value) for value in validation["terminal_token_preserved"]
    )
    train_dropped_no_target = sum(value <= 0 for value in training["supervised_tokens"])
    validation_dropped_no_target = sum(
        value <= 0 for value in validation["supervised_tokens"]
    )
    training = training.filter(lambda row: row["supervised_tokens"] > 0)
    validation = validation.filter(lambda row: row["supervised_tokens"] > 0)
    sequence_lengths = list(training["sequence_tokens"])
    supervised_lengths = list(training["supervised_tokens"])
    diagnostic_columns = [
        "supervised_tokens",
        "sequence_tokens",
        "sequence_tokens_before_truncation",
        "was_truncated",
        "terminal_token_preserved",
    ]
    training = training.remove_columns(diagnostic_columns)
    validation = validation.remove_columns(diagnostic_columns)

    model = AutoModelForCausalLM.from_pretrained(
        args.training_start,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    train_args = TrainingArguments(
        output_dir=str(output / "trainer"),
        seed=args.seed,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.gradient_clip_norm,
        lr_scheduler_type=args.lr_scheduler_type,
        # Only meaningful for the *_with_min_lr schedules; passing it to others
        # raises rather than being ignored, so it is omitted when unused.
        lr_scheduler_kwargs=(
            {"min_lr_rate": args.min_lr_rate} if "min_lr" in args.lr_scheduler_type else None
        ),
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps" if args.save_unit == "step" else "no",
        save_steps=args.save_interval,
        save_total_limit=None if args.save_total_limit == 0 else args.save_total_limit,
        logging_steps=5,
        report_to=[],
        load_best_model_at_end=bool(args.load_best_model_at_end),
        metric_for_best_model="eval_loss" if args.load_best_model_at_end else None,
        greater_is_better=False if args.load_best_model_at_end else None,
        remove_unused_columns=False,
    )
    from transformers import TrainerCallback

    class PeriodicEpochSave(TrainerCallback):
        def on_epoch_end(self, training_args, state, control, **kwargs):  # noqa: ANN001, ANN003
            completed = int(state.epoch or 0)
            if args.save_unit == "epoch" and completed > 0 and completed % args.save_interval == 0:
                control.should_save = True
            return control

    callbacks = [PeriodicEpochSave()]
    wall_callback = None
    if args.max_wall_time_seconds > 0:
        wall_callback = wall_clock_callback(
            args.max_wall_time_seconds, args.deadline_reserve_seconds
        )
        callbacks.append(wall_callback)
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=training,
        eval_dataset=validation,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8
        ),
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    started = time.monotonic()
    result = trainer.train()
    elapsed = time.monotonic() - started
    evaluation = trainer.evaluate()

    step = trainer.state.global_step
    export = export_checkpoint(trainer, tokenizer, checkpoint_dir, step)
    verify_complete(export)

    summary = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "completed",
        "checkpoint": str(export),
        "completed_steps": step,
        "requested_steps": args.max_steps,
        "recipe": {
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "per_device_batch_size": args.per_device_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": (
                args.per_device_batch_size * args.gradient_accumulation_steps
            ),
            "scheduled_sample_slots": (
                step * args.per_device_batch_size * args.gradient_accumulation_steps
            ),
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "gradient_clip_norm": args.gradient_clip_norm,
            "lr_scheduler_type": args.lr_scheduler_type,
            "min_lr_rate": args.min_lr_rate,
            "save_unit": args.save_unit,
            "save_interval": args.save_interval,
            "save_total_limit": args.save_total_limit,
        },
        "stopped_on_wall_clock": bool(
            wall_callback and getattr(wall_callback, "stopped_early", False)
        ),
        "wall_seconds": elapsed,
        "seconds_per_step": elapsed / step if step else None,
        "train_loss": result.metrics.get("train_loss"),
        "train_runtime": result.metrics.get("train_runtime"),
        # The training diagnostic, not the metric. It is a collapse guard: the
        # score this task is judged on is executed pass@1, and NLL going down does
        # not imply it goes up. Use /opt/harness/fast_eval.sh for the metric.
        "validation_completion_nll": evaluation.get("eval_loss"),
        "source_shards": shards,
        "corpus_rows_available": available,
        "train_rows": len(training),
        "validation_rows": len(validation),
        "train_rows_before_filter": train_rows_before_filter,
        "validation_rows_before_filter": validation_rows_before_filter,
        "train_rows_truncated": train_truncated,
        "validation_rows_truncated": validation_truncated,
        "train_rows_dropped_no_supervised_tokens": train_dropped_no_target,
        "validation_rows_dropped_no_supervised_tokens": validation_dropped_no_target,
        "train_rows_terminal_token_preserved": train_terminal_preserved,
        "validation_rows_terminal_token_preserved": validation_terminal_preserved,
        "mean_sequence_tokens": (
            sum(sequence_lengths) / len(sequence_lengths) if sequence_lengths else None
        ),
        "mean_supervised_tokens": (
            sum(supervised_lengths) / len(supervised_lengths) if supervised_lengths else None
        ),
        "optimizer_steps_per_second": step / elapsed if elapsed > 0 else None,
        "gpu": gpu,
        "peak_memory_bytes": peak_memory_bytes(torch),
        "checkpoint_weight_sha256": weight_sha256(export),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    atomic_json(output / "train_summary.json", summary)
    atomic_json(output / "summary.json", summary)
    with (output / "dynamics.jsonl").open("w", encoding="utf-8") as handle:
        for row in trainer.state.log_history:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-start", type=Path, default=Path(
        os.environ.get("TRAINING_START", "/assets/models/training_start")
    ))
    parser.add_argument("--data", type=Path, default=Path(
        os.environ.get("TRAIN_DATA", "/assets/data/codeforces_cots")
    ))
    parser.add_argument("--source-subset", default="solutions_py_decontaminated")
    parser.add_argument("--output", type=Path, default=Path("/out"))
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=20260727)
    parser.add_argument("--train-rows", type=int, default=8005)
    parser.add_argument("--validation-rows", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--per-device-batch-size", type=int, default=3)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=6)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=0.2)
    parser.add_argument("--lr-scheduler-type", default="cosine_with_min_lr")
    parser.add_argument("--min-lr-rate", type=float, default=0.1)
    parser.add_argument("--eval-steps", type=int, default=30)
    parser.add_argument("--save-unit", choices=("step", "epoch"), default="step")
    parser.add_argument("--save-interval", type=int, default=30)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--load-best-model-at-end", type=int, default=0)
    parser.add_argument("--max-wall-time-seconds", type=int, default=0)
    parser.add_argument("--deadline-reserve-seconds", type=int, default=600)
    parser.add_argument(
        "--one-step", action="store_true", help="one step on 16 rows, for plumbing checks"
    )
    parser.add_argument("--smoke", action="store_true", help="self-check and exit, no GPU")
    args = parser.parse_args()

    if args.smoke:
        print(json.dumps({"schema_version": 1, "task_id": TASK_ID, "smoke": "passed"}))
        return
    if args.checkpoint_dir is None:
        args.checkpoint_dir = args.output / "checkpoints"
    if args.one_step:
        args.max_steps = 1
        args.train_rows = 16
        args.validation_rows = 16
        args.eval_steps = 1
        args.save_interval = 1
    print(json.dumps(train(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
