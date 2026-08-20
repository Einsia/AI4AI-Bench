"""Run the pinned single-GPU OpenUnlearning NPO recipe and retain its receipt."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

UPSTREAM = Path("/workspace/open-unlearning")
EXPECTED_START_WEIGHT = "35b0cd9f7db2735538ff78b4879fab61b23b7ca316df27b059585bf656700a8d"
DATA_HASHES = {
    "forget10.json": "0044c8c2e70a38be93f62ec6cb1c1cc2a1f55a8df2fb549a4da2da8dde9d92f6",
    "retain90.json": "debcf019c41db7b30b62ca6ad41fd8a4bcd29a5b321c21d2961974e308dfca55",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def env_number(name: str, cast: type[int] | type[float]) -> int | float:
    value = cast(os.environ[name])
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def nonnegative_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def parse_gpu_sample(stdout: str) -> tuple[str, float, float]:
    rows = [line.strip() for line in stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"expected telemetry for exactly one visible GPU, got {len(rows)}")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 3:
        raise RuntimeError(f"malformed nvidia-smi telemetry row: {rows[0]!r}")
    uuid, memory_text, utilization_text = fields
    try:
        memory_mib = float(memory_text)
        utilization_percent = float(utilization_text)
    except ValueError as exc:
        raise RuntimeError(f"non-numeric nvidia-smi telemetry row: {rows[0]!r}") from exc
    if (
        not uuid
        or not math.isfinite(memory_mib)
        or not math.isfinite(utilization_percent)
        or memory_mib < 0
        or not 0 <= utilization_percent <= 100
    ):
        raise RuntimeError(f"invalid nvidia-smi telemetry row: {rows[0]!r}")
    return uuid, memory_mib, utilization_percent


def gpu_sample() -> tuple[str, float, float]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "GPU telemetry unavailable: nvidia-smi executable not found"
        ) from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic"
        raise RuntimeError(
            f"GPU telemetry unavailable: nvidia-smi exited {result.returncode}: {detail}"
        )
    return parse_gpu_sample(result.stdout)


def acquire_training_lock(output: Path):
    lock_path = output.parent / f".{output.name}.train.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(f"another trainer owns OUTPUT_DIR {output}") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps({"pid": os.getpid(), "output_dir": str(output)}) + "\n")
    handle.flush()
    return handle


def terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def has_model_weights(directory: Path) -> bool:
    return bool(list(directory.glob("*.safetensors")) or list(directory.glob("*.bin")))


def completed_trainer_checkpoints(output: Path) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    for checkpoint in output.glob("checkpoint-*"):
        raw = checkpoint.name.removeprefix("checkpoint-")
        if raw.isdigit() and checkpoint.is_dir() and has_model_weights(checkpoint):
            found.append((int(raw), checkpoint))
    return sorted(found)


def copy_hf_files(source: Path, target: Path) -> None:
    """Copy one complete HF export without copying sibling Trainer checkpoints."""

    target.mkdir()
    for item in source.iterdir():
        if item.is_file():
            shutil.copy2(item, target / item.name)


def top_up_tokenizer_and_config(target: Path, training_start: Path) -> None:
    for name in (
        "config.json", "generation_config.json", "tokenizer.json", "tokenizer.model",
        "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
    ):
        source = training_start / name
        if source.is_file() and not (target / name).exists():
            shutil.copy2(source, target / name)


def main() -> None:
    start = Path(os.environ["TRAINING_START"]).resolve()
    data = Path(os.environ["TRAIN_DATA"]).resolve()
    output = Path(os.environ["OUTPUT_DIR"]).resolve()
    if not str(start).startswith("/assets/") or not str(data).startswith("/assets/"):
        raise ValueError("fixed model and data must be mounted below /assets")
    if not str(output).startswith("/out/"):
        raise ValueError("OUTPUT_DIR must be a unique directory below /out")
    training_lock = acquire_training_lock(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")

    start_weight = start / "model.safetensors"
    if sha256(start_weight) != EXPECTED_START_WEIGHT:
        raise ValueError("training-start model.safetensors does not match the published anchor")
    observed_data = {name: sha256(data / name) for name in DATA_HASHES}
    if observed_data != DATA_HASHES:
        raise ValueError(f"training data hash mismatch: {observed_data}")
    if not (UPSTREAM / "src/train.py").is_file():
        raise FileNotFoundError(UPSTREAM / "src/train.py")

    max_steps = int(os.environ.get("MAX_STEPS", "-1"))
    if max_steps == 0 or max_steps < -1:
        raise ValueError("MAX_STEPS must be -1 or a positive integer")
    save_unit = os.environ.get("SAVE_UNIT", "epoch").strip().lower()
    save_interval = int(os.environ.get("SAVE_INTERVAL", "5"))
    save_total_limit = int(os.environ.get("SAVE_TOTAL_LIMIT", "3"))
    if save_unit not in {"step", "epoch"} or save_interval <= 0 or save_total_limit < 0:
        raise ValueError("invalid SAVE_UNIT, SAVE_INTERVAL or SAVE_TOTAL_LIMIT")
    max_wall_seconds = nonnegative_env("MAX_WALL_TIME_SECONDS", 0)
    reserve_seconds = nonnegative_env("DEADLINE_RESERVE_SECONDS", 900)
    if max_wall_seconds and max_wall_seconds <= reserve_seconds:
        raise ValueError(
            "MAX_WALL_TIME_SECONDS must exceed DEADLINE_RESERVE_SECONDS"
        )
    train_budget_seconds = (
        max_wall_seconds - reserve_seconds if max_wall_seconds else None
    )

    recipe: dict[str, Any] = {
        "learning_rate": env_number("LEARNING_RATE", float),
        "beta": env_number("NPO_BETA", float),
        "alpha": env_number("NPO_ALPHA", float),
        "gamma": env_number("RETAIN_GAMMA", float),
        "epochs": env_number("EPOCHS", int),
        "per_device_batch_size": env_number("PER_DEVICE_BATCH", int),
        "gradient_accumulation_steps": env_number("GRAD_ACCUM", int),
        "seed": int(os.environ["SEED"]),
        "max_steps": max_steps,
        "save_unit": save_unit,
        "save_interval": save_interval,
        "save_total_limit": save_total_limit,
        "max_wall_time_seconds": max_wall_seconds,
        "deadline_reserve_seconds": reserve_seconds,
        "weight_decay": 0.01,
        "warmup_epochs": 1.0,
        "optimizer": "paged_adamw_32bit",
        "deepspeed": "zero_stage3_offload_config.json",
        "deepspeed_zero_stage": 3,
        "deepspeed_offload": "disabled",
        "attention_backend": "flash_attention_2",
        "torch_dtype": "bfloat16",
    }
    recipe["effective_batch_size"] = (
        recipe["per_device_batch_size"] * recipe["gradient_accumulation_steps"]
    )
    output.mkdir(parents=True, exist_ok=False)
    work = output.parent / f".{output.name}-work"
    work.mkdir(parents=True, exist_ok=True)
    command = [
        "accelerate", "launch",
        "--config_file", "configs/accelerate/default_config.yaml",
        "--num_processes", "1",
        "--main_process_port", str(free_port()),
        "src/train.py", "--config-name=unlearn.yaml",
        "experiment=unlearn/tofu/default.yaml", "mode=unlearn", "trainer=NPO",
        "task_name=ai4ai_llama32_1b_npo", "model=Llama-3.2-1B-Instruct",
        "forget_split=forget10", "retain_split=retain90",
        f"model.model_args.pretrained_model_name_or_path={start}",
        f"model.tokenizer_args.pretrained_model_name_or_path={start}",
        "model.model_args.attn_implementation=flash_attention_2",
        "model.model_args.torch_dtype=bfloat16",
        f"hydra.run.dir={work / 'hydra'}", "hydra.output_subdir=.hydra",
        f"paths.output_dir={output}",
        "data.forget.TOFU_QA_forget.args.hf_args.path=json",
        "~data.forget.TOFU_QA_forget.args.hf_args.name",
        f"+data.forget.TOFU_QA_forget.args.hf_args.data_files={data / 'forget10.json'}",
        "data.retain.TOFU_QA_retain.args.hf_args.path=json",
        "~data.retain.TOFU_QA_retain.args.hf_args.name",
        f"+data.retain.TOFU_QA_retain.args.hf_args.data_files={data / 'retain90.json'}",
        "~eval.tofu", "trainer.args.do_eval=false", "trainer.args.eval_on_start=false",
        "trainer.args.eval_strategy=no",
        f"trainer.args.save_strategy={'steps' if save_unit == 'step' else 'epoch'}",
        f"+trainer.args.save_steps={save_interval}",
        "trainer.args.save_only_model=true",
        f"+trainer.args.save_total_limit={save_total_limit if save_unit == 'step' and save_total_limit else max(1, save_total_limit * save_interval) if save_total_limit else 1000000}",
        "trainer.args.remove_unused_columns=false",
        f"trainer.args.learning_rate={recipe['learning_rate']}",
        f"trainer.args.num_train_epochs={recipe['epochs']}",
        f"trainer.args.per_device_train_batch_size={recipe['per_device_batch_size']}",
        f"trainer.args.gradient_accumulation_steps={recipe['gradient_accumulation_steps']}",
        "trainer.args.weight_decay=0.01", "trainer.args.warmup_epochs=1.0",
        "+trainer.args.max_grad_norm=1.0", "trainer.args.optim=paged_adamw_32bit",
        f"trainer.args.seed={recipe['seed']}", "trainer.args.logging_steps=1",
        "trainer.args.report_to=none", "trainer.args.ddp_find_unused_parameters=null",
        "trainer.args.gradient_checkpointing=true",
        f"trainer.method_args.beta={recipe['beta']}",
        f"trainer.method_args.alpha={recipe['alpha']}",
        f"trainer.method_args.gamma={recipe['gamma']}",
        "trainer.method_args.retain_loss_type=NLL",
    ]
    if max_steps > 0:
        command.append(f"+trainer.args.max_steps={max_steps}")

    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    log_path = work / "training.log"
    log_path.touch()
    gpu_uuid: str | None = None
    telemetry_samples = 0
    peak_memory_mib = 0.0
    peak_utilization = 0.0
    telemetry_failure: str | None = None
    run_env = os.environ.copy()
    run_env.update({
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1", "WANDB_MODE": "disabled",
        "HOME": str(work / "home"), "HF_HOME": str(work / "huggingface"),
        "TMPDIR": str(work / "tmp"), "XDG_CACHE_HOME": str(work / "xdg"),
    })
    for path in ("home", "huggingface", "tmp", "xdg"):
        (work / path).mkdir(parents=True, exist_ok=True)
    returncode = 125
    wall_clock_stop = False
    try:
        gpu_uuid, memory, utilization = gpu_sample()
        telemetry_samples = 1
        peak_memory_mib = memory
        peak_utilization = utilization
    except Exception as exc:
        telemetry_failure = f"{type(exc).__name__}: {exc}"
    if telemetry_failure is None:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=UPSTREAM,
                env=run_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            try:
                while process.poll() is None:
                    if (
                        train_budget_seconds is not None
                        and time.monotonic() - started >= train_budget_seconds
                    ):
                        wall_clock_stop = True
                        terminate_process_group(process)
                        break
                    observed_uuid, memory, utilization = gpu_sample()
                    if observed_uuid != gpu_uuid:
                        raise RuntimeError(
                            f"visible GPU changed during training: {gpu_uuid} -> {observed_uuid}"
                        )
                    telemetry_samples += 1
                    peak_memory_mib = max(peak_memory_mib, memory)
                    peak_utilization = max(peak_utilization, utilization)
                    time.sleep(5)
                returncode = process.wait()
            except Exception as exc:
                telemetry_failure = f"{type(exc).__name__}: {exc}"
                terminate_process_group(process)
                returncode = process.returncode if process.returncode is not None else 125
    runtime = time.monotonic() - started
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    dynamics = {
        key: float(matches[-1])
        for key in (
            "train_runtime",
            "train_samples_per_second",
            "train_steps_per_second",
            "train_loss",
        )
        if (matches := re.findall(rf"['\"]?{key}['\"]?\s*[:=]\s*([0-9.eE+-]+)", log_text))
    }
    expected_wall_exit = wall_clock_stop and returncode in {-signal.SIGTERM, -signal.SIGKILL, 143}
    receipt = {
        "schema_version": 1,
        "status": "pending_export",
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "returncode": returncode,
        "wall_clock_stop": wall_clock_stop,
        "runtime_seconds": runtime,
        "gpu_telemetry": {
            "gpu_uuid": gpu_uuid,
            "sample_count": telemetry_samples,
            "peak_memory_used_mib": peak_memory_mib,
            "peak_utilization_percent": peak_utilization,
            "source": "nvidia-smi inside the single-GPU training container",
            "failure": telemetry_failure,
        },
        "recipe": recipe,
        "training_start_weight_sha256": EXPECTED_START_WEIGHT,
        "training_data_sha256": observed_data,
        "command": command,
        "training_dynamics": dynamics,
        "log": str(log_path),
    }
    atomic_json(work / "training-receipt.json", receipt)
    fcntl.flock(training_lock.fileno(), fcntl.LOCK_UN)
    training_lock.close()
    if telemetry_failure is not None:
        raise RuntimeError(f"training telemetry failed closed: {telemetry_failure}")
    if returncode != 0 and not expected_wall_exit:
        raise SystemExit(returncode)
    periodic = completed_trainer_checkpoints(output)
    if has_model_weights(output):
        final_source = output
    elif expected_wall_exit and periodic:
        # SIGTERM may prevent Trainer's normal root export.  Preserve the newest
        # already-atomic periodic model as the unconditional planned-stop candidate.
        final_source = periodic[-1][1]
    else:
        raise RuntimeError(f"trainer produced no complete model checkpoint in {output}")
    final_model = output.parent / "final-model"
    if final_model.exists():
        raise FileExistsError(f"refusing to overwrite {final_model}")
    copy_hf_files(final_source, final_model)
    top_up_tokenizer_and_config(final_model, start)
    has_tokenizer = (final_model / "tokenizer.json").is_file() or (
        final_model / "tokenizer.model"
    ).is_file()
    if (
        not (final_model / "config.json").is_file()
        or not (final_model / "tokenizer_config.json").is_file()
        or not has_tokenizer
        or not has_model_weights(final_model)
    ):
        raise RuntimeError(f"final export is not a self-contained HF checkpoint: {final_model}")
    for checkpoint in output.glob("checkpoint-*"):
        if not checkpoint.is_dir():
            continue
        for name in (
            "config.json", "generation_config.json", "tokenizer.json",
            "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json",
        ):
            source = final_model / name
            target = checkpoint / name
            if source.is_file() and not target.exists():
                shutil.copy2(source, target)
    states = []
    for state_path in output.glob("checkpoint-*/trainer_state.json"):
        try:
            states.append(json.loads(state_path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass
    if states:
        receipt["completed_steps"] = max(int(state.get("global_step", 0)) for state in states)
        receipt["completed_epoch"] = max(int(float(state.get("epoch", 0))) for state in states)
    elif max_steps > 0:
        receipt["completed_steps"] = max_steps
    else:
        receipt["completed_epoch"] = int(recipe["epochs"])
    receipt["status"] = "passed"
    receipt["final_source"] = str(final_source)
    receipt["planned_wall_clock_stop_accepted"] = expected_wall_exit
    atomic_json(work / "training-receipt.json", receipt)
    atomic_json(output / "training_metadata.json", receipt)
    print(
        json.dumps(
            {"checkpoint": str(output), "runtime_seconds": runtime, **dynamics},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
