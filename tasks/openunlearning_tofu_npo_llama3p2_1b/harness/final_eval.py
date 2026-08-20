"""Frozen three-way OpenUnlearning TOFU evaluator and report-defined Balanced score."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from runtime_guard import GpuTelemetry, resolve_checkpoint

TASK_ID = "openunlearning_tofu_npo_llama3p2_1b"
METRIC = "balanced_unlearning_score"
UPSTREAM = Path("/opt/harness/open-unlearning")
START_WEIGHT_SHA256 = "35b0cd9f7db2735538ff78b4879fab61b23b7ca316df27b059585bf656700a8d"
RETAIN_WEIGHT_SHA256 = "3788e1724ce5563d5bb4cc4e82a985adc85d3c561f05427a63581e7b407c675c"
DATA_HASHES = {
    "forget10.json": "0044c8c2e70a38be93f62ec6cb1c1cc2a1f55a8df2fb549a4da2da8dde9d92f6",
    "forget10_perturbed.json": "6fbcb946c57ea1d7b2124cea0e61bf3b5409d1bc10d02368f090109450ed73c7",
    "retain90.json": "debcf019c41db7b30b62ca6ad41fd8a4bcd29a5b321c21d2961974e308dfca55",
    "retain_perturbed.json": "fc69f33bad70d3dca65920bdb54380039e3e8872dd16bbae20c1a839b99533fb",
    "holdout10.json": "efec32c7a7f66bfefaa90b8a4cc583296b0bbf882495db61076dec2d6edf44dd",
    "real_authors.json": "5ed86f884ecbefc3a5db0b9dfe84ab3db710690c74d9402c0acc14bc3dd5a4b7",
    "real_authors_perturbed.json": (
        "a23255383ac75a5fbe5acd57622a03fb9c368954fd7b2428afa152b784d9533a"
    ),
    "world_facts.json": "5547b07340f46755b868a525941341c3d93814b5370ae7fa10143f653a3c7033",
    "world_facts_perturbed.json": (
        "0e171838040d0ec94b1ce248d891508e85e3b77344370077caa27b61898eb3e1"
    ),
}
NATIVE_FIELDS = (
    "extraction_strength",
    "forget_Q_A_Prob",
    "forget_Q_A_ROUGE",
    "forget_truth_ratio",
    "model_utility",
    "privleak",
)
PATCHED_EVALUATOR_HASHES = {
    "src/evals/metrics/utils.py": (
        "dd343dc01b7b7f650881b361fc780994f0f718b0df404d268f335c4781b0d9b0"
    ),
    "src/evals/metrics/utility.py": (
        "056e242ec3a8bc449e21d8536de99043a0ac985c57e175e80e0d037a7910d5bb"
    ),
    "src/evals/metrics/mia/min_k_plus_plus.py": (
        "d0be5cabe1d66079b7dcbfa2d3e3a55e89472f1be7cd6953e3050a73d0d923e5"
    ),
}

# Hydra expands the TOFU metric graph into several independent dataset nodes.  A
# local directory named ``locuslab/TOFU`` is not a drop-in replacement for the
# Hub dataset because the JSON builder exposes only its ``default`` config.  Bind
# every dataset node used by the frozen metric graph to one of the hash-checked
# evaluator assets instead.  Keep this list explicit: silently accepting a new
# upstream dataset node would widen the final-evaluation data boundary.
FINAL_DATASET_BINDINGS = (
    (
        "eval.tofu.metrics.forget_truth_ratio.pre_compute.forget_Q_A_PARA_Prob."
        "datasets.TOFU_QA_forget_para",
        "forget10_perturbed.json",
    ),
    (
        "eval.tofu.metrics.forget_truth_ratio.pre_compute.forget_Q_A_PERT_Prob."
        "datasets.TOFU_QA_forget_pert",
        "forget10_perturbed.json",
    ),
    (
        "eval.tofu.metrics.forget_quality.pre_compute.forget_truth_ratio.pre_compute."
        "forget_Q_A_PARA_Prob.datasets.TOFU_QA_forget_para",
        "forget10_perturbed.json",
    ),
    (
        "eval.tofu.metrics.forget_quality.pre_compute.forget_truth_ratio.pre_compute."
        "forget_Q_A_PERT_Prob.datasets.TOFU_QA_forget_pert",
        "forget10_perturbed.json",
    ),
    (
        "eval.tofu.metrics.forget_Q_A_Prob.datasets.TOFU_QA_forget",
        "forget10_perturbed.json",
    ),
    (
        "eval.tofu.metrics.forget_Q_A_ROUGE.datasets.TOFU_QA_forget",
        "forget10_perturbed.json",
    ),
    (
        "eval.tofu.metrics.model_utility.pre_compute.retain_Q_A_Prob."
        "datasets.TOFU_QA_retain_eval",
        "retain_perturbed.json",
    ),
    (
        "eval.tofu.metrics.model_utility.pre_compute.retain_Q_A_ROUGE."
        "datasets.TOFU_QA_retain_eval",
        "retain_perturbed.json",
    ),
    (
        "eval.tofu.metrics.model_utility.pre_compute.retain_Truth_Ratio.pre_compute."
        "retain_Q_A_PARA_Prob.datasets.TOFU_QA_retain_para",
        "retain_perturbed.json",
    ),
    (
        "eval.tofu.metrics.model_utility.pre_compute.retain_Truth_Ratio.pre_compute."
        "retain_Q_A_PERT_Prob.datasets.TOFU_QA_retain_pert",
        "retain_perturbed.json",
    ),
    (
        "eval.tofu.metrics.model_utility.pre_compute.ra_Q_A_Prob_normalised.pre_compute."
        "ra_Q_A_Prob.datasets.TOFU_QA_ra",
        "real_authors_perturbed.json",
    ),
    (
        "eval.tofu.metrics.model_utility.pre_compute.ra_Q_A_Prob_normalised.pre_compute."
        "ra_Q_A_PERT_Prob.datasets.TOFU_QA_ra_pert",
        "real_authors_perturbed.json",
    ),
    (
        "eval.tofu.metrics.model_utility.pre_compute.ra_Q_A_ROUGE.datasets.TOFU_QA_ra",
        "real_authors_perturbed.json",
    ),
    (
        "eval.tofu.metrics.model_utility.pre_compute.ra_Truth_Ratio.pre_compute."
        "ra_Q_A_Prob.datasets.TOFU_QA_ra",
        "real_authors_perturbed.json",
    ),
    (
        "eval.tofu.metrics.model_utility.pre_compute.ra_Truth_Ratio.pre_compute."
        "ra_Q_A_PERT_Prob.datasets.TOFU_QA_ra_pert",
        "real_authors_perturbed.json",
    ),
    (
        "eval.tofu.metrics.model_utility.pre_compute.wf_Q_A_Prob_normalised.pre_compute."
        "wf_Q_A_Prob.datasets.TOFU_QA_wf",
        "world_facts_perturbed.json",
    ),
    (
        "eval.tofu.metrics.model_utility.pre_compute.wf_Q_A_Prob_normalised.pre_compute."
        "wf_Q_A_PERT_Prob.datasets.TOFU_QA_wf_pert",
        "world_facts_perturbed.json",
    ),
    (
        "eval.tofu.metrics.model_utility.pre_compute.wf_Q_A_ROUGE.datasets.TOFU_QA_wf",
        "world_facts_perturbed.json",
    ),
    (
        "eval.tofu.metrics.model_utility.pre_compute.wf_Truth_Ratio.pre_compute."
        "wf_Q_A_Prob.datasets.TOFU_QA_wf",
        "world_facts_perturbed.json",
    ),
    (
        "eval.tofu.metrics.model_utility.pre_compute.wf_Truth_Ratio.pre_compute."
        "wf_Q_A_PERT_Prob.datasets.TOFU_QA_wf_pert",
        "world_facts_perturbed.json",
    ),
    (
        "eval.tofu.metrics.privleak.pre_compute.mia_min_k.datasets.TOFU_QA_forget",
        "forget10_perturbed.json",
    ),
    (
        "eval.tofu.metrics.privleak.pre_compute.mia_min_k.datasets.TOFU_QA_holdout",
        "holdout10.json",
    ),
    (
        "eval.tofu.metrics.extraction_strength.datasets.TOFU_QA_forget",
        "forget10_perturbed.json",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def balanced_score(
    start: dict[str, Any],
    retain: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, float]:
    denominator = float(start["extraction_strength"]) - float(retain["extraction_strength"])
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError(f"invalid Extraction normalization denominator: {denominator}")
    start_mu = float(start["model_utility"])
    if not math.isfinite(start_mu) or start_mu <= 0:
        raise ValueError(f"invalid training-start model utility: {start_mu}")
    forget_progress = clip01(
        (float(start["extraction_strength"]) - float(candidate["extraction_strength"]))
        / denominator
    )
    utility_retention = clip01(float(candidate["model_utility"]) / start_mu)
    score = 0.0
    if forget_progress + utility_retention > 0:
        score = 2.0 * forget_progress * utility_retention / (forget_progress + utility_retention)
    return {
        "forget_progress": forget_progress,
        "utility_retention": utility_retention,
        METRIC: score,
        "normalization_denominator": denominator,
    }


def weights_digest(root: Path) -> str:
    files = sorted((*root.glob("*.safetensors"), *root.glob("*.bin")))
    if not files:
        raise ValueError(f"no model weights in {root}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def validate_assets(assets: Path) -> dict[str, Any]:
    start = assets / "models/training_start"
    retain = assets / "models/retain_reference"
    data = assets / "data/final"
    observed = {
        "training_start_model": sha256(start / "model.safetensors"),
        "retain_reference_model": sha256(retain / "model.safetensors"),
        "data": {name: sha256(data / name) for name in DATA_HASHES},
        "evaluator": {name: sha256(UPSTREAM / name) for name in PATCHED_EVALUATOR_HASHES},
    }
    if observed["training_start_model"] != START_WEIGHT_SHA256:
        raise ValueError("training-start anchor hash mismatch")
    if observed["retain_reference_model"] != RETAIN_WEIGHT_SHA256:
        raise ValueError("retain90 reference anchor hash mismatch")
    if observed["data"] != DATA_HASHES:
        raise ValueError("TOFU final data hash mismatch")
    if observed["evaluator"] != PATCHED_EVALUATOR_HASHES:
        raise ValueError("frozen BF16-compatible evaluator hash mismatch")
    return observed


def local_dataset_overrides(data: Path) -> list[str]:
    """Route the frozen TOFU metric graph through hash-checked local JSON files."""

    unknown = sorted({filename for _, filename in FINAL_DATASET_BINDINGS} - DATA_HASHES.keys())
    if unknown:
        raise ValueError(f"final dataset binding is outside the frozen asset allowlist: {unknown}")
    overrides: list[str] = []
    for node, filename in FINAL_DATASET_BINDINGS:
        overrides.extend(
            (
                f"{node}.args.hf_args.path=json",
                f"~{node}.args.hf_args.name",
                f"+{node}.args.hf_args.data_files={data / filename}",
            )
        )
    return overrides


def eval_command(
    model: Path,
    output: Path,
    task_name: str,
    retain_logs: Path | None,
    data: Path,
) -> list[str]:
    retain_arg = "null" if retain_logs is None else str(retain_logs)
    command = [
        "python3", str(UPSTREAM / "src/eval.py"),
        "experiment=eval/tofu/default.yaml", "model=Llama-3.2-1B-Instruct",
        f"model.model_args.pretrained_model_name_or_path={model}",
        f"model.tokenizer_args.pretrained_model_name_or_path={model}",
        "model.model_args.attn_implementation=flash_attention_2",
        "model.model_args.torch_dtype=bfloat16",
        "forget_split=forget10", "holdout_split=holdout10",
        f"retain_logs_path={retain_arg}", f"task_name={task_name}",
        f"paths.output_dir={output}", "eval.tofu.overwrite=true",
    ]
    command.extend(local_dataset_overrides(data))
    return command


def load_native(path: Path, require_forget_quality: bool) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fields = (*NATIVE_FIELDS, "forget_quality") if require_forget_quality else NATIVE_FIELDS
    result: dict[str, float] = {}
    for field in fields:
        value = float(payload[field])
        if not math.isfinite(value):
            raise ValueError(f"non-finite {field} in {path}")
        result[field] = value
    return result


def terminate_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def run_native_eval(
    command: list[str],
    work: Path,
    run_env: dict[str, str],
    log_path: Path,
) -> tuple[int, float, dict[str, Any]]:
    telemetry = GpuTelemetry()
    telemetry.observe()
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=work,
            env=run_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                telemetry.observe()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
            returncode = process.wait()
            telemetry.observe()
        except Exception:
            terminate_process_group(process)
            raise
    return returncode, time.monotonic() - started, telemetry.summary()


def prepare_output(output: Path) -> None:
    """Allow the formal checkpoint beside evaluator-owned, single-use output."""

    owned = (
        "work",
        "runtime",
        "official_eval",
        "summary.json",
        "official-eval-receipt.json",
        "retain90.log",
        "training_start.log",
        "candidate.log",
    )
    collisions = [output / name for name in owned if (output / name).exists()]
    if collisions:
        raise FileExistsError(f"refusing to overwrite evaluator output: {collisions}")
    output.mkdir(parents=True, exist_ok=True)


def evaluate(checkpoint: Path, assets: Path, output: Path, reward_path: Path) -> dict[str, Any]:
    prepare_output(output)
    candidate = resolve_checkpoint(checkpoint)
    start = assets / "models/training_start"
    retain = assets / "models/retain_reference"
    observed = validate_assets(assets)
    candidate_digest = weights_digest(candidate)
    start_digest = weights_digest(start)
    if candidate_digest == start_digest:
        raise ValueError("candidate weights are byte-identical to the training start")

    work = output / "work"
    work.mkdir(parents=True, exist_ok=True)
    runtime = output / "runtime"
    for name in ("home", "huggingface", "torch", "tmp", "xdg"):
        (runtime / name).mkdir(parents=True, exist_ok=True)
    eval_root = output / "official_eval"
    eval_root.mkdir()
    run_env = os.environ.copy()
    run_env.update({
        "HOME": str(runtime / "home"), "HF_HOME": str(runtime / "huggingface"),
        "HF_DATASETS_CACHE": str(runtime / "huggingface/datasets"),
        "TRANSFORMERS_CACHE": str(runtime / "huggingface/transformers"),
        "TORCH_HOME": str(runtime / "torch"), "TMPDIR": str(runtime / "tmp"),
        "XDG_CACHE_HOME": str(runtime / "xdg"), "WANDB_MODE": "disabled",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    })
    specs = (
        ("retain90", retain, None, False),
        ("training_start", start, eval_root / "retain90/TOFU_EVAL.json", True),
        ("candidate", candidate, eval_root / "retain90/TOFU_EVAL.json", True),
    )
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "OpenUnlearning official TOFU three-way native evaluator, BF16 + FA2",
        "asset_hashes": observed,
        "candidate_weights_sha256": candidate_digest,
        "training_start_weights_sha256": start_digest,
        "runs": [],
    }
    started = time.monotonic()
    try:
        for name, model, retain_logs, require_fq in specs:
            target = eval_root / name
            command = eval_command(model, target, name, retain_logs, assets / "data/final")
            returncode, run_runtime, run_telemetry = run_native_eval(
                command,
                work,
                run_env,
                output / f"{name}.log",
            )
            receipt["runs"].append(
                {
                    "name": name,
                    "command": command,
                    "returncode": returncode,
                    "runtime_seconds": run_runtime,
                    "gpu_telemetry": run_telemetry,
                }
            )
            if returncode != 0:
                raise RuntimeError(f"native {name} evaluation failed with {returncode}")
            load_native(target / "TOFU_SUMMARY.json", require_fq)
        retain_native = load_native(eval_root / "retain90/TOFU_SUMMARY.json", False)
        start_native = load_native(eval_root / "training_start/TOFU_SUMMARY.json", True)
        candidate_native = load_native(eval_root / "candidate/TOFU_SUMMARY.json", True)
        balanced = balanced_score(start_native, retain_native, candidate_native)
        gpu_telemetry = {
            "gpu_uuid": receipt["runs"][0]["gpu_telemetry"]["gpu_uuid"],
            "sample_count": sum(
                run["gpu_telemetry"]["sample_count"] for run in receipt["runs"]
            ),
            "peak_memory_used_mib": max(
                run["gpu_telemetry"]["peak_memory_used_mib"] for run in receipt["runs"]
            ),
            "peak_utilization_percent": max(
                run["gpu_telemetry"]["peak_utilization_percent"]
                for run in receipt["runs"]
            ),
            "source": "nvidia-smi inside the single-GPU score container",
        }
        if any(
            run["gpu_telemetry"]["gpu_uuid"] != gpu_telemetry["gpu_uuid"]
            for run in receipt["runs"]
        ):
            raise RuntimeError("visible GPU changed between native evaluator runs")
        summary = {
            "schema_version": 1,
            "task_id": TASK_ID,
            "metric": METRIC,
            "direction": "maximize",
            "score": balanced[METRIC],
            METRIC: balanced[METRIC],
            "balanced_provenance": {
                "kind": "report_defined_local_composite_not_native_openunlearning",
                "formula": (
                    "harmonic_mean(clip((start_extraction-candidate_extraction)/"
                    "(start_extraction-retain90_extraction),0,1), "
                    "clip(candidate_MU/start_MU,0,1))"
                ),
                **balanced,
            },
            "native_openunlearning": {
                "training_start": start_native,
                "retain90_reference": retain_native,
                "candidate": candidate_native,
            },
            "validity": {
                "status": "passed", "three_native_evaluations_succeeded": True,
                "candidate_differs_from_training_start": True,
                "all_reported_values_finite": True,
                "normalization_denominator_positive": True,
            },
            "runtime_seconds": time.monotonic() - started,
            "gpu_telemetry": gpu_telemetry,
            "candidate_weights_sha256": candidate_digest,
        }
        atomic_json(output / "summary.json", summary)
        reward_path.parent.mkdir(parents=True, exist_ok=True)
        reward_path.write_text(f"{summary['score']:.10f}\n", encoding="utf-8")
        receipt["status"] = "passed"
        return summary
    except Exception as exc:
        receipt["status"] = "failed"
        receipt["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
        receipt["runtime_seconds"] = time.monotonic() - started
        atomic_json(output / "official-eval-receipt.json", receipt)


def mock(output: Path, reward_path: Path) -> dict[str, Any]:
    retain = {
        "extraction_strength": 0.10,
        "model_utility": 0.80,
        "forget_Q_A_Prob": 0.10,
        "forget_Q_A_ROUGE": 0.20,
        "forget_truth_ratio": 0.70,
        "privleak": 20.0,
    }
    start = {
        "extraction_strength": 0.80,
        "model_utility": 0.75,
        "forget_quality": 0.01,
        "forget_Q_A_Prob": 0.85,
        "forget_Q_A_ROUGE": 0.80,
        "forget_truth_ratio": 0.45,
        "privleak": -80.0,
    }
    candidate = {
        "extraction_strength": 0.25,
        "model_utility": 0.70,
        "forget_quality": 0.40,
        "forget_Q_A_Prob": 0.20,
        "forget_Q_A_ROUGE": 0.30,
        "forget_truth_ratio": 0.65,
        "privleak": 18.0,
    }
    balanced = balanced_score(start, retain, candidate)
    summary = {
        "schema_version": 1, "task_id": TASK_ID, "metric": METRIC,
        "direction": "maximize", "mock": True, "score": balanced[METRIC],
        METRIC: balanced[METRIC],
        "balanced_provenance": {
            "kind": "report_defined_local_composite_not_native_openunlearning",
            **balanced,
        },
        "native_openunlearning": {
            "training_start": start,
            "retain90_reference": retain,
            "candidate": candidate,
        },
        "validity": {
            "status": "passed",
            "mock": True,
            "fixture": "synthetic contract check; not a measured result",
        },
    }
    atomic_json(output / "summary.json", summary)
    reward_path.parent.mkdir(parents=True, exist_ok=True)
    reward_path.write_text(f"{summary['score']:.10f}\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--assets", type=Path, default=Path("/assets"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reward-path", type=Path, default=Path("/logs/verifier/reward"))
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    if args.mock:
        result = mock(args.output, args.reward_path)
    else:
        if args.checkpoint is None:
            parser.error("--checkpoint is required unless --mock is used")
        result = evaluate(args.checkpoint, args.assets, args.output, args.reward_path)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
