"""Produce one pruned OPT-6.7B from the frozen dense start.

Ported from the reference protocol's baseline/method/train.py. It is named prune.py because
that is what it does: there is no training loop in this task, no optimizer, no
learning rate and no checkpoint sequence. It runs one activation-aware pruning pass
and exports one model.

Three things the old version did are gone, all of them parts of the v1 constraint
model that was dropped:

  load_recipe()        read baseline/recipe.toml and rejected the run if
                       prune_method was outside {wanda, wanda_owl}, if
                       sparsity_ratio was not 0.7, or if sparsity_type was not
                       "unstructured". Recipe-side enforcement of "still the same
                       algorithm". v1 does not do that -- the boundary is the mount
                       list, plus one artifact-side sparsity check in the scoring
                       harness that this file cannot reach.
  run_one_step()       a GPU-backed admission gate that pruned one linear module
                       without exporting, so a candidate could be checked before it
                       was allowed to run. There is no candidate admission step in
                       v1.
  PROFILES             {"proxy": 128, "formal": 128} -- two names for one number.
                       Calibration width is now a run.sh default.

What stayed: the subprocess call into owl_opt.py with cwd set to the OWL tree, the
log parsing, and the summary. The log parsing stayed because upstream OWL reports its
achieved sparsity and per-layer allocation to stdout and nowhere else.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SPARSITY_PATTERN = re.compile(r"sparsity sanity check (?P<value>[0-9.]+)")
LAYER_PATTERN = re.compile(r"layer (?P<layer>[0-9]+) sparsity (?P<value>[0-9.]+)")
MEMORY_PATTERN = re.compile(r"llmab gpu memory peak bytes (?P<value>[0-9]+)")
# The window the scoring harness enforces on the exported weights. Checked here too,
# from the pruning log rather than from the tensors, so a run that missed it fails in
# minutes instead of at score time. This copy is a convenience and is NOT the
# authority: harness/final_eval.py recomputes it from the weights, and an Agent may
# delete these four lines without changing what is enforced.
SPARSITY_WINDOW = (0.699, 0.701)


def parse_log(text: str) -> dict[str, Any]:
    sparsity_matches = list(SPARSITY_PATTERN.finditer(text))
    if not sparsity_matches:
        raise ValueError(
            "the OWL log reports no 'sparsity sanity check' line, so the achieved "
            "sparsity is unknown"
        )
    layers = [float(match.group("value")) for match in LAYER_PATTERN.finditer(text)]
    memory_matches = list(MEMORY_PATTERN.finditer(text))
    return {
        "actual_global_sparsity": float(sparsity_matches[-1].group("value")),
        "layer_sparsities": layers,
        "gpu_memory_peak_bytes": (
            int(memory_matches[-1].group("value")) if memory_matches else None
        ),
    }


def as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def prune(args: argparse.Namespace) -> None:
    dense_model = Path(args.dense_model).resolve()
    calibration = Path(args.calibration_data).resolve()
    owl_source = Path(args.owl_source).resolve()
    output = Path(args.output).resolve()
    pruned_dir = Path(args.pruned_dir).resolve()

    for required in (
        owl_source / "main.py",
        owl_source / "lib/prune_all.py",
        dense_model / "config.json",
        calibration / "en/c4-train.00000-of-01024.json.gz",
    ):
        if not required.exists():
            raise FileNotFoundError(f"required input is missing: {required}")
    output.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(Path(__file__).with_name("owl_opt.py")),
        "--model",
        str(dense_model),
        "--model_name_or_path",
        str(dense_model),
        "--seed",
        str(int(args.calibration_seed)),
        "--nsamples",
        str(int(args.calibration_samples)),
        "--sparsity_ratio",
        str(float(args.sparsity_ratio)),
        "--sparsity_type",
        str(args.sparsity_type),
        "--prune_method",
        str(args.prune_method),
        "--Lamda",
        str(float(getattr(args, "lambda"))),
        "--Hyper_m",
        str(float(args.hyper_m)),
        "--save",
        str(output / "upstream_metrics"),
        "--save_model",
        str(pruned_dir),
    ]
    if as_bool(args.use_variant):
        command.append("--use_variant")
    command.extend(args.extra)

    environment = os.environ.copy()
    environment.update(
        {
            "OWL_SOURCE": str(owl_source),
            "CALIBRATION_DATA": str(calibration),
        }
    )

    resolved = {
        "schema_version": 1,
        "task_id": "owl_wanda_opt6p7b_70pct",
        "dense_model": str(dense_model),
        "calibration_data": str(calibration),
        "calibration_samples": int(args.calibration_samples),
        "calibration_seed": int(args.calibration_seed),
        "sequence_length": 2048,
        "prune_method": str(args.prune_method),
        "sparsity_ratio": float(args.sparsity_ratio),
        "sparsity_type": str(args.sparsity_type),
        "lambda": float(getattr(args, "lambda")),
        "hyper_m": float(args.hyper_m),
        "use_variant": as_bool(args.use_variant),
        "command": command,
        # Recorded for the run log, not read by anything that grades. The scoring
        # harness computes sparsity from the weights and ignores every field here.
        "note": "descriptive only; the scoring harness reads the weights, not this file",
    }
    (output / "retrain_resolved_config.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    started = time.monotonic()
    completed = subprocess.run(
        command, cwd=str(owl_source), env=environment, text=True, capture_output=True, check=False
    )
    elapsed = time.monotonic() - started
    (output / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    # Upstream prints the whole pruning trace; echo the tail so a failure is visible
    # without opening a file.
    sys.stdout.write(completed.stdout[-4000:])
    sys.stderr.write(completed.stderr[-4000:])
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode, command, completed.stdout, completed.stderr
        )

    parsed = parse_log(completed.stdout)
    sparsity = float(parsed["actual_global_sparsity"])
    if not pruned_dir.is_dir():
        raise RuntimeError(f"the pruning pass exported nothing to {pruned_dir}")

    layers = parsed.pop("layer_sparsities")
    metrics = {
        **parsed,
        "layer_sparsity_min": min(layers) if layers else None,
        "layer_sparsity_max": max(layers) if layers else None,
        "layer_sparsity_mean": sum(layers) / len(layers) if layers else None,
        "layer_count": len(layers),
        "calibration_samples": int(args.calibration_samples),
        "calibration_seed": int(args.calibration_seed),
        "prune_wall_seconds": elapsed,
    }
    (output / "retrain_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    low, high = SPARSITY_WINDOW
    in_window = low <= sparsity <= high
    (output / "retrain_summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_id": "owl_wanda_opt6p7b_70pct",
                "status": "pruned" if in_window else "pruned_outside_sparsity_window",
                "artifact": str(pruned_dir),
                "actual_global_sparsity": sparsity,
                "sparsity_in_window": in_window,
                "prune_wall_seconds": elapsed,
                "offline": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"prune: exported {pruned_dir} at sparsity {sparsity:.6f} in {elapsed:.0f}s "
        f"({'in' if in_window else 'OUTSIDE'} [{low}, {high}])"
    )
    if not in_window:
        raise RuntimeError(
            f"achieved sparsity {sparsity:.6f} is outside [{low}, {high}]; the "
            "scoring harness invalidates this artifact"
        )


def smoke() -> None:
    fixture = "\n".join(
        [
            "layer 0 sparsity 0.69000000",
            "layer 1 sparsity 0.71000000",
            "sparsity sanity check 0.7000",
            "llmab gpu memory peak bytes 123456789",
        ]
    )
    parsed = parse_log(fixture)
    assert parsed["actual_global_sparsity"] == 0.7, parsed
    assert parsed["gpu_memory_peak_bytes"] == 123456789, parsed
    assert parsed["layer_sparsities"] == [0.69, 0.71], parsed
    assert as_bool("true") and as_bool("1") and not as_bool("false") and not as_bool("")
    try:
        parse_log("nothing useful here")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("a log with no sparsity line was accepted")
    print("prune.py smoke passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Not `required=True`: --smoke has to run without them, and the build-time check
    # calls it. Validated after parsing instead.
    parser.add_argument("--dense-model")
    parser.add_argument("--calibration-data")
    parser.add_argument("--owl-source", default="/workspace/owl")
    parser.add_argument("--output", default="/out")
    parser.add_argument("--pruned-dir", default="/out/pruned")
    parser.add_argument("--prune-method", default="wanda_owl")
    parser.add_argument("--sparsity-ratio", default="0.7")
    parser.add_argument("--sparsity-type", default="unstructured")
    parser.add_argument("--lambda", dest="lambda", default="0.08")
    parser.add_argument("--hyper-m", default="5.0")
    parser.add_argument("--use-variant", default="false")
    parser.add_argument("--calibration-samples", default="128")
    parser.add_argument("--calibration-seed", default="0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("extra", nargs="*", help="passed through to owl_opt.py")
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return
    missing = [
        flag
        for flag, value in (
            ("--dense-model", args.dense_model),
            ("--calibration-data", args.calibration_data),
        )
        if not value
    ]
    if missing:
        parser.error(f"{', '.join(missing)} required outside --smoke")
    prune(args)


if __name__ == "__main__":
    main()
