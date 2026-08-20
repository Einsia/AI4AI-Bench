"""Build the DiGress Hydra command, enforce the wall clock and export checkpoints.

The method and vendored training tree are editable.  The shipped recipe installs a configurable
Lightning callback in that editable tree, verifies every completed checkpoint, and atomically
publishes standard candidates.  The frozen evaluator independently reloads accepted candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

TASK_ID = "digress_qm9_graph_diffusion"
UPSTREAM_REVISION = "780242b8d3e7d78316bb5cf90c639fb0cd4c6079"
# How long Lightning gets between SIGTERM and SIGKILL. Taken out of the reserve
# rather than added to it, so the whole shutdown still finishes before the wall.
GRACE_SECONDS = 300


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_metrics(text: str) -> dict[str, float | None]:
    """The run's own dynamics, for the log. The score comes from the harness."""

    def last(pattern: str, percent: bool = False) -> float | None:
        matches = re.findall(pattern, text, flags=re.MULTILINE)
        if not matches:
            return None
        value = float(matches[-1])
        return value / 100.0 if percent else value

    return {
        "validity": last(r"Validity over \d+ molecules:\s*([0-9.]+)%", percent=True),
        "relaxed_validity": last(
            r"Relaxed validity over \d+ molecules:\s*([0-9.]+)%", percent=True
        ),
        "uniqueness": last(r"Uniqueness over \d+ valid molecules:\s*([0-9.]+)%", percent=True),
        "novelty": last(r"Novelty over \d+ unique valid molecules:\s*([0-9.]+)%", percent=True),
        "val_nll": last(r"(?:Val|Test) NLL\s+(-?[0-9.]+)"),
        "epochs_seen": last(r"[Ee]poch\s+(\d+)"),
    }


def run_with_wall_clock(
    command: list[str],
    cwd: Path,
    environment: dict[str, str],
    log_path: Path,
    wall_seconds: int,
    reserve_seconds: int,
) -> dict[str, Any]:
    """Run the trainer, stopping it in time to write a complete checkpoint.

    `wall_seconds` of 0 means no deadline: the container's own timeout is then the
    only limit, and the last checkpoint is whichever epoch happened to finish.

    The child gets its own process group (`start_new_session`) so the signal reaches
    Lightning's dataloader workers too. Signalling only the parent leaves worker
    processes holding the GPU, which turns a clean stop into a hang.
    """

    deadline = None
    if wall_seconds > 0:
        budget = max(wall_seconds - reserve_seconds, 60)
        deadline = time.monotonic() + budget
        print(
            f"train.py: training for at most {budget}s, then SIGTERM ({reserve_seconds}s reserve, "
            f"{GRACE_SECONDS}s grace)",
            flush=True,
        )

    started = time.monotonic()
    stopped_by_wall_clock = False
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while True:
                status = process.poll()
                if status is not None:
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    stopped_by_wall_clock = True
                    print(
                        "train.py: wall clock reached; SIGTERM so Lightning can "
                        "finish the epoch and write last.ckpt",
                        flush=True,
                    )
                    _terminate(process)
                    break
                time.sleep(5)
        except KeyboardInterrupt:
            _terminate(process)
            raise
        status = process.wait()

    return {
        "exit_status": status,
        "stopped_by_wall_clock": stopped_by_wall_clock,
        "wall_seconds": time.monotonic() - started,
    }


def _terminate(process: subprocess.Popen[Any]) -> None:
    """SIGTERM the whole group, then SIGKILL what is left after the grace period."""

    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        process.terminate()
    try:
        process.wait(timeout=GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        print(
            f"train.py: still running {GRACE_SECONDS}s after SIGTERM; SIGKILL. The last complete "
            "checkpoint on disk is what gets exported.",
            flush=True,
        )
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        process.kill()


def newest_checkpoint(root: Path, exclude: Path) -> Path | None:
    """The most recently written last.ckpt below `root`, ignoring the export dir."""

    found = [
        path
        for path in root.rglob("last.ckpt")
        if path.is_file() and exclude not in path.parents and path.parent != exclude
    ]
    if not found:
        found = [path for path in root.rglob("*.ckpt") if path.is_file() and path.parent != exclude]
    if not found:
        return None
    return sorted(found, key=lambda path: path.stat().st_mtime)[-1]


IMPORT_OLD = '''from pytorch_lightning.callbacks import ModelCheckpoint
'''

IMPORT_NEW = '''from pytorch_lightning.callbacks import ModelCheckpoint, Timer
'''

CALLBACK_OLD = '''    callbacks = []
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                              filename='{epoch}',
                                              monitor='val/epoch_NLL',
                                              save_top_k=5,
                                              mode='min',
                                              every_n_epochs=1)
        last_ckpt_save = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}", filename='last', every_n_epochs=1)
        callbacks.append(last_ckpt_save)
        callbacks.append(checkpoint_callback)
'''

CALLBACK_NEW = '''    callbacks = []
    if cfg.train.save_model:
        checkpoint_dir = os.environ['AI4AI_DIGRESS_CHECKPOINT_WORK']
        checkpoint_unit = os.environ.get('AI4AI_SAVE_UNIT', 'epoch')
        checkpoint_interval = int(os.environ.get('AI4AI_SAVE_INTERVAL', '50'))
        checkpoint_kwargs = dict(
            dirpath=checkpoint_dir,
            filename='periodic-{epoch}-{step}',
            save_top_k=-1,
            save_weights_only=False,
        )
        if checkpoint_unit == 'step':
            checkpoint_kwargs.update(every_n_train_steps=checkpoint_interval, every_n_epochs=0)
        elif checkpoint_unit == 'epoch':
            checkpoint_kwargs.update(
                every_n_epochs=checkpoint_interval,
                every_n_train_steps=0,
                save_on_train_epoch_end=True,
            )
        else:
            raise ValueError(f"unsupported AI4AI_SAVE_UNIT: {checkpoint_unit}")
        callbacks.append(ModelCheckpoint(**checkpoint_kwargs))
        # Stop through Lightning before the host's hard deadline.  fit() then returns
        # normally and the final save below captures the exact stopped progress.
        train_budget = int(os.environ.get('AI4AI_DIGRESS_TRAIN_BUDGET', '0'))
        if train_budget > 0:
            callbacks.append(Timer(duration=dict(seconds=train_budget), interval='step'))
'''

FIT_OLD = '''        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
        if cfg.general.name not in ['debug', 'test']:
'''

FIT_NEW = '''        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
        # A normal completion always has an exact final, complete Lightning checkpoint.
        trainer.save_checkpoint(os.path.join(
            os.environ['AI4AI_DIGRESS_CHECKPOINT_WORK'], 'final.ckpt'
        ))
        if cfg.general.name not in ['debug', 'test']:
'''


def configure_checkpointing(upstream: Path) -> None:
    """Configure the pinned callback without hiding it from an Agent.

    Both this wrapper and ``upstream`` are candidate-owned.  Exact replacements deliberately
    fail if an Agent changes the relevant code without also supplying its own saving policy.
    """

    target = upstream / "src/main.py"
    text = target.read_text(encoding="utf-8")
    if text.count(IMPORT_OLD) == 1:
        text = text.replace(IMPORT_OLD, IMPORT_NEW)
    elif text.count(IMPORT_NEW) != 1:
        raise RuntimeError(
            "DiGress Lightning imports changed; update train.py's saving integration too"
        )
    if text.count(CALLBACK_OLD) == 1:
        text = text.replace(CALLBACK_OLD, CALLBACK_NEW)
    elif text.count(CALLBACK_NEW) != 1:
        raise RuntimeError(
            "DiGress checkpoint callback changed; update train.py's saving integration too"
        )
    if text.count(FIT_OLD) == 1:
        text = text.replace(FIT_OLD, FIT_NEW)
    elif text.count(FIT_NEW) != 1:
        raise RuntimeError(
            "DiGress fit path changed; update train.py's final-checkpoint integration too"
        )
    target.write_text(text, encoding="utf-8")


def export_checkpoints(
    upstream: Path,
    source_root: Path,
    standard_root: Path,
    save_unit: str,
    save_total_limit: int,
) -> list[dict[str, Any]]:
    """Verify and atomically publish all distinct progress points."""

    allow_unpickling(upstream)
    by_progress: dict[int, tuple[Path, dict[str, Any]]] = {}
    for source in sorted(source_root.glob("*.ckpt"), key=lambda path: path.stat().st_mtime):
        structure = verify_loadable(source)
        raw_progress = structure.get("epoch") if save_unit == "epoch" else structure.get("global_step")
        if raw_progress is None:
            raw_progress = structure.get("global_step") or structure.get("epoch")
        if raw_progress is None:
            continue
        progress = int(raw_progress)
        # The final checkpoint has the newest mtime and therefore wins a duplicate progress.
        by_progress[progress] = (source, structure)

    if not by_progress:
        raise RuntimeError(f"training produced no loadable checkpoint in {source_root}")

    exported: list[dict[str, Any]] = []
    helper = Path("/opt/harness/save_checkpoint.py")
    for progress, (source, structure) in sorted(by_progress.items()):
        subprocess.run(
            [
                sys.executable,
                str(helper),
                "--output",
                str(standard_root.parent),
                "--progress",
                str(progress),
                "--source",
                str(source),
                "--payload-name",
                "model.ckpt",
                "--retention",
                str(save_total_limit),
            ],
            check=True,
        )
        exported.append(
            {
                "progress": progress,
                "source": str(source),
                "path": str(standard_root / f"checkpoint-{progress}" / "model.ckpt"),
                "sha256": file_sha256(source),
                "structure": structure,
            }
        )
    retained = {path.name for path in standard_root.glob("checkpoint-*") if path.is_dir()}
    return [item for item in exported if f"checkpoint-{item['progress']}" in retained]


def allow_unpickling(upstream: Path) -> None:
    """Put the trainer's own tree on sys.path so its checkpoint can be torch.loaded.

    MEASURED on the B300, and it broke every retrain on this task. Lightning pickles the
    LightningModule's hyper_parameters, and DiGress's are live objects rather than plain
    values:

        cfg                 omegaconf.dictconfig.DictConfig
        dataset_infos       datasets.qm9_dataset.QM9infos
        visualization_tools analysis.visualization.MolecularVisualization
        extra_features      diffusion.extra_features.ExtraFeatures
        domain_features     diffusion.extra_features_molecular.ExtraMolecularFeatures

    Unpickling imports each defining module under the name it had when it was pickled.
    The trainer is launched as `python3 <upstream>/src/main.py`, so its sys.path[0] was
    <upstream>/src and those names were TOP-LEVEL there. This file runs as
    `python3 /workspace/train.py`, so sys.path[0] is /workspace and the image's ENV
    PYTHONPATH is /workspace/digress -- neither provides <upstream>/src, and
    verify_loadable died with

        ModuleNotFoundError: No module named 'datasets'

    AFTER copying the checkpoint and BEFORE writing training_metadata.json. So an 11.7 h
    retrain exited nonzero with a perfectly good checkpoint on disk and no metadata beside
    it, and the phase after it never ran.

    BOTH entries are needed: datasets/qm9_dataset.py itself does `import src.utils`, which
    wants the tree ROOT, not its src/. Adding only one moves the error rather than fixing
    it.

    The tree used is THIS RUN'S upstream -- the copy that produced the checkpoint and whose
    class definitions the pickle names. Not /opt/harness/digress: if a candidate changed a
    class, the checkpoint's pickle refers to the candidate's version of it.

    APPENDED rather than inserted at 0, because this tree carries modules with ordinary
    names -- utils, models, metrics -- and they must not shadow a real dependency.
    """

    for entry in (upstream / "src", upstream):
        text = str(entry)
        if entry.is_dir() and text not in sys.path:
            sys.path.append(text)


def verify_loadable(path: Path) -> dict[str, Any]:
    """Load the exported checkpoint back, so a truncated file fails here and loudly.

    The reserve exists to prevent a checkpoint that is half written. This is the
    check that the reserve worked: an export that produced an unloadable file is the
    failure mode that otherwise surfaces an hour later inside the scoring phase.

    Call allow_unpickling() first, or this raises on the pickled hyper_parameters
    rather than on anything about the file being whole.
    """

    import torch

    blob = torch.load(str(path), map_location="cpu")
    if not isinstance(blob, dict) or "state_dict" not in blob:
        raise RuntimeError(
            f"the exported checkpoint at {path} is not a Lightning checkpoint"
        )
    tensors = sum(1 for value in blob["state_dict"].values() if hasattr(value, "shape"))
    if tensors == 0:
        raise RuntimeError(f"the exported checkpoint holds no tensors: {path}")
    return {
        "tensors": tensors,
        "epoch": blob.get("epoch"),
        "global_step": blob.get("global_step"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=Path("/workspace/digress"))
    parser.add_argument("--data", type=Path, default=Path("/assets/data/qm9_no_h"))
    parser.add_argument("--output", type=Path, default=Path("/out"))
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--run-name", default="digress_qm9")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--num-layers", type=int, default=9)
    parser.add_argument("--diffusion-steps", type=int, default=500)
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1.0e-12)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help="positive values bound normal-mode training batches; 0 leaves the full epoch",
    )
    parser.add_argument(
        "--max-wall-seconds",
        type=int,
        default=0,
        help="0 disables the wall-clock stop. The retrain phase exports this.",
    )
    parser.add_argument("--reserve-seconds", type=int, default=600)
    parser.add_argument("--save-unit", choices=("step", "epoch"), default="epoch")
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--smoke", action="store_true", help="self-check and exit")
    args, hydra_overrides = parser.parse_known_args()

    if args.smoke:
        print(json.dumps({"schema_version": 1, "task_id": TASK_ID, "smoke": "passed"}))
        return 0

    output = args.output.resolve()
    checkpoint_dir = (args.checkpoint_dir or (output / "work/digress")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    upstream = args.upstream.resolve()
    if not (upstream / "src/main.py").is_file():
        raise FileNotFoundError(upstream / "src/main.py")
    if not args.data.is_dir():
        raise FileNotFoundError(args.data)
    if args.save_interval <= 0 or args.save_total_limit < 0:
        raise ValueError("save interval must be positive and total limit must be nonnegative")
    # Configure a private training copy.  Running a probe must never mutate
    # /workspace and accidentally add harness-generated edits to candidate.patch.
    training_upstream = checkpoint_dir / "source"
    if training_upstream.exists():
        raise FileExistsError(f"refusing to reuse DiGress training source: {training_upstream}")
    shutil.copytree(upstream, training_upstream, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    configure_checkpointing(training_upstream)
    lightning_checkpoints = checkpoint_dir / "lightning"
    lightning_checkpoints.mkdir(parents=True, exist_ok=True)

    # Hydra writes below its run dir and Lightning below the checkpoint dir; both
    # are under /out. The pinned tree at /workspace/digress is not written to at
    # run time, so an edit to it is the Agent's edit and nothing else.
    command = [
        sys.executable,
        str(training_upstream / "src/main.py"),
        "+experiment=qm9_no_h",
        "hydra.run.dir={}".format(output / "hydra"),
        # Hydra 1.3 no longer changes into hydra.run.dir by default. Upstream writes
        # checkpoints through relative paths, so keep them on the persistent mount.
        "hydra.job.chdir=true",
        f"general.name={args.run_name}",
        "general.wandb=disabled",
        "general.gpus=1",
        f"general.samples_to_generate={args.samples}",
        f"general.final_model_samples_to_generate={args.samples}",
        "general.final_model_samples_to_save=0",
        "general.final_model_chains_to_save=0",
        f"train.seed={args.seed}",
        f"train.n_epochs={args.epochs}",
        f"train.batch_size={args.batch_size}",
        f"train.lr={args.learning_rate}",
        f"train.ema_decay={args.ema_decay}",
        f"train.weight_decay={args.weight_decay}",
        f"train.clip_grad={args.clip_grad}",
        f"model.n_layers={args.num_layers}",
        f"model.diffusion_steps={args.diffusion_steps}",
        f"dataset.datadir={args.data}",
    ]
    # Anything unrecognised goes to Hydra untouched, last, so it wins. This is the
    # escape hatch for every config key not named above -- including the checkpoint
    # interval, which matters for the wall-clock stop: the SIGTERM lands at an epoch
    # boundary, so if upstream only refreshes last.ckpt every N epochs then up to N
    # epochs of work is what a stop costs. Set it to 1 if the run is long.
    command.extend(hydra_overrides)
    if args.max_train_batches > 0:
        command.append(f"+general.limit_train_batches={args.max_train_batches}")

    environment = os.environ.copy()
    environment.update(
        {
            "WANDB_MODE": "offline",
            "HYDRA_FULL_ERROR": "1",
            "PYTHONPATH": str(training_upstream),
            "AI4AI_DIGRESS_CHECKPOINT_WORK": str(lightning_checkpoints),
            "AI4AI_SAVE_UNIT": args.save_unit,
            "AI4AI_SAVE_INTERVAL": str(args.save_interval),
            "AI4AI_DIGRESS_TRAIN_BUDGET": str(
                max(args.max_wall_seconds - args.reserve_seconds - GRACE_SECONDS, 60)
                if args.max_wall_seconds > 0 else 0
            ),
        }
    )

    log_path = output / "train-stdout.log"
    result = run_with_wall_clock(
        command,
        training_upstream,
        environment,
        log_path,
        args.max_wall_seconds,
        args.reserve_seconds,
    )
    log_text = log_path.read_text(errors="replace")

    checkpoints = export_checkpoints(
        training_upstream,
        lightning_checkpoints,
        output / "checkpoints",
        args.save_unit,
        args.save_total_limit,
    )

    if result["exit_status"] != 0 and not result["stopped_by_wall_clock"]:
        # A crash with a usable checkpoint is still reported as a failure: the run
        # did not do what it was asked, and a silent partial success is how a budget
        # gets spent without anybody noticing.
        print(
            "train.py: trainer exited {}; a checkpoint was exported anyway from "
            "{}".format(result["exit_status"], checkpoints[-1]["path"]),
            file=sys.stderr,
        )

    metadata = {
        "schema_version": 1,
        "task_id": TASK_ID,
        # Informational description of the shipped method. The evaluator does not
        # gate on this candidate-authored value.
        "algorithm_family": "digress_discrete_graph_diffusion",
        "upstream_revision": UPSTREAM_REVISION,
        "seed": args.seed,
        "requested_epochs": args.epochs,
        "resolved_config": {
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "num_layers": args.num_layers,
            "diffusion_steps": args.diffusion_steps,
            "ema_decay": args.ema_decay,
            "weight_decay": args.weight_decay,
            "clip_grad": args.clip_grad,
            "max_train_batches": args.max_train_batches,
            "save_unit": args.save_unit,
            "save_interval": args.save_interval,
            "save_total_limit": args.save_total_limit,
        },
        "hydra_overrides": list(hydra_overrides),
        "checkpoints": checkpoints,
        "checkpoint_structure": checkpoints[-1]["structure"],
        "metrics": parse_metrics(log_text),
        "offline_wandb": True,
        **result,
    }
    atomic_json(checkpoint_dir / "training_metadata.json", metadata)
    atomic_json(output / "summary.json", metadata)
    print(json.dumps(metadata, sort_keys=True, default=str))
    return 0 if result["exit_status"] == 0 or result["stopped_by_wall_clock"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
