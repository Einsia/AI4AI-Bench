"""Aggregate independent final evaluations for a bounded artifact series."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
from typing import Any

from task import load_task

SUMMARY_SCHEMA_VERSION = 2
SELECTION_RULE = "best_valid_of_up_to_3"


class FinalScoreError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _timestamp(path: Path) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(
            path.read_text(encoding="utf-8").strip().replace("Z", "+00:00")
        )
    except (OSError, ValueError):
        return None


def _elapsed(root: Path) -> int | None:
    started = _timestamp(root / ".started")
    completed = _timestamp(root / ".complete")
    return int((completed - started).total_seconds()) if started and completed else None


def _numeric(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def classify_summary(summary: dict[str, Any], expected_metric: str) -> str:
    metric = summary.get("metric")
    if metric != expected_metric:
        raise FinalScoreError(
            f"score summary metric {metric!r} does not match task metric {expected_metric!r}"
        )
    status = str(summary.get("status", "valid"))
    if status in {"invalid", "rejected"}:
        return "invalid"
    if status in {"error", "failed", "incomplete"}:
        raise FinalScoreError(f"score summary is non-terminal infrastructure state {status!r}")
    if _numeric(summary.get("score")):
        return "valid"
    raise FinalScoreError(
        "score summary has no numeric score and does not explicitly declare invalid"
    )


def is_terminal_summary(task: Path, summary_path: Path) -> bool:
    config = load_task(task)
    expected_metric = str(config.get("metadata", {}).get("final_metric", ""))
    summary = _read_json(summary_path)
    if not expected_metric or summary is None:
        return False
    try:
        classify_summary(summary, expected_metric)
    except FinalScoreError:
        return False
    return True


def aggregate(
    *,
    task: Path,
    artifacts_path: Path,
    score_root: Path,
    evaluation_config_path: Path | None = None,
) -> dict[str, Any]:
    config = load_task(task)
    metadata = config.get("metadata", {})
    metric = str(metadata.get("final_metric", ""))
    direction = str(metadata.get("final_direction", ""))
    if not metric:
        raise FinalScoreError(f"{task}/task.toml declares no metadata.final_metric")
    if direction not in {"max", "min"}:
        raise FinalScoreError(
            f"{task}/task.toml final_direction must be 'max' or 'min', got {direction!r}"
        )
    artifacts = _read_json(artifacts_path)
    if artifacts is None or not isinstance(artifacts.get("accepted"), list):
        raise FinalScoreError(f"invalid artifact receipt: {artifacts_path}")

    results: list[dict[str, Any]] = []
    valid: list[tuple[float, int, dict[str, Any]]] = []
    external_checkpoints = artifacts.get("provenance") == "external_checkpoint"
    for artifact in artifacts["accepted"]:
        try:
            progress = int(artifact["progress"])
            artifact_path = str(
                artifact["payload_path"]
                if external_checkpoints
                else artifact.get("published_path")
                or artifact.get("path")
                or artifact["payload_path"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise FinalScoreError(f"invalid accepted artifact entry: {artifact!r}") from exc
        root = score_root / f"artifact-{progress}"
        summary_path = root / "out/summary.json"
        summary = _read_json(summary_path)
        if summary is None:
            raise FinalScoreError(f"score is incomplete or unreadable: {summary_path}")
        state = classify_summary(summary, metric)
        row = {
            "progress": progress,
            "artifact_path": artifact_path,
            "status": state,
            "score": summary.get("score"),
            "stderr": summary.get("stderr"),
            "n": summary.get("n"),
            "correct": summary.get("correct"),
            "elapsed_seconds": _elapsed(root),
            "summary_path": str(summary_path.resolve()),
            "reason": summary.get("reason")
            or summary.get("failure_reason")
            or summary.get("error"),
        }
        results.append(row)
        if state == "valid":
            valid.append((float(summary["score"]), progress, summary))

    selected: tuple[float, int, dict[str, Any]] | None = None
    if valid:
        # Later progress is the deterministic tie-break for both directions.
        selected = max(valid, key=lambda item: (item[0], item[1])) if direction == "max" else min(
            valid, key=lambda item: (item[0], -item[1])
        )

    payload: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "selection_rule": SELECTION_RULE,
        "metric": metric,
        "direction": direction,
        "status": "valid" if selected else "invalid",
        "score": selected[0] if selected else None,
        "selected_progress": selected[1] if selected else None,
        "selected_artifact": (
            next(row["artifact_path"] for row in results if row["progress"] == selected[1])
            if selected
            else None
        ),
        "accepted_artifact_count": len(results),
        "valid_artifact_count": len(valid),
        "invalid_artifact_count": len(results) - len(valid),
        "artifact_results": results,
    }
    if selected:
        for key in ("stderr", "n", "correct"):
            if key in selected[2]:
                payload[key] = selected[2][key]
    else:
        payload["reason"] = "no accepted artifact produced a valid final score"
    if evaluation_config_path is not None:
        evaluation_config = _read_json(evaluation_config_path)
        if evaluation_config is None:
            raise FinalScoreError(
                f"invalid evaluation config: {evaluation_config_path}"
            )
        classification = evaluation_config.get("result_classification")
        verification = evaluation_config.get("verification")
        if classification not in {
            "official_self_hosted",
            "non_official_local",
        } or not isinstance(verification, dict):
            raise FinalScoreError(
                "evaluation config is missing result classification evidence"
            )
        payload["result_classification"] = classification
        payload["verification"] = verification
    return payload


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    terminal = subparsers.add_parser("terminal")
    terminal.add_argument("--task", type=Path, required=True)
    terminal.add_argument("--summary", type=Path, required=True)
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--task", type=Path, required=True)
    aggregate_parser.add_argument("--artifacts", type=Path, required=True)
    aggregate_parser.add_argument("--score-root", type=Path, required=True)
    aggregate_parser.add_argument("--output", type=Path, required=True)
    aggregate_parser.add_argument("--evaluation-config", type=Path)
    args = parser.parse_args()

    if args.command == "terminal":
        return 0 if is_terminal_summary(args.task, args.summary) else 1
    try:
        payload = aggregate(
            task=args.task,
            artifacts_path=args.artifacts,
            score_root=args.score_root,
            evaluation_config_path=args.evaluation_config,
        )
        write_summary(args.output, payload)
    except FinalScoreError as exc:
        parser.exit(1, f"final-score: {exc}\n")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
