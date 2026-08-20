#!/usr/bin/env python3
"""Build the paper's fixed 8,192-pair RewardBench-decontaminated training asset."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SOURCE_ROWS = 61_135
FINAL_ROWS = 2_985
RESERVED_ROWS = 128
TRAIN_ROWS = 8_192


def text(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("conversation text must be a string")
    result = " ".join(unicodedata.normalize("NFKC", value).split())
    if not result:
        raise ValueError("conversation text cannot be empty")
    return result


def canonical(value: str) -> str:
    return text(value).casefold()


def digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def assistant(value: Any) -> str:
    if isinstance(value, str):
        return text(value)
    if not isinstance(value, list):
        raise TypeError("chosen/rejected must be a string or message list")
    for message in reversed(value):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return text(message.get("content"))
    raise ValueError("conversation has no assistant response")


def prompt(row: dict[str, Any]) -> str:
    if isinstance(row.get("prompt"), str) and row["prompt"].strip():
        return text(row["prompt"])
    for field in ("chosen", "rejected"):
        messages = row.get(field)
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, dict) and message.get("role") == "user":
                    return text(message.get("content"))
    raise ValueError("UltraFeedback row has no prompt")


def pair(row: dict[str, Any]) -> tuple[dict[str, Any], str, str, str, str]:
    question = prompt(row)
    chosen = assistant(row.get("chosen"))
    rejected = assistant(row.get("rejected"))
    if canonical(chosen) == canonical(rejected):
        raise ValueError("chosen and rejected collapse to the same response")
    projected = {
        "chosen": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": chosen},
        ],
        "rejected": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": rejected},
        ],
    }
    projected["pair_sha256"] = digest(
        {"chosen": projected["chosen"], "rejected": projected["rejected"]}
    )
    identity = digest(
        {
            "prompt": canonical(question),
            "chosen": canonical(chosen),
            "rejected": canonical(rejected),
        }
    )
    return projected, canonical(question), canonical(chosen), canonical(rejected), identity


def signatures(rows: Iterable[dict[str, Any]]) -> tuple[set[str], set[str]]:
    prompts: set[str] = set()
    responses: set[str] = set()
    for row in rows:
        prompts.add(canonical(text(row["prompt"])))
        responses.add(canonical(assistant(row["chosen"])))
        responses.add(canonical(assistant(row["rejected"])))
    return prompts, responses


def load_rows(root: Path, pattern: str) -> list[dict[str, Any]]:
    files = sorted(root.rglob(pattern))
    if not files:
        raise FileNotFoundError(f"no {pattern} below {root}")
    from datasets import load_dataset

    data = load_dataset("parquet", data_files=[str(path) for path in files], split="train")
    return [dict(row) for row in data]


def build(ultrafeedback: Path, rewardbench: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    source = load_rows(ultrafeedback, "*train_prefs*.parquet")
    final = load_rows(rewardbench, "filtered-*.parquet")
    if len(source) != SOURCE_ROWS or len(final) != FINAL_ROWS:
        raise ValueError(
            f"unexpected source rows: UltraFeedback={len(source)}, RewardBench={len(final)}"
        )
    final_prompts, final_responses = signatures(final)
    selected: dict[str, dict[str, Any]] = {}
    for row in source:
        try:
            projected, question, chosen, rejected, identity = pair(row)
        except (KeyError, TypeError, ValueError):
            continue
        if question in final_prompts or chosen in final_responses or rejected in final_responses:
            continue
        incumbent = selected.get(identity)
        if incumbent is None or projected["pair_sha256"] < incumbent["pair_sha256"]:
            selected[identity] = projected
    ordered = sorted(selected.values(), key=lambda row: row["pair_sha256"])
    if len(ordered) < RESERVED_ROWS + TRAIN_ROWS:
        raise ValueError("too few clean unique preference pairs")
    training = ordered[RESERVED_ROWS : RESERVED_ROWS + TRAIN_ROWS]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in training),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ultrafeedback", type=Path, required=True)
    parser.add_argument("--rewardbench", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.ultrafeedback, args.rewardbench, args.output)


if __name__ == "__main__":
    main()
