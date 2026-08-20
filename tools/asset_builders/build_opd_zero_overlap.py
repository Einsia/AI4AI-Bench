#!/usr/bin/env python3
"""Reproduce the frozen OPD DAPO/AIME zero-overlap training projection."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SOURCE_SHA256 = "534375d6bb8630d22ab46a56e11f2ffec1d288d8f7d04099bc82d68948705941"
SOURCE_ROWS = 1_791_700
CANONICAL_UNIQUE_ROWS = 16_395
OUTPUT_ROWS = 15_285
AIME24_SHA256 = "025484a99fea498e7d0c3b0ee42afcbec0176405c19c5dbf557b9f6ca6445675"
AIME25_I_SHA256 = "b91b3c96f05d9635d2a0692b124ebe023c1ff59cb19c074275e6c4b349d0659e"
AIME25_II_SHA256 = "16a2dcfbbf9db1b11f8a69a3ba5e4cac73e3641b19a37e2307e9c12240bbed5e"
PROTOCOL_REVISION = "opd-dapo-aime-zero-overlap-v2"
PYARROW_VERSION = "23.0.1"
NGRAM_SIZE = 8
RESERVED_ROWS = 16 + 64


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def one_file(root: Path, pattern: str) -> Path:
    matches = [path for path in root.rglob(pattern) if path.is_file() and not path.is_symlink()]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {pattern} below {root}, found {len(matches)}")
    return matches[0]


def flatten(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        if isinstance(value.get("content"), str):
            return value["content"]
        return " ".join(flatten(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return " ".join(flatten(item) for item in value)
    return "" if value is None else str(value)


def problem(value: Any) -> str:
    text = flatten(value).strip()
    prefix = (
        "Solve the following math problem step by step. The last line of your "
        "response should be of the form Answer:"
    )
    if text.startswith(prefix) and "\n\n" in text:
        text = text.split("\n\n", 1)[1]
    marker = '\n\nRemember to put your answer on its own line after "Answer:".'
    if marker in text:
        text = text.split(marker, 1)[0]
    return text.strip()


def words(value: Any) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", problem(value)).casefold()
    return tuple(re.findall(r"[^\W_]+", normalized, flags=re.UNICODE))


def canonical(value: Any) -> str:
    return " ".join(words(value))


def ngrams(value: Any) -> set[str]:
    tokens = words(value)
    return {
        " ".join(tokens[index : index + NGRAM_SIZE])
        for index in range(max(0, len(tokens) - NGRAM_SIZE + 1))
    }


def zero_overlap_audit(training: list[Any], final: list[str]) -> dict[str, int]:
    """Return semantic violations for the published training projection."""

    normalized = [canonical(item) for item in training]
    final_canonical = [canonical(item) for item in final]
    final_exact = set(final_canonical)
    final_ngrams = set().union(*(ngrams(item) for item in final))
    return {
        "rows": len(normalized),
        "empty": sum(not item for item in normalized),
        "duplicates": len(normalized) - len(set(normalized)),
        "aime_exact": sum(item in final_exact for item in normalized),
        "aime_containment": sum(
            bool(item)
            and any(item in final_item or final_item in item for final_item in final_canonical)
            for item in normalized
        ),
        "aime_shared_8gram": sum(
            bool(ngrams(item) & final_ngrams) for item in normalized
        ),
    }


def load_aime(aime24: Path, aime25: Path) -> list[str]:
    import json

    import pyarrow.parquet as parquet

    source24 = one_file(aime24, "train-00000-of-00001.parquet")
    if sha256(source24) != AIME24_SHA256:
        raise ValueError("AIME 2024 source hash mismatch")
    table = parquet.read_table(source24)
    prompts = [
        str(prompt).strip()
        for prompt, url in zip(
            table.column("problem").to_pylist(), table.column("url").to_pylist(), strict=True
        )
        if "2024" in str(url)
    ]
    for name, expected in (
        ("aime2025-I.jsonl", AIME25_I_SHA256),
        ("aime2025-II.jsonl", AIME25_II_SHA256),
    ):
        path = one_file(aime25, name)
        if sha256(path) != expected:
            raise ValueError(f"AIME 2025 source hash mismatch: {name}")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        prompts.extend(str(row["question"]).strip() for row in rows)
    if len(prompts) != 60 or len({canonical(item) for item in prompts}) != 60:
        raise ValueError("AIME sources do not contain 60 unique frozen questions")
    return prompts


def build(dapo: Path, aime24: Path, aime25: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    import pyarrow
    import pyarrow as pa
    import pyarrow.parquet as parquet

    # The public materializer deliberately runs inside the immutable task image.
    # Keep the writer version aligned with that image so the declared public build
    # path is executable instead of depending on a second, unpublished environment.
    if pyarrow.__version__ != PYARROW_VERSION:
        raise RuntimeError(
            f"expected pyarrow {PYARROW_VERSION}, found {pyarrow.__version__}"
        )
    source = one_file(dapo, "*.parquet")
    if sha256(source) != SOURCE_SHA256:
        raise ValueError("DAPO source hash mismatch")
    table = parquet.read_table(source).replace_schema_metadata(None)
    if table.num_rows != SOURCE_ROWS or "prompt" not in table.column_names:
        raise ValueError("DAPO source schema or row count mismatch")
    prompts = table.column("prompt").to_pylist()
    final = load_aime(aime24, aime25)
    final_canonical = [canonical(item) for item in final]
    exact = set(final_canonical)
    final_ngrams = set().union(*(ngrams(item) for item in final))

    first: dict[str, int] = {}
    safe: list[int] = []
    for index, prompt in enumerate(prompts):
        normalized = canonical(prompt)
        if not normalized:
            raise ValueError(f"DAPO row {index} has an empty prompt")
        if normalized in first:
            continue
        first[normalized] = index
        containment = any(normalized in item or item in normalized for item in final_canonical)
        if normalized in exact or containment or ngrams(prompt) & final_ngrams:
            continue
        safe.append(index)
    if len(first) != CANONICAL_UNIQUE_ROWS:
        raise ValueError(
            f"DAPO canonical unique rows {len(first)} != expected "
            f"{CANONICAL_UNIQUE_ROWS}"
        )
    ranked = sorted(
        safe,
        key=lambda index: (
            hashlib.sha256(
                (PROTOCOL_REVISION + "\0" + canonical(prompts[index])).encode("utf-8")
            ).hexdigest(),
            index,
        ),
    )
    held_out = set(ranked[:RESERVED_ROWS])
    training = [index for index in safe if index not in held_out]
    if len(training) != OUTPUT_ROWS:
        raise ValueError(f"OPD training rows {len(training)} != expected {OUTPUT_ROWS}")
    selected = table.take(pa.array(training, type=pa.int64()))
    audit = zero_overlap_audit(selected.column("prompt").to_pylist(), final)
    violations = {key: value for key, value in audit.items() if key != "rows" and value}
    if audit["rows"] != OUTPUT_ROWS or violations:
        raise ValueError(f"OPD zero-overlap audit failed: {audit}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    parquet.write_table(
        selected,
        temporary,
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
        write_statistics=True,
        data_page_version="1.0",
    )
    os.replace(temporary, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dapo", type=Path, required=True)
    parser.add_argument("--aime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.dapo, args.aime / "aime2024", args.aime / "aime2025", args.output)


if __name__ == "__main__":
    main()
