#!/usr/bin/env python3
"""Atomically place one recipe-owned checkpoint in the v1.5 standard directory."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def numeric_checkpoints(root: Path) -> list[tuple[int, Path]]:
    rows: list[tuple[int, Path]] = []
    for path in root.glob("checkpoint-*"):
        suffix = path.name.removeprefix("checkpoint-")
        if suffix.isdigit() and path.is_dir():
            rows.append((int(suffix), path))
    return sorted(rows)


def publish(
    output: Path, progress: int, source: Path, *, payload_name: str,
    retention: int,
) -> Path:
    if progress < 0:
        raise ValueError("progress must be non-negative")
    if retention < 0:
        raise ValueError("retention must be zero or positive")
    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    root = output.resolve() / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"checkpoint-{progress}"
    if target.exists():
        # A periodic save and the unconditional final save may name the same
        # progress. Reuse the already-complete atomic publication.
        return target
    temporary = root / f".checkpoint-{progress}.{os.getpid()}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        if source.is_dir():
            if payload_name != ".":
                raise ValueError("payload-name must be '.' when source is a directory")
            for item in source.iterdir():
                destination = temporary / item.name
                if item.is_dir():
                    shutil.copytree(item, destination, symlinks=True)
                else:
                    shutil.copy2(item, destination, follow_symlinks=False)
        else:
            if payload_name in {"", "."} or Path(payload_name).name != payload_name:
                raise ValueError("a file source needs one basename as --payload-name")
            shutil.copy2(source, temporary / payload_name)
        temporary.replace(target)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    if retention:
        rows = numeric_checkpoints(root)
        for _, old in rows[:-retention]:
            shutil.rmtree(old)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/out"))
    parser.add_argument("--progress", type=int, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--payload-name", default=".")
    parser.add_argument("--retention", type=int, default=3)
    args = parser.parse_args()
    print(publish(
        args.output, args.progress, args.source,
        payload_name=args.payload_name, retention=args.retention,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
