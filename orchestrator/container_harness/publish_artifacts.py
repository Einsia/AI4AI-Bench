#!/usr/bin/env python3
"""Atomically publish the latest scoreable formal artifacts below /out.

Task code calls this only after it has finished writing and validating a source
artifact.  The host independently validates every publication before scoring.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
from pathlib import Path

NAME = re.compile(r"artifact-(\d+)$")


def inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def publish(output: Path, progress: int, source: Path) -> Path:
    output = output.resolve()
    source = source.resolve()
    if not inside(source, output):
        raise ValueError(f"artifact source must be below {output}, got {source}")
    if not source.exists():
        raise FileNotFoundError(f"artifact source does not exist: {source}")
    root = output / "final-artifacts"
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"artifact-{progress}"
    if target.exists():
        raise FileExistsError(f"refusing to replace published artifact: {target}")
    temporary = root / f".artifact-{progress}.{os.getpid()}.incomplete"
    if source.is_file():
        shutil.copy2(source, temporary)
    elif source.is_dir():
        shutil.copytree(source, temporary)
    else:
        raise ValueError(f"artifact source is neither a regular file nor directory: {source}")
    temporary.replace(target)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep", type=int, default=3)
    parser.add_argument(
        "--artifact", nargs=2, action="append", metavar=("PROGRESS", "PATH"), default=[]
    )
    parser.add_argument(
        "--artifact-json", nargs=3, action="append", default=[],
        metavar=("JSON", "DOTTED_KEY", "PATH"),
        help="read an artifact's numeric progress from a JSON receipt",
    )
    parser.add_argument(
        "--discover", action="append", default=[], metavar="GLOB",
        help="discover numbered source paths; only the latest --keep are copied",
    )
    parser.add_argument(
        "--subpath", default="",
        help="scoreable path below each --discover match, e.g. actor/huggingface",
    )
    args = parser.parse_args(argv)
    if args.keep < 1 or args.keep > 3:
        parser.error("--keep must be between one and three")
    parsed: list[tuple[int, Path]] = []
    for raw_progress, raw_path in args.artifact:
        try:
            progress = int(raw_progress)
        except ValueError:
            parser.error(f"artifact progress must be an integer, got {raw_progress!r}")
        if progress < 0:
            parser.error("artifact progress must be non-negative")
        parsed.append((progress, Path(raw_path)))
    for raw_json, dotted_key, raw_path in args.artifact_json:
        try:
            value = json.loads(Path(raw_json).read_text(encoding="utf-8"))
            for part in dotted_key.split("."):
                value = value[part]
            progress = int(value)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            parser.error(f"cannot read integer {dotted_key!r} from {raw_json}: {exc}")
        if progress < 0:
            parser.error("artifact progress must be non-negative")
        parsed.append((progress, Path(raw_path)))
    for pattern in args.discover:
        for raw_path in sorted(glob.glob(pattern)):
            matched = Path(raw_path)
            match = re.search(r"(\d+)$", matched.name)
            if match:
                source = matched / args.subpath if args.subpath else matched
                if source.exists():
                    parsed.append((int(match.group(1)), source))
    if not parsed:
        parser.error(
            "at least one --artifact, --artifact-json or numbered --discover match is required"
        )
    # An explicit final export and a trainer-managed checkpoint can legitimately
    # describe the same progress. Explicit entries are parsed first and take
    # precedence over later discovery matches.
    deduplicated: dict[int, Path] = {}
    for progress, path in parsed:
        deduplicated.setdefault(progress, path)
    parsed = list(deduplicated.items())

    selected = sorted(parsed)[-args.keep :]
    published = [str(publish(args.output, progress, path)) for progress, path in selected]
    root = args.output.resolve() / "final-artifacts"
    receipt = {
        "schema_version": 1,
        "keep": args.keep,
        "published": published,
        # Publication is deliberately non-destructive. The host receipt records
        # older, rejected and incomplete candidates; it never silently erases them.
        "removed": [],
    }
    temporary = root / f".publication.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(root / "publication.json")
    for path in published:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
