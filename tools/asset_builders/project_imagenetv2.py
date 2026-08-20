#!/usr/bin/env python3
"""Project ImageNetV2 matched-frequency offsets 0-1 into the proxy alias."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def build(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    classes = sorted(path for path in source.iterdir() if path.is_dir())
    if len(classes) != 1_000:
        raise ValueError(f"expected 1000 ImageNetV2 classes, found {len(classes)}")
    for directory in classes:
        images = sorted(path for path in directory.iterdir() if path.is_file())
        if len(images) != 10:
            raise ValueError(f"expected 10 images in class {directory.name}, found {len(images)}")
        target = output / directory.name
        target.mkdir(parents=True)
        for image in images[:2]:
            shutil.copyfile(image, target / image.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.output)


if __name__ == "__main__":
    main()
