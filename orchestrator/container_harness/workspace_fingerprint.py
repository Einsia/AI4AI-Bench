#!/usr/bin/env python3
"""Hash the Git-relevant state of an Agent workspace."""

from __future__ import annotations

import argparse
import hashlib
import os
import stat
from pathlib import Path

IGNORED_DIRECTORIES = {".git", "__pycache__"}


def fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        base = Path(directory)
        kept_directories: list[str] = []
        for name in sorted(directory_names):
            path = base / name
            if name in IGNORED_DIRECTORIES:
                continue
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                digest.update(relative.encode("utf-8", "surrogateescape") + b"\0L")
                digest.update(os.readlink(path).encode("utf-8", "surrogateescape") + b"\0")
            else:
                kept_directories.append(name)
        directory_names[:] = kept_directories

        for name in sorted(file_names):
            if name.endswith((".pyc", ".pyo")):
                continue
            path = base / name
            relative = path.relative_to(root).as_posix()
            metadata = path.lstat()
            digest.update(relative.encode("utf-8", "surrogateescape") + b"\0")
            if stat.S_ISLNK(metadata.st_mode):
                digest.update(b"L" + os.readlink(path).encode("utf-8", "surrogateescape") + b"\0")
                continue
            if not stat.S_ISREG(metadata.st_mode):
                digest.update(f"O{stat.S_IFMT(metadata.st_mode):o}\0".encode("ascii"))
                continue
            executable = bool(metadata.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))
            digest.update(b"F1\0" if executable else b"F0\0")
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("/workspace"))
    args = parser.parse_args()
    print(fingerprint(args.root.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
