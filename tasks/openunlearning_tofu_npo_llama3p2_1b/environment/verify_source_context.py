#!/usr/bin/env python3
"""Fail the image build if the exported OpenUnlearning source tree drifts."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

EXPECTED_SHA256 = "957a22a7fbf3fe45d442382c27d1d870f0afcd0d92769ddc5082c5db152ae545"
EXPECTED_FILES = 214
EXPECTED_BYTES = 489850


def main() -> int:
    root = Path(sys.argv[1])
    records = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if ".git" in relative.parts or ".cache" in relative.parts:
            continue
        if path.is_symlink():
            raise SystemExit(f"source context contains a symlink: {relative}")
        if not path.is_file():
            continue
        payload = path.read_bytes()
        records.append(
            {
                "path": relative.as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    encoded = json.dumps(records, separators=(",", ":"), sort_keys=True).encode()
    observed = hashlib.sha256(encoded).hexdigest()
    size = sum(record["size_bytes"] for record in records)
    if (observed, len(records), size) != (EXPECTED_SHA256, EXPECTED_FILES, EXPECTED_BYTES):
        raise SystemExit(
            "OpenUnlearning source context does not match assets.lock.yaml: "
            f"sha256={observed} files={len(records)} bytes={size}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
