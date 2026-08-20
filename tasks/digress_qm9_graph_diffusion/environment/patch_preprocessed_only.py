"""Allow the pinned QM9 loader to consume a complete processed-only export."""

from __future__ import annotations

import py_compile
import sys
from pathlib import Path

OLD = '''    def download(self):
        """
        Download raw qm9 files. Taken from PyG QM9 class
        """
        try:
'''
NEW = '''    def download(self):
        """
        Download raw qm9 files. Taken from PyG QM9 class
        """
        # A locked processed-only export needs no raw source. PyG checks raw paths
        # before processed paths, so without this guard it attempts network access
        # even when all three processed tensors are already present.
        if files_exist(self.processed_paths):
            return
        try:
'''


def patch(root: Path) -> None:
    target = root / "src/datasets/qm9_dataset.py"
    text = target.read_text(encoding="utf-8")
    if text.count(OLD) != 1:
        raise RuntimeError(f"unexpected pinned QM9 loader shape: {target}")
    target.write_text(text.replace(OLD, NEW), encoding="utf-8")
    py_compile.compile(str(target), doraise=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: patch_preprocessed_only.py EDITABLE_ROOT GRADER_ROOT")
    for argument in sys.argv[1:]:
        patch(Path(argument))
