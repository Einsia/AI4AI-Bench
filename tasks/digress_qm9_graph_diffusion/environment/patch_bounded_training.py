"""Add an optional normal-mode train-batch limit to the pinned DiGress trainer."""

from __future__ import annotations

import py_compile
import sys
from pathlib import Path

OLD = '''                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
'''

NEW = '''                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      limit_train_batches=cfg.general.get('limit_train_batches', 1.0),
                      fast_dev_run=cfg.general.name == 'debug',
'''


def patch(root: Path) -> None:
    target = root / "src/main.py"
    text = target.read_text(encoding="utf-8")
    if text.count(OLD) != 1:
        raise RuntimeError(f"unexpected pinned DiGress Trainer shape: {target}")
    target.write_text(text.replace(OLD, NEW), encoding="utf-8")
    py_compile.compile(str(target), doraise=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: patch_bounded_training.py EDITABLE_ROOT GRADER_ROOT")
    for argument in sys.argv[1:]:
        patch(Path(argument))
