"""Load a task's declarative phase mapping after its raw bytes were verified."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


class PhaseDeclarationError(RuntimeError):
    pass


def load_phases(task_dir: Path, *, namespace: str = "runtime") -> dict[str, Any]:
    path = task_dir / "declaration.py"
    if not path.is_file():
        raise PhaseDeclarationError(f"no declaration.py at {path}")
    name = f"{task_dir.name}_{namespace}_declaration"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PhaseDeclarationError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    phases = getattr(module, "PHASES", None)
    if not isinstance(phases, dict):
        raise PhaseDeclarationError(f"{path} does not define a PHASES dict")
    return phases
