#!/usr/bin/env python3
"""Adapt the shared single-GPU loader to Transformers 4.28.1.

This task deliberately pins an older Transformers release for the upstream OWL
implementation.  That release accepts explicit device maps, but not the shorthand
``device_map="cuda"`` used by newer task images.
"""

from __future__ import annotations

import sys
from pathlib import Path


path = Path(sys.argv[1])
old = 'torch_dtype=torch.bfloat16, device_map="cuda",'
new = 'torch_dtype=torch.bfloat16, device_map={"": 0},'
text = path.read_text(encoding="utf-8")
if text.count(old) != 1:
    raise SystemExit(f"expected exactly one shared device-map expression in {path}")
path.write_text(text.replace(old, new), encoding="utf-8")
