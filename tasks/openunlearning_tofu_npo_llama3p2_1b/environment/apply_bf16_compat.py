"""Apply the report's BF16-to-NumPy boundary casts and attest exact file hashes."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

PATCHES = {
    "src/evals/metrics/utils.py": (
        (
            "    avg_losses = avg_losses.cpu().numpy().tolist()\n"
            "    normalized_probs = normalized_probs.cpu().numpy().tolist()",
            "    # NumPy has no native bfloat16 dtype. Keep model evaluation in BF16 and\n"
            "    # cast only the detached aggregate values at the serialization boundary.\n"
            "    avg_losses = avg_losses.float().cpu().numpy().tolist()\n"
            "    normalized_probs = normalized_probs.float().cpu().numpy().tolist()",
        ),
    ),
    "src/evals/metrics/utility.py": (
        (
            (
                "        scores = F.softmax(outputs.logits, dim=-1)[:, class_id]"
                ".cpu().numpy().tolist()"
            ),
            "        scores = (\n"
            "            F.softmax(outputs.logits, dim=-1)[:, class_id]\n"
            "            .float()\n"
            "            .cpu()\n"
            "            .numpy()\n"
            "            .tolist()\n"
            "        )",
        ),
    ),
    "src/evals/metrics/mia/min_k_plus_plus.py": (
        (") / torch.sqrt(sigma).cpu().numpy()", ") / torch.sqrt(sigma).float().cpu().numpy()"),
    ),
}
EXPECTED = {
    "src/evals/metrics/utils.py": (
        "dd343dc01b7b7f650881b361fc780994f0f718b0df404d268f335c4781b0d9b0"
    ),
    "src/evals/metrics/utility.py": (
        "056e242ec3a8bc449e21d8536de99043a0ac985c57e175e80e0d037a7910d5bb"
    ),
    "src/evals/metrics/mia/min_k_plus_plus.py": (
        "d0be5cabe1d66079b7dcbfa2d3e3a55e89472f1be7cd6953e3050a73d0d923e5"
    ),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    for relative, replacements in PATCHES.items():
        path = args.root / relative
        text = path.read_text(encoding="utf-8")
        for old, new in replacements:
            if text.count(old) != 1:
                raise RuntimeError(f"expected exactly one BF16 patch target in {relative}: {old}")
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")
        observed = digest(path)
        if observed != EXPECTED[relative]:
            raise RuntimeError(
                f"patched {relative} hash {observed} != official {EXPECTED[relative]}"
            )


if __name__ == "__main__":
    main()
