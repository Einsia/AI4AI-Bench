"""Resolve and load the scored artifact for both evaluator tiers.

Two shapes are accepted:

    a merged model      *.safetensors + config.json, loaded directly
    a PEFT adapter      adapter_config.json + adapter_model.safetensors, merged
                        onto the reference from /assets at load time

An adapter is assembled on `/assets/models/policy_start`; a merged model is loaded
directly. The load itself is the final compatibility check. Method-family rules are
specified in instruction.md and are not inferred from candidate-written metadata.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

MERGED_PATTERNS = ("*.safetensors", "pytorch_model*.bin")
ADAPTER_CONFIG = "adapter_config.json"


def weight_files(directory: Path) -> list[Path]:
    """Model weight files directly in `directory`, adapter files excluded.

    Sorted and de-duplicated, so the hash below is a function of content rather
    than of glob order.
    """

    found = {
        path
        for pattern in MERGED_PATTERNS
        for path in directory.glob(pattern)
        if not path.name.startswith("adapter_")
    }
    return sorted(found)


def weight_sha256(directory: Path) -> str:
    """Hash the weight files, name and bytes, in sorted order."""

    files = weight_files(directory) or sorted(
        {
            path
            for pattern in ("adapter_model*.safetensors", "adapter_model*.bin")
            for path in directory.glob(pattern)
        }
    )
    if not files:
        raise RuntimeError(f"no model weights under {directory}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def describe(checkpoint: Path, reference: Path | None = None) -> dict[str, Any]:
    """Work out which of the two shapes this is, without loading anything.

    Runs before a device is claimed, so an unusable submission costs a directory
    listing rather than a model load and 413 generations.
    """

    if not checkpoint.is_dir():
        raise FileNotFoundError(f"{checkpoint} is not a directory")
    merged = weight_files(checkpoint)
    adapter = (checkpoint / ADAPTER_CONFIG).is_file()
    if merged and not adapter:
        kind = "merged_model"
    elif adapter:
        kind = "peft_adapter"
        if not any(checkpoint.glob("adapter_model*.safetensors")) and not any(
            checkpoint.glob("adapter_model*.bin")
        ):
            raise FileNotFoundError(f"{checkpoint} has adapter config but no adapter weights")
        if reference is None or not (reference / "config.json").is_file():
            raise FileNotFoundError(
                f"{checkpoint} is a PEFT adapter, so the reference policy is needed to "
                f"merge it, and {reference} does not hold a model config"
            )
    else:
        raise FileNotFoundError(
            f"{checkpoint} holds neither model weights nor {ADAPTER_CONFIG}. A scored "
            "artifact is either a self-contained model directory or a PEFT adapter."
        )
    report: dict[str, Any] = {
        "kind": kind,
        "path": str(checkpoint),
        "weight_files": [path.name for path in merged] or None,
        "weight_sha256": weight_sha256(checkpoint),
        "tokenizer_present": any(
            (checkpoint / name).is_file()
            for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
        ),
    }
    if kind == "peft_adapter":
        config = json.loads((checkpoint / ADAPTER_CONFIG).read_text(encoding="utf-8"))
        # Recorded, not checked. `r` and `target_modules` were frozen by a recipe
        # allowlist on the reference protocol and are the candidate's to choose now.
        report["adapter"] = {
            "peft_type": config.get("peft_type"),
            "r": config.get("r"),
            "target_modules": sorted(config.get("target_modules") or []),
        }
        report["merged_against"] = str(reference)
    return report


def load_model(checkpoint: Path, reference: Path | None = None) -> tuple[Any, Any]:
    """Load the artifact onto the current device, merging an adapter if that is what
    it is.

    Evaluation uses the fixed bf16 and SDPA execution protocol.
    """

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    report = describe(checkpoint, reference)
    # The tokenizer travels with the artifact when it has one -- a candidate that
    # changed the chat template has changed the model's interface, and the eval
    # should see the interface it shipped. Falling back to the reference's tokenizer
    # covers an adapter saved without one.
    tokenizer_source = checkpoint if report["tokenizer_present"] else reference
    if tokenizer_source is None:
        raise FileNotFoundError(f"no tokenizer in {checkpoint} and no reference to fall back to")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    source = checkpoint if report["kind"] == "merged_model" else reference
    model = AutoModelForCausalLM.from_pretrained(
        source,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda",
    )
    if report["kind"] == "peft_adapter":
        from peft import PeftModel

        # Merge so adapter and merged submissions share the same inference shape.
        model = PeftModel.from_pretrained(model, checkpoint).merge_and_unload()
    model.eval()
    return model, tokenizer


def smoke() -> None:
    """Exercise the shape detection on directories, with no weights to load."""

    import tempfile

    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        empty = root / "empty"
        empty.mkdir()
        try:
            describe(empty)
        except FileNotFoundError:
            pass
        else:
            raise RuntimeError("an empty directory must be refused")

        merged = root / "merged"
        merged.mkdir()
        (merged / "model.safetensors").write_bytes(b"weights")
        (merged / "config.json").write_text("{}\n", encoding="utf-8")
        (merged / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
        report = describe(merged)
        if report["kind"] != "merged_model" or not report["tokenizer_present"]:
            raise RuntimeError(f"merged model misread: {report}")

        reference = root / "reference"
        reference.mkdir()
        (reference / "config.json").write_text("{}\n", encoding="utf-8")
        adapter = root / "adapter"
        adapter.mkdir()
        (adapter / ADAPTER_CONFIG).write_text(
            json.dumps({"peft_type": "LORA", "r": 128, "target_modules": ["q_proj"]}),
            encoding="utf-8",
        )
        (adapter / "adapter_model.safetensors").write_bytes(b"delta")
        report = describe(adapter, reference)
        if report["kind"] != "peft_adapter" or report["adapter"]["r"] != 128:
            raise RuntimeError(f"adapter misread: {report}")
        try:
            describe(adapter, None)
        except FileNotFoundError:
            pass
        else:
            raise RuntimeError("an adapter with no reference must be refused")
    print("checkpoint.py smoke passed")


if __name__ == "__main__":
    smoke()
