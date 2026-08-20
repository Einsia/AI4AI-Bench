"""Validate and load a scalar reward-model artifact.

The formal artifact may be either a PEFT delta applied to the fixed mounted base
or a complete sequence-classification checkpoint produced by the fresh replay.
The gate checks only what scoring requires: the files load and the model emits one
scalar reward per sequence. It does not require LoRA or a Bradley-Terry objective.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Substrings that identify the classifier head inside a PEFT export. TaskType
# SEQ_CLS puts it in modules_to_save, so it appears with a wrapper prefix.
HEAD_MARKERS = ("score", "classifier")


class ArtifactViolation(ValueError):
    """The submitted checkpoint cannot be scored as a scalar reward model."""


def adapter_weight_files(checkpoint: Path) -> list[Path]:
    return sorted(
        {
            path
            for pattern in ("adapter_model*.safetensors", "adapter_model*.bin")
            for path in checkpoint.glob(pattern)
        }
    )


def full_model_weight_files(checkpoint: Path) -> list[Path]:
    return sorted(
        {
            path
            for pattern in ("model*.safetensors", "pytorch_model*.bin", "consolidated*.safetensors")
            for path in checkpoint.glob(pattern)
        }
    )


def read_tensor_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    """Tensor names and shapes, without materializing the weights.

    safetensors keeps a JSON header, so shapes cost a header read rather than a
    load. A .bin needs torch, and is loaded with weights_only=True because a
    pickle from a candidate is untrusted input.
    """

    if path.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(str(path), framework="numpy") as handle:
            return {key: tuple(handle.get_slice(key).get_shape()) for key in handle.keys()}
    import torch

    state = torch.load(str(path), map_location="cpu", weights_only=True)
    return {key: tuple(value.shape) for key, value in state.items()}


def inspect_adapter(shapes: dict[str, tuple[int, ...]], hidden_size: int) -> dict[str, Any]:
    """Check that a PEFT tensor inventory includes a compatible scalar head.

    Tensor names are deliberately not used to constrain the PEFT method. LoRA,
    IA3, prompt tuning and future PEFT formats use different parameter names;
    ``PeftModel.from_pretrained`` is the compatibility authority. This inexpensive
    check only establishes the task-level requirement that the delta also exports
    a scalar sequence-classification head.
    """

    if not shapes:
        raise ArtifactViolation("the checkpoint contains no tensors")

    head_names = [
        name for name in sorted(shapes) if any(marker in name for marker in HEAD_MARKERS)
    ]
    adapter_names = [name for name in sorted(shapes) if name not in head_names]
    if not adapter_names:
        raise ArtifactViolation("the checkpoint has a scalar head but no PEFT parameters")
    if not head_names:
        raise ArtifactViolation(
            "the checkpoint has no scalar head. A reward model's head is trained from "
            "scratch, so it must be exported with the adapter -- PEFT does this via "
            "modules_to_save when task_type is SEQ_CLS."
        )

    # The head maps hidden_size -> 1. Weight is (out, in); bias, if present, (out,).
    head_shapes = {name: shapes[name] for name in head_names}
    outputs = {shape[0] for shape in head_shapes.values() if shape}
    if outputs != {1}:
        raise ArtifactViolation(
            f"the head is not scalar: output dimensions {sorted(outputs)} from "
            f"{head_shapes}. The frozen evaluator requires one reward per sequence."
        )
    matrices = {name: shape for name, shape in head_shapes.items() if len(shape) == 2}
    if not matrices:
        raise ArtifactViolation(f"the head has no weight matrix, only {head_shapes}")
    wrong = {name: shape for name, shape in matrices.items() if shape[1] != hidden_size}
    if wrong:
        raise ArtifactViolation(
            f"the head does not fit the pinned base model's hidden size {hidden_size}: "
            f"{wrong}. This delta was trained against a different backbone."
        )

    parameters = sum(int(_product(shape)) for shape in shapes.values())
    return {
        "tensors": len(shapes),
        "parameters": parameters,
        "adapter_tensors": len(adapter_names),
        "head_tensors": len(head_names),
        "head_shapes": head_shapes,
        "hidden_size": hidden_size,
    }


def inspect_full_model(shapes: dict[str, tuple[int, ...]], hidden_size: int) -> dict[str, Any]:
    """Check that a full checkpoint contains a scalar sequence-reward head."""

    if not shapes:
        raise ArtifactViolation("the checkpoint contains no tensors")
    head_shapes = {
        name: shape
        for name, shape in shapes.items()
        if any(marker in name for marker in HEAD_MARKERS)
    }
    matrices = {name: shape for name, shape in head_shapes.items() if len(shape) == 2}
    scalar = {name: shape for name, shape in matrices.items() if shape == (1, hidden_size)}
    if not scalar:
        raise ArtifactViolation(
            f"the full checkpoint has no scalar reward head with shape (1, {hidden_size}); "
            f"found {head_shapes}"
        )
    return {
        "format": "full_model",
        "tensors": len(shapes),
        "parameters": sum(int(_product(shape)) for shape in shapes.values()),
        "head_shapes": head_shapes,
        "hidden_size": hidden_size,
    }


def _product(shape: tuple[int, ...]) -> int:
    total = 1
    for dimension in shape:
        total *= int(dimension)
    return total


def base_hidden_size(base_model: Path) -> int:
    config_path = base_model / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    hidden = config.get("hidden_size")
    if not isinstance(hidden, int) or hidden <= 0:
        raise ValueError(f"{config_path} has no usable hidden_size")
    return hidden


def check(checkpoint: Path, base_model: Path) -> dict[str, Any]:
    """Full artifact-side check against the mounted base. Raises on a violation."""

    if not checkpoint.is_dir():
        raise ArtifactViolation(f"no checkpoint directory at {checkpoint}")
    adapters = adapter_weight_files(checkpoint)
    full_models = full_model_weight_files(checkpoint)
    if adapters and full_models:
        raise ArtifactViolation("checkpoint mixes adapter and full-model weight files")
    files = adapters or full_models
    if not files:
        raise ArtifactViolation(
            f"no adapter or full-model weights under {checkpoint}"
        )
    shapes: dict[str, tuple[int, ...]] = {}
    for path in files:
        for name, shape in read_tensor_shapes(path).items():
            if name in shapes:
                raise ArtifactViolation(f"tensor {name} appears in more than one file")
            shapes[name] = shape
    if adapters:
        report = inspect_adapter(shapes, base_hidden_size(base_model))
        report["format"] = "adapter"
    else:
        report = inspect_full_model(shapes, base_hidden_size(base_model))
    report["weight_files"] = [path.name for path in files]
    return report


def load_model(checkpoint: Path, base_model: Path) -> tuple[Any, Any]:
    """Load either a candidate adapter on the pinned base or a full checkpoint."""

    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if adapter_weight_files(checkpoint):
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=1,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        model = PeftModel.from_pretrained(model, checkpoint, local_files_only=True)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        )
    model.config.pad_token_id = tokenizer.pad_token_id
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model, tokenizer


def smoke() -> None:
    hidden = 4096
    good = {
        "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": (16, hidden),
        "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": (hidden, 16),
        "base_model.model.score.modules_to_save.default.weight": (1, hidden),
    }
    report = inspect_adapter(good, hidden)
    assert report["parameters"] == 16 * hidden * 2 + hidden, report

    def rejects(shapes: dict[str, tuple[int, ...]], expect: str) -> None:
        try:
            inspect_adapter(shapes, hidden)
        except ArtifactViolation as error:
            assert expect in str(error), f"wrong reason for {expect}: {error}"
        else:  # pragma: no cover
            raise AssertionError(f"accepted an artifact that should fail: {expect}")

    # A two-class head instead of a scalar reward.
    rejects(
        {**good, "base_model.model.score.modules_to_save.default.weight": (2, hidden)}, "not scalar"
    )
    # A non-LoRA PEFT tensor is allowed; the real loader validates its config.
    ia3 = {
        "base_model.model.layers.0.mlp.down_proj.ia3_l.default": (1, hidden),
        "base_model.model.score.modules_to_save.default.weight": (1, hidden),
    }
    assert inspect_adapter(ia3, hidden)["adapter_tensors"] == 1
    # A delta trained against a different backbone width.
    rejects(
        {**good, "base_model.model.score.modules_to_save.default.weight": (1, 2048)}, "hidden size"
    )
    # No head at all: the reward would be an untrained random projection.
    rejects({k: v for k, v in good.items() if "score" not in k}, "no scalar head")
    rejects(
        {"base_model.model.score.modules_to_save.default.weight": (1, hidden)},
        "no PEFT parameters",
    )
    full = {
        "model.layers.0.self_attn.q_proj.weight": (hidden, hidden),
        "score.weight": (1, hidden),
    }
    assert inspect_full_model(full, hidden)["format"] == "full_model"
    print("artifact.py smoke passed")


if __name__ == "__main__":
    smoke()
