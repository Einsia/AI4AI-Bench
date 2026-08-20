"""Fail-closed build check for the offline NPO trainer and frozen evaluator."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

EXPECTED_REVISION = "4ad738aaf60f6a4385f6e2506d01da99e76c31f3"
PATCHED = {
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    result = subprocess.run(
        ["python3", "-m", "pip", "check"], text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"pip check failed:\n{result.stdout}\n{result.stderr}")

    import accelerate
    import bitsandbytes
    import datasets
    import deepspeed
    import flash_attn
    import torch
    import transformers

    versions = {
        "torch": torch.__version__, "transformers": transformers.__version__,
        "datasets": datasets.__version__, "accelerate": accelerate.__version__,
        "deepspeed": deepspeed.__version__, "flash_attn": flash_attn.__version__,
        "bitsandbytes": bitsandbytes.__version__,
    }
    if bitsandbytes.__version__ != "0.49.0":
        raise RuntimeError(f"expected bitsandbytes 0.49.0, got {bitsandbytes.__version__}")
    bnb_root = Path(bitsandbytes.__file__).resolve().parent
    if not (bnb_root / "libbitsandbytes_cuda128.so").is_file():
        raise FileNotFoundError(bnb_root / "libbitsandbytes_cuda128.so")

    revision = Path("/opt/harness/open-unlearning.revision").read_text(encoding="utf-8").strip()
    if revision != EXPECTED_REVISION:
        raise RuntimeError(f"unexpected OpenUnlearning revision: {revision}")
    frozen = Path("/opt/harness/open-unlearning")
    editable = Path("/workspace/open-unlearning")
    for required in (
        frozen / "src/eval.py",
        editable / "src/train.py",
        Path("/workspace/run.sh"),
        Path("/opt/harness/runtime_guard.py"),
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    if any(Path("/workspace").glob("**/.git")):
        raise RuntimeError("nested .git remained in the editable workspace")
    for relative, expected in PATCHED.items():
        if sha256(frozen / relative) != expected:
            raise RuntimeError(f"frozen evaluator hash mismatch: {relative}")

    import yaml

    accelerate_path = editable / "configs/accelerate/default_config.yaml"
    accelerate_config = yaml.safe_load(accelerate_path.read_text(encoding="utf-8"))
    if accelerate_config.get("distributed_type") != "DEEPSPEED":
        raise RuntimeError(f"NPO accelerate config is not DEEPSPEED: {accelerate_config}")
    deepspeed_config = accelerate_config.get("deepspeed_config", {})
    relative_deepspeed = deepspeed_config.get("deepspeed_config_file")
    if relative_deepspeed != "configs/accelerate/zero_stage3_offload_config.json":
        raise RuntimeError(f"unexpected NPO DeepSpeed config: {relative_deepspeed!r}")
    if deepspeed_config.get("zero3_init_flag") is not True:
        raise RuntimeError("NPO accelerate config must enable zero3_init_flag")
    zero = json.loads((editable / relative_deepspeed).read_text(encoding="utf-8"))
    zero_optimization = zero.get("zero_optimization", {})
    if zero_optimization.get("stage") != 3:
        raise RuntimeError(f"NPO DeepSpeed stage is not 3: {zero_optimization}")
    # The pinned file name contains "offload", but the official revision explicitly
    # disables both forms of offload. Lock the effective values rather than inferring
    # behavior from the filename or silently changing the official protocol.
    for name in ("offload_optimizer", "offload_param"):
        if zero_optimization.get(name, {}).get("device") != "none":
            raise RuntimeError(f"unexpected NPO DeepSpeed {name} device")

    sys.path.insert(0, "/opt/harness")
    spec = importlib.util.spec_from_file_location("npo_final_eval", "/opt/harness/final_eval.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import final evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import runtime_guard

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        summary = module.mock(root / "out", root / "reward")
        if not 0.0 <= summary["balanced_unlearning_score"] <= 1.0:
            raise RuntimeError(f"invalid Balanced mock: {summary}")
        if not summary["validity"]["fixture"].startswith("synthetic"):
            raise RuntimeError("final mock contains measured task results")

        loadable = root / "candidate/checkpoint-10"
        loadable.mkdir(parents=True)
        for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
            (loadable / name).write_text("{}\n", encoding="utf-8")
        (loadable / "model.safetensors").write_bytes(b"build-check")
        incomplete = root / "candidate/checkpoint-20"
        incomplete.mkdir()
        (incomplete / "config.json").write_text("{}\n", encoding="utf-8")
        if runtime_guard.resolve_checkpoint(root / "candidate") != loadable:
            raise RuntimeError("checkpoint resolver did not skip an incomplete higher step")
    print(json.dumps({"image_check": "passed", "versions": versions}, sort_keys=True))


if __name__ == "__main__":
    main()
