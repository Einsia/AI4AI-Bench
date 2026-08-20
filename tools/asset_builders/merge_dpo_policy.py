#!/usr/bin/env python3
"""Merge the pinned Zephyr QLoRA adapter into its pinned Mistral base."""

from __future__ import annotations

import argparse
from pathlib import Path


def build(base: Path, adapter: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        base,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    merged = PeftModel.from_pretrained(model, adapter, local_files_only=True).merge_and_unload(
        safe_merge=True
    )
    output.mkdir(parents=True)
    merged.save_pretrained(output, safe_serialization=True, max_shard_size="5GB")
    tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
    tokenizer.save_pretrained(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.base, args.adapter, args.output)


if __name__ == "__main__":
    main()
