"""The one scoring path, shared by fast_eval and the hidden final.

Both tiers call `perplexity` here on a WikiText2 parquet, so a proxy score and a
final score differ only in which split they read. Nothing in this file chooses a
split, a checkpoint or an output location -- that is the caller's half. Keep it that
way: this file is the reason "the evaluator does not change" can be checked by
reading one thing.

Three details are load-bearing because changing any of them changes the metric:

  "\\n\\n".join(text)   how the rows are concatenated before tokenising. This is
                      Wanda's and OWL's convention for WikiText2 perplexity.
  use_fast=False       the slow OPT tokeniser. Fast and slow tokenisers do not
                      always agree on token count, and the token count sets the
                      number of 2048-token blocks, which sets the score.
  2048                 the block length. Both the calibration and the evaluation use
                      it upstream.

One thing did change, deliberately. The old evaluator read the parquet through
`datasets.load_dataset("parquet", ...)`, which builds an arrow cache and made the
score depend on a `datasets` version and a writable cache directory. pandas reads
the same column with no cache and no third dependency, so the scoring container now
needs torch, transformers and pandas and nothing else.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

SEQUENCE_LENGTH = 2048
# Exact artifact window around the 0.7 target.
SPARSITY_WINDOW = (0.699, 0.701)


def load_text(parquet: Path) -> str:
    """Read one WikiText2 parquet into the single string the protocol scores.

    Accepts a directory so a mount can hold the file under its published name.
    """

    import pandas as pd

    candidates = [parquet] if parquet.is_file() else sorted(parquet.glob("*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"no WikiText2 parquet at {parquet}")
    frame = pd.read_parquet(candidates[0])
    if "text" not in frame.columns:
        raise ValueError(f"unexpected WikiText2 schema: {sorted(frame.columns)}")
    return "\n\n".join(frame["text"].astype(str).tolist())


def decoder_sparsity(model: Any) -> tuple[float, list[float]]:
    """Global and per-layer zero fraction over the OPT decoder's linear weights.

    THE ARTIFACT-SIDE CHECK. Computed from the weights that are about to be scored,
    never from metadata the candidate wrote. The reference protocol had both: this function,
    which is real, and a `load_metadata` beside it that read sparsity_ratio out of a
    `training_metadata.json` the pruning script itself produced. The second one only
    proved the candidate could write 0.7 into a file, and it is gone.

    What it defends: 70% sparsity is not a rule about method, it is what makes the
    perplexity comparable at all. A model pruned to 50% would score far better and
    would not be the same measurement. Because a mount cannot hold this -- it is a
    property of the produced tensors -- it is checked at score time and a violation
    invalidates the trial.
    """

    import torch

    layers = model.model.decoder.layers
    total_zero = 0
    total_count = 0
    ratios: list[float] = []
    for layer in layers:
        layer_zero = 0
        layer_count = 0
        for module in layer.modules():
            if isinstance(module, torch.nn.Linear):
                weights = module.weight.data
                layer_zero += int((weights == 0).sum().item())
                layer_count += weights.numel()
        if not layer_count:
            raise RuntimeError("OPT decoder layer contains no linear weights")
        ratios.append(layer_zero / layer_count)
        total_zero += layer_zero
        total_count += layer_count
    if not total_count:
        raise RuntimeError("OPT decoder has no linear weights")
    return total_zero / total_count, ratios


def load_model(checkpoint: Path) -> tuple[Any, Any]:
    """Load a pruned checkpoint in fp16 with its own tokeniser.

    The tokeniser comes from the checkpoint rather than from the dense asset because
    the scoring container does not mount the dense model -- and because a checkpoint
    whose tokeniser cannot be loaded is not scoreable, which is worth finding out
    here rather than three minutes in.
    """

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"no config.json under {checkpoint}")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False, local_files_only=True)
    model.eval()
    return model, tokenizer


def block_nlls(model: Any, tokenizer: Any, text: str) -> list[float]:
    """Total negative log likelihood per complete 2048-token block.

    Returns one number per block rather than a mean, so the caller can aggregate over
    any subset of blocks. That is what makes a proxy-inside-final split reportable
    without scoring twice -- unused here, since validation and test are disjoint, but
    it is the shape the spec asks for and it costs nothing.
    """

    import torch
    import torch.nn as nn

    encoded = tokenizer(text, return_tensors="pt").input_ids
    blocks = encoded.numel() // SEQUENCE_LENGTH
    if blocks <= 0:
        raise RuntimeError("the evaluation text has no complete 2048-token block")
    device = next(model.parameters()).device
    loss_fn = nn.CrossEntropyLoss()
    nlls: list[float] = []
    with torch.no_grad():
        for index in range(blocks):
            inputs = encoded[:, index * SEQUENCE_LENGTH : (index + 1) * SEQUENCE_LENGTH].to(device)
            logits = model(inputs).logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = inputs[:, 1:]
            loss = loss_fn(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            ).float()
            # The upstream convention: the mean token loss over the block, scaled
            # back up by the block length. Note it multiplies by SEQUENCE_LENGTH
            # while the loss averaged over SEQUENCE_LENGTH - 1 shifted positions.
            # This is Wanda's and OWL's evaluator arithmetic.
            nlls.append(float(loss.item() * SEQUENCE_LENGTH))
    return nlls


def perplexity(nlls: list[float]) -> float:
    """exp of the mean per-token NLL over the blocks given."""

    if not nlls:
        raise ValueError("no blocks to score")
    value = math.exp(sum(nlls) / (len(nlls) * SEQUENCE_LENGTH))
    if not math.isfinite(value):
        raise RuntimeError(f"perplexity is non-finite: {value}")
    return value


def sparsity_in_window(actual: float) -> bool:
    low, high = SPARSITY_WINDOW
    return low <= actual <= high


def summarize(nlls: list[float], actual_sparsity: float, layer_sparsities: list[float]) -> dict:
    """The fields both tiers report. No `correct` and no `n`: this is not accuracy.

    `spread` is the block-to-block standard deviation of per-token NLL, reported
    because it is the only dispersion this metric has. It is NOT an error bar on the
    score: the blocks are one fixed text, not a sample from a population. Candidate
    robustness still depends on the pruning calibration seed.
    """

    per_token = [value / SEQUENCE_LENGTH for value in nlls]
    mean = sum(per_token) / len(per_token)
    if len(per_token) > 1:
        variance = sum((value - mean) ** 2 for value in per_token) / (len(per_token) - 1)
    else:
        variance = 0.0
    return {
        "score": perplexity(nlls),
        "blocks": len(nlls),
        "sequence_length": SEQUENCE_LENGTH,
        "block_nll_per_token_mean": mean,
        "block_nll_per_token_sd": math.sqrt(variance),
        "actual_global_sparsity": actual_sparsity,
        "sparsity_in_window": sparsity_in_window(actual_sparsity),
        "sparsity_window": list(SPARSITY_WINDOW),
        "layer_sparsity_min": min(layer_sparsities) if layer_sparsities else None,
        "layer_sparsity_max": max(layer_sparsities) if layer_sparsities else None,
        "layer_sparsity_mean": (
            sum(layer_sparsities) / len(layer_sparsities) if layer_sparsities else None
        ),
        "layers": len(layer_sparsities),
    }


def smoke() -> None:
    # A uniform NLL must give exactly exp of the per-token value.
    flat = [2.0 * SEQUENCE_LENGTH] * 10
    assert abs(perplexity(flat) - math.exp(2.0)) < 1e-9, perplexity(flat)
    summary = summarize(flat, 0.7, [0.68, 0.72])
    assert summary["blocks"] == 10, summary
    assert summary["sparsity_in_window"] is True, summary
    assert summary["block_nll_per_token_sd"] == 0.0, summary
    assert abs(summary["layer_sparsity_mean"] - 0.70) < 1e-12, summary
    # The window is exclusive of a 50%-pruned model, which is the case the check
    # exists for.
    assert not sparsity_in_window(0.5)
    assert not sparsity_in_window(0.6989)
    assert sparsity_in_window(0.699) and sparsity_in_window(0.701)
    # A varied set must report a positive spread, or the field is not doing anything.
    varied = summarize([1.0 * SEQUENCE_LENGTH, 3.0 * SEQUENCE_LENGTH], 0.7, [0.7])
    assert varied["block_nll_per_token_sd"] > 0.0, varied
    print("grade.py smoke passed")


if __name__ == "__main__":
    smoke()
