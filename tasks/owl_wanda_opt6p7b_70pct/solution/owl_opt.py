"""Run the pinned OWL tree against local assets, with the OPT decoder path.

Ported from the reference protocol's baseline/method/offline_entry.py. Upstream OWL commit
dddb7a4 routes uniform Wanda through LLaMA-only calibration helpers and fetches its
calibration data from the Hub. This adapter keeps upstream's scoring and mask
algorithm and changes two things: it selects the OPT decoder path, and it replaces
Hub dataset aliases with the local mount.

**This file is yours to change.** So is /workspace/owl, which is where the actual
mask arithmetic lives -- `lib/prune_all.py` holds `WrappedGPT`, the Wanda metric
`abs(W) * sqrt(scaler_row)`, OWL's per-layer outlier allocation, and
`return_given_alpha`. Editing that tree takes effect on the next run with no
reinstall.

What is gone from the old version: `run_one_step()` and
`_activation_aware_one_module_wanda()`, which existed to satisfy a candidate
admission gate that v1 does not have. They pruned one linear module of layer 0 from
one calibration sequence and exported nothing, purely to prove a sealed candidate
could still execute Wanda before it was permitted to run.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import torch
from datasets import load_dataset

OWL_SOURCE = Path(os.environ["OWL_SOURCE"])
CALIBRATION_ROOT = Path(os.environ["CALIBRATION_DATA"])
# The shipped wanda_owl path and OPT model setup both hardcode 2048 upstream. Keep
# the adapter's uniform-Wanda control on the same truthful fixed protocol.
SEQUENCE_LENGTH = 2048


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"pinned OWL asset is missing: {path}")
    return path


def _install_offline_loaders() -> None:
    """Point upstream's `get_loaders` at the mounted C4 shard.

    The sampling here is upstream's, kept deliberately: draw a random document, keep
    it only if it tokenises longer than the block length, then take a random window.
    THE MOUNT HOLDS THE WHOLE SHARD, so `nsamples` is a real parameter -- the old task
    fixed it at 128 and a candidate can now raise it. What it cannot do is reach a
    different shard: shard 0 of C4's 1024 en train shards is what is mounted.
    """

    sys.path.insert(0, str(OWL_SOURCE))
    import lib.data as data_module
    import lib.prune_all as prune_module
    from lib.data import TokenizerWrapper

    def get_c4(nsamples, seed, seqlen, tokenizer):
        train_path = _require(CALIBRATION_ROOT / "en/c4-train.00000-of-01024.json.gz")
        validation_path = _require(CALIBRATION_ROOT / "en/c4-validation.00000-of-00008.json.gz")
        train_data = load_dataset("json", data_files=str(train_path), split="train")
        validation_data = load_dataset("json", data_files=str(validation_path), split="train")
        random.seed(seed)
        train_loader = []
        for _ in range(nsamples):
            while True:
                row = train_data[random.randint(0, len(train_data) - 1)]
                encoded = tokenizer(row["text"], return_tensors="pt")
                if encoded.input_ids.shape[1] > seqlen:
                    break
            start = random.randint(0, encoded.input_ids.shape[1] - seqlen - 1)
            sample = encoded.input_ids[:, start : start + seqlen]
            target = sample.clone()
            target[:, :-1] = -100
            train_loader.append((sample, target))
        # C4 validation, returned because upstream's signature has two return values.
        # The pruning path uses only the train loader and discards this. It is C4, not
        # WikiText2 -- no evaluation text is reachable from this container.
        validation_enc = tokenizer(
            " ".join(validation_data[:1100]["text"]), return_tensors="pt"
        ).input_ids[:, : 256 * seqlen]
        return train_loader, TokenizerWrapper(validation_enc)

    def get_loaders(name, nsamples=128, seed=0, seqlen=SEQUENCE_LENGTH, tokenizer=None):
        if "c4" in name:
            return get_c4(nsamples, seed, seqlen, tokenizer)
        raise ValueError(
            f"unsupported dataset {name!r}: only the mounted C4 shard is available, "
            "and there is no network"
        )

    data_module.get_loaders = get_loaders
    prune_module.get_loaders = get_loaders


def _no_upstream_eval(*_args, **_kwargs):
    """Stub out upstream's post-prune WikiText2 evaluation.

    NOT A BOUNDARY, a mechanical necessity. Upstream's main() evaluates WikiText2
    perplexity immediately after pruning and would abort the run when it cannot find
    the data -- and in the pruning container there is no WikiText2 mount at all, which
    is what actually keeps the evaluation splits out of this process.

    The scoring harness does the measuring, on text this container cannot see. During
    exploration the validation split IS mounted, and /opt/harness/fast_eval.sh is the
    supported way to read it.
    """

    print("owl_opt: upstream evaluation skipped; scoring happens in /opt/harness")
    return float("nan")


def _check_sparsity(model) -> float:
    """Upstream's definition, selecting the OPT decoder layers.

    Prints one line per layer and one 'sparsity sanity check' total, because that is
    the only channel prune.py has for reading the achieved allocation back out.
    """

    from lib.prune_all import find_layers

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers if hasattr(model.model, "decoder") else model.model.layers
    zero_count = 0
    parameter_count = 0
    for index, layer in enumerate(layers):
        layer_zero_count = 0
        layer_parameter_count = 0
        for module in find_layers(layer).values():
            weights = module.weight.data
            layer_zero_count += int((weights == 0).sum().item())
            layer_parameter_count += weights.numel()
        zero_count += layer_zero_count
        parameter_count += layer_parameter_count
        print(f"layer {index} sparsity {layer_zero_count / layer_parameter_count:.8f}")
    model.config.use_cache = use_cache
    return zero_count / parameter_count


def _prune_wanda_opt(args, model, tokenizer, device, prune_n=0, prune_m=0) -> None:
    """Upstream's Wanda metric and mask logic with the OPT forward signature.

    The arithmetic is upstream's, unchanged:

        metric = abs(W) * sqrt(scaler_row)

    where `scaler_row` is the mean squared input activation per input channel,
    accumulated by `WrappedGPT.add_batch` through a forward hook on each linear
    module. Then the lowest `sparsity_ratio` fraction of each row is zeroed.

    Everything in it is open to change. If you replace the metric, the score, the
    allocation or the mask rule, the only thing that has to survive is that the
    exported model lands at 70% global sparsity over these layers -- and that is
    checked against the weights, not against this file.
    """

    from lib.prune_all import (
        WrappedGPT,
        find_layers,
        get_loaders,
        prepare_calibration_input_opt,
        return_given_alpha,
    )

    use_cache = model.config.use_cache
    model.config.use_cache = False
    print(f"loading {args.nsamples} calibration sequences at seed {args.seed}")
    dataloader, _ = get_loaders(
        "c4",
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=SEQUENCE_LENGTH,
        tokenizer=tokenizer,
    )
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, _ = prepare_calibration_input_opt(model, dataloader, device)

    layers = model.model.decoder.layers
    for index, layer in enumerate(layers):
        subset = find_layers(layer)
        wrapped_layers = {name: WrappedGPT(module) for name, module in subset.items()}

        def add_batch(name, layer_wrappers=wrapped_layers):
            def hook(_, inputs, output):
                layer_wrappers[name].add_batch(inputs[0].data, output.data)

            return hook

        handles = [module.register_forward_hook(add_batch(name)) for name, module in subset.items()]
        for sample_index in range(args.nsamples):
            with torch.no_grad():
                outs[sample_index] = layer(
                    inps[sample_index].unsqueeze(0), attention_mask=attention_mask
                )[0]
        for handle in handles:
            handle.remove()

        for name, module in subset.items():
            print(f"pruning layer {index} name {name}")
            metric = torch.abs(module.weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )
            mask = torch.zeros_like(metric, dtype=torch.bool)
            if prune_n:
                for column in range(metric.shape[1]):
                    if column % prune_m == 0:
                        group = metric[:, column : column + prune_m].float()
                        indices = torch.topk(group, prune_n, dim=1, largest=False)[1]
                        mask.scatter_(1, column + indices, True)
            else:
                sorted_metric = torch.sort(metric, dim=-1, stable=True)
                if args.use_variant:
                    cumulative_metric = torch.cumsum(sorted_metric[0], dim=1)
                    row_sum = metric.sum(dim=1)
                    alpha = 0.4
                    alpha_bounds = [0.0, 0.8]
                    mask, current_sparsity = return_given_alpha(
                        alpha, sorted_metric, metric, cumulative_metric, row_sum
                    )
                    while (
                        torch.abs(current_sparsity - args.sparsity_ratio) > 0.001
                        and alpha_bounds[1] - alpha_bounds[0] >= 0.001
                    ):
                        if current_sparsity > args.sparsity_ratio:
                            alpha_bounds[1] = alpha
                            alpha = (alpha + alpha_bounds[0]) / 2.0
                        else:
                            alpha_bounds[0] = alpha
                            alpha = (alpha + alpha_bounds[1]) / 2.0
                        mask, current_sparsity = return_given_alpha(
                            alpha, sorted_metric, metric, cumulative_metric, row_sum
                        )
                else:
                    prune_count = int(metric.shape[1] * args.sparsity_ratio)
                    mask.scatter_(1, sorted_metric[1][:, :prune_count], True)
            module.weight.data[mask] = 0

        for sample_index in range(args.nsamples):
            with torch.no_grad():
                outs[sample_index] = layer(
                    inps[sample_index].unsqueeze(0), attention_mask=attention_mask
                )[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def main() -> None:
    _install_offline_loaders()
    import main as upstream_main

    upstream_main.check_sparsity = _check_sparsity
    upstream_main.prune_wanda = _prune_wanda_opt
    upstream_main.eval_ppl = _no_upstream_eval
    upstream_main.main()
    print(f"llmab gpu memory peak bytes {torch.cuda.max_memory_allocated()}")


if __name__ == "__main__":
    main()
