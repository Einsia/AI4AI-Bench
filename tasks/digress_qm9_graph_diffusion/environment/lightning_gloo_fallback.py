"""Make Lightning pick a process-group backend NCCL can actually run.

Run at image build time, as root, against the installed lightning_fabric. Not imported
at run time by anything.

WHY THIS EXISTS

Upstream DiGress hardcodes `strategy="ddp_find_unused_parameters_true"` in src/main.py, so
DDP initialises even at `devices=1` -- and Lightning's default backend for any CUDA device
is NCCL. That combination cannot run on the declared target GPU; the reason is
worth writing down, because the obvious diagnosis is wrong.

The card reports capability (10, 3). This image is torch 2.0.1/cu118, whose arch list is

    sm_37 sm_50 sm_60 sm_61 sm_70 sm_75 sm_80 sm_86 sm_90 compute_37

The trailing `compute_37` is PTX, so the driver JIT-compiles torch's own kernels forward
onto sm_103 and they run. Measured, 24 of 25 ops under CUDA_LAUNCH_BLOCKING=1: matmul, cat,
sort, multinomial, scatter_add_, embedding, layer_norm, MultiheadAttention, a Linear
forward+backward, an AdamW step. torch core is healthy on this card.

The one failure was the NCCL collective:

    enqueue.cc:100 NCCL WARN Cuda failure 'named symbol not found'
    NCCL version 2.14.3+cuda11.8

NCCL is statically linked into libtorch_cuda.so and its device code ships as real-arch
cubins with NO PTX entry, so unlike torch core it cannot be JIT-compiled onto an
architecture newer than its build. It stops at its cubin list and the first collective dies
looking up a symbol that was never compiled for sm_103. gloo carries the same CUDA tensors
without complaint, measured on all_reduce and broadcast_object_list.

This was originally diagnosed as a torch_scatter/torch_sparse cubin problem. It is not:
none of torch_scatter, torch_sparse, torch_cluster, torch_spline_conv or pyg_lib is
installed here at all. torch-geometric 2.3.1 is pip-installed with no compiled companions
and falls back to pure torch, so there is no third-party cubin in this image to blame.

WHY THE BACKEND AND NOT THE STRATEGY

`strategy="single_device"` would also avoid NCCL, and it changes more. Lightning wraps the
dataloader in DistributedSampler under DDP, which shuffles from
`torch.Generator().manual_seed(seed + epoch)`; a single-device run uses RandomSampler off
the global RNG. Different shuffle, different batch composition. Switching only the
transport leaves the sampler, and therefore the data the model sees, alone. At world_size=1
a collective is a copy either way.

NOTE ON PROVENANCE: that sampler argument is from reading pytorch_lightning 2.0.4 and
lightning_fabric 2.0.4 in this image, NOT from a differential run comparing the two
strategies. It is the conservative choice on a reading, not a measured equivalence.

WHY THE CONDITION IS SHAPED LIKE THIS

The test is not "is this card an sm_103" and not a device allowlist. It is a statement of
the fault: if the running card's capability is absent from torch's own arch list, then no
cubin in this install targets it, so NCCL -- which has only cubins -- cannot launch, while
torch core survives on PTX. That is why the fallback is gloo and not CPU.

On hardware the install was built for, the condition is false and this returns "nccl",
byte-identically to upstream. An sm_90 device against this wheel finds its cubin and is
unaffected.
"""

from __future__ import annotations

import sys
from pathlib import Path

ORIGINAL = '''def _get_default_process_group_backend_for_device(device: torch.device) -> str:
    return "nccl" if device.type == "cuda" else "gloo"
'''

PATCHED = '''def _get_default_process_group_backend_for_device(device: torch.device) -> str:
    """nccl when NCCL has device code for this card, gloo when it does not.

    PATCHED FOR THIS IMAGE -- see environment/lightning_gloo_fallback.py for the full
    reasoning. Short version: NCCL's device code is cubin-only with no PTX entry, so it
    cannot JIT forward onto an arch newer than its build, while torch core here carries a
    compute_37 PTX entry and can. On a card absent from torch's arch list the first NCCL
    collective dies with "named symbol not found" although every core kernel runs. The
    condition below is that fault, not a device allowlist: on hardware this install was
    built for it is false and the return is "nccl", exactly as upstream.
    """
    if device.type != "cuda":
        return "gloo"
    try:
        major, minor = torch.cuda.get_device_capability(device)
        arch_list = torch.cuda.get_arch_list()
    except Exception:
        # A probe must never be the thing that breaks the run. Upstream's answer.
        return "nccl"
    if not arch_list:
        return "nccl"
    if f"sm_{major}{minor}" in arch_list:
        return "nccl"
    # A cubin is forward compatible within its major version -- an sm_100 cubin runs on
    # sm_103 -- so an equal-major, lower-or-equal-minor entry is loadable and NCCL is fine.
    for entry in arch_list:
        if not entry.startswith("sm_"):
            continue
        digits = entry[3:]
        if not digits.isdigit() or len(digits) < 2:
            continue
        if int(digits[:-1]) == major and int(digits[-1]) <= minor:
            return "nccl"
    return "gloo"
'''


def patch_text(text: str) -> str:
    """Replace the backend chooser, or refuse if the file is not what we expect.

    Exact-count, like harness/final_eval.py:patch_upstream_main. A patch that
    half-applies leaves a file that imports and misbehaves, which is the failure mode
    that costs a batch of runs rather than a build.
    """

    # Order matters. Checking the count first makes a re-run report "lightning_fabric has
    # changed", which is both wrong and the most expensive wrong answer available: it points
    # whoever reads it at upstream instead of at their own second invocation.
    if "PATCHED FOR THIS IMAGE" in text:
        raise SystemExit(
            "lightning_gloo_fallback: already patched, nothing to do. This script is not "
            "idempotent by design -- applying it twice would mean the first application was "
            "not what is on disk."
        )
    found = text.count(ORIGINAL)
    if found != 1:
        raise SystemExit(
            "lightning_gloo_fallback: the upstream backend chooser appears "
            f"{found} times, expected exactly 1. lightning_fabric has changed; "
            "re-read the function before editing this patch."
        )
    return text.replace(ORIGINAL, PATCHED, 1)


def self_test() -> None:
    """Exercise both branches without a GPU, because the build host has none.

    The function only reads torch.cuda.get_device_capability and get_arch_list, so
    stubbing those two covers the whole decision. Constructing torch.device("cuda", 0)
    needs no driver. Without this the build could ship a patch that imports cleanly and
    chooses wrong, and the first evidence would be a failed retrain.
    """

    import torch
    from lightning_fabric.utilities.distributed import (
        _get_default_process_group_backend_for_device as choose,
    )

    real_capability = torch.cuda.get_device_capability
    real_arch_list = torch.cuda.get_arch_list
    cuda = torch.device("cuda", 0)

    cases = (
        # target GPU: sm_103 absent from a cu118 arch list -> gloo
        ((10, 3), ["sm_37", "sm_80", "sm_90", "compute_37"], "gloo"),
        # an sm_90 device against this wheel: cubin present -> upstream nccl behavior
        ((9, 0), ["sm_37", "sm_80", "sm_90", "compute_37"], "nccl"),
        # exact match anywhere in the list -> nccl
        ((8, 6), ["sm_80", "sm_86", "sm_90"], "nccl"),
        # cubin forward-compatible within a major: sm_100 cubin on sm_103 -> nccl
        ((10, 3), ["sm_90", "sm_100"], "nccl"),
        # same major but the only cubin is NEWER, so it does not load -> gloo
        ((10, 0), ["sm_90", "sm_103"], "gloo"),
        # no arch list at all: say nothing, keep upstream's answer
        ((10, 3), [], "nccl"),
    )
    try:
        for capability, arch_list, expected in cases:
            torch.cuda.get_device_capability = lambda _device=None, _c=capability: _c
            torch.cuda.get_arch_list = lambda _a=arch_list: _a
            got = choose(cuda)
            if got != expected:
                raise SystemExit(
                    "lightning_gloo_fallback: self-test failed for capability "
                    f"{capability} arch_list {arch_list}: expected {expected}, got {got}"
                )
        if choose(torch.device("cpu")) != "gloo":
            raise SystemExit("lightning_gloo_fallback: cpu must still choose gloo")
    finally:
        torch.cuda.get_device_capability = real_capability
        torch.cuda.get_arch_list = real_arch_list

    print(
        "lightning_gloo_fallback: self-test passed "
        f"({len(cases)} capability/arch cases plus cpu)"
    )


def main() -> int:
    import lightning_fabric.utilities.distributed as target_module

    target = Path(target_module.__file__)
    source = target.read_text(encoding="utf-8")
    target.write_text(patch_text(source), encoding="utf-8")
    print(f"lightning_gloo_fallback: patched {target}")

    # Re-import from disk so the self-test exercises the file that shipped, not the
    # module object already in memory from the line above.
    for name in [n for n in sys.modules if n.startswith("lightning_fabric")]:
        del sys.modules[name]
    self_test()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
