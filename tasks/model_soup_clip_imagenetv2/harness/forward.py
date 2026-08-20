"""Turning a state dict into predictions. Shared by fast_eval and the final.

Split out from both so that a proxy score and a final score are the same forward
pass over different rows, and split out from grade.py so that grade.py stays free
of torch -- the arithmetic of the metric should be readable, and testable, without
a GPU in the room.

The pinned model-soups tree is baked into the image at /opt/harness/model_soups
rather than mounted, for the reason OPD bakes the JustRL grader into its image:
`get_model_from_sd` decides what a state dict *means* -- which module it becomes,
which head is attached, what the 1000 logits are -- so it is part of the
evaluator's definition of the architecture rather than part of the method. Both
this file and solution/soup.py import from it and neither can edit it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

UPSTREAM = Path("/opt/harness/model_soups")
UPSTREAM_REVISION = "d5398f181ea51c5cd9d95ebacc6ea7132bb108ec"
# Fixed evaluator batch size.
BATCH_SIZE = 256


def add_upstream_to_path() -> None:
    if not UPSTREAM.is_dir():
        raise FileNotFoundError(f"the pinned model-soups tree is missing: {UPSTREAM}")
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))


def load_clip(clip_cache: Path) -> tuple[Any, Any]:
    """CLIP ViT-B/32 and its preprocessing, from the mounted cache.

    `download_root` points at the asset mount and the container has no network, so
    this resolves locally or fails. It never fetches.
    """

    add_upstream_to_path()
    import clip  # type: ignore[import-not-found]

    return clip.load("ViT-B/32", "cpu", jit=False, download_root=str(clip_cache))


def build_model(state: Any, base_model: Any) -> Any:
    add_upstream_to_path()
    import torch
    from utils import get_model_from_sd  # type: ignore[import-not-found]

    model = get_model_from_sd(state, base_model)
    return model.cuda().eval() if torch.cuda.is_available() else model.eval()


def score_rows(
    model: Any,
    rows: list[tuple[Path, int]],
    preprocess: Any,
    offsets: tuple[int, ...],
    *,
    batch_size: int = BATCH_SIZE,
) -> list[dict[str, Any]]:
    """One forward pass over the given (path, label) rows, in order.

    Each row carries the offset it came from, because the final partitions on
    offsets afterwards to produce score(P) and score(F\\P). The offset is derived
    from position rather than from the filename: `select` emits rows class-major
    then offset, so position modulo the offset count is the offset. Deriving it
    here and in the materialization of the proxy asset from the same rule is what
    keeps the two agreeing without a manifest between them.
    """

    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    class Images(Dataset):
        def __init__(self, samples: list[tuple[Path, int]]) -> None:
            self.samples = samples

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int) -> tuple[Any, int]:
            path, label = self.samples[index]
            with Image.open(path) as image:
                return preprocess(image.convert("RGB")), label

    loader = DataLoader(Images(rows), batch_size=batch_size, num_workers=0, pin_memory=True)
    use_cuda = torch.cuda.is_available()
    graded: list[dict[str, Any]] = []
    position = 0
    with torch.no_grad():
        for images, labels in loader:
            if use_cuda:
                images = images.cuda(non_blocking=True)
            predictions = model(images).argmax(dim=1).cpu()
            for label, prediction in zip(labels.tolist(), predictions.tolist(), strict=True):
                path, _ = rows[position]
                graded.append(
                    {
                        "label": label,
                        "offset": offsets[position % len(offsets)],
                        "image": path.name,
                        "prediction": prediction,
                        "correct": label == prediction,
                    }
                )
                position += 1
    if position != len(rows):
        raise RuntimeError(f"graded {position} rows of {len(rows)}")
    return graded
