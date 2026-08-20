#!/usr/bin/env python3
"""Create DiGress' frozen hydrogen-free QM9 export from public raw files."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

EXPECTED_README_SHA256 = "049212f3675916f611c025d63f4af5705150991e84d56f235b32f2037f4906b2"
EXPECTED_SPLIT_SHA256 = {
    "train.csv": "43604483853746a0cf21e597d8aea030e8bcc4213103c7edd9199b015cede13e",
    "val.csv": "c6c11be1c5ddd368ee8a46d88edff7438c30c41042ed255878f8d26fbde20137",
    "test.csv": "7cec851383f60c337270b016f46660f41bd5dc421433498a79cc5e76cb6fedef",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exactly_one(root: Path, name: str) -> Path:
    matches = [path for path in root.rglob(name) if path.is_file() and not path.is_symlink()]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {name} below {root}, found {len(matches)}")
    return matches[0]


def build(qm9: Path, uncharacterized: Path, readme: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    raw = output / "raw"
    raw.mkdir(parents=True)
    for name in ("gdb9.sdf", "gdb9.sdf.csv"):
        shutil.copyfile(exactly_one(qm9, name), raw / name)
    candidates = [path for path in uncharacterized.rglob("*") if path.is_file()]
    if len(candidates) != 1:
        raise FileNotFoundError("the QM9 uncharacterized source must contain one file")
    shutil.copyfile(candidates[0], raw / "uncharacterized.txt")
    # DiGress' frozen export retained this public provenance file even though PyG
    # does not list it in raw_file_names. It is pinned separately from the QM9 zip.
    shutil.copyfile(exactly_one(readme, "QM9_README"), raw / "QM9_README")
    if sha256(raw / "QM9_README") != EXPECTED_README_SHA256:
        raise ValueError("QM9_README does not match the pinned public file")

    import numpy as np
    import pandas as pd

    dataset = pd.read_csv(raw / "gdb9.sdf.csv")
    number = len(dataset)
    train_rows = 100_000
    test_rows = int(0.1 * number)
    validation_rows = number - train_rows - test_rows
    train, validation, test = np.split(
        dataset.sample(frac=1, random_state=42),
        [train_rows, train_rows + validation_rows],
    )
    train.to_csv(raw / "train.csv")
    validation.to_csv(raw / "val.csv")
    test.to_csv(raw / "test.csv")
    for name, expected in EXPECTED_SPLIT_SHA256.items():
        if sha256(raw / name) != expected:
            raise RuntimeError(f"pandas produced a non-canonical QM9 split: {name}")

    sys.path.insert(0, "/opt/harness/digress")
    from src.datasets.qm9_dataset import QM9Dataset, compute_qm9_smiles
    from torch_geometric.loader import DataLoader

    partitions = {
        stage: QM9Dataset(stage=stage, root=str(output), remove_h=True)
        for stage in ("train", "val", "test")
    }
    loader = DataLoader(partitions["train"], batch_size=512, shuffle=False, num_workers=0)
    smiles = compute_qm9_smiles(["C", "N", "O", "F"], loader, True)
    np.save(output / "train_smiles_no_h.npy", np.asarray(smiles))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qm9", type=Path, required=True)
    parser.add_argument("--uncharacterized", type=Path, required=True)
    parser.add_argument("--readme", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.qm9, args.uncharacterized, args.readme, args.output)


if __name__ == "__main__":
    main()
