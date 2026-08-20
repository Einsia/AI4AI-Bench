"""Materialize the 512-pair proxy asset from the full RewardBench snapshot.

A trusted preparation step, run on the host once per RewardBench revision -- not in
any Agent container, and not at image build time. It is the analogue of the previous
branch's `judge/tools/project_public_assets.py`, and much smaller, because it
selects rather than de-contaminates: the rows it writes are RewardBench rows,
unmodified.

    python3 environment/build_proxy_asset.py \
      --rewardbench <TRUSTED_REWARDBENCH> \
      --output <assets>/data/rewardbench_proxy

Which 512 rows is decided by `harness/grade.select_proxy_ids`, the same function
`final_eval.py` calls to recompute membership when it reports score(P) and
score(F\\P). This script does not get its own copy of that logic; if it did, the
proxy asset and the final's idea of the proxy could drift, and the three-number
report would be describing two different row sets.

The manifest carries a digest of the selected ids so `fast_eval` can refuse a proxy
file that is not the row set this harness selects -- a stale asset left over from a
different RewardBench revision, for instance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

import grade  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_rewardbench(root: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    files = sorted((root / "data").glob("filtered-*.parquet")) or sorted(
        root.glob("filtered-*.parquet")
    )
    if not files:
        raise FileNotFoundError(f"no filtered RewardBench parquet under {root}")
    table = pq.read_table([str(path) for path in files])
    required = {"id", "subset", "prompt", "chosen", "rejected"}
    if not required <= set(table.column_names):
        raise ValueError(f"unexpected RewardBench schema: {sorted(table.column_names)}")
    return [
        {key: str(record[key]) for key in sorted(required)}
        for record in table.select(sorted(required)).to_pylist()
    ]


def build(rewardbench: Path, output: Path, size: int) -> dict[str, Any]:
    rows = read_rewardbench(rewardbench)
    if len(rows) != grade.FINAL_ROWS:
        raise ValueError(f"RewardBench has {len(rows)} rows, expected {grade.FINAL_ROWS}")
    selected_ids = grade.select_proxy_ids(rows, size)
    selected = [row for row in rows if row["id"] in selected_ids]
    if len(selected) != size:
        raise RuntimeError(
            f"{len(selected)} rows matched {size} selected ids; RewardBench ids are not unique"
        )

    output.mkdir(parents=True, exist_ok=True)
    pairs_path = output / "proxy.jsonl"
    # Sorted by id so the file is byte-identical across runs on the same revision.
    pairs_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in sorted(selected, key=lambda item: item["id"])
        ),
        encoding="utf-8",
    )
    counts: dict[str, int] = {}
    for row in selected:
        counts[row["subset"]] = counts.get(row["subset"], 0) + 1
    manifest = {
        "schema_version": 1,
        "task_id": "ultrafeedback_bt_rm_rewardbench",
        "source": "allenai/reward-bench",
        "source_revision": "168d848cdbbea9764fae4a544dc9ca1e6cca4931",
        "final_rows": grade.FINAL_ROWS,
        "proxy_rows": size,
        "overlap_fraction": size / grade.FINAL_ROWS,
        "selection": "stratified over all 23 subsets, largest remainder, one-row floor",
        "selection_digest": grade.selection_digest(selected_ids),
        "subset_rows": dict(sorted(counts.items())),
        "files": {
            pairs_path.name: {"rows": size, "sha256": file_sha256(pairs_path)},
        },
    }
    (output / "proxy-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rewardbench", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=grade.PROXY_ROWS)
    args = parser.parse_args()
    manifest = build(args.rewardbench.resolve(), args.output.resolve(), args.size)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
