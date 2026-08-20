"""Materialize the 128-row IFEval proxy asset from the full 541-row source.

A trusted preparation step, run on the host once per IFEval revision -- not in any
Agent container, and not at image build time. It replaces the reference protocol's
`environment/project_public_ifeval.py`, which cut a *different* slice: `first_128`
of the hash ordering, the tier that is now retired.

    python3 environment/build_proxy_asset.py \
      --ifeval <TRUSTED_ASSETS>/data/ifeval_final/ifeval_input_data.jsonl \
      --output <TRUSTED_ASSETS>/data/ifeval_proxy

Which 128 rows is decided by `harness/grade.split_source`, the same function
`final_eval.py` calls to recompute membership when it reports score(P) and
score(F\\P). This script does not get its own copy of that logic; if it did, the
proxy asset and the final's idea of the proxy could drift, and the three-number
report would describe two different row sets. That is precisely what happened on the
reference protocol, where the projection script and the evaluator each carried their own
`first_128` and the ordering had to agree by inspection.

Two checks are worth the trouble here:

  * `grade.require_legacy_ordering` reproduces the `public_keys_sha256` recorded in
    the reference protocol's `assets.lock.yaml`. If the source file is not
    google/IFEval@966cd895, that digest will not match, and every tier would
    otherwise be silently re-cut.
  * the emitted proxy is checked to be disjoint from the retired public128 and a
    subset of the final. The proxy and the retired tier are *adjacent* in the same
    ordering and the same size, so mixing them up is the easy mistake.
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

IFEVAL_SOURCE = "google/IFEval"
IFEVAL_REVISION = "966cd89545d6b6acfd7638bc708b98261ca58e84"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_source(path: Path) -> Path:
    """Accept either the jsonl itself or the directory holding exactly one."""

    if path.is_file():
        return path
    candidates = sorted(path.glob("*.jsonl"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected exactly one jsonl under {path}, found {len(candidates)}")
    return candidates[0]


def build(ifeval: Path, output: Path) -> dict[str, Any]:
    source_path = resolve_source(ifeval)
    rows = [
        json.loads(line)
        for line in source_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    split = grade.split_source(rows)
    ordering_digest = grade.require_legacy_ordering(split["legacy_public"])

    proxy = split["proxy"]
    proxy_keys = {grade.canonical_key(row) for row in proxy}
    final_keys = {grade.canonical_key(row) for row in split["final"]}
    legacy_keys = {grade.canonical_key(row) for row in split["legacy_public"]}
    if not proxy_keys <= final_keys:
        raise RuntimeError("the proxy is not a subset of the final")
    if proxy_keys & legacy_keys:
        raise RuntimeError("the proxy overlaps the retired public128")
    if len(proxy) != grade.PROXY_ROWS:
        raise RuntimeError(f"selected {len(proxy)} rows, expected {grade.PROXY_ROWS}")

    output.mkdir(parents=True, exist_ok=True)
    proxy_path = output / "proxy.jsonl"
    # Written in selection order, which is the order fast_eval scores them in and the
    # order the digest below is taken over.
    proxy_path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in proxy),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "task_id": grade.TASK_ID,
        "source": IFEVAL_SOURCE,
        "source_revision": IFEVAL_REVISION,
        "source_file": source_path.name,
        "source_sha256": file_sha256(source_path),
        "source_rows": grade.SOURCE_ROWS,
        "final_rows": grade.FINAL_ROWS,
        "proxy_rows": grade.PROXY_ROWS,
        "held_out_rows": grade.HELD_OUT_ROWS,
        "overlap_fraction": grade.PROXY_ROWS / grade.FINAL_ROWS,
        "ordering": "sha256_of_canonical_key",
        "selection": "first_128_after_first128 (a prefix of the sealed 413)",
        "legacy_public_keys_sha256": ordering_digest,
        "selection_digest": grade.keys_digest(grade.canonical_key(row) for row in proxy),
        "files": {
            proxy_path.name: {
                "rows": grade.PROXY_ROWS,
                "sha256": file_sha256(proxy_path),
            }
        },
    }
    (output / "proxy-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ifeval", type=Path, required=True, help="the 541-row source jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.ifeval.resolve(), args.output.resolve())
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
