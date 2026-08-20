"""The one scoring path, shared by fast_eval and the hidden final.

Both stages score IFEval rows with the same scorer, the same generation protocol
and the same aggregation, so a proxy number and a final number differ only in
which rows were scored. That is new: on the reference protocol the proxy tier was
`first_64` of a 128-row public projection that was *disjoint* from the final's 413
rows, and the two tiers therefore had no rows in common at all.

Nothing here loads a model or a checkpoint. Keep it that way: this file is the
reason "the evaluator does not change" can be checked by reading one thing.

## The row split, and why it is a prefix of the sealed set

The source is `google/IFEval` at 966cd895, 541 rows. The reference protocol ordered
them by `sha256(str(row["key"]))` ascending and took `first_128` as the public
projection; the remaining 413 were the sealed final. Both numbers are recorded in
`environment/assets.lock.yaml` and in the judge-side `final.toml`
(`partition = "sha256_key_order_after_first128"`), so 541 = 128 + 413, disjoint.

v1 keeps the ordering byte for byte -- see `LEGACY_PUBLIC_KEYS_SHA256`, which is
checked rather than trusted -- and re-cuts the tiers on top of it:

    ordered[:128]      the retired public128. Not scored by either tier now.
    ordered[128:]      FINAL, all 413 rows. This is the reward.
    ordered[128:256]   PROXY, 128 rows, a prefix of the final.
    ordered[256:]      the 285 rows no exploration container mounts.

The proxy is a *prefix of the final* rather than a spread-out sample because
`sha256` of the row key is already a fixed random permutation of the source: a
prefix, a suffix and an every-third-row sample are the same draw, so taking a
prefix costs nothing and adds no second selection rule to keep in sync. The
digest check above is what makes that claim testable.

Overlap is 128/413 = 31.0%, inside the spec's 50% ceiling.

## Proxy interpretation

The proxy is a visible subset of the final. `final_eval.py` reports the whole
final, visible proxy and held-out complement separately. A proxy change is not a
held-out result.

The reported split includes score(F),
score(P) and score(F\\P).

## No stratification, unlike BT-RM

`ifeval_prompt_level_strict_accuracy` is an unweighted mean of independent 0/1
row outcomes, so a uniform subsample is an unbiased estimator of the full set and
a prefix of a hash order is a uniform subsample. BT-RM has to stratify because
RewardBench weights subset accuracies inside four sections, which makes the row
mix part of the metric; here it is not.

## Uncertainty

`summarize` reports the binomial standard error of one absolute row mean. It does
not compute a paired interval and does not include training-seed or replay
variance. The row-level records are retained for matched comparisons.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

TASK_ID = "dpo_preference_alignment"
METRIC = "ifeval_strict_accuracy_hidden413"

# google/IFEval@966cd89545d6b6acfd7638bc708b98261ca58e84, the whole input file.
SOURCE_ROWS = 541
# The reference protocol's public projection. Retired: neither tier scores these 128
# rows now. Kept as a constant because the ordering is verified through it.
LEGACY_PUBLIC_ROWS = 128
# The sealed final: everything after the retired prefix. 413 rows, and the reward.
FINAL_ROWS = SOURCE_ROWS - LEGACY_PUBLIC_ROWS
# The proxy: the first 128 rows of the final, in the same hash order.
PROXY_ROWS = 128
HELD_OUT_ROWS = FINAL_ROWS - PROXY_ROWS

# From environment/assets.lock.yaml on the reference protocol:
# derivation.public_keys_sha256, over the 128 keys of the retired projection, newline
# joined in selection order. Recomputing it here is what proves this file's ordering
# is the one that produced every recorded IFEval number -- an ordering change would
# silently re-cut all three tiers and every baseline in task.toml would describe rows
# that are no longer being scored.
LEGACY_PUBLIC_KEYS_SHA256 = "c395d8bcee8447ddd5bcb179ce1ceeea45a9eb5ca3c763ea61e92761b8e29b9a"

# The generation protocol, from baseline/evidence.json's metric.protocol:
# "541 rows; greedy; model chat template; max_new_tokens=1280; deterministic
# disjoint shards; fresh metric state per row". Every one of these is part of the
# measurement, so they live here rather than in either evaluator.
MAX_NEW_TOKENS = 1280
DO_SAMPLE = False
TOKENIZATION_PROTOCOL = "llmab_deterministic_ifeval_tokenization_v1"
PROTOCOL = (
    "413 rows (proxy 128, held out 285); greedy; model chat template; "
    "max_new_tokens=1280; deterministic disjoint shards; fresh metric state per row"
)


class _DeterministicSentenceTokenizer:
    """Frozen sentence splitter for IFEval's sentence-count constraints.

    Carried verbatim from the reference protocol's evaluator. It exists because
    lighteval's IFEval reaches for `nltk.download` at scoring time, and the
    scoring container has no network -- so the alternative to a frozen splitter is
    a scorer that either fails or silently falls back.
    """

    _boundary = re.compile(
        r"(?<=[.!?])(?:[\"')\]]*)\s+(?=(?:[\"'(\[]?[A-Z0-9]|[-*#>]))|\n{2,}"
    )

    def tokenize(self, text: str) -> list[str]:
        stripped = text.strip()
        if not stripped:
            return []
        return [segment.strip() for segment in self._boundary.split(stripped) if segment.strip()]


_SENTENCE_TOKENIZER = _DeterministicSentenceTokenizer()


def deterministic_word_tokenize(text: str, *_args: Any, **_kwargs: Any) -> list[str]:
    """Frozen word splitter, replacing `nltk.word_tokenize`. Same reason as above."""

    return re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*|[^\w\s]", text, flags=re.UNICODE)


# --------------------------------------------------------------------------- #
# row selection: one ordering, three callers
# --------------------------------------------------------------------------- #


def canonical_key(row: dict[str, Any]) -> str:
    """The row's stable identity, as the projection defined it.

    IFEval keys are integers in the source file. `str()` of them is what the
    reference protocol hashed, so it is what has to be hashed here.
    """

    key = row.get("key")
    if isinstance(key, bool) or not isinstance(key, (str, int)) or key == "":
        raise ValueError(f"IFEval row has no stable key: {key!r}")
    return str(key)


def order_key(row: dict[str, Any]) -> tuple[str, str]:
    """`(sha256(canonical key), canonical key)`, the frozen ordering.

    Lifted from `environment/project_public_ifeval.py` on the reference protocol. The
    second element only breaks digest ties, which cannot happen for distinct keys;
    it is kept so the sort is total on paper as well as in practice.
    """

    canonical = canonical_key(row)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), canonical


def keys_digest(keys: Any) -> str:
    """Digest of an ordered key list. Order-sensitive, because order is the thing
    being pinned.

    Two callers compare it: `require_legacy_ordering` against the value recorded on
    the reference protocol, and `fast_eval.load_proxy` against the proxy asset it was
    handed. Writing the expression twice is how the two come to disagree.
    """

    return hashlib.sha256("\n".join(str(key) for key in keys).encode("utf-8")).hexdigest()


def ordered_source(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """All 541 source rows in the frozen order, refusing anything else.

    The row count, the schema and key uniqueness are all checked. The previous
    branch checked the count in the projection script and the schema in the
    evaluator; both matter to the same claim, so both are here.
    """

    if len(rows) != SOURCE_ROWS:
        raise ValueError(f"IFEval source holds {len(rows)} rows, expected {SOURCE_ROWS}")
    required = {"key", "prompt", "instruction_id_list", "kwargs"}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not required <= set(row):
            raise ValueError(f"IFEval row {index} is missing {sorted(required - set(row or {}))}")
    keys = [canonical_key(row) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("IFEval source keys are not unique, so the ordering is undefined")
    return sorted(rows, key=order_key)


def split_source(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Cut the 541 into the retired prefix, the final, the proxy and the held out.

    One definition, three callers: `environment/build_proxy_asset.py` materializes
    `proxy`, `fast_eval` re-derives it to check the asset it was given, and
    `final_eval` re-derives all four so that score(P) and score(F\\P) are split by
    the rule that chose the rows rather than by whatever a file happens to hold.
    """

    ordered = ordered_source(rows)
    legacy = ordered[:LEGACY_PUBLIC_ROWS]
    final = ordered[LEGACY_PUBLIC_ROWS:]
    proxy = final[:PROXY_ROWS]
    held_out = final[PROXY_ROWS:]
    counts = (len(legacy), len(final), len(proxy), len(held_out))
    expected = (LEGACY_PUBLIC_ROWS, FINAL_ROWS, PROXY_ROWS, HELD_OUT_ROWS)
    if counts != expected:
        raise RuntimeError(f"split produced {counts}, expected {expected}")
    return {"legacy_public": legacy, "final": final, "proxy": proxy, "held_out": held_out}


def require_legacy_ordering(legacy_rows: list[dict[str, Any]]) -> str:
    """Check the ordering against the digest the reference protocol recorded.

    This is the one check that ties every number in task.toml to the rows being
    scored now. If it fails, the sort changed and all three tiers moved.
    """

    digest = keys_digest(canonical_key(row) for row in legacy_rows)
    if digest != LEGACY_PUBLIC_KEYS_SHA256:
        raise ValueError(
            "the IFEval ordering does not reproduce the recorded public128 projection\n"
            f"  expected {LEGACY_PUBLIC_KEYS_SHA256}\n  actual   {digest}\n"
            "Every recorded baseline was measured on tiers cut from that ordering, so a "
            "mismatch means the mounted source is not google/IFEval@966cd895."
        )
    return digest


def shard(rows: list[dict[str, Any]], shards: int) -> list[list[dict[str, Any]]]:
    """Deterministic disjoint shards, round robin, as the protocol string says.

    Sharding cannot change any row's score here, and the reason is narrower than
    "generation is deterministic": each row is generated on its own, greedy, with
    no padding and no batch neighbours, so there is no reduction whose order a
    different shard layout could change. That is what makes the recorded two-device
    hidden413 run comparable with a one-device one -- OPD cannot say the same,
    because its rows share a batch.
    """

    if shards < 1:
        raise ValueError("at least one shard is required")
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(shards)]
    for position, row in enumerate(rows):
        buckets[position % shards].append(row)
    return buckets


# --------------------------------------------------------------------------- #
# scoring one completion
# --------------------------------------------------------------------------- #

_SCORER_STATE: tuple[Any, Any, Any] | None = None


def install_ifeval_scorer() -> tuple[Any, Any, Any]:
    """Patch lighteval's IFEval registry for literal-symbol frequency, once.

    Carried from the reference protocol. `keywords:letter_frequency` rows in IFEval
    can ask about a non-alphabetic character, and upstream's LetterFrequencyChecker
    rejects those, so the rows would fail on the checker rather than on the model.

    Installed once rather than per row: the patch is idempotent, and "fresh metric
    state per row" is about the scorer instance, which `score_completion` still
    builds per row. The reference protocol re-ran the whole install for all 541 rows.
    """

    global _SCORER_STATE
    if _SCORER_STATE is not None:
        return _SCORER_STATE

    import nltk

    # No network in this container. Upstream calls download() during scoring.
    nltk.download = lambda *_args, **_kwargs: False
    nltk.word_tokenize = deterministic_word_tokenize

    from lighteval.models.model_output import ModelResponse
    from lighteval.tasks.tasks.ifeval import instructions_registry, instructions_utils
    from lighteval.tasks.tasks.ifeval.instructions import LetterFrequencyChecker
    from lighteval.tasks.tasks.ifeval.main import IFEvalMetrics, ifeval_prompt

    instructions_utils._get_sentence_tokenizer = lambda: _SENTENCE_TOKENIZER
    instructions_utils.nltk.download = lambda *_args, **_kwargs: False
    instructions_utils.nltk.word_tokenize = deterministic_word_tokenize

    class LiteralSymbolFrequencyChecker(LetterFrequencyChecker):
        """Accept a single non-alphabetic character as the counted symbol."""

        def build_description(self, *, letter=None, let_frequency=None, let_relation=None):
            symbol = letter.strip() if isinstance(letter, str) else ""
            if len(symbol) != 1 or symbol.isalpha():
                return super().build_description(
                    letter=letter, let_frequency=let_frequency, let_relation=let_relation
                )
            if let_frequency is None or let_frequency < 0:
                raise ValueError("literal symbol frequency must be non-negative")
            if let_relation not in {"less than", "at least"}:
                raise ValueError("literal symbol relation is invalid")
            self._letter = symbol.lower()
            self._frequency = let_frequency
            self._comparison_relation = let_relation
            self._description_pattern = (
                "In your response, the character {letter} should appear "
                "{let_relation} {let_frequency} times."
            )
            return self._description_pattern.format(
                letter=self._letter,
                let_relation=self._comparison_relation,
                let_frequency=self._frequency,
            )

    instructions_registry.INSTRUCTION_DICT["keywords:letter_frequency"] = (
        LiteralSymbolFrequencyChecker
    )
    _SCORER_STATE = (ModelResponse, IFEvalMetrics, ifeval_prompt)
    return _SCORER_STATE


def score_completion(source: dict[str, Any], completion: str) -> dict[str, Any]:
    """Score one completion against one IFEval row, with fresh metric state."""

    model_response, metric_type, prompt_builder = install_ifeval_scorer()
    scorer = metric_type()
    doc = prompt_builder(source, task_name="ifeval")
    return scorer.compute(doc, model_response(text=[completion]))


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate scored rows into the headline metric plus its parts.

    Prompt-level strict accuracy is an unweighted mean of independent 0/1 outcomes,
    so the error is a plain binomial -- no clustering as in OPD, where K samples of
    one question are correlated, and no weight propagation as in BT-RM.

    Instruction-level accuracy is reported beside it because prompt-level strict
    accuracy requires every instruction in a row to pass. The headline remains
    prompt-level strict accuracy.
    """

    if not rows:
        raise ValueError("no rows to summarize")
    keys = [str(row["key"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("a row key appears more than once, so rows were merged twice")

    strict_prompt = [float(row["scores"]["prompt_level_strict_acc"]) for row in rows]
    loose_prompt = [float(row["scores"]["prompt_level_loose_acc"]) for row in rows]
    strict_instruction = [
        float(value) for row in rows for value in row["scores"]["inst_level_strict_acc"]
    ]
    loose_instruction = [
        float(value) for row in rows for value in row["scores"]["inst_level_loose_acc"]
    ]
    total = len(strict_prompt)
    score = sum(strict_prompt) / total
    if not math.isfinite(score):
        raise RuntimeError("score is non-finite")
    generated = [int(row.get("generated_tokens", 0)) for row in rows]
    return {
        "score": score,
        "stderr": math.sqrt(max(score * (1.0 - score), 0.0) / total),
        "stderr_kind": "binomial_descriptive_not_seed_or_paired_uncertainty",
        "metrics": {
            METRIC: score,
            "ifeval_prompt_level_loose_accuracy": sum(loose_prompt) / total,
            "ifeval_instruction_level_strict_accuracy": (
                sum(strict_instruction) / len(strict_instruction)
            ),
            "ifeval_instruction_level_loose_accuracy": (
                sum(loose_instruction) / len(loose_instruction)
            ),
        },
        "n": total,
        "correct": int(sum(strict_prompt)),
        "instructions": len(strict_instruction),
        "step_size": 1.0 / total,
        "mean_generated_tokens": sum(generated) / total,
        # Exact count whose generated-token length reached the fixed cap.
        "length_clipped": sum(1 for value in generated if value >= MAX_NEW_TOKENS),
    }


def synthetic_scores(index: int, instructions: int = 2) -> dict[str, Any]:
    """Deterministic stand-in for a scored row, for --mock and --smoke.

    Keeps every consumer of `summarize` runnable on a host with no GPU, no
    lighteval and no question file.
    """

    strict = float(index % 3 != 0)
    loose = float(index % 5 != 0)
    return {
        "prompt_level_strict_acc": strict,
        "prompt_level_loose_acc": max(strict, loose),
        "inst_level_strict_acc": [strict] * instructions,
        "inst_level_loose_acc": [max(strict, loose)] * instructions,
    }


def synthetic_source(count: int = SOURCE_ROWS) -> list[dict[str, Any]]:
    """A source-shaped row set with plausible integer keys, for --mock and --smoke.

    The keys are not IFEval's, so `require_legacy_ordering` is deliberately not
    applied to these. Everything else about the split is exercised.
    """

    return [
        {
            "key": 1000 + 7 * index,
            "prompt": f"synthetic instruction row {index}",
            "instruction_id_list": ["punctuation:no_comma"],
            "kwargs": [{}],
        }
        for index in range(count)
    ]


def smoke() -> None:
    if (LEGACY_PUBLIC_ROWS, FINAL_ROWS, PROXY_ROWS, HELD_OUT_ROWS) != (128, 413, 128, 285):
        raise RuntimeError(
            f"tier sizes drifted: {LEGACY_PUBLIC_ROWS}/{FINAL_ROWS}/{PROXY_ROWS}/{HELD_OUT_ROWS}"
        )
    if PROXY_ROWS / FINAL_ROWS > 0.5:
        raise RuntimeError("proxy/final overlap is over the spec's 50% ceiling")

    rows = synthetic_source()
    split = split_source(rows)
    proxy_keys = {canonical_key(row) for row in split["proxy"]}
    final_keys = {canonical_key(row) for row in split["final"]}
    held_keys = {canonical_key(row) for row in split["held_out"]}
    legacy_keys = {canonical_key(row) for row in split["legacy_public"]}
    if not proxy_keys <= final_keys:
        raise RuntimeError("the proxy must be a subset of the final")
    if proxy_keys & held_keys:
        raise RuntimeError("proxy and held-out rows overlap")
    if proxy_keys | held_keys != final_keys:
        raise RuntimeError("proxy and held-out rows do not partition the final")
    if legacy_keys & final_keys:
        raise RuntimeError("the retired public128 must be disjoint from the final")
    if len(legacy_keys | final_keys) != SOURCE_ROWS:
        raise RuntimeError("the tiers do not cover the source")
    # Order independence: the split is a function of the keys, not of file order.
    if [canonical_key(r) for r in split_source(list(reversed(rows)))["proxy"]] != [
        canonical_key(r) for r in split["proxy"]
    ]:
        raise RuntimeError("the split depends on source row order")

    shards = shard(split["final"], 3)
    if sum(len(bucket) for bucket in shards) != FINAL_ROWS:
        raise RuntimeError("sharding lost rows")
    if {canonical_key(r) for bucket in shards for r in bucket} != final_keys:
        raise RuntimeError("sharding is not a partition")

    scored = [
        {"key": canonical_key(row), "scores": synthetic_scores(index), "generated_tokens": 64}
        for index, row in enumerate(split["final"])
    ]
    summary = summarize(scored)
    if summary["n"] != FINAL_ROWS or not 0.0 < summary["score"] < 1.0:
        raise RuntimeError(f"unexpected summary: {summary}")
    if summary["stderr"] <= 0.0:
        raise RuntimeError("stderr must be positive on varied data")
    perfect = [
        {"key": row["key"], "scores": synthetic_scores(1), "generated_tokens": 8} for row in scored
    ]
    if abs(summarize(perfect)["score"] - 1.0) > 1e-12:
        raise RuntimeError("an all-correct set must score 1.0")
    if summarize(perfect)["stderr"] != 0.0:
        raise RuntimeError("an all-correct set has zero binomial spread")
    print("grade.py smoke passed")


if __name__ == "__main__":
    smoke()
