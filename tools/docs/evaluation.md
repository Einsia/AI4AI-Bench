# Evaluation and replay

AI4AI-Bench separates exploratory feedback from the frozen evaluation used to report a
run. A proxy evaluation is visible to the coding agent during Explore and is intended
for iteration. Final evaluation runs only after a fresh Formal retrain (or on explicitly
supplied checkpoints), uses the task's frozen evaluator, and is the score reported by
the lifecycle. This release supports self-hosted evaluation; an official blind service
is not yet available.

## Replay an existing patch

Use `--candidate-patch` to skip the four-hour Agent Explore and replay source changes
from the fixed task starting point:

```bash
bash orchestrator/trial.sh replay-name \
  --task tasks/ddpo_sd15_aesthetic \
  --assets /data/ai4ai/assets/ddpo_sd15_aesthetic \
  --root /data/ai4ai/runs --gpu 0 \
  --candidate-patch candidate.patch
```

The patch must be a non-empty regular file no larger than 64 MiB. It is streamed into
the run-owned copy, hashed, marked
with `external_patch` provenance, and passed through the same patch safety check,
fresh Formal retrain, checkpoint validation, and final evaluation as an Agent-submitted
candidate. No Explore completion or Agent identity is fabricated. Resuming the same
run name with different patch bytes is rejected. After a successful import, that
run-owned copy can be used to resume even if the original temporary path is gone.

Important receipts below `<root>/<name>/` are:

- `run-config.json`: immutable task, asset, image, budget, and patch identity;
- `explore/out/external_patch.json`: imported path, byte count, and SHA-256;
- `manifest.json`: lifecycle status, `execution_mode`, and `candidate_provenance`;
- `retrain/artifacts.json`: checkpoint discovery, validation, and selection;
- `score/artifact-*/out/summary.json`: one frozen evaluation per artifact;
- `score/out/summary.json`: best valid result among up to three artifacts.

## Evaluate existing checkpoints

Checkpoint-only evaluation accepts one to three unique, non-negative progress labels:

```bash
bash orchestrator/evaluate.sh eval-name \
  --task tasks/ddpo_sd15_aesthetic \
  --assets /data/ai4ai/assets/ddpo_sd15_aesthetic \
  --root /data/ai4ai/evaluations --gpu 0 \
  --checkpoint 1000=/path/to/checkpoint-1000 \
  --checkpoint 2000=/path/to/checkpoint-2000
```

Each checkpoint is independently hashed, checked by the frozen task-specific loader,
and sent to final evaluation only if validation succeeds. The external path is hashed
again immediately before and after validation and scoring; changing its contents during
or after initialization invalidates the evaluation rather than silently changing its
provenance. Among valid final results, the task's declared `max` or `min` direction and
the existing `best_valid_of_up_to_3` rule select the aggregate result.

The public defaults (`source-check=warn`, `image-check=warn`, and
`hardware-check=warn`) are for local compatibility and produce **non-official local**
results. An official self-hosted result requires all three checks to be `strict`, the
real `score` phase, and the declared score timeout with no override; `--score-mock` and
any explicit `--score-timeout` are non-official. A successful strict run has also
passed the locked asset checks. The CLI prints this classification and the path to
`evaluation-config.json`; the config and aggregate summary both retain the same
machine-readable `result_classification` and `verification` fields. Phase-specific observed
source, image, asset, and hardware evidence is in each `preflight.json`.

Receipts below `<root>/<name>/` are:

- `evaluation-config.json`: immutable evaluation and checkpoint identities;
- `artifacts.json`: external provenance, content hashes, validation receipts, and the
  accepted set;
- `validation/checkpoint-*/out/validation.json`: loader result for each checkpoint;
- `score/artifact-*/out/summary.json`: independent final result;
- `score/out/summary.json`: aggregate result and selected artifact.

Task evaluators may add diagnostics, but terminal score receipts normally expose
`status`, `metric`, `score`, sample count `n`, and uncertainty such as `stderr` or a
confidence interval when available. The aggregate additionally records
`selection_rule`, `selected_artifact`, `selected_progress`, and per-artifact results.
Treat the JSON receipts—not console output—as the machine-readable record.

## Interpreting failures

A missing result is not a score of zero. A terminal JSON receipt written immediately
before a crash is safely rebound and its missing completion stamp is restored. If a
validator or scorer leaves only partial, non-terminal output, the orchestrator writes
`interrupted.json`, preserves the evidence, and refuses in-place reuse; resume with a
new evaluation/run name. An `invalid` validation or final summary is a terminal
artifact/metric failure and is retained with its reason. Aggregation can therefore
report `status: invalid` and `score: null` when no supplied checkpoint produces a valid
final score. Do not coerce missing, invalid, rejected, or infrastructure-failed attempts
into the task metric.
