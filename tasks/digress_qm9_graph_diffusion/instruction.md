# DiGress graph diffusion on QM9

Improve molecular generation on the fixed QM9-without-hydrogens data. The shipped solution uses discrete graph diffusion; that implementation is the starting point rather than a prose-level algorithm restriction.

You have up to four hours for exploration. Do not run work only to consume time, but do not treat a submit-ready candidate as completion. Preserve each trustworthy candidate as a fallback and continue scientifically meaningful exploration while the remaining budget can support experiments whose results can be completed and interpreted.

Before submitting, check the remaining budget and the plausible directions that have not yet been tested. A candidate being better than the current reference, loadable, reproducible, or artifact-valid establishes that it is a fallback; none of those facts alone establishes that exploration is complete. The default action when substantial usable budget remains is to continue exploring, analyzing, or validating.

Early submission is appropriate only when no further meaningful experiment can be completed and interpreted within the remaining budget. Do not submit merely because the current candidate is good enough or has passed its validation checks.

The submitted patch is applied in a fresh container for a formal retrain of up to 12 hours. Formal retraining starts from fresh Hydra state and fixed data, not from an exploration checkpoint.

Your submission must encode a long-running recipe designed to make meaningful use of the formal training budget. It must not normally terminate early only because of a short fixed step or epoch limit.

Your formal recipe may decide when and how often to save complete and loadable checkpoints. Save each checkpoint under `/out/checkpoints/checkpoint-<progress>/`, where `<progress>` is numeric and increases with training or construction progress.

If more than three valid checkpoints are produced, only the three with the greatest `<progress>` values will be accepted. Every accepted checkpoint will be evaluated independently, and the run's official result is the best valid final score among them. The harness handles final artifact collection and final evaluation.

Each checkpoint directory must contain a complete `model.ckpt` loadable by the frozen DiGress evaluator.

## Evaluation and data boundary

Exploration and retraining receive a derived train/validation-only asset. The pinned DiGress datamodule requires a test-named tensor during setup, so this asset contains a byte-identical alias of validation at the expected test filename; it does not contain the real test tensor or raw test data. Only the score phase mounts the complete frozen asset containing the real test split.

The headline fast metric is `validity_uniqueness_novelty`, the product of those three rates, and higher is better. Fast evaluation also reports validation NLL, where lower is better. The final metric is the upstream test `nll`, also lower-is-better. The fast headline and final metric are therefore not the same quantity; use validation NLL and molecule diagnostics to interpret a proxy result rather than treating the composite as a direct estimate of final NLL.

Sampling in the shipped implementation is not deterministic. Do not assume that setting a training seed makes two generated molecule sets identical; use the observed variability when deciding whether another comparison is informative. Formal scoring uses the frozen evaluator outside the submitted workspace. Do not import external molecule data or weights, reconstruct the real test tensor, access evaluator-only files during training, implement an evaluation-specific lookup, or change metric direction.

## Shipped solution reference

The current shipped solution has the following B300 results under the declared upstream test-NLL protocol:

| Measurement | Result |
|---|---:|
| One-epoch training start, final test NLL | `78.05` |
| Current shipped solution, final test NLL (lower is better) | `69.57` |
| Change from training start | `-8.48` (`10.86%` lower) |
| Formal training time | `3305.81 s` |
| Final scoring time | `2108.34 s` |
| Sampled molecules in final scoring | `10,000` |
| Validity / uniqueness / upstream novelty | `0.9805 / 0.9769 / 0.5190` |

## Work surface

Read `/workspace/run.sh`, `/workspace/train.py`, the shipped configuration, datamodule, objective, and sampler. Everything under `/workspace` is editable, including row use and augmentation, architecture, transition or generation model, objective, batching, optimizer, schedule, checkpointing, and sampling implementation. These examples are illustrative, not exhaustive; they do not restrict any other change within the fixed task boundaries.

The available train/validation asset defines the training-data universe; a candidate may select, reweight, resample, augment, or otherwise use those rows. It must not infer or fabricate the real test split from the compatibility alias. The formal artifact only needs to be loadable by the frozen scoring interface and produce the declared upstream test NLL; candidate-authored algorithm labels or tensor-name heuristics should not determine validity.

## Running experiments

Use a new output directory for every probe:

```bash
OUTPUT_DIR=/out/probe-name bash /workspace/run.sh
/opt/harness/fast_eval.sh /out/probe-name/checkpoint/last.ckpt
/opt/harness/timer.sh
```

Preserve optimizer and epoch counts, train and validation NLL, molecule diagnostics, examples per second, peak memory, resolved Hydra configuration, checkpoint hash, evaluator payload, and failures. Stop a failed candidate on non-finite NLL, OOM, fixed-data drift, evaluator-contract failure, or a missing or unloadable checkpoint. Stopping one candidate does not by itself end exploration.

## Formal replay

Formal replay applies only `candidate.patch`, starts with fresh Hydra and output state, and invokes exactly:

```bash
bash /workspace/run.sh
```

Exploration checkpoints, generated molecules, caches, and environment-only overrides do not cross phases. The real test asset remains absent until the separate score phase.

## Submission

A smoke or startup check proves only that the code can begin; it is not performance evidence.

Before ending exploration, wait for every training, evaluation, and background command and read its result, or stop it explicitly and record why. Preserve the best trustworthy candidate as a fallback while exploring other directions.

Before the final action, verify that the source is runnable and that the patch contains no generated data, checkpoint, cache, or evaluator artifact.

Before submitting, verify that the patch encodes the long formal recipe and checkpoint-saving policy described above.

When no further meaningful experiment can be completed and interpreted within the remaining budget, verify the final source and artifacts, then run `/opt/harness/submit.sh` as the final action. If no candidate is trustworthy, use `/opt/harness/no_candidate.sh "reason"`. Deadline capture is recovery only and is not a normal submission path.
