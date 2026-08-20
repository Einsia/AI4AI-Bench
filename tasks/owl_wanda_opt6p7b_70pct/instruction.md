# OWL/Wanda pruning at 70 percent sparsity

Produce a better unstructured sparse OPT-6.7B artifact from the fixed dense model and available calibration asset. The shipped solution uses activation-aware OWL/Wanda pruning without fine-tuning; that is a baseline method, not a restriction on candidate training.

You have up to four hours for exploration. Do not run work only to consume time, but do not treat a submit-ready candidate as completion. Preserve each trustworthy candidate as a fallback and continue scientifically meaningful exploration while the remaining budget can support experiments whose results can be completed and interpreted.

Before submitting, check the remaining budget and the plausible directions that have not yet been tested. A candidate being better than the current reference, loadable, reproducible, or artifact-valid establishes that it is a fallback; none of those facts alone establishes that exploration is complete. The default action when substantial usable budget remains is to continue exploring, analyzing, or validating.

Early submission is appropriate only when no further meaningful experiment can be completed and interpreted within the remaining budget. Do not submit merely because the current candidate is good enough or has passed its validation checks.

The submitted patch is applied in a fresh formal container for a construction run of up to 12 hours. It reconstructs each sparse artifact from the fixed dense model and calibration data, and exploration weights are not reused.

Your submission must encode a long-running recipe designed to make meaningful use of the formal training budget. It must not normally terminate early only because of a short fixed step or epoch limit.

Your formal recipe may decide when and how often to save complete and loadable checkpoints. Save each checkpoint under `/out/checkpoints/checkpoint-<progress>/`, where `<progress>` is numeric and increases with training or construction progress.

If more than three valid checkpoints are produced, only the three with the greatest `<progress>` values will be accepted. Every accepted checkpoint will be evaluated independently, and the run's official result is the best valid final score among them. The harness handles final artifact collection and final evaluation.

Producing one complete pruning construction is valid; its checkpoint must be a complete model loadable by the frozen evaluator.

## Evaluation boundary

The exploration metric is `wikitext2_validation_perplexity`; the final metric is `wikitext2_test_perplexity`. Lower is better for both. Validation text is available during exploration; test text is mounted only during scoring.

The hard artifact gate requires decoder sparsity in `[0.699, 0.701]` and a loadable model. The fixed dense model is the pre-pruning start and a useful quality reference, but its zero sparsity makes it ineligible as a final task artifact. Dense and sparse perplexities may be compared to quantify sparsification cost, not as two gate-valid submissions.

The dense training start and available C4 calibration shard are fixed. Candidates may change pruning, search, calibration use, objectives, and may train using only the available asset. Formal test scoring uses the frozen evaluator outside the submitted workspace. Do not import external data or weights, read test text during exploration or construction, exceed the declared sparsity window, or implement an evaluation-specific lookup.

## Shipped solution reference

The fixed dense reference and shipped sparse solution have the following test results:

| Artifact | Sparsity gate | Final test perplexity |
|---|---|---:|
| Fixed dense reference | Fails: zero sparsity | `10.860456` |
| Shipped sparse solution, seed `0` | Passes | `53.997456` |
| Shipped sparse solution, seed `1` | Passes | `52.617988` |
| Shipped sparse solution, seed `2` | Passes | `53.461515` |
| Shipped sparse solution, mean | Passes | `53.358987` |

The sparse mean is `+42.498531` perplexity above the dense reference. This is the observed quality cost of producing a 70%-sparse artifact, not an improvement delta between two gate-valid task artifacts.

| Resource measurement | Result |
|---|---:|
| Prune and export time | `203.67-208.50 s` |
| Final scoring time | `101.80-104.52 s` |
| Peak GPU memory | `23,340 MiB` |

The spread across calibration seeds is material relative to small perplexity changes. Use that observed spread when deciding whether another calibration seed, deeper analysis, or a different candidate is the best use of the remaining budget.

## Work surface

Read `/workspace/run.sh`, activation collection, scoring, layer allocation, calibration sampling, mask construction, optional training, artifact export, and sparsity checker. Everything under `/workspace` is editable; OWL/Wanda is the shipped implementation rather than an enforced family. These examples are illustrative, not exhaustive; they do not restrict any other change within the fixed task boundaries.

A candidate must preserve the declared global sparsity window and export a loadable model. Layer-wise allocation and training method may change, but a lower perplexity obtained by retaining extra weights or using test text is not comparable.

## Running experiments

Use a new output directory for every attempt:

```bash
OUTPUT_DIR=/out/probe-name bash /workspace/run.sh
/opt/harness/fast_eval.sh /out/probe-name/pruned
/opt/harness/timer.sh
```

Preserve calibration identity, layer sparsities, mask statistics, construction time, peak memory, artifact hash, sparsity-check payload, evaluator payload, and failures. Stop a failed candidate on load failure, sparsity failure, non-finite scores, external-weight use, or clear matched-evaluation regression. Stopping one candidate does not by itself end exploration.

## Formal replay

Formal construction applies `candidate.patch` to a fresh `/workspace` and invokes exactly:

```bash
bash /workspace/run.sh
```

It does not reuse exploration weights, masks, caches, output directories, or shell exports. The command must reconstruct every saved sparse checkpoint from the fixed dense model and calibration shard; each accepted checkpoint is gate-checked and scored independently.

## Submission

A smoke or startup check proves only that the code can begin; it is not performance evidence.

Before ending exploration, wait for every training, evaluation, and background command and read its result, or stop it explicitly and record why. Preserve the best trustworthy candidate as a fallback while exploring other directions.

Before the final action, verify the final sparsity report and artifact loadability.

Before submitting, verify that the patch encodes the long formal recipe and checkpoint-saving policy described above.

When no further meaningful experiment can be completed and interpreted within the remaining budget, verify the final source and artifacts, then run `/opt/harness/submit.sh` as the final action. If no candidate is trustworthy, use `/opt/harness/no_candidate.sh "reason"`. Deadline capture is recovery only and is not a normal submission path.
