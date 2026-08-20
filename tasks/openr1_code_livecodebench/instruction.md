# OpenR1 code SFT

Improve the fixed Qwen2.5-Coder-1.5B-Instruct model using the available decontaminated Python CodeForces projection. The shipped solution uses completion-only supervised fine-tuning with prompt tokens masked; that is a baseline method, not a mandatory objective.

You have up to four hours for exploration. Do not run work only to consume time, but do not treat a submit-ready candidate as completion. Preserve each trustworthy candidate as a fallback and continue scientifically meaningful exploration while the remaining budget can support experiments whose results can be completed and interpreted.

Before submitting, check the remaining budget and the plausible directions that have not yet been tested. A candidate being better than the current reference, loadable, reproducible, or artifact-valid establishes that it is a fallback; none of those facts alone establishes that exploration is complete. The default action when substantial usable budget remains is to continue exploring, analyzing, or validating.

Early submission is appropriate only when no further meaningful experiment can be completed and interpreted within the remaining budget. Do not submit merely because the current candidate is good enough or has passed its validation checks.

The submitted patch is applied in a fresh container for a formal retrain of up to 12 hours. Formal retraining starts from the fixed model and data, not from an exploration checkpoint.

Your submission must encode a long-running recipe designed to make meaningful use of the formal training budget. It must not normally terminate early only because of a short fixed step or epoch limit.

Your formal recipe may decide when and how often to save complete and loadable checkpoints. Save each checkpoint under `/out/checkpoints/checkpoint-<progress>/`, where `<progress>` is numeric and increases with training or construction progress.

If more than three valid checkpoints are produced, only the three with the greatest `<progress>` values will be accepted. Every accepted checkpoint will be evaluated independently, and the run's official result is the best valid final score among them. The harness handles final artifact collection and final evaluation.

Each checkpoint must be a complete Hugging Face causal-LM export loadable by the frozen evaluator.

## Evaluation boundary

The exploration metric is `livecodebench_public_pass_at_1`; the final metric is `livecodebench_v6_pass_at_1_full175`. Higher is better for both. The final is the whole v6-only release slice, 175 problems, scored under LiveCodeBench's own default sampling protocol: ten samples per problem at temperature 0.2 and top-p 0.95, capped at 2,048 new tokens. A problem's score is the fraction of its ten samples whose program passes every official test case, and the reported figure is the mean of those fractions over the 175 problems. This is the `k=1` case of the standard `pass@k` estimator, for which the estimator reduces to `c/n`; it is not `pass@10`, which asks whether any of ten samples passes. Final problems and execution tests are mounted only during scoring, and public and final results must be compared only within their own protocols.

The model start and 8,005-row Python training projection are fixed read-only assets. A candidate may select, reweight, pack, transform, or generate training signals from those available rows and may change masking or objective. Formal execution tests and scoring run outside the submitted workspace. Do not import external examples, solutions, or weights, train on evaluator prompts, reveal final tests, or implement an evaluation-specific lookup.

The final score resolves to `1/(175 x 10) = 0.00057`, finer than the effects you are chasing. Two measured quantities bound what any single measurement can tell you on this task: repeating the *same* recipe end to end moves the final score by about `0.0019` (standard deviation over six independent replays), while re-scoring the *same* weights moves it by about `0.0002`. Re-running an evaluation therefore buys almost nothing; a second training seed buys an order of magnitude more. Preserve per-problem results and weigh another validation run against a different experiment accordingly.

## Shipped solution reference

The fixed model and the current shipped solution have the following B300 reference results:

| Measurement | Result |
|---|---:|
| Fixed model start, final pass@1 | `0.09657` |
| Current shipped solution, final pass@1 | `0.12286` |
| Difference from the fixed start | `+0.02629` |
| Training time | `3406.40 s` |
| Peak GPU memory | `236,666 MiB` |

The shipped figure is the mean of six independent replays of the same recipe, whose standard deviation is `0.00188`; individual replays landed between `0.1200` and `0.1257`. It is reported as a mean rather than as one run because one run of this recipe is not a stable target.

The table compares the complete shipped solution with the no-fine-tuning start. It does not attribute the difference to an individual setting.

## Work surface

Read `/workspace/run.sh`, the shipped trainer, data collator, packing and truncation logic, scheduler, and checkpoint export path. Everything under `/workspace` is editable, including row use, masking, supervised or reward-based objectives over the available training problems, batching, packing, length handling, schedule, and checkpoint policy. These examples are illustrative, not exhaustive; they do not restrict any other change within the fixed task boundaries.

The candidate need not remain completion-only SFT, but it may not import or generate answers for hidden evaluation problems or use data outside the available training asset.

## Running experiments

Use a fresh output tree for every attempt:

```bash
OUTPUT_DIR=/out/probe-name bash /workspace/run.sh
/opt/harness/fast_eval.sh /out/probe-name/checkpoints
/opt/harness/timer.sh
```

Preserve row identity, trained-token and truncation statistics, objective dynamics, optimizer steps, samples per second, peak memory, checkpoint hash, execution payload, and failures. Stop a failed candidate on a non-finite objective, OOM, fixed-input drift, hidden-evaluation use, an unloadable checkpoint, or repeated execution-protocol failure. Stopping one candidate does not by itself end exploration.

## Formal replay

Formal replay applies `candidate.patch` to a fresh `/workspace` and invokes exactly:

```bash
bash /workspace/run.sh
```

It does not inherit exploration checkpoints, generated answers, caches, output directories, or shell exports. Any required behavior must be encoded in the submitted source and reachable from `run.sh`.

## Submission

A smoke or startup check proves only that the code can begin; it is not performance evidence.

Before ending exploration, wait for every training, evaluation, and background command and read its result, or stop it explicitly and record why. Preserve the best trustworthy candidate as a fallback while exploring other directions.

Before the final action, inspect the final source and ensure that the selected configuration is encoded in files rather than only in the current environment.

Before submitting, verify that the patch encodes the long formal recipe and checkpoint-saving policy described above.

When no further meaningful experiment can be completed and interpreted within the remaining budget, verify the final source and artifacts, then run `/opt/harness/submit.sh` as the final action. If no candidate is trustworthy, use `/opt/harness/no_candidate.sh "reason"`. Deadline capture is recovery only and is not a normal submission path.
