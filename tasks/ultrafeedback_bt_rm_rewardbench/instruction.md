# Bradley-Terry reward modeling on UltraFeedback

Improve a scalar reward model from the fixed Mistral training start and available decontaminated UltraFeedback preference pairs. The shipped solution uses a Bradley-Terry objective; that is a baseline method rather than a required loss. The formal artifact may be a complete model or a compatible parameter-efficient delta and must expose one scalar reward per sequence.

You have up to four hours for exploration. Do not run work only to consume time, but do not treat a submit-ready candidate as completion. Preserve each trustworthy candidate as a fallback and continue scientifically meaningful exploration while the remaining budget can support experiments whose results can be completed and interpreted.

Before submitting, check the remaining budget and the plausible directions that have not yet been tested. A candidate being better than the current reference, loadable, reproducible, or artifact-valid establishes that it is a fallback; none of those facts alone establishes that exploration is complete. The default action when substantial usable budget remains is to continue exploring, analyzing, or validating.

Early submission is appropriate only when no further meaningful experiment can be completed and interpreted within the remaining budget. Do not submit merely because the current candidate is good enough or has passed its validation checks.

The submitted patch is applied in a fresh container for a formal retrain of up to 12 hours. Formal retraining starts from the fixed base and pair file, not from an exploration checkpoint.

Your submission must encode a long-running recipe designed to make meaningful use of the formal training budget. It must not normally terminate early only because of a short fixed step or epoch limit.

Your formal recipe may decide when and how often to save complete and loadable checkpoints. Save each checkpoint under `/out/checkpoints/checkpoint-<progress>/`, where `<progress>` is numeric and increases with training or construction progress.

If more than three valid checkpoints are produced, only the three with the greatest `<progress>` values will be accepted. Every accepted checkpoint will be evaluated independently, and the run's official result is the best valid final score among them. The harness handles final artifact collection and final evaluation.

Each checkpoint must include the complete loadable reward-model or adapter payload, including the scalar reward head.

## Evaluation boundary

The exploration metric is `rewardbench_proxy_512`; the final metric is `rewardbench_v1_score` with fixed section aggregation. Higher is better for both. The visible 512-pair proxy is a subset of the 2,985-pair final, and the other 2,473 final pairs are held out until scoring. Compare proxy results only with proxy results and final results only with the full final protocol.

The artifact must be loadable as a scalar reward model on the fixed Mistral architecture. Any overlap between training rows and RewardBench invalidates the run. Candidates may select, reweight, schedule, or transform available preference rows and change the reward-model objective. Formal scoring uses the frozen RewardBench harness outside the submitted workspace. Do not import external preference/evaluation rows or reward checkpoints, change RewardBench section weights, or implement an evaluation-specific lookup.

RewardBench is sensitive to training seed, especially in its reasoning subsets. Preserve section-level scores and treat one high draw as limited evidence; decide whether another seed or a different experiment has greater expected value.

## Shipped solution reference

The current shipped solution has the following B300 reference result. RewardBench is seed-sensitive, so treat it as a single-run reference rather than a replicated effect.

The fixed Mistral base has no trained scalar reward head, so it does not define a meaningful RewardBench training-start score or a start-to-solution delta.

| Measurement | Result |
|---|---:|
| Current shipped solution, final RewardBench score | `74.568936` |
| Training time | `3353.29 s` |
| Final scoring time | `208.24 s` |
| Peak training GPU memory | `26,592 MiB` |

Preserve RewardBench section scores so an aggregate improvement cannot hide a section collapse.

## Work surface

Read `/workspace/run.sh`, the scalar reward head, data collation, shipped Bradley-Terry loss, trainable-parameter configuration, optimizer, scheduler, export path, overlap checker, and artifact loader. Everything under `/workspace` is editable, including row use, pairwise or alternative reward objectives, centering, margins, sampling, optimization, schedule, and full-parameter or parameter-efficient capacity. These examples are illustrative, not exhaustive; they do not restrict any other change within the fixed task boundaries.

The candidate need not preserve the Bradley-Terry loss, but the exported model must emit one scalar per sequence so the frozen RewardBench comparison remains defined. An easier training objective is not enough evidence; inspect section-level evaluation and artifact validity.

## Running experiments

Use a fresh output directory for every attempt:

```bash
OUTPUT_DIR=/out/probe-name bash /workspace/run.sh
/opt/harness/fast_eval.sh /out/probe-name/checkpoint
/opt/harness/timer.sh
```

Preserve pair order, truncation statistics, loss and margin dynamics, throughput, peak memory, checkpoint and scalar-head hashes, overlap-check output, artifact-check payload, evaluator payload, and failures. Stop a failed candidate on fixed-input or overlap failure, non-finite loss, OOM, a missing scalar head, or an unloadable checkpoint. Stopping one candidate does not by itself end exploration.

## Formal replay

Formal replay applies `candidate.patch` to a fresh `/workspace` and invokes exactly:

```bash
bash /workspace/run.sh
```

It does not inherit exploration checkpoints, caches, output directories, or shell exports. Each accepted formal artifact must be either a complete model or a compatible delta on the fixed backbone. The candidate patch must not contain any model weights or generated evaluation data.

## Submission

A smoke or startup check proves only that the code can begin; it is not performance evidence.

Before ending exploration, wait for every training, evaluation, and background command and read its result, or stop it explicitly and record why. Preserve the best trustworthy candidate as a fallback while exploring other directions.

Before the final action, verify the artifact and overlap checks on the final source.

Before submitting, verify that the patch encodes the long formal recipe and checkpoint-saving policy described above.

When no further meaningful experiment can be completed and interpreted within the remaining budget, verify the final source and artifacts, then run `/opt/harness/submit.sh` as the final action. If no candidate is trustworthy, use `/opt/harness/no_candidate.sh "reason"`. Deadline capture is recovery only and is not a normal submission path.
