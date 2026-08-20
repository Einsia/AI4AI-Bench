# DPO preference alignment

Improve preference alignment of the fixed Zephyr/Mistral policy using the available UltraFeedback preference asset. The shipped solution uses pairwise DPO with the policy start as its frozen reference; that is a baseline method, not a required objective.

You have up to four hours for exploration. Do not run work only to consume time, but do not treat a submit-ready candidate as completion. Preserve each trustworthy candidate as a fallback and continue scientifically meaningful exploration while the remaining budget can support experiments whose results can be completed and interpreted.

Before submitting, check the remaining budget and the plausible directions that have not yet been tested. A candidate being better than the current reference, loadable, reproducible, or artifact-valid establishes that it is a fallback; none of those facts alone establishes that exploration is complete. The default action when substantial usable budget remains is to continue exploring, analyzing, or validating.

Early submission is appropriate only when no further meaningful experiment can be completed and interpreted within the remaining budget. Do not submit merely because the current candidate is good enough or has passed its validation checks.

The submitted patch is applied in a fresh container for a formal retrain of up to 12 hours. Formal retraining starts from the fixed policy, not from an exploration checkpoint.

Your submission must encode a long-running recipe designed to make meaningful use of the formal training budget. It must not normally terminate early only because of a short fixed step or epoch limit.

Your formal recipe may decide when and how often to save complete and loadable checkpoints. Save each checkpoint under `/out/checkpoints/checkpoint-<progress>/`, where `<progress>` is numeric and increases with training or construction progress.

If more than three valid checkpoints are produced, only the three with the greatest `<progress>` values will be accepted. Every accepted checkpoint will be evaluated independently, and the run's official result is the best valid final score among them. The harness handles final artifact collection and final evaluation.

Each checkpoint must be a complete PEFT adapter or merged Hugging Face causal-LM model loadable with the fixed policy base.

## Evaluation boundary

The exploration metric is `ifeval_strict_accuracy_public128`; the final metric is `ifeval_strict_accuracy_hidden413`. Higher is better for both. The 128-row public proxy is a visible subset of the 413-row final, and the other 285 final rows are held out until scoring. Compare proxy and final results only within their own protocols.

The policy start and available UltraFeedback asset are read-only; formal replay must start from that policy. A candidate may choose any subset, weighting, curriculum, reference-policy treatment, preference objective, and trainable-parameter strategy. The complete IFEval source and the 285 held-out final rows are mounted only during scoring. Formal scoring uses the frozen harness outside the submitted workspace. Do not import external rows or weights, train on held-out prompts, or implement an evaluation-specific lookup.

Preference loss and reward margin are training diagnostics, not selection metrics. Inspect truncation and stability, then require same-protocol evaluator evidence.

## Shipped solution reference

The fixed policy and the current shipped solution have the following B300 reference results under the declared final protocol.

| Measurement | Result |
|---|---:|
| Fixed policy start, final strict accuracy | `164/413 = 0.397094` |
| Current shipped solution, final strict accuracy | `210/413 = 0.508475` |
| Difference from the fixed start | `+46/413 = +0.111380` |
| Training time | `3352.75 s` |
| Final scoring time | `2090.04 s` |
| Peak training GPU memory | `19,720 MiB` |

The start and solution use the same final metric. The difference is a single-run reference, not a replicated estimate, and does not identify which individual setting caused it. `/workspace/run.sh` and the implementation are the authority for the shipped method.

## Work surface

Read `/workspace/run.sh`, the shipped trainer, data collation, reference handling, trainable-parameter configuration, and export path. Everything under `/workspace` is editable, including row selection, preference or auxiliary objectives, reference-free methods, trainable capacity, batching, truncation, optimizer, schedule, attention implementation, and checkpoint policy. These examples are illustrative, not exhaustive; they do not restrict any other change within the fixed task boundaries.

The shipped solution uses the fixed policy start with its adapter disabled as the reference. A candidate may use that reference differently or choose an objective that requires no reference model, but no additional model weights are available or permitted. This does not change initialization: fresh formal replay starts from the fixed policy and available training asset. Only source changes cross phases. That replay may export either a newly merged model or an adapter compatible with the fixed policy start. Resource utilization alone is not an objective; judge changes by stability and same-protocol evaluator results.

## Running experiments

Use a fresh output directory for every attempt:

```bash
OUTPUT_DIR=/out/probe-name bash /workspace/run.sh
/opt/harness/fast_eval.sh /out/probe-name/checkpoint
/opt/harness/timer.sh
```

Preserve row identity, effective batch, token and truncation statistics, loss and margin dynamics, throughput, peak memory, checkpoint hash, evaluator payload, and failures. Stop a failed candidate on fixed-input drift, non-finite loss, OOM, incomplete export, an unloadable checkpoint, or clear matched-evaluation regression. Stopping one candidate does not by itself end exploration.

## Formal replay

Formal replay applies `candidate.patch` to a fresh `/workspace` and invokes exactly:

```bash
bash /workspace/run.sh
```

Exploration checkpoints, adapters, merged weights, caches, output directories, and shell exports do not cross phases, and `candidate.patch` must not contain model weights. Any required change must be present in the submitted source and reachable from `run.sh`. Scoring consumes only the accepted formal artifacts produced by this fresh replay.

## Submission

A smoke or startup check proves only that the code can begin; it is not performance evidence.

Before ending exploration, wait for every training, evaluation, and background command and read its result, or stop it explicitly and record why. Preserve the best trustworthy candidate as a fallback while exploring other directions.

Before the final action, inspect the final patch and verify that the selected behavior is encoded in source rather than only in the current environment.

Before submitting, verify that the patch encodes the long formal recipe and checkpoint-saving policy described above.

When no further meaningful experiment can be completed and interpreted within the remaining budget, verify the final source and artifacts, then run `/opt/harness/submit.sh` as the final action. If no candidate is trustworthy, use `/opt/harness/no_candidate.sh "reason"`. Deadline capture is recovery only and is not a normal submission path.
