# CLIP model soup on ImageNetV2

Construct a better weight-space soup from 72 fixed CLIP ViT-B/32 ingredients. The shipped solution is a reference construction, not a restriction on the candidate's selection or weighting logic.

You have up to four hours for exploration. Do not run work only to consume time, but do not treat a submit-ready candidate as completion. Preserve each trustworthy candidate as a fallback and continue scientifically meaningful exploration while the remaining budget can support experiments whose results can be completed and interpreted.

Before submitting, check the remaining budget and the plausible directions that have not yet been tested. A candidate being better than the current reference, loadable, reproducible, or artifact-valid establishes that it is a fallback; none of those facts alone establishes that exploration is complete. The default action when substantial usable budget remains is to continue exploring, analyzing, or validating.

Early submission is appropriate only when no further meaningful experiment can be completed and interpreted within the remaining budget. Do not submit merely because the current candidate is good enough or has passed its validation checks.

The submitted source patch is applied in a fresh formal container for a construction run of up to 12 hours. It reconstructs each soup from the fixed ingredients, and exploration artifacts are not reused.

Your submission must encode a long-running recipe designed to make meaningful use of the formal training budget. It must not normally terminate early only because of a short fixed step or epoch limit.

Your formal recipe may decide when and how often to save complete and loadable checkpoints. Save each checkpoint under `/out/checkpoints/checkpoint-<progress>/`, where `<progress>` is numeric and increases with training or construction progress.

If more than three valid checkpoints are produced, only the three with the greatest `<progress>` values will be accepted. Every accepted checkpoint will be evaluated independently, and the run's official result is the best valid final score among them. The harness handles final artifact collection and final evaluation.

Producing one complete construction is valid; each checkpoint must contain the complete soup state dict expected by the frozen evaluator.

## Evaluation boundary

The exploration metric is `imagenetv2_top1_proxy2000`; the final metric is `imagenetv2_top1_full10000`. Higher is better for both. The 2,000-image proxy is available during exploration, while the full 10,000-image set is mounted only during scoring.

Exploration and formal construction receive the 72 read-only ingredients, the fixed CLIP payload, and the 2,000 proxy images; the other 8,000 final-only images are absent until scoring. The frozen score phase applies the fixed architecture, preprocessing, and class mapping. Its artifact contract is one compatible state dict in the affine hull of the ingredients. This constrains the submitted artifact, not the search or optimization used to choose it. Coefficients may be negative or extrapolative; their sign and distance from the ingredient mean are diagnostics, not additional validity gates. Formal scoring runs outside the submitted workspace. Do not import external images, labels, weights, or ingredients, reconstruct the final-only inputs, or implement an evaluation-specific lookup.

The build is deterministic for fixed coefficients and inputs. A repeated score of the same state dict is therefore a reproducibility check, not a training-seed sweep. Proxy improvement alone does not establish a final improvement, especially when the coefficient search uses the proxy repeatedly.

## Shipped solution reference

The training-start reference is the best single ingredient, `model_69`; it is not the original CLIP base. The shipped solution is the uniform average of all 72 ingredients.

| Measurement | Result |
|---|---:|
| Best-single start, full ImageNetV2 top-1 | `0.6874` |
| Shipped uniform soup, full ImageNetV2 top-1 | `0.6859` |
| Difference from best single | `-0.0015` |
| Uniform soup construction time | `248.13 s` |
| Full scoring time | `1198.02 s` |
| Peak GPU memory | `2,222 MiB` |

The `-0.0015` difference is only about `0.56` paired standard errors and should be described as a statistical tie, not a demonstrated regression.

## Work surface

Read `/workspace/run.sh`, the ingredient loader, coefficient construction, and the read-only artifact checker at `/opt/harness/soup_check.py`. Everything under `/workspace` is editable for any search or optimization strategy that produces a valid artifact. These examples are illustrative, not exhaustive; they do not restrict any other change within the fixed task boundaries.

The gate evaluates the final state dict, not how its coefficients were found. A proxy gain from an incompatible state dict, non-finite coefficient vector, or state outside the permitted hull cannot become a result. Construction speed and memory matter only insofar as they enable stronger valid searches within the budget.

## Running experiments

Use a separate output directory for each candidate:

```bash
OUTPUT_DIR=/out/probe-name bash /workspace/run.sh
/opt/harness/fast_eval.sh /out/probe-name
/opt/harness/timer.sh
```

Record ingredient identities, coefficients, selection trajectory, proxy queries, construction time, peak memory, state-dict hash, artifact-check payload, evaluator payload, and failed constraints. Repeated proxy use increases selection uncertainty but is not itself an artifact failure. Stop a failed candidate on non-finite coefficients, incompatible tensors, or artifact-gate failure. Stopping one candidate does not by itself end exploration.

## Formal replay

Formal construction applies `candidate.patch` to a fresh `/workspace` and invokes exactly:

```bash
bash /workspace/run.sh
```

It does not reuse exploration soups, caches, output directories, or shell exports. The command must reconstruct every saved soup checkpoint from the fixed ingredients, and each must be loadable.

## Submission

A smoke or startup check proves only that the code can begin; it is not performance evidence.

Before ending exploration, wait for every training, evaluation, and background command and read its result, or stop it explicitly and record why. Preserve the best trustworthy candidate as a fallback while exploring other directions.

Before the final action, verify the coefficient rule and artifact gate on the final source.

Before submitting, verify that the patch encodes the long formal recipe and checkpoint-saving policy described above.

When no further meaningful experiment can be completed and interpreted within the remaining budget, verify the final source and artifacts, then run `/opt/harness/submit.sh` as the final action. If no candidate is trustworthy, use `/opt/harness/no_candidate.sh "reason"`. Deadline capture is recovery only and is not a normal submission path.
