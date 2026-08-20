# Official Llama NPO on TOFU

Improve unlearning for Llama-3.2-1B-Instruct on the fixed TOFU forget10 protocol. The shipped solution uses the official OpenUnlearning NPO recipe; that is the reference method, not a mandatory loss family.

You have up to four hours for exploration. Do not run work only to consume time, but do not treat a submit-ready candidate as completion. Preserve each trustworthy candidate as a fallback and continue scientifically meaningful exploration while the remaining budget can support experiments whose results can be completed and interpreted.

Before submitting, check the remaining budget and the plausible directions that have not yet been tested. A candidate being better than the current reference, loadable, reproducible, or artifact-valid establishes that it is a fallback; none of those facts alone establishes that exploration is complete. The default action when substantial usable budget remains is to continue exploring, analyzing, or validating.

Early submission is appropriate only when no further meaningful experiment can be completed and interpreted within the remaining budget. Do not submit merely because the current candidate is good enough or has passed its validation checks.

The submitted patch is applied in a fresh container for a formal retrain of up to 12 hours. Formal retraining starts from the fixed published full anchor, not from an exploration checkpoint.

Your submission must encode a long-running recipe designed to make meaningful use of the formal training budget. It must not normally terminate early only because of a short fixed step or epoch limit.

Your formal recipe may decide when and how often to save complete and loadable checkpoints. Save each checkpoint under `/out/checkpoints/checkpoint-<progress>/`, where `<progress>` is numeric and increases with training or construction progress.

If more than three valid checkpoints are produced, only the three with the greatest `<progress>` values will be accepted. Every accepted checkpoint will be evaluated independently, and the run's official result is the best valid final score among them. The harness handles final artifact collection and final evaluation.

Each checkpoint must be a complete Hugging Face model export; optimizer or Trainer state alone is not sufficient.

## Evaluation boundary

Final evaluation outputs the pinned OpenUnlearning `Extraction` and `Model Utility` (`MU`) components. Lower Extraction indicates stronger forgetting; higher MU indicates better retained capability. The benchmark also computes `balanced_unlearning_score`, the harmonic mean of normalized forgetting progress and utility retention, as its primary scalar for comparing candidates. This local composite is not an OpenUnlearning metric or an additional validity gate; the final payload retains Extraction and MU separately.

The visible forget/retain answer-NLL diagnostic is an exploration tool and is not numerically interchangeable with Extraction or MU.

Exploration and retraining mount only the published full Llama anchor and the TOFU train-role projection; these define the complete training start and data universe. Candidates may select, reweight, schedule, or transform the available rows and change the unlearning objective or reference treatment. The separate score phase additionally mounts the evaluator-only retain90 anchor and final-role data, which are absent from the earlier phases. Formal scoring runs outside the submitted workspace. Do not import external data or checkpoints, implement an evaluation-specific lookup, or replace the declared final score with a candidate-defined metric.

## Shipped solution reference

The fixed full anchor and the current shipped Llama NPO solution have the following B300 reference results under the official component evaluation:

| Component | Full-anchor start | Shipped NPO solution |
|---|---:|---:|
| Extraction, lower is better | `0.707805` | `0.063436` |
| Model Utility, higher is better | `0.597131` | `0.478673` |

Relative to the full-anchor start, Extraction changes by `-0.644369` in the desired direction while MU changes by `-0.118458`, recording the retained-utility cost.

| Resource measurement | Result |
|---|---:|
| Formal training time | `569.18 s` |
| Native scoring time | `363.53 s` |
| Peak GPU memory | `33,242 MiB` |

Do not compare these values with results from a different model family or with the answer-NLL diagnostic.

## Work surface

Read `/workspace/run.sh`, `/workspace/train.py`, the shipped NPO loss, distributed configuration, data-role loader, and checkpoint exporter. Everything under `/workspace` is editable, including row use, forget/retain/reference treatment, unlearning or auxiliary objectives, optimization, batching, schedule, and checkpoint selection. These examples are illustrative, not exhaustive; they do not restrict any other change within the fixed task boundaries.

The shipped execution configuration uses ZeRO-3 with optimizer and parameter offload disabled; candidates may change execution and objective details. Stop immediately on an anchor-role or data-role violation.

## Running experiments

Use a separate output directory for every attempt:

```bash
OUTPUT_DIR=/out/probe-name bash /workspace/run.sh
/opt/harness/fast_eval.sh /out/probe-name
/opt/harness/timer.sh
```

Preserve resolved configuration, forget and retain losses, throughput, peak memory, checkpoint hash, component diagnostics, evaluator payload, and failures. Missing telemetry is an evidence gap, not by itself a scientific failure. Stop a failed candidate on non-finite loss, clear retained-utility collapse in the available diagnostics, fixed-role violation, or an unloadable checkpoint. Stopping one candidate does not by itself end exploration.

## Formal replay

Formal replay applies `candidate.patch`, mounts only the full anchor and train-role data, and invokes exactly:

```bash
bash /workspace/run.sh
```

Exploration checkpoints, caches, output directories, and shell exports do not cross phases. The separate score phase then mounts the evaluator-only retain90 anchor and final-role data and runs the official Extraction/MU evaluation for each accepted checkpoint.

## Submission

A smoke or startup check proves only that the code can begin; it is not performance evidence.

Before ending exploration, wait for every training, evaluation, and background command and read its result, or stop it explicitly and record why. Preserve the best trustworthy candidate as a fallback while exploring other directions.

Before the final action, verify that the final patch preserves the official anchors, component separation, and data roles.

Before submitting, verify that the patch encodes the long formal recipe and checkpoint-saving policy described above.

When no further meaningful experiment can be completed and interpreted within the remaining budget, verify the final source and artifacts, then run `/opt/harness/submit.sh` as the final action. If no candidate is trustworthy, use `/opt/harness/no_candidate.sh "reason"`. Deadline capture is recovery only and is not a normal submission path.
