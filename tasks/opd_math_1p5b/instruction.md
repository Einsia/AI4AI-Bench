# OPD: on-policy distillation of a 1.5B math reasoner

Improve the fixed 1.5B student on the available math-training assets. The shipped solution uses sampled-token on-policy distillation with the mounted teacher; that is the reference method rather than a mandatory objective.

You have up to four hours for exploration. Do not run work only to consume time, but do not treat a submit-ready candidate as completion. Preserve each trustworthy candidate as a fallback and continue scientifically meaningful exploration while the remaining budget can support experiments whose results can be completed and interpreted.

Before submitting, check the remaining budget and the plausible directions that have not yet been tested. A candidate being better than the current reference, loadable, reproducible, or artifact-valid establishes that it is a fallback; none of those facts alone establishes that exploration is complete. The default action when substantial usable budget remains is to continue exploring, analyzing, or validating.

Early submission is appropriate only when no further meaningful experiment can be completed and interpreted within the remaining budget. Do not submit merely because the current candidate is good enough or has passed its validation checks.

The submitted source patch is applied in a fresh container for a formal retrain of up to 12 hours. Formal retraining starts from the fixed student model, not from an exploration checkpoint.

Your submission must encode a long-running recipe designed to make meaningful use of the formal training budget. It must not normally terminate early only because of a short fixed step or epoch limit.

Your formal recipe may decide when and how often to save complete and loadable checkpoints. Save each checkpoint under `/out/checkpoints/checkpoint-<progress>/`, where `<progress>` is numeric and increases with training or construction progress.

If more than three valid checkpoints are produced, only the three with the greatest `<progress>` values will be accepted. Every accepted checkpoint will be evaluated independently, and the run's official result is the best valid final score among them. The harness handles final artifact collection and final evaluation.

Each checkpoint must be a loadable actor/Hugging Face export; trainer state alone is not a checkpoint.

## Evaluation boundary

The exploration metric is `math500_pass_at_1`: all 500 MATH-500 questions, four samples per question, with a fixed 12,288-token generation cap. The final metric is `aime24_25_at32`: 60 AIME 2024 and 2025 questions, 32 samples per question, with a fixed 31,744-token generation cap. Both are higher-is-better and use the same rule grader with verifier fallback, but they are different benchmarks and must not be compared numerically with each other.

You may read both evaluator implementations under `/opt/harness`. MATH-500 is mounted during exploration; the AIME inputs used for final scoring are not. Do not reconstruct, import, or tune against the final question set. Changing the fast evaluator's sample count, question set, grader, or generation cap produces a different measurement and is not comparable with the declared fast metric.

Use `stderr`, `clip_rate`, training dynamics, and any validation you choose when judging a candidate. A small change from one stochastic evaluation is weak evidence, and an improvement in a short probe does not by itself establish an improvement after formal replay. Decide whether additional validation or a different direction has greater expected value.

## Shipped solution reference

The fixed student and a full replay of the unmodified shipped solution were both scored under the declared AIME final protocol on B300. The table therefore gives a direct final-to-final comparison. The MATH-500 row is a separate early-checkpoint reference for exploration and must not be compared numerically with AIME.

| Measurement | Result |
|---|---:|
| Fixed student start, AIME24+25 final | `484/1920 = 0.252083` |
| Current shipped solution, AIME24+25 final | `820/1920 = 0.427083 +/- 0.051615` |
| Difference from the fixed student start | `+336/1920 = +0.175000` |
| AIME24 component | `0.4958` |
| AIME25 component | `0.3583` |
| MATH-500 fast reference at step 40 | `0.8410 +/- 0.0138`, clip rate `0.0645` |
| Formal training | 2,057 steps in 11.51 hours |
| Final scoring time | about 54 minutes at the declared batch width |
| Training throughput | about 20.14 seconds per step |
| Peak training GPU memory | `39,568 MiB` (`38.6 GiB`) |
| Checkpoint storage | about 3 GiB for exported weights; about 27 GiB including optimizer state |

The start-to-solution difference is meaningful only within the AIME final protocol. The step-40 MATH-500 value is included solely to orient short exploration runs; it is neither the training start nor the endpoint of the formal replay. Every new run writes its own dynamics under `${OUTPUT_DIR}/metrics`; use those current measurements rather than treating this reference as a target.

## Work surface

Start with `/workspace/run.sh`, which is the formal training entry point, and trace the implementation it calls. Everything under `/workspace` is editable, including the launcher, training driver, loss, optimizer, data pipeline, and the vendored `verl` implementation. These examples are illustrative, not exhaustive; they do not restrict any other change within the fixed task boundaries. Source edits under `/workspace` take effect directly through `PYTHONPATH`; no package reinstall is required.

The student training start and the available teacher and training-data assets are read-only mounts under `/assets`. A candidate may change which training rows are used, their sampling or curriculum, and how or whether the teacher, rewards, references, or generated samples enter the objective. Formal scoring uses the frozen evaluator outside the submitted workspace, and the environment has no network. Do not import external checkpoints or data, train on evaluation questions, or implement an evaluation-specific lookup.

The candidate need not preserve the shipped on-policy loss or teacher-use pattern. It must start the formal replay from the fixed student, use only the available assets, and export a loadable checkpoint. Resource utilization is not itself an objective; prefer changes that improve useful work, stability, and evaluation evidence within the fixed wall clock.

## Running experiments

The main tools are:

```bash
bash /workspace/run.sh
/opt/harness/fast_eval.sh <checkpoint-or-checkpoint-root>
/opt/harness/timer.sh
/opt/harness/submit.sh
```

Environment overrides and trailing Hydra overrides are available for probes. Give concurrent probes distinct `OUTPUT_DIR`, `CKPT_DIR`, and `EXPERIMENT_NAME` values. `run.sh` holds a shared GPU workload lock and exclusive locks on its output and checkpoint directories; conflicting runs fail with exit code 75. `fast_eval.sh` takes the GPU lock exclusively and refuses to start while a compute process still owns the GPU, so training must release the device before evaluation.

Checkpoints are written under `CKPT_DIR` as `global_step_N`. The evaluator accepts an exported Hugging Face directory, a loadable `global_step_N`, or a checkpoint root containing loadable steps. Old step directories may remain after their weights have been pruned; these stubs are not valid checkpoints. Confirm that the checkpoint selected for evaluation actually contains model weights.

Stop a failed candidate after an OOM, NaN, repeated startup failure, clearly worse training dynamics, or a result that cannot be made comparable to the declared metric. Stopping one candidate does not by itself end exploration.

## Formal replay

Formal replay applies `candidate.patch` to a fresh `/workspace` and invokes exactly:

```bash
bash /workspace/run.sh
```

It does not call `train.py`, inherit exploration shell exports, reuse exploration checkpoints, Ray state, compiler caches, or output directories. Any required change must therefore be present in the submitted files and reachable from `run.sh`; an environment-only experiment is not a reproducible candidate.

The orchestrator supplies `STUDENT_MODEL`, `TEACHER_MODEL`, `TRAIN_DATA`, `MAX_WALL_TIME_SECONDS`, and `DEADLINE_RESERVE_SECONDS` during formal replay. Editing only the fallback defaults for those variables does not change the formal value. `MAX_WALL_TIME_SECONDS` stops training before the outer deadline so a checkpoint can finish writing. Ensure the configured training horizon does not end unintentionally before that wall-clock stop.

The submitted artifact is source code only. Do not include model weights, optimizer state, generated datasets, evaluation outputs, caches, logs, or other run artifacts in `/workspace`. Formal replay will independently verify the resolved fixed inputs and select a complete checkpoint from the new run for scoring.

## Submission

A smoke or startup check proves only that the code can begin; it is not performance evidence.

Before ending exploration, wait for every training, evaluation, and background command and read its result, or stop it explicitly and record why. Preserve the best trustworthy candidate as a fallback while exploring other directions.

Before the final action, inspect the patch and verify that the best-supported change is encoded in source rather than only in the current shell or output directory.

Before submitting, verify that the patch encodes the long formal recipe and checkpoint-saving policy described above.

When no further meaningful experiment can be completed and interpreted within the remaining budget, verify the final source and artifacts, then run `/opt/harness/submit.sh` as the final action. If no candidate is trustworthy, use `/opt/harness/no_candidate.sh "reason"`. Deadline capture is recovery only and is not a normal submission path.
