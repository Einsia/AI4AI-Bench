# RAGEN on Sokoban

Improve the fixed Qwen2.5-3B-Instruct policy on the frozen Sokoban evaluation protocol. The shipped solution uses multi-turn on-policy GRPO and generates its training boards and trajectories online; that is the reference method rather than a mandatory objective.

You have up to four hours for exploration. Do not run work only to consume time, but do not treat a submit-ready candidate as completion. Preserve each trustworthy candidate as a fallback and continue scientifically meaningful exploration while the remaining budget can support experiments whose results can be completed and interpreted.

Before submitting, check the remaining budget and the plausible directions that have not yet been tested. A candidate being better than the current reference, loadable, reproducible, or artifact-valid establishes that it is a fallback; none of those facts alone establishes that exploration is complete. The default action when substantial usable budget remains is to continue exploring, analyzing, or validating.

Early submission is appropriate only when no further meaningful experiment can be completed and interpreted within the remaining budget. Do not submit merely because the current candidate is good enough or has passed its validation checks.

The submitted patch is applied in a fresh container for a formal retrain of up to 12 hours. Formal retraining starts from the fixed policy, regenerates boards, and does not reuse exploration rollouts or checkpoints.

Your submission must encode a long-running recipe designed to make meaningful use of the formal training budget. It must not normally terminate early only because of a short fixed step or epoch limit.

Your formal recipe may decide when and how often to save complete and loadable checkpoints. Save each checkpoint under `/out/checkpoints/checkpoint-<progress>/`, where `<progress>` is numeric and increases with training or construction progress.

If more than three valid checkpoints are produced, only the three with the greatest `<progress>` values will be accepted. Every accepted checkpoint will be evaluated independently, and the run's official result is the best valid final score among them. The harness handles final artifact collection and final evaluation.

Only a merged, loadable Hugging Face model is a checkpoint; raw FSDP shards are not.

## Evaluation boundary

The exploration metric is `public_four_bank_solve_rate`; the final metric is `held_out_512_board_solve_rate`. Higher is better for both. The public banks and held-out boards use different fixed environment-seed protocols, so compare each metric only with results from the same tier.

The policy start, frozen score-time Sokoban engine, action decoding, evaluation behavior, and final seeds are fixed. Candidates may change training-board construction, curriculum, rollout collection, reward shaping, objectives, and on- or off-policy updates using only information available in the training container. Formal scoring runs outside the submitted workspace. Do not import external boards, demonstrations, trajectories, or weights, reconstruct or train on final seeds, or implement an evaluation-specific lookup.

Training is stochastic at both board and policy levels. Preserve board identities and per-board outcomes, and do not treat one training seed as a complete noise estimate. A valid artifact is a merged, loadable Hugging Face checkpoint; trainer shards alone are not a result.

## Shipped solution reference

The fixed policy and the current shipped solution have the following B300 reference results:

| Measurement | Result |
|---|---:|
| Fixed policy start, final solve rate | `60/512 = 0.117188` |
| Current shipped solution, final solve rate | `87/512 = 0.169922` |
| Difference from the fixed start | `+27/512 = +0.052734` |
| Training time | `2746.19 s` |
| Final scoring time | `339.15 s` |
| Peak GPU memory during final scoring | `247,684 MiB` |

The memory number is the final-scoring peak. Training is stochastic, so report exact solved-board counts and judge a small claimed improvement against the available uncertainty before deciding what to test next.

## Work surface

Read `/workspace/run.sh`, training-board generation, rollout or data collection, advantage and reward computation, loss reduction, optimizer, checkpoint merge, and environment integration. Everything under `/workspace` is editable, including curriculum, on- or off-policy objectives, filtering, reward shaping, batching, optimization, schedule, and merge logic. These examples are illustrative, not exhaustive; they do not restrict any other change within the fixed task boundaries.

The candidate need not preserve GRPO, on-policy sampling, or the shipped training environment behavior. Formal replay must start from the fixed policy, use no external or hidden-final data, and export a merged checkpoint scored by the frozen Sokoban evaluator. Systems gains are useful only when the resulting checkpoint is evaluated under that unchanged final protocol.

## Running experiments

Give every attempt its own output tree:

```bash
OUTPUT_DIR=/out/probe-name bash /workspace/run.sh
/opt/harness/fast_eval.sh /out/probe-name/checkpoints
/opt/harness/timer.sh
```

Preserve board-bank identities, trajectory lengths, action and reward distributions, filter statistics, update timing, throughput, peak memory, trainer state, merged-checkpoint hash, evaluator payload, and failures. Training and evaluation share the GPU lock. Stop a failed candidate on environment, merge, or load failure, non-finite loss, rollout collapse, action collapse, or repeated solve-rate regression. Stopping one candidate does not by itself end exploration.

## Formal replay

Formal replay applies `candidate.patch` to a fresh `/workspace`, regenerates boards, and invokes exactly:

```bash
bash /workspace/run.sh
```

It does not inherit exploration rollouts, checkpoints, Ray state, caches, output directories, or shell exports.

## Submission

A smoke or startup check proves only that the code can begin; it is not performance evidence.

Before ending exploration, wait for every training, evaluation, and background command and read its result, or stop it explicitly and record why. Preserve the best trustworthy candidate as a fallback while exploring other directions.

Before the final action, verify that the final source starts cleanly and can merge its checkpoint.

Before submitting, verify that the patch encodes the long formal recipe and checkpoint-saving policy described above.

When no further meaningful experiment can be completed and interpreted within the remaining budget, verify the final source and artifacts, then run `/opt/harness/submit.sh` as the final action. If no candidate is trustworthy, use `/opt/harness/no_candidate.sh "reason"`. Deadline capture is recovery only and is not a normal submission path.
