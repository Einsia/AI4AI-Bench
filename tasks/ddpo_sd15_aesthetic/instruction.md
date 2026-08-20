# DDPO aesthetic alignment

Improve the final aesthetic score of Stable Diffusion v1.5 from the fixed training start and available prompt/model assets. The shipped solution uses on-policy DDPO with LoRA; that is the starting method, not an algorithm label the candidate must preserve.

You have up to four hours for exploration. Do not run work only to consume time, but do not treat a submit-ready candidate as completion. Preserve each trustworthy candidate as a fallback and continue scientifically meaningful exploration while the remaining budget can support experiments whose results can be completed and interpreted.

Before submitting, check the remaining budget and the plausible directions that have not yet been tested. A candidate being better than the current reference, loadable, reproducible, or artifact-valid establishes that it is a fallback; none of those facts alone establishes that exploration is complete. The default action when substantial usable budget remains is to continue exploring, analyzing, or validating.

Early submission is appropriate only when no further meaningful experiment can be completed and interpreted within the remaining budget. Do not submit merely because the current candidate is good enough or has passed its validation checks.

The submitted source patch is applied in a fresh container for a formal retrain of up to 12 hours. Formal retraining starts from the fixed Stable Diffusion model, not from an exploration checkpoint.

Your submission must encode a long-running recipe designed to make meaningful use of the formal training budget. It must not normally terminate early only because of a short fixed step or epoch limit.

Your formal recipe may decide when and how often to save complete and loadable checkpoints. Save each checkpoint under `/out/checkpoints/checkpoint-<progress>/`, where `<progress>` is numeric and increases with training or construction progress.

If more than three valid checkpoints are produced, only the three with the greatest `<progress>` values will be accepted. Every accepted checkpoint will be evaluated independently, and the run's official result is the best valid final score among them. The harness handles final artifact collection and final evaluation.

Each checkpoint must contain complete Diffusers LoRA weights loadable by the frozen evaluator with the fixed Stable Diffusion base model.

## Evaluation boundary

The exploration metric is `mean_aesthetic_score_public64`, evaluated on 64 generated images. The final metric is `mean_aesthetic_score_final256`, evaluated on 256 images. Higher is better for both. CLIP alignment and pairwise image distance are reported as auxiliary diagnostics, not validity gates or terms in the headline score.

The Stable Diffusion training start and CLIP/aesthetic reward assets are fixed and available during training. The exact final prompt/latent stream is mounted only during scoring. Formal scoring uses frozen harness code outside the submitted workspace, even when a candidate changes its training-time reward construction. Training-time prompt construction and sampling, reward shaping and normalization, auxiliary losses, update rule, and trainable parameters may change. Do not import external data or weights, reconstruct the final stream, or implement an evaluation-specific lookup.

Generation and training are stochastic. Compare candidates at matched seeds and retain the complete aesthetic and auxiliary-diagnostic payload. Treat a small difference in light of the observed noise and decide whether further validation is worth its opportunity cost.

## Shipped solution reference

The fixed model and the current shipped solution have the following B300 reference results under the declared final protocol. The result is stochastic, so another training seed may not preserve the difference.

| Measurement | Result |
|---|---:|
| Fixed-model training start, final aesthetic score | `5.397311` |
| Current shipped solution, final aesthetic score | `5.526373` |
| Difference from fixed-model start | `+0.129062` |
| Training time | `3340.09 s` |
| Final scoring time | `528.28 s` |
| Peak training GPU memory | `58,598 MiB` |

This single-run difference does not identify which individual setting caused it. Report the auxiliary alignment and diversity diagnostics with the aesthetic score so mode collapse or prompt drift remains visible without invalidating an otherwise scoreable result.

## Work surface

Read `/workspace/ddpo_config.py`, `/workspace/train.py`, `/workspace/run.sh`, `/workspace/launch_training.sh`, and the editable implementation before deciding what limits the run. Everything under `/workspace` is editable, including prompt selection, objective and reward construction, sampling and training shapes, optimizer, schedule, update rule, and trainable-parameter strategy. These examples are illustrative, not exhaustive; they do not restrict any other change within the fixed task boundaries.

The shipped DDPO implementation is a reference, not a method gate. A candidate may change the training algorithm as long as formal replay starts from the fixed model, uses only the available assets, does not train on the hidden final stream, and exports a checkpoint the frozen scorer can load. Resource utilization is not itself an objective; prefer changes that improve useful work, stability, and same-protocol evaluation evidence within the wall clock.

## Running experiments

Use a distinct output directory for every attempt:

```bash
DDPO_PROFILE=proxy OUTPUT_DIR=/out/probe-name bash /workspace/run.sh
/opt/harness/fast_eval.sh /out/probe-name/checkpoint
/opt/harness/timer.sh
```

Preserve the resolved configuration, prompt/sample inventory, optimizer work, reward and auxiliary-diagnostic dynamics, throughput, peak memory, checkpoint hash, evaluator payload, and failure reason. Stop a failed candidate after an OOM, non-finite objective, unloadable checkpoint, hidden-input use, or fixed-start drift. Stopping one candidate does not by itself end exploration.

## Formal replay

Formal replay applies `candidate.patch` to a fresh `/workspace` and invokes exactly:

```bash
bash /workspace/run.sh
```

It does not inherit exploration checkpoints, generated images, caches, output directories, or shell exports. Any required change must be present in the submitted files and reachable from the formal command. Generated samples and other exploration outputs are not formal checkpoints.

## Submission

A smoke or startup check proves only that the code can begin; it is not performance evidence.

Before ending exploration, wait for every training, evaluation, and background command and read its result, or stop it explicitly and record why. Preserve the best trustworthy candidate as a fallback while exploring other directions.

Before the final action, verify that the source is runnable, the selected change is encoded in the files, and the patch contains no generated images, model weights, logs, caches, or evaluator output.

Before submitting, verify that the patch encodes the long formal recipe and checkpoint-saving policy described above.

When no further meaningful experiment can be completed and interpreted within the remaining budget, verify the final source and artifacts, then run `/opt/harness/submit.sh` as the final action. If no candidate is trustworthy, use `/opt/harness/no_candidate.sh "reason"`. Deadline capture is recovery only and is not a normal submission path.
