# Evaluation image (vLLM backend)

`Dockerfile.eval` builds `docker.io/chiyizhe/ai4ai:openr1_code_livecodebench-eval-v1.6-full175-vllm`, a
scoring-only companion to the training image. It is published as
`docker.io/chiyizhe/ai4ai:openr1_code_livecodebench-eval-v1.6-full175-vllm`; see
`tools/docs/published-images.md` for the immutable digest.

## Why it is separate

The training image pins `trl==0.9.6`, which requires `transformers==4.46.1`.
vLLM 0.11.0 requires `transformers>=4.55.2`. The two cannot coexist, and the
evaluation path imports neither `trl` nor `peft`. Upstream `huggingface/open-r1`
splits the same way: training runs on trl, evaluation runs on lighteval + vLLM.

NumPy is *not* a conflict. vLLM 0.11.0 runs on numpy 1.26.4, which is what the
sibling RAGEN image already ships.

## What changes, and what does not

The image is `FROM` the training image, so the read-only `/opt/harness` -- the
evaluator and the pinned LiveCodeBench@28fef95e tree -- is carried over byte for
byte. A build-time assertion fails the build if that commit ever moves. The
protocol is therefore unchanged: same rows, same prompt, same `extract_code`,
same official test execution, same `> 0` pass rule.

The only change is the generation backend. The shipped `grade.generate()` runs
one prompt per forward pass, which leaves a 275 GB card at 1.4 percent
utilisation and makes the official LiveCodeBench protocol (n=10, temperature
0.2, top_p 0.95) impractical: 175 problems ran for over six hours without
finishing. Continuous batching does the same work in about two minutes at
roughly 5,800 tokens/s, with the KV cache filling 219 GB.

Greedy n=1 is not reproducible across backends at problem level: the same
weights score 20/175 under transformers and 22/175 under vLLM, with only 1
percent of completions identical and eight problems flipping in both
directions. That is floating-point reduction order, not a defect in either
backend, and it is one more reason to prefer the sampled protocol the upstream
benchmark actually defines.

## Build

Offline, one named context, as with every other task image:

```bash
docker buildx build --network=none \
  --build-context wheelhouse=<wheelhouse> \
  -f tasks/openr1_code_livecodebench/environment/Dockerfile.eval \
  -t docker.io/chiyizhe/ai4ai:openr1_code_livecodebench-eval-v1.6-full175-vllm .
```

`eval-requirements.lock` holds the 86 exact pins. They were resolved with
`pip install --dry-run --report` inside the training image, which answers what
must be added to that base rather than resolving in a vacuum; `torch` is
correctly absent from the set.

## Use

`runner.py` already accepts `--image`, so no schema change is needed. Train with
the default image and score with this one:

```bash
python3 orchestrator/runner.py retrain --task tasks/openr1_code_livecodebench ...
python3 orchestrator/runner.py score   --task tasks/openr1_code_livecodebench ... \
  --image docker.io/chiyizhe/ai4ai:openr1_code_livecodebench-eval-v1.6-full175-vllm
```

`--skip-digest-check` is needed while `[environment].digest` still names the
training image. Giving the eval image its own digest entry in `task.toml` is the
follow-up that removes that flag.
