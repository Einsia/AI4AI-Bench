# Published task images

The ten AI4AI v1.5 task images were published to Docker Hub on 2026-08-18, and one scoring
companion image on 2026-08-20. OpenR1 was rebuilt on 2026-08-20 when its final metric moved
from a 128-row prefix to the whole 175-row v6 slice, so both of its rows below carry the new
identity rather than the 2026-08-18 one. All eleven live in a single repository, `docker.io/chiyizhe/ai4ai`,
with the task name carried in the tag. Task declarations use the immutable digest references
below; the versioned tags are retained only as convenient human-readable aliases. All images
target Linux `amd64`.

The digests are unchanged from the 2026-08-18 publication: the manifests were replayed into the
shared repository byte for byte, so each reference still names the same audited bytes it did
when it was first pushed.

| Task | Immutable public reference | Versioned tag |
|---|---|---|
| DDPO | `docker.io/chiyizhe/ai4ai@sha256:16fbe1972620c7e47a8fd2135fc3f0eb350c77d45ab30a648ea8416fedfbd5bb` | `docker.io/chiyizhe/ai4ai:ddpo_sd15_aesthetic-v1.5-e0d3edf7b2a1` |
| DiGress | `docker.io/chiyizhe/ai4ai@sha256:fea474c6027c689e3364e07a5dfa73f2d5cbd8c523498f0a46c6cd854da29ede` | `docker.io/chiyizhe/ai4ai:digress_qm9_graph_diffusion-v1.5-3234eae2dad4` |
| DPO | `docker.io/chiyizhe/ai4ai@sha256:b020d14e02f9088aca1c0015eeefb4a78758a4e18516114a0a2c8f2f3d9be820` | `docker.io/chiyizhe/ai4ai:dpo_preference_alignment-v1.5-eda539bfcfd8` |
| Model Soup | `docker.io/chiyizhe/ai4ai@sha256:25d70889946756e8c1ef386a30f1d4aaa7c165f9371eb9638d39c3bbcccca2e1` | `docker.io/chiyizhe/ai4ai:model_soup_clip_imagenetv2-v1.5-ee8e2614d42a` |
| OPD | `docker.io/chiyizhe/ai4ai@sha256:87c4dbbd1216502c6e18c893b77b37ea3b1d0becd7b08ee9d2514193a198a8f6` | `docker.io/chiyizhe/ai4ai:opd_math_1p5b-v1.5-5b0bae505624` |
| OpenR1 | `docker.io/chiyizhe/ai4ai@sha256:f15cd8f7d74d006dd8e4b5abb545406f29cca08c436225b8696e8dd13fd87005` | `docker.io/chiyizhe/ai4ai:openr1_code_livecodebench-v1.5-0f6e3c906b07` |
| OpenUnlearning | `docker.io/chiyizhe/ai4ai@sha256:fe21d8982535c908ac81af20e4ba8c72bb0ff30c8d5ac0e5ebd2751ed00ff884` | `docker.io/chiyizhe/ai4ai:openunlearning_tofu_npo_llama3p2_1b-v1.5-evaluator-hotfix-347a00db1add` |
| OWL | `docker.io/chiyizhe/ai4ai@sha256:f5c2af25c85a114a3504b99048a477316142f874b110de7ef18e85460a7b3bac` | `docker.io/chiyizhe/ai4ai:owl_wanda_opt6p7b_70pct-v1.5-1df4a0c59c0f` |
| RAGEN | `docker.io/chiyizhe/ai4ai@sha256:bbdf9daec78a4e56cbdcb555bf918185fdee242d796b1cb400f95f5c152fedfc` | `docker.io/chiyizhe/ai4ai:ragen_sokoban_grpo-v1.5-aef0ed20015b` |
| Reward | `docker.io/chiyizhe/ai4ai@sha256:bf2f8c0cc63ffd01dcac67f9d04acbf3f8db07137d83f874e8f217e9d75e4ac9` | `docker.io/chiyizhe/ai4ai:ultrafeedback_bt_rm_rewardbench-v1.5-523db00590dd` |

## Scoring companion image

One task also publishes a scoring-only image. It is a second image rather than a change to the
first because the training image pins `trl==0.9.6`, which requires `transformers==4.46.1`,
while vLLM requires `transformers>=4.55.2`; the two cannot coexist and the scoring path imports
neither `trl` nor `peft`. Upstream `huggingface/open-r1` separates the same way, training on
trl and evaluating on lighteval with vLLM.

The image builds `FROM` the OpenR1 task image, so the read-only `/opt/harness` -- the evaluators
and the pinned LiveCodeBench tree -- is carried over byte for byte, and the build fails if that
commit moves. Only the generation backend changes; rows, prompt, extraction, test execution and
the pass rule are untouched. `runner.py` already accepts `--image`, so no schema change is needed:

```bash
python3 orchestrator/runner.py score --task tasks/openr1_code_livecodebench ... \
  --image docker.io/chiyizhe/ai4ai:openr1_code_livecodebench-eval-v1.6-full175-vllm --skip-digest-check
```

| Task | Immutable public reference | Versioned tag |
|---|---|---|
| OpenR1 (scoring) | `docker.io/chiyizhe/ai4ai@sha256:dc908a3ea5b97b07bcdd0a90627903e1a7b97c886b76cca2ab026540175be826` | `docker.io/chiyizhe/ai4ai:openr1_code_livecodebench-eval-v1.6-full175-vllm` |

`--skip-digest-check` is required while `[environment].digest` names only the training image;
giving the scoring image its own entry in `task.toml` removes that flag. The build recipe and
its hash-locked package set are in `tasks/openr1_code_livecodebench/environment/`
(`Dockerfile.eval`, `eval-requirements.in`, `eval-requirements.lock`).

For example:

```bash
docker pull \
  docker.io/chiyizhe/ai4ai@sha256:f15cd8f7d74d006dd8e4b5abb545406f29cca08c436225b8696e8dd13fd87005
```

Post-push verification pulled every digest and confirmed that its local image ID matched the
corresponding audited release candidate. A second check using an empty Docker configuration
confirmed that every manifest is anonymously readable. Publication does not imply that
runtime model and dataset assets are bundled, nor that disclosed license or vulnerability
risks have been eliminated; consult `tools/docs/assets.md` and `THIRD_PARTY_NOTICES.md`.

In particular, OPD and RAGEN were published after an explicit publisher decision to accept
the unresolved OPD provenance/licensing findings and the RAGEN CUDA Toolkit EULA review risk.
The technical notices retained in the release source and embedded images deliberately still
describe those issues as release holds: publication records the publisher's decision, not a
claim that the findings were resolved or that downstream redistribution is cleared.
