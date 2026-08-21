# AI4AI-Bench

[![arXiv](https://img.shields.io/badge/arXiv-2608.20318-b31b1b?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2608.20318)
[![Paper](https://img.shields.io/badge/Paper-PDF-EC1C24?style=flat-square&logo=files&logoColor=white)](https://arxiv.org/pdf/2608.20318)
[![Homepage](https://img.shields.io/badge/Homepage-lab.einsia.ai-0969DA?style=flat-square&logo=homepage&logoColor=white)](https://lab.einsia.ai/ai4ai/)
[![Tasks](https://img.shields.io/badge/Tasks-10-2E7D32?style=flat-square)](https://lab.einsia.ai/ai4ai/tasks/)
[![Trajectories](https://img.shields.io/badge/Trajectories-290-6E56CF?style=flat-square)](https://lab.einsia.ai/ai4ai/trajectories/)
[![Docker](https://img.shields.io/badge/Docker-chiyizhe%2Fai4ai-2496ED?style=flat-square&logo=docker&logoColor=white)](https://hub.docker.com/r/chiyizhe/ai4ai)

AI4AI-Bench asks whether a coding agent can improve an existing AI training recipe—not
merely edit code that passes a fixed test. Its ten tasks span generation, alignment,
reasoning, unlearning, pruning, reinforcement learning, reward modeling, and model merging.

Each run separates open-ended experimentation from reproducible measurement. The agent may
explore for up to four hours, but only a source patch crosses into a fresh formal environment;
formal training then runs for up to twelve hours, publishes at most three checkpoints, and
hands each checkpoint to frozen validation and final evaluation.

```text
fixed task + assets
        │
        ├─ 4 h Explore ──> candidate.patch
        │                         │
        └──────── fresh Formal <──┘
                  (up to 12 h)
                         │
             up to 3 checkpoints
                         │
             validation ──> final score
```

## Quickstart

AI4AI-Bench targets Linux `amd64`, Python 3.10+, Docker with NVIDIA Container Toolkit, and
an NVIDIA GPU. Official runs use one B300; other GPUs are useful for local development when
the selected task fits. Install a native Codex or Claude CLI before starting an agent run.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[assets]'
```

Prepare one task's pinned public assets. Set `HF_TOKEN` first if an upstream repository is
gated.

```bash
export AI4AI_ASSET_STORE=/data/ai4ai/assets
python tools/prepare_assets.py \
  --task ddpo_sd15_aesthetic \
  --assets "$AI4AI_ASSET_STORE/ddpo_sd15_aesthetic" \
  --execute
python tools/verify_assets.py \
  --task ddpo_sd15_aesthetic \
  --assets "$AI4AI_ASSET_STORE/ddpo_sd15_aesthetic" \
  --hash
```

Run the no-API GPU smoke test first. It pulls one published image if needed and exercises
Docker, GPU passthrough, a CUDA kernel, host mounts, and the mock score lifecycle.

```bash
bash tools/smoke.sh --root /data/ai4ai/smoke --gpu 0
```

Then configure the selected Agent and check the complete host without displaying credentials.

```bash
export OPENAI_API_KEY=your-key
python tools/check_setup.py \
  --task ddpo_sd15_aesthetic \
  --assets "$AI4AI_ASSET_STORE/ddpo_sd15_aesthetic" \
  --root /data/ai4ai/runs \
  --gpu 0 --agent codex --mode local
```

Start a real run. The default lifecycle performs Explore, fresh formal retraining,
checkpoint validation, and final evaluation.

```bash
bash orchestrator/trial.sh ddpo-codex \
  --task tasks/ddpo_sd15_aesthetic \
  --assets "$AI4AI_ASSET_STORE/ddpo_sd15_aesthetic" \
  --root /data/ai4ai/runs --gpu 0 \
  --agent codex --model gpt-5.6-sol --reasoning-effort high
```

For Claude, use `--agent claude --model claude-opus-5` and set `ANTHROPIC_API_KEY` (or
`ANTHROPIC_AUTH_TOKEN`). See the [runtime guide](tools/docs/runtime-and-storage.md) for alternate
paths, endpoints, local/official verification modes, and storage requirements.

## Evaluation and replay

Replay an existing source patch from the same fixed start:

```bash
bash orchestrator/trial.sh replay-name \
  --task tasks/ddpo_sd15_aesthetic \
  --assets /data/ai4ai/assets/ddpo_sd15_aesthetic \
  --root /data/ai4ai/runs --gpu 0 \
  --candidate-patch candidate.patch
```

Evaluate one to three existing checkpoints independently:

```bash
bash orchestrator/evaluate.sh eval-name \
  --task tasks/ddpo_sd15_aesthetic \
  --assets /data/ai4ai/assets/ddpo_sd15_aesthetic \
  --root /data/ai4ai/evaluations --gpu 0 \
  --checkpoint 1000=/path/to/checkpoint-1000
```

These are self-hosted final evaluations. The public `warn` defaults label their receipts as
non-official local results; strict official-mode requirements are documented separately. See
[evaluation and receipts](tools/docs/evaluation.md) for checkpoint selection, result states, and the
current absence of a blind evaluation service.

## Tasks and documentation

- [Ten benchmark tasks](tasks/README.md)
- [Runtime assets](tools/docs/assets.md)
- [Runtime, storage, and mounts](tools/docs/runtime-and-storage.md)
- [Evaluation and receipts](tools/docs/evaluation.md)
- [Published task images](tools/docs/published-images.md)
- [Troubleshooting](tools/docs/troubleshooting.md)

## Citation

```bibtex
@article{chi2026ai4ai,
  title   = {AI4AI-Bench: Benchmarking LLM Agents in Algorithmic Design for Recursive Self-Improvement},
  author  = {Yizhe Chi and Wenyi Li and Deyao Hong and Xiaoqiu Wang and Mingju Gao and Kaisen Yang and Bingxiang He and Youjie Zheng and Calvin Xiao and Qinhuai Na},
  journal = {arXiv preprint arXiv:2608.20318},
  year    = {2026}
}
```

## License

The benchmark code is released under [Apache-2.0](LICENSE). Models, datasets, and image
contents remain subject to their upstream terms; see [third-party notices](THIRD_PARTY_NOTICES.md).
