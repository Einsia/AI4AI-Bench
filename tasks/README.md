# Tasks

Each task fixes the model and data identities, exploration/final evaluator,
resource ceiling and artifact contract. The coding Agent may edit only the
workspace exposed by that task.

| Task | Method | Exploration metric | Final metric |
|---|---|---|---|
| [ddpo_sd15_aesthetic](ddpo_sd15_aesthetic/) | Stable Diffusion 1.5 DDPO LoRA | mean aesthetic score | mean aesthetic score with alignment/diversity gates |
| [digress_qm9_graph_diffusion](digress_qm9_graph_diffusion/) | discrete graph diffusion | molecule validity, uniqueness and novelty | QM9 negative log-likelihood (lower is better) |
| [dpo_preference_alignment](dpo_preference_alignment/) | pairwise DPO on Zephyr/Mistral 7B | IFEval strict accuracy, public 128 | IFEval strict accuracy, final 413 |
| [model_soup_clip_imagenetv2](model_soup_clip_imagenetv2/) | CLIP weight-space soup | ImageNetV2 top-1, 2,000 rows | ImageNetV2 top-1, 10,000 rows |
| [opd_math_1p5b](opd_math_1p5b/) | sampled-token on-policy distillation | MATH-500 pass@1 | AIME 2024/2025 at 32 samples |
| [openr1_code_livecodebench](openr1_code_livecodebench/) | completion-only code SFT | public code pass@1 | LiveCodeBench v6 pass@1 |
| [openunlearning_tofu_npo_llama3p2_1b](openunlearning_tofu_npo_llama3p2_1b/) | official Llama NPO | fixed forget/retain NLL diagnostic | TOFU Extraction and MU plus a local balanced composite |
| [owl_wanda_opt6p7b_70pct](owl_wanda_opt6p7b_70pct/) | OWL/Wanda pruning | WikiText-2 validation perplexity | WikiText-2 test perplexity with exact-sparsity gate |
| [ragen_sokoban_grpo](ragen_sokoban_grpo/) | multi-turn on-policy GRPO | public four-bank solve rate | held-out 512-board solve rate |
| [ultrafeedback_bt_rm_rewardbench](ultrafeedback_bt_rm_rewardbench/) | Bradley-Terry reward modeling | RewardBench 512-pair proxy | RewardBench v1 score with artifact and overlap gates |

The NPO exploration value is a training diagnostic, not a smaller version of
the native final. Its final report keeps Extraction and MU separate and uses a
local balanced composite as the benchmark's primary scalar.

Every task has the same directory contract:

```text
task.toml       immutable identities, budgets, metrics and provenance
instruction.md  brief shown to the Agent
declaration.py  phase mounts, commands and exports
environment/    reproducible image inputs
solution/       Agent-editable implementation
harness/        read-only evaluation and artifact checks
```
