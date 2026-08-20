# Third-party notices

AI4AI task images combine this Apache-2.0 repository with pinned upstream source,
base images, and Python/system packages. The authoritative versions and download
locations are in each task's `environment/assets.lock.yaml`, Dockerfile, and
`runtime-requirements.lock`. Those files are part of the release record.

Principal task sources include:

- `kvablack/ddpo-pytorch` (MIT)
- `cvignac/DiGress` (MIT)
- `huggingface/alignment-handbook` (Apache-2.0)
- `mlfoundations/model-soups` (MIT)
- `thunlp/OPD` and `thunlp/JustRL` (see the unresolved OPD notice below)
- `volcengine/verl` (Apache-2.0)
- `LiveCodeBench/LiveCodeBench` evaluator code (MIT)
- `locuslab/open-unlearning` (BSD-3-Clause)
- `luuyin/OWL` (MIT)
- `mll-lab-nu/RAGEN` and its pinned dependencies (see its upstream notices)
- `RLHFlow/RLHF-Reward-Modeling` (Apache-2.0)

Model and dataset licenses are separate from code licenses. In particular, Stable
Diffusion v1.5 uses CreativeML Open RAIL-M; Llama 3.2-derived checkpoints use the
Llama 3.2 Community License; OPT weights use the OPT license; ImageNet/ImageNetV2,
RewardBench component datasets, LiveCodeBench data, and Model Soup ingredient
checkpoints carry their respective upstream terms. Runtime assets are not embedded
in the task images.

## Embedded-image review holds

The following findings concern bytes embedded in a task image, rather than separately
downloaded runtime assets:

- **OPD is not cleared for public image upload.** The image copies JustRL evaluator files
  from a repository whose root declares no license covering those files. Its pinned base
  image also contains `/opt/opd` and `/opt/llm-algobench` in lower layers; deleting those
  paths in a later Docker layer does not remove their bytes from the distributed image.
  Provenance and redistribution terms for those lower-layer files have not been established.
  The separately copied `rg` binary likewise needs an explicit license record. Replace or
  clear all of these inputs before publishing the OPD image.
- **RAGEN remains on a publisher-review hold.** It embeds NVIDIA CUDA Toolkit `ptxas`,
  identified in the task asset lock as `LicenseRef-NVIDIA-Proprietary`. Confirm that the
  intended registry distribution complies with the applicable CUDA Toolkit EULA and include
  any required notice before publishing this image.

These are release holds, not a legal conclusion that every other image is cleared. Base
images and the complete SBOM still require the publisher's normal license review.

## Runtime-asset redistribution

Runtime assets are not embedded in the task images. Users may materialize supported assets
directly from their upstream providers, subject to provider access controls and terms. A
project-operated mirror or bundled asset store is a separate redistribution decision. In
particular, Model Soup checkpoint/ImageNet-derived components, OpenR1 Codeforces and
LiveCodeBench data, Llama-derived OpenUnlearning weights, OPT/C4/WikiText assets, and the
mixed-license RewardBench components require their recorded component terms to be reviewed.
Do not treat this notice or a successful hash check as permission to redistribute an asset.

The complete dependency inventory for a published image is its generated SBOM.
Retain that SBOM, the image digest, scanner database timestamp, and any accepted
vulnerability findings with the release.
