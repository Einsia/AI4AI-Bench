"""The ml_collections config upstream's tools/train.py loads via --config.

Every value comes from an environment variable that run.sh declares with a default,
so run.sh is the whole configuration surface in one file. This layer exists because
upstream takes its configuration as a Python config object rather than as flags; it is
not a boundary and nothing checks it.

The selected defaults live in run.sh; the fallbacks here match them so direct config
inspection reports the same recipe.
"""

from __future__ import annotations

import os

from config.base import get_config as get_base_config


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default) not in ("", "0", "false", "False")


def get_config():
    config = get_base_config()

    config.run_name = os.environ.get("RUN_NAME", "ddpo_sd15_aesthetic")
    config.seed = int(os.environ.get("SEED", "43"))
    config.logdir = os.environ["LOG_DIR"]
    config.num_epochs = int(os.environ.get("NUM_EPOCHS", "13"))
    # The image patch makes save_freq=1 include the first completed epoch.
    config.save_freq = int(os.environ.get("SAVE_FREQ", "1"))
    checkpoint_limit = int(os.environ.get("NUM_CHECKPOINT_LIMIT", "3"))
    config.num_checkpoint_limit = None if checkpoint_limit == 0 else checkpoint_limit
    config.mixed_precision = os.environ.get("MIXED_PRECISION", "fp16")

    config.pretrained.model = os.environ["DDPO_MODEL"]
    # None, not "main": a revision string sends diffusers looking for a git ref, and
    # the model here is a local directory with no VCS.
    config.pretrained.revision = None
    config.use_lora = True

    config.sample.num_steps = int(os.environ.get("SAMPLE_STEPS", "50"))
    config.sample.batch_size = int(os.environ.get("SAMPLE_BATCH_SIZE", "8"))
    config.sample.num_batches_per_epoch = int(os.environ.get("SAMPLES_PER_EPOCH", "4"))
    config.sample.guidance_scale = float(os.environ.get("GUIDANCE_SCALE", "5.0"))

    config.train.batch_size = int(os.environ.get("TRAIN_BATCH_SIZE", "4"))
    config.train.gradient_accumulation_steps = int(os.environ.get("GRAD_ACCUM", "4"))
    config.train.learning_rate = float(os.environ.get("LEARNING_RATE", "3e-4"))
    config.train.num_inner_epochs = 1
    # Upstream declares adv_clip_max as an int and clamps with it directly. Round
    # rather than truncate so 4.9 does not silently become 4.
    config.train.adv_clip_max = int(round(float(os.environ.get("ADV_CLIP_MAX", "5"))))
    config.train.clip_range = float(os.environ.get("PPO_CLIP_RANGE", "1e-4"))

    config.prompt_fn = "simple_animals"
    config.reward_fn = "aesthetic_score"
    if _flag("PER_PROMPT_STAT_TRACKING", "1"):
        tracking = get_base_config().per_prompt_stat_tracking
        tracking.buffer_size = int(os.environ.get("PER_PROMPT_BUFFER_SIZE", "32"))
        tracking.min_count = int(os.environ.get("PER_PROMPT_MIN_COUNT", "16"))
        config.per_prompt_stat_tracking = tracking
    else:
        config.per_prompt_stat_tracking = None
    return config
