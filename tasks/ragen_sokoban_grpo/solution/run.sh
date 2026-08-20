#!/usr/bin/env bash
# Multi-turn on-policy RL on Sokoban | vLLM rollout | FSDP training.
#
# Absorbed from the reference protocol's baseline/method/train.py + recipe.toml +
# method/hooks.toml. Those three files existed to let an Agent change nine named
# scalars and one hook while a checker refused everything else. There is no checker
# now, so the recipe is gone and its values are the defaults below.
#
# Every knob is an environment variable with a default. Change a default, export a
# different value, or append Hydra overrides as positional arguments -- the last
# wins, same as upstream.
#
# This script trains and then merges. RAGEN writes sharded FSDP state that nothing
# can load; `verl.model_merger` turns it into HF weights, and only merged weights
# score. So the merge is part of a run, not a step after it.

set -euo pipefail

# ---- paths ----
# /assets/models/policy is a read-only mount and the only weights in the container.
POLICY_MODEL=${POLICY_MODEL:-/assets/models/policy}
OUTPUT_DIR=${OUTPUT_DIR:-/out}
WORK_DIR=${WORK_DIR:-${OUTPUT_DIR}/work}
CKPT_DIR=${CKPT_DIR:-${OUTPUT_DIR}/checkpoints}
# The framework, editable. An edit here takes effect on the next run.
RAGEN_DIR=${RAGEN_DIR:-/workspace/ragen}

resolved_output=$(readlink -m "${OUTPUT_DIR}")
case "${resolved_output}" in
  /out|/out/*) ;;
  *)
    if [[ "${ALLOW_OUTPUT_OUTSIDE_OUT:-0}" != "1" ]]; then
      echo "run.sh: OUTPUT_DIR must be /out or below it, got '${OUTPUT_DIR}'." >&2
      echo "run.sh: formal receipts and merged checkpoints persist only below /out." >&2
      exit 78
    fi
    ;;
esac
OUTPUT_DIR=${resolved_output}
WORK_DIR=$(readlink -m "${WORK_DIR}")
CKPT_DIR=$(readlink -m "${CKPT_DIR}")
for path in "${WORK_DIR}" "${CKPT_DIR}"; do
  case "${path}" in
    /out|/out/*) ;;
    *)
      if [[ "${ALLOW_OUTPUT_OUTSIDE_OUT:-0}" != "1" ]]; then
        echo "run.sh: work/checkpoint paths must stay below /out, got '${path}'." >&2
        exit 78
      fi
      ;;
  esac
done

# Hold one lock across training and checkpoint merge. fast_eval.sh takes the same
# lock and fails quickly rather than contending for this process's GPU or Ray state.
PHASE_LOCK=${AI4AI_GPU_PHASE_LOCK:-/out/.ai4ai-gpu-phase.lock}
if [[ "${AI4AI_GPU_PHASE_LOCK_HELD:-0}" != "1" ]]; then
  export AI4AI_GPU_PHASE_LOCK_HELD=1
  exec python3 /opt/harness/gpu_phase_lock.py \
    --lock "${PHASE_LOCK}" \
    --label "ragen-train" \
    -- bash "$(readlink -f "${BASH_SOURCE[0]}")" "$@"
fi

# ---- algorithm ----
# GRPO with rollout filtering. Nothing here
# is an allowlist: adv_estimator takes whatever verl's registry has, and the loss
# itself is in ${RAGEN_DIR}. Dr.GRPO is NORMALIZE_ADVANTAGE=false.
ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}
NORMALIZE_ADVANTAGE=${NORMALIZE_ADVANTAGE:-true}
ACTOR_LR=${ACTOR_LR:-1e-6}
ENTROPY_COEFF=${ENTROPY_COEFF:-0.001}
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.28}
LOSS_AGG_MODE=${LOSS_AGG_MODE:-seq-mean-token-mean}
# The pinned upstream main-table GRPO recipe uses softmax top-p filtering over reward
# variance. Keep the filter semantics explicit so a framework default cannot silently
# move this control back to the older linear rule.
ROLLOUT_FILTER_STRATEGY=${ROLLOUT_FILTER_STRATEGY:-top_p}
ROLLOUT_FILTER_VALUE=${ROLLOUT_FILTER_VALUE:-0.9}
ROLLOUT_FILTER_TOP_P_PROB_MODE=${ROLLOUT_FILTER_TOP_P_PROB_MODE:-softmax}
ROLLOUT_FILTER_INCLUDE_ZERO=${ROLLOUT_FILTER_INCLUDE_ZERO:-true}
ROLLOUT_FILTER_TYPE=${ROLLOUT_FILTER_TYPE:-largest}
ROLLOUT_FILTER_METRIC=${ROLLOUT_FILTER_METRIC:-reward_variance}

# ---- shapes ----
TRAIN_ENV_GROUPS=${TRAIN_ENV_GROUPS:-8}
ROLLOUT_N=${ROLLOUT_N:-16}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-3600}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-400}
# Preserve the logical upstream batch with bounded single-device concurrency.
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.65}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-128}

# ---- schedule ----
# This single-device adaptation runs 80 updates and saves at an editable interval.
MAX_STEPS=${MAX_STEPS:-80}
SAVE_UNIT="${SAVE_UNIT:-step}"               # step
SAVE_INTERVAL="${SAVE_INTERVAL:-40}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"  # 0 means unlimited
[[ "${SAVE_UNIT}" == "step" ]] || { echo "RAGEN supports SAVE_UNIT=step" >&2; exit 78; }
[[ "${SAVE_INTERVAL}" =~ ^[1-9][0-9]*$ ]] || { echo "SAVE_INTERVAL must be positive" >&2; exit 78; }
[[ "${SAVE_TOTAL_LIMIT}" =~ ^[0-9]+$ ]] || { echo "SAVE_TOTAL_LIMIT must be nonnegative" >&2; exit 78; }
SAVE_FREQ=${SAVE_FREQ:-${SAVE_INTERVAL}}
# In-training validation, on the first proxy bank. Diagnostic only -- v1 takes the
# last checkpoint, so nothing selects on it. Keep it disabled to reserve the budget
# for on-policy updates.
TEST_FREQ=${TEST_FREQ:--1}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-false}
VAL_BOARDS=${VAL_BOARDS:-64}
VAL_ENVIRONMENT_SEED=${VAL_ENVIRONMENT_SEED:-4242}
VAL_ENGINE_SEED=${VAL_ENGINE_SEED:-0}
if (( SAVE_TOTAL_LIMIT == 0 )); then
  FRAMEWORK_CKPT_LIMIT=1000000
else
  FRAMEWORK_CKPT_LIMIT=${SAVE_TOTAL_LIMIT}
fi

# The retrain phase exports these; 0 disables the wall-clock stop. RAGEN's trainer
# has no deadline of its own -- OPD's has one because a patch added it -- so the stop
# is enforced from outside with `timeout`, and the reserve is what leaves room to
# merge. Without it a run killed at the container wall leaves sharded FSDP state and
# no loadable weights, which scores as nothing.
MAX_WALL_TIME_SECONDS=${MAX_WALL_TIME_SECONDS:-0}
DEADLINE_RESERVE_SECONDS=${DEADLINE_RESERVE_SECONDS:-1800}

SEED=${SEED:-10000}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-ragen_sokoban_grpo-${SEED}-$(date +%Y%m%d_%H%M%S)}

# W&B starts local services and unix sockets even offline. Keep telemetry
# process-free.
export WANDB_MODE=${WANDB_MODE:-disabled}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export PYTHONPATH=${RAGEN_DIR}:${RAGEN_DIR}/verl${PYTHONPATH:+:${PYTHONPATH}}
unset RAY_ADDRESS RAY_NAMESPACE RAY_JOB_ID RAY_RUNTIME_ENV_HOOK || true

# /tmp is a 256 MiB tmpfs. Ray's session directory plus the torch, Triton and vLLM
# caches will overrun it during a real run, so send them to the output mount.
export TMPDIR=${OUTPUT_DIR}/tmp
export RAY_TMPDIR=${OUTPUT_DIR}/tmp/ray
export TORCHINDUCTOR_CACHE_DIR=${OUTPUT_DIR}/tmp/inductor
export TRITON_CACHE_DIR=${OUTPUT_DIR}/tmp/triton
export VLLM_CACHE_ROOT=${OUTPUT_DIR}/tmp/vllm
export HF_HOME=${OUTPUT_DIR}/tmp/huggingface
mkdir -p "${TMPDIR}" "${RAY_TMPDIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" \
  "${VLLM_CACHE_ROOT}" "${HF_HOME}" "${WORK_DIR}" "${CKPT_DIR}"

# The policy is the one fixed input with a path. /assets is a read-only mount, but
# that only stops it being edited -- nothing stopped this line from pointing
# elsewhere. Refuse a source outside /assets. Set ALLOW_UNPINNED_POLICY=1 to
# override, which the orchestrator never does. This is a guard-rail you own and can
# delete; what actually fixes the policy is that there is nowhere else to load from.
case "${POLICY_MODEL}" in
  /assets/*) ;;
  *)
    if [[ "${ALLOW_UNPINNED_POLICY:-0}" != "1" ]]; then
      echo "run.sh: the policy must live under /assets, got '${POLICY_MODEL}'." >&2
      echo "run.sh: the model source and the evaluator are the fixed inputs;" >&2
      echo "run.sh: everything else about the method is yours." >&2
      exit 78
    fi
    echo "run.sh: WARNING policy outside /assets: ${POLICY_MODEL}" >&2
    ;;
esac

# Match the pinned upstream main-table GRPO recipe. The filtered batch may contain
# more than one mini-batch; it does not need to collapse to a single update batch.
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-16}
ENABLE_GRADIENT_CHECKPOINTING=${ENABLE_GRADIENT_CHECKPOINTING:-true}
ENTROPY_FROM_LOGITS_WITH_CHUNKING=${ENTROPY_FROM_LOGITS_WITH_CHUNKING:-true}
COLLAPSE_COMPUTE_FREQ=${COLLAPSE_COMPUTE_FREQ:-30}

########################### parameter arrays ###########################

MODEL=(
  "model_path=${POLICY_MODEL}"
  # The base image has no flash-attn build that matches this torch, and the
  # compatibility patch applied at build time changes RAGEN's own default to sdpa.
  # This override covers the path that reads the model config instead.
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa
  "actor_rollout_ref.model.enable_gradient_checkpointing=${ENABLE_GRADIENT_CHECKPOINTING}"
)

ALGORITHM=(
  "algorithm.adv_estimator=${ADV_ESTIMATOR}"
  "algorithm.norm_adv_by_std_in_grpo=${NORMALIZE_ADVANTAGE}"
  "actor_rollout_ref.actor.optim.lr=${ACTOR_LR}"
  "actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}"
  "actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}"
  "actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}"
  "actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE}"
  "actor_rollout_ref.actor.entropy_from_logits_with_chunking=${ENTROPY_FROM_LOGITS_WITH_CHUNKING}"
)

ROLLOUT=(
  "actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE}"
  "actor_rollout_ref.rollout.rollout_filter_strategy=${ROLLOUT_FILTER_STRATEGY}"
  "actor_rollout_ref.rollout.rollout_filter_value=${ROLLOUT_FILTER_VALUE}"
  "actor_rollout_ref.rollout.rollout_filter_top_p_prob_mode=${ROLLOUT_FILTER_TOP_P_PROB_MODE}"
  "actor_rollout_ref.rollout.rollout_filter_include_zero=${ROLLOUT_FILTER_INCLUDE_ZERO}"
  "actor_rollout_ref.rollout.rollout_filter_type=${ROLLOUT_FILTER_TYPE}"
  "actor_rollout_ref.rollout.rollout_filter_metric=${ROLLOUT_FILTER_METRIC}"
  "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}"
  "actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LENGTH}"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}"
  "actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}"
  "ctx_manager.generation.gen_config.response_length=${MAX_RESPONSE_LENGTH}"
)

ENVIRONMENTS=(
  "seed.train=${SEED}"
  "es_manager.train.env_groups=${TRAIN_ENV_GROUPS}"
  "es_manager.train.group_size=${ROLLOUT_N}"
  "es_manager.train.env_configs.n_groups=[${TRAIN_ENV_GROUPS}]"
  "ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
  "micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU}"
  # In-training validation. Same shape the evaluators use, so a number logged here
  # is comparable with one bank of fast_eval.
  "seed.val=${VAL_ENVIRONMENT_SEED}"
  "+sampling_seed=${VAL_ENGINE_SEED}"
  "es_manager.val.env_groups=${VAL_BOARDS}"
  es_manager.val.group_size=1
  "es_manager.val.env_configs.n_groups=[${VAL_BOARDS}]"
)

TRAINER=(
  "trainer.total_training_steps=${MAX_STEPS}"
  "trainer.save_freq=${SAVE_FREQ}"
  "trainer.test_freq=${TEST_FREQ}"
  "trainer.val_before_train=${VAL_BEFORE_TRAIN}"
  "trainer.max_actor_ckpt_to_keep=${FRAMEWORK_CKPT_LIMIT}"
  "trainer.max_critic_ckpt_to_keep=${FRAMEWORK_CKPT_LIMIT}"
  "trainer.default_local_dir=${WORK_DIR}/verl_checkpoints"
  "trainer.local_log_dir=${WORK_DIR}/ragen_results"
  "trainer.experiment_name=${EXPERIMENT_NAME}"
  "collapse_detection.compute_freq=${COLLAPSE_COMPUTE_FREQ}"
  'trainer.logger=[console]'
  "hydra.run.dir=${WORK_DIR}/hydra"
)

RAY=(
  # Ray's automatic node-IP discovery probes external DNS, and this container is
  # --network=none. Bind it to loopback instead of weakening the isolation.
  +ray_kwargs.ray_init._node_ip_address=127.0.0.1
  # Do not let Ray size its control plane from the host's full inventory. The
  # container has a bounded /dev/shm and a memory ceiling; the unconstrained
  # defaults terminated raylet during startup while the GPU work itself fit.
  +ray_kwargs.ray_init.num_cpus=32
  +ray_kwargs.ray_init.object_store_memory=1073741824
  +ray_kwargs.ray_init.include_dashboard=false
)

########################### launch ###########################

repro_dir="${OUTPUT_DIR}/repro"
mkdir -p "${repro_dir}"
cp "$(readlink -f "${BASH_SOURCE[0]}")" "${repro_dir}/run.sh"
{
  printf '%q ' "$(readlink -f "${BASH_SOURCE[0]}")" "$@"
  printf '\n'
} > "${repro_dir}/command.txt"
(env | grep -v -E '(^|_)(API_)?KEY=|TOKEN=|PASSWORD=|PASS=|SECRET=|CREDENTIAL=|COOKIE=' | sort || true) \
  > "${repro_dir}/env.txt"

# The wall-clock stop. `timeout` sends SIGTERM with DEADLINE_RESERVE_SECONDS left,
# so training ends with the last complete save on disk and time to merge it. 124 is
# coreutils' timeout status and is expected, not a failure.
train_budget=0
if [[ "${MAX_WALL_TIME_SECONDS}" != "0" ]]; then
  train_budget=$(( MAX_WALL_TIME_SECONDS - DEADLINE_RESERVE_SECONDS ))
  if (( train_budget < 60 )); then
    echo "run.sh: MAX_WALL_TIME_SECONDS ${MAX_WALL_TIME_SECONDS} leaves no room past" >&2
    echo "run.sh: the ${DEADLINE_RESERVE_SECONDS}s merge reserve" >&2
    exit 78
  fi
fi

train_command=(
  python3 "${RAGEN_DIR}/train.py"
  --config-name _2_sokoban
  "system.CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"
  "${MODEL[@]}"
  "${ALGORITHM[@]}"
  "${ROLLOUT[@]}"
  "${ENVIRONMENTS[@]}"
  "${TRAINER[@]}"
  "${RAY[@]}"
  "$@"
)

set -x
status=0
if (( train_budget > 0 )); then
  # --kill-after gives the trainer a minute to die on SIGTERM before SIGKILL, so a
  # hung worker cannot eat the merge reserve.
  timeout --signal=TERM --kill-after=60 "${train_budget}" \
    "${train_command[@]}" 2>&1 | tee "${WORK_DIR}/train.log" || status=$?
else
  "${train_command[@]}" 2>&1 | tee "${WORK_DIR}/train.log" || status=$?
fi
set +x

# Ray/vLLM children can survive their Python parent. This container owns the GPU
# phase lock, so a container-wide Ray stop cannot interrupt another valid phase.
service_cleanup_log="${OUTPUT_DIR}/service-cleanup-train.log"
cleanup_status=0
if command -v ray >/dev/null 2>&1; then
  timeout --signal=TERM --kill-after=10 60 ray stop --force \
    >"${service_cleanup_log}" 2>&1 || cleanup_status=$?
else
  printf '%s\n' "ray executable not found; training process cleanup only" \
    >"${service_cleanup_log}"
fi
printf 'cleanup_status=%s\n' "${cleanup_status}" >>"${service_cleanup_log}"

if (( status == 124 )); then
  echo "run.sh: training hit its ${train_budget}s wall; merging the last complete save"
elif (( status != 0 )); then
  echo "run.sh: training failed with exit ${status}; see ${WORK_DIR}/train.log" >&2
  exit "${status}"
fi

# Merge and finalize. Nothing scores an unmerged FSDP checkpoint.
python3 /workspace/finalize.py \
  --work "${WORK_DIR}" \
  --checkpoints "${CKPT_DIR}" \
  --policy "${POLICY_MODEL}" \
  --ragen "${RAGEN_DIR}" \
  --keep "${SAVE_TOTAL_LIMIT}" \
  --train-log "${WORK_DIR}/train.log" \
  --train-status "${status}"
