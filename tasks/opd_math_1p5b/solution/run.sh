#!/usr/bin/env bash
# On-policy distillation | text | vLLM rollout | FSDP training.
#
# Absorbed from thunlp/OPD@4532fd35 verl_example/opd.sh. That launcher is not a
# dependency: it is pure configuration, and its
# `reward.custom_reward_function.path=verl/recipe/r1_ascend/deepscaler.py`
# override points at a file that exists in neither verl copy. The values below
# are the *effective* ones -- opd.sh's arrays merged with the overrides the old
# trusted runner appended after them, since `"$@"` came last and won.
#
# Every knob is an environment variable with a default. Change a default, export
# a different value, or append Hydra overrides as positional arguments -- the
# last wins, same as upstream.

set -euo pipefail

# ---- paths (read-only mounts) ----
STUDENT_MODEL=${STUDENT_MODEL:-/assets/models/student}
TEACHER_MODEL=${TEACHER_MODEL:-/assets/models/teacher}
TRAIN_DATA=${TRAIN_DATA:-/assets/data/train.parquet}
# test_freq is -1, so val_files is never read. Point it at the train file rather
# than an eval set so no evaluation data is reachable from the training config.
VAL_DATA=${VAL_DATA:-${TRAIN_DATA}}
OUTPUT_DIR=${OUTPUT_DIR:-/out}
CKPT_DIR=${CKPT_DIR:-${OUTPUT_DIR}/work/opd-checkpoints}

# ---- algorithm ----
DISTILLATION_LOSS_MODE=${DISTILLATION_LOSS_MODE:-k1}
USE_POLICY_GRADIENT=${USE_POLICY_GRADIENT:-True}
USE_TASK_REWARDS=${USE_TASK_REWARDS:-False}
DISTILLATION_TOPK=${DISTILLATION_TOPK:-16}
LOSS_MAX_CLAMP=${LOSS_MAX_CLAMP:-10.0}
LOG_PROB_MIN_CLAMP=${LOG_PROB_MIN_CLAMP:--10.0}
ACTOR_LR=${ACTOR_LR:-1e-6}

# ---- shapes ----
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-7168}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}
ROLLOUT_N=${ROLLOUT_N:-4}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}

# ---- single-GPU split ----
# One device holds the actor, the vLLM rollout engine and the teacher engine.
# The teacher sleeps while the actor trains; see single_gpu_wall_clock.patch.
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-1}
ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.20}
TEACHER_TP=${TEACHER_TP:-1}
TEACHER_GPU_MEM_UTIL=${TEACHER_GPU_MEM_UTIL:-0.16}
COLOCATE_TEACHER=${COLOCATE_TEACHER:-true}

# ---- schedule ----
# TOTAL_TRAINING_STEPS deliberately exceeds the wall-clock horizon.
# MAX_WALL_TIME_SECONDS is what actually stops
# training -- it ends one step early so the last checkpoint is whole -- so too many
# steps costs nothing while too few silently forfeits the budget.
#
# Overshooting is only free because the LR schedule does not depend on this number.
# VERL pushes total_training_steps into
# actor.optim.total_training_steps, which would shape a decay -- but
# lr_scheduler_type defaults to "constant" and lr_warmup_steps_ratio to 0.0. If you switch
# to a cosine schedule, this number starts setting the decay horizon and a
# wall-clock stop will leave the LR partway down.
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-2200}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-999}
SAVE_UNIT="${SAVE_UNIT:-step}"               # step
SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"  # 0 means unlimited
[[ "${SAVE_UNIT}" == "step" ]] || { echo "OPD supports SAVE_UNIT=step" >&2; exit 78; }
[[ "${SAVE_INTERVAL}" =~ ^[1-9][0-9]*$ ]] || { echo "SAVE_INTERVAL must be positive" >&2; exit 78; }
[[ "${SAVE_TOTAL_LIMIT}" =~ ^[0-9]+$ ]] || { echo "SAVE_TOTAL_LIMIT must be nonnegative" >&2; exit 78; }
SAVE_FREQ=${SAVE_FREQ:-${SAVE_INTERVAL}}
if (( SAVE_TOTAL_LIMIT == 0 )); then
  FRAMEWORK_SAVE_LIMIT=1000000
else
  FRAMEWORK_SAVE_LIMIT=${SAVE_TOTAL_LIMIT}
fi
TEST_FREQ=${TEST_FREQ:--1}
# Stop one step early rather than being killed mid-write, so the last
# checkpoint is always complete. 0 disables the wall-clock stop.
MAX_WALL_TIME_SECONDS=${MAX_WALL_TIME_SECONDS:-0}
DEADLINE_RESERVE_SECONDS=${DEADLINE_RESERVE_SECONDS:-1200}

PROJECT_NAME=${PROJECT_NAME:-opd_math_1p5b}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-$(date +%Y%m%d_%H%M%S)}

# W&B and SwanLab both start local services and create unix sockets even when
# offline. Keep telemetry process-free and use VERL's synchronous file logger.
export WANDB_MODE=${WANDB_MODE:-disabled}
export SWANLAB_MODE=${SWANLAB_MODE:-disabled}
# OPD imports its CUDA stack before Ray creates the zero-GPU control actor. Ray
# normally strips CUDA_VISIBLE_DEVICES from that actor, which leaves the already
# imported stack inconsistent. Preserve it for zero-GPU actors only; GPU workers
# still get Ray's normal device isolation.
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=${RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO:-0}

# /tmp is a 256 MiB tmpfs. Ray's session directory, plus torch and Triton
# caches, will overrun it during a real run, so send them to the output mount.
#
# HOME is /tmp in this image, so every
# library that defaults to ~/.cache/<name> writes into the tmpfs. Naming one variable
# per library is incomplete, so redirect HOME and XDG_CACHE_HOME as well as the
# specific variables below.
export HOME=${HOME_OVERRIDE:-${OUTPUT_DIR}/tmp/home}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${OUTPUT_DIR}/tmp/cache}
export TMPDIR=${TMPDIR:-${OUTPUT_DIR}/tmp}
export RAY_TMPDIR=${RAY_TMPDIR:-${OUTPUT_DIR}/tmp/ray}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-${OUTPUT_DIR}/tmp/inductor}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${OUTPUT_DIR}/tmp/triton}
export FLASHINFER_CACHE_DIR=${FLASHINFER_CACHE_DIR:-${OUTPUT_DIR}/tmp/flashinfer}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-${OUTPUT_DIR}/tmp/flashinfer}
# verl's FileLogger defaults to the working directory. Keep metrics beside the
# checkpoints so runtime logs never enter candidate.patch.
export VERL_FILE_LOGGER_ROOT=${VERL_FILE_LOGGER_ROOT:-${OUTPUT_DIR}/metrics}
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${TMPDIR}" "${RAY_TMPDIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${FLASHINFER_CACHE_DIR}" \
  "${VERL_FILE_LOGGER_ROOT}"

# Let `trainer.default_local_dir=` on the command line win over CKPT_DIR, so the
# repro directory below lands next to the checkpoints it describes.
for arg in "$@"; do
  case "$arg" in
    trainer.default_local_dir=*) CKPT_DIR=${arg#trainer.default_local_dir=} ;;
  esac
done

# A probe owns its output and checkpoint trees, but shares the assigned GPU with
# other training probes. The shared GPU lock lets those probes coexist while the
# evaluator takes the same lock exclusively. Per-directory locks fail fast on the
# destructive case: two trainers writing the same Hydra state or checkpoint.
GPU_WORKLOAD_LOCK=${GPU_WORKLOAD_LOCK:-/out/.gpu-workload.lock}
mkdir -p "${OUTPUT_DIR}" "${CKPT_DIR}"
exec 9>"${GPU_WORKLOAD_LOCK}"
if ! flock -n -s 9; then
  echo "run.sh: evaluation is using the assigned GPU; stop it before training." >&2
  exit 75
fi
exec 8>"${OUTPUT_DIR}/.training.lock"
if ! flock -n -x 8; then
  echo "run.sh: another training process owns OUTPUT_DIR=${OUTPUT_DIR}." >&2
  exit 75
fi
exec 7>"${CKPT_DIR}/.training.lock"
if ! flock -n -x 7; then
  echo "run.sh: another training process owns CKPT_DIR=${CKPT_DIR}." >&2
  exit 75
fi

# The training data is one of the three fixed inputs. /assets is a read-only
# mount, but that only stops it being edited -- nothing stopped this script from
# being pointed somewhere else, and a parquet written under /workspace would ride
# into the retrain container inside candidate.patch, which is generated with
# --binary. So refuse a source outside /assets.
#
# Set ALLOW_UNPINNED_TRAIN_DATA=1 to override, which the orchestrator never does.
for _path in "${TRAIN_DATA}" "${VAL_DATA}"; do
  case "${_path}" in
    /assets/*) ;;
    *)
      if [[ "${ALLOW_UNPINNED_TRAIN_DATA:-0}" != "1" ]]; then
        echo "run.sh: training data must live under /assets, got '${_path}'." >&2
        echo "run.sh: the model, the data and the evaluator are the three fixed" >&2
        echo "run.sh: inputs; everything else about the method is yours." >&2
        exit 78
      fi
      echo "run.sh: WARNING training data outside /assets: ${_path}" >&2
      ;;
  esac
done

max_num_tokens=$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1 ))
train_files="['${TRAIN_DATA}']"
val_files="['${VAL_DATA}']"

########################### parameter arrays ###########################

DATA=(
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=False
  data.train_files="${train_files}"
  data.val_files="${val_files}"
  data.train_batch_size="${TRAIN_BATCH_SIZE}"
  data.max_prompt_length="${MAX_PROMPT_LENGTH}"
  data.max_response_length="${MAX_RESPONSE_LENGTH}"
  data.filter_overlong_prompts=True
  data.truncation=error
  data.shuffle=False
  +data.apply_chat_template_kwargs.enable_thinking=False
)

MODEL=(
  actor_rollout_ref.model.path="${STUDENT_MODEL}"
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
  # Upstream opd.sh compiles. Compilation is disabled here because the teacher
  # and the actor share one device and recompile on every wake/sleep cycle.
  actor_rollout_ref.actor.use_torch_compile=False
  actor_rollout_ref.actor.fsdp_config.use_torch_compile=False
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}"
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}"
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
  actor_rollout_ref.actor.fsdp_config.param_offload=True
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
  actor_rollout_ref.actor.checkpoint.save_contents='["model","optimizer","extra","hf_model"]'
)

ROLLOUT=(
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}"
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}"
  actor_rollout_ref.rollout.n="${ROLLOUT_N}"
  actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}"
  actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.rollout.top_k=-1
  actor_rollout_ref.rollout.max_model_len="${max_num_tokens}"
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
  actor_rollout_ref.rollout.agent.num_workers=8
  # B300 can JIT the supported FlashInfer kernels locally. Disable vLLM's
  # TRTLLM-attention auto-detect so initialization does not probe NVIDIA's
  # external cubin repository before choosing the local path.
  +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.use_trtllm_attention=False
)

TRAINER=(
  trainer.balance_batch=True
  trainer.logger='["console","file"]'
  trainer.project_name="${PROJECT_NAME}"
  trainer.experiment_name="${EXPERIMENT_NAME}"
  trainer.n_gpus_per_node="${NGPUS_PER_NODE}"
  trainer.nnodes="${NNODES}"
  trainer.val_before_train=False
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}"
  trainer.total_epochs="${TOTAL_EPOCHS}"
  trainer.save_freq="${SAVE_FREQ}"
  trainer.test_freq="${TEST_FREQ}"
  # Producer-side retention is editable and distinct from the harness acceptance limit.
  trainer.max_actor_ckpt_to_keep="${FRAMEWORK_SAVE_LIMIT}"
  trainer.max_critic_ckpt_to_keep="${FRAMEWORK_SAVE_LIMIT}"
  trainer.default_local_dir="${CKPT_DIR}"
  trainer.rollout_data_dir="${OUTPUT_DIR}/rollouts"
  hydra.run.dir="${OUTPUT_DIR}/hydra"
  +trainer.colocate_teacher="${COLOCATE_TEACHER}"
  # main_ppo calls ray.init() with this table. A private temp directory and no
  # fixed head/dashboard ports give every concurrent probe its own local cluster.
  +ray_kwargs.ray_init._temp_dir="${RAY_TMPDIR}"
  +ray_kwargs.ray_init.include_dashboard=False
  +ray_kwargs.ray_init._node_ip_address=127.0.0.1
)

EXTRA=(
  distillation.enabled=True
  distillation.n_gpus_per_node="${TEACHER_WORLD_SIZE}"
  distillation.nnodes="${NNODES}"
  distillation.teacher_models.teacher_model.model_path="${TEACHER_MODEL}"
  distillation.teacher_models.teacher_model.inference.name=vllm
  distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size="${TEACHER_TP}"
  distillation.teacher_models.teacher_model.inference.gpu_memory_utilization="${TEACHER_GPU_MEM_UTIL}"
  distillation.teacher_models.teacher_model.inference.max_model_len="${max_num_tokens}"
  +distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.attention_config.use_trtllm_attention=False
  distillation.distillation_loss.loss_mode="${DISTILLATION_LOSS_MODE}"
  distillation.distillation_loss.topk="${DISTILLATION_TOPK}"
  distillation.distillation_loss.use_policy_gradient="${USE_POLICY_GRADIENT}"
  distillation.distillation_loss.use_task_rewards="${USE_TASK_REWARDS}"
  distillation.distillation_loss.loss_max_clamp="${LOSS_MAX_CLAMP}"
  distillation.distillation_loss.log_prob_min_clamp="${LOG_PROB_MIN_CLAMP}"
  # The distillation objective supplies the advantage. There is no task reward,
  # so no reward function is loaded. Upstream opd.sh names a deepscaler path
  # that exists in neither verl tree; null is the value that actually ran.
  reward.custom_reward_function.path=null
)

if [[ "${MAX_WALL_TIME_SECONDS}" != "0" ]]; then
  TRAINER+=(
    +trainer.max_wall_time_seconds="${MAX_WALL_TIME_SECONDS}"
    +trainer.deadline_reserve_seconds="${DEADLINE_RESERVE_SECONDS}"
  )
fi

########################### launch ###########################

# Do not attach to a Ray head inherited from the shell. main_ppo starts a private
# local cluster from ray_kwargs above; process exit tears down only that cluster.
unset RAY_ADDRESS

repro_dir="${CKPT_DIR}/repro"
mkdir -p "${repro_dir}" "${OUTPUT_DIR}/rollouts"
cp "$(readlink -f "${BASH_SOURCE[0]}")" "${repro_dir}/run.sh"
{
  printf '%q ' "$(readlink -f "${BASH_SOURCE[0]}")" "$@"
  printf '\n'
} > "${repro_dir}/command.txt"
{
  printf 'project_name=%s\n' "${PROJECT_NAME}"
  printf 'experiment_name=%s\n' "${EXPERIMENT_NAME}"
  printf 'ckpt_dir=%s\n' "${CKPT_DIR}"
  printf 'student_model=%s\n' "${STUDENT_MODEL}"
  printf 'teacher_model=%s\n' "${TEACHER_MODEL}"
  printf 'train_data=%s\n' "${TRAIN_DATA}"
  printf 'loss_mode=%s\n' "${DISTILLATION_LOSS_MODE}"
  printf 'total_training_steps=%s\n' "${TOTAL_TRAINING_STEPS}"
} > "${repro_dir}/run_metadata.txt"
(env | grep -v -E '(^|_)(API_)?KEY=|TOKEN=|PASSWORD=|PASS=|SECRET=|CREDENTIAL=|COOKIE=' | sort || true) \
  > "${repro_dir}/env.txt"

set -x
TRAIN_PID=""
terminate_training() {
  if [[ -n "${TRAIN_PID}" ]]; then
    kill -TERM -- "-${TRAIN_PID}" 2>/dev/null || kill -TERM "${TRAIN_PID}" 2>/dev/null || true
    wait "${TRAIN_PID}" 2>/dev/null || true
  fi
  exit 143
}
trap terminate_training TERM INT HUP
setsid python3 -m verl.trainer.main_ppo \
  "${DATA[@]}" \
  "${MODEL[@]}" \
  "${ACTOR[@]}" \
  "${ROLLOUT[@]}" \
  "${TRAINER[@]}" \
  "${EXTRA[@]}" \
  "$@" &
TRAIN_PID=$!
set +e
wait "${TRAIN_PID}"
train_status=$?
set -e
trap - TERM INT HUP
TRAIN_PID=""
if (( train_status != 0 )); then
  exit "${train_status}"
fi
set +x

# Only complete actor/Hugging Face exports are public candidates.  The optimizer,
# FSDP and other trainer state stays below /out/work and is never offered to scoring.
found=0
while IFS=$'\t' read -r progress candidate; do
  [[ -n "${progress}" ]] || continue
  python3 /opt/harness/save_checkpoint.py \
    --output "${OUTPUT_DIR}" \
    --progress "${progress}" \
    --source "${candidate}" \
    --payload-name . \
    --retention "${SAVE_TOTAL_LIMIT}"
  found=1
done < <(
  for candidate in "${CKPT_DIR}"/global_step_*/actor/huggingface; do
    [[ -d "${candidate}" ]] || continue
    name=$(basename "$(dirname "$(dirname "${candidate}")")")
    progress=${name#global_step_}
    [[ "${progress}" =~ ^[0-9]+$ ]] || continue
    printf '%s\t%s\n' "${progress}" "${candidate}"
  done | sort -n -k1,1
)
if (( found == 0 )); then
  echo "run.sh: OPD produced no complete actor/Hugging Face export" >&2
  exit 1
fi
