#!/bin/bash
set -x
ulimit -n 65535

# ---------- Ray Configuration ----------
export RAY_TMPDIR="${RAY_TMPDIR}"            # Move Ray's temporary directory to a place with sufficient inodes.
export RAY_LOG_TO_STDERR_ONLY=1              # Avoid writing massive log files
export RAY_OBJECT_STORE_MEMORY=1000000000    # Set the explicit size of the object store (can be increased according to machine memory).
export RAY_DISABLE_IMPORT_WARNING=1          # Clear output
export RAY_BACKEND_LOG_LEVEL=warning         # Lower the worker log level (strongly recommended)

# ---------- PyTorch / CUDA / vLLM Configuration ----------
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export NCCL_DEBUG=WARN
# export RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING=1  # To debug the execution details of an actor, you can enable

# ---------- Start Ray ----------
echo "[INFO] Starting Ray head node..."
ray start --head \
  --port=6379 \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265 \
  --include-dashboard=true \
  --temp-dir="${RAY_TMPDIR}"

MODEL_DIR=${1}
WANDB_KEY=${2}
ROOT_DIR=$(pwd)

DATE=$(date +%m%d)
ADVANTAGE="group_norm"
# SHORT_NAME="Qwen2.5-3B-Instruct"
SHORT_NAME="Qwen3-4B"

TASK="search_r1"
ALGO="tool-search-r1-async-ready-task"

PRETRAIN="${MODEL_DIR}/${SHORT_NAME}"
EXP="${DATE}-${TASK}-${SHORT_NAME}-${ADVANTAGE}-${ALGO}"

SAVE_PATH="${ROOT_DIR}/outputs/${ADVANTAGE}-${ALGO}/${DATE}/${SHORT_NAME}/model"

PROMPT_DATA="json@${ROOT_DIR}/local/${TASK}"
TENSORBOARD="${ROOT_DIR}/logs/tensorboard/${ADVANTAGE}-${ALGO}-${DATE}-${SHORT_NAME}"
CKPT_PATH="${ROOT_DIR}/outputs/${ADVANTAGE}-${ALGO}//${DATE}/${SHORT_NAME}/ckpt"

mkdir -p "${ROOT_DIR}/logs"
mkdir -p "${ROOT_DIR}/logs/std"
mkdir -p "${ROOT_DIR}/logs/tensorboard"
mkdir -p "${ROOT_DIR}/outputs"

PROMPT_MAX_LEN=16384
GENERATE_MAX_LEN=4096


ENV_JSON=$(cat <<EOF
{
  "working_dir": "${ROOT_DIR}",
  "excludes": ["/data/", "/outputs/", ".git/"],
  "pip": ["hydra-core", "antlr4-python3-runtime==4.9.3", "shortuuid", "class_registry", "json5"]
}
EOF
)

ray job submit --address="http://localhost:8265" \
    --runtime-env-json="${ENV_JSON}" \
    -- python -m marti.cli.commands.train --config-name "tool_search" \
    agent_workflow="tool" \
    default_agent.ref_num_nodes=1 \
    default_agent.pg_strategy="SPREAD" \
    default_agent.ref_num_gpus_per_node=8 \
    default_agent.critic_num_nodes=1 \
    default_agent.critic_num_gpus_per_node=8 \
    default_agent.actor_num_nodes=1 \
    default_agent.actor_num_gpus_per_node=8 \
    default_agent.vllm_num_engines=8 \
    default_agent.vllm_tensor_parallel_size=1 \
    default_agent.vllm_sync_backend="nccl" \
    default_agent.colocate_all_models=True \
    default_agent.vllm_enable_sleep=True \
    default_agent.deepspeed_enable_sleep=True \
    default_agent.vllm_gpu_memory_utilization=0.6 \
    default_agent.pretrain="${PRETRAIN}" \
    default_agent.save_path="${SAVE_PATH}" \
    default_agent.micro_train_batch_size=4 \
    default_agent.train_batch_size=512 \
    default_agent.num_episodes=1 \
    default_agent.save_steps=100 \
    default_agent.eval_steps=50 \
    default_agent.logging_steps=1 \
    default_agent.max_samples=1000000 \
    default_agent.micro_rollout_batch_size=4 \
    default_agent.rollout_batch_size=256 \
    default_agent.training_mode="rl" \
    default_agent.n_samples_per_prompt=4 \
    default_agent.max_epochs=1 \
    default_agent.prompt_max_len=${PROMPT_MAX_LEN} \
    default_agent.generate_max_len=${GENERATE_MAX_LEN} \
    default_agent.advantage_estimator=${ADVANTAGE} \
    default_agent.temperature=1.0 \
    default_agent.eval_temperature=1.0 \
    default_agent.lambd=1.0 \
    default_agent.gamma=1.0 \
    default_agent.zero_stage=3 \
    default_agent.bf16=True \
    default_agent.actor_learning_rate=1e-6 \
    default_agent.critic_learning_rate=9e-6 \
    default_agent.init_kl_coef=0.001 \
    default_agent.use_kl_loss=True \
    default_agent.max_ckpt_num=100 \
    default_agent.normalize_reward=True \
    default_agent.lr_warmup_steps=0.03 \
    default_agent.adam_offload=True \
    default_agent.gradient_checkpointing=True \
    default_agent.ckpt_path="${CKPT_PATH}" \
    tools_config.tools.search.base_url="http://127.0.0.1:8080/retrieve" \
    tools_config.max_turns=2 \
    mask_truncated_completions=True \
    eval_before_training=False \
    eval_workers=1 \
    verify_task="search_r1" \
    verify_task_eval="search_r1" \
    allow_label_list=True \
    separate_label_list="LABEL_SEP" \
    packing_samples=True \
    prompt_data="${PROMPT_DATA}" \
    input_key="problem" \
    label_key="answer" \
    add_prompt_suffix=null \
    apply_chat_template=False \
    wandb_project="MARTI-Dev" \
    wandb_run_name="${EXP}" \
    extra_eval_tasks=["nq","musique","bamboogle"] \
    extra_eval_dir="${ROOT_DIR}/local/bench" \
    use_tensorboard="${TENSORBOARD}" 2>&1 | tee "${ROOT_DIR}/logs/std/${DATE}-${EXP}.log"

  # use_wandb="${WANDB_KEY}" \

echo "Model Training Finished. Shutting down..."