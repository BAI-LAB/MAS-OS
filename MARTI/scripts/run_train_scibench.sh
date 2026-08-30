#!/bin/bash
# =============================================================================
# SciBench 训练脚本
# =============================================================================
#
# SciBench 数据集训练，使用 GRPO 算法。
# 
# 用法:
#   bash scripts/run_train_scibench.sh <MODEL_DIR> <WANDB_KEY>
#
# 示例:
#   bash scripts/run_train_scibench.sh /path/to/Qwen2.5-3B YOUR_WANDB_KEY
#
# 注意:
#   - 需要至少 8 张 80GB GPU
#   - 数据位于 data/SciBench/train_marti.jsonl
# =============================================================================

set -x
ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 --include-dashboard=true

MODEL_DIR=${1}
WANDB_KEY=${2}
ROOT_DIR=$(pwd)

DATE=$(date +%m%d)
ADVANTAGE="group_norm"
SHORT_NAME=$(basename ${MODEL_DIR})

TASK="SciBench"
ALGO="single-agent"

PRETRAIN="${MODEL_DIR}"
EXP="${DATE}-${TASK}-${SHORT_NAME}-${ADVANTAGE}-${ALGO}"

SAVE_PATH="${ROOT_DIR}/outputs/${ADVANTAGE}-${ALGO}/${DATE}/${SHORT_NAME}-${TASK}/model"

# 使用转换后的 MARTI 格式数据 (必须指定 train_marti.jsonl，而非目录)
PROMPT_DATA="json@${ROOT_DIR}/data/${TASK}/train_marti.jsonl"
TENSORBOARD="${ROOT_DIR}/logs/${ADVANTAGE}-${ALGO}-${DATE}-${SHORT_NAME}-${TASK}"
CKPT_PATH="${ROOT_DIR}/outputs/${ADVANTAGE}-${ALGO}/${DATE}/${SHORT_NAME}-${TASK}/ckpt"

mkdir -p "${ROOT_DIR}/logs"
mkdir -p "${ROOT_DIR}/outputs"

PROMPT_MAX_LEN=1024
GENERATE_MAX_LEN=2048

export PYTORCH_NVML_BASED_CUDA_CHECK=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=WARN

ENV_JSON=$(cat <<EOF
{
  "working_dir": "${ROOT_DIR}",
  "excludes": ["/data/", "/outputs/", ".git/"],
  "pip": ["hydra-core", "antlr4-python3-runtime==4.9.3", "shortuuid", "class_registry"]
}
EOF
)

ray job submit --address="http://localhost:8265" \
    --runtime-env-json="${ENV_JSON}" \
    -- python -m marti.cli.commands.train \
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
    default_agent.train_batch_size=128 \
    default_agent.num_episodes=10 \
    default_agent.save_steps=500 \
    default_agent.eval_steps=5 \
    default_agent.logging_steps=1 \
    default_agent.max_samples=100000 \
    default_agent.micro_rollout_batch_size=4 \
    default_agent.rollout_batch_size=128 \
    default_agent.training_mode="rl" \
    default_agent.n_samples_per_prompt=8 \
    default_agent.max_epochs=1 \
    default_agent.prompt_max_len=${PROMPT_MAX_LEN} \
    default_agent.generate_max_len=${GENERATE_MAX_LEN} \
    default_agent.advantage_estimator=${ADVANTAGE} \
    default_agent.temperature=1.0 \
    default_agent.eval_temperature=0.6 \
    default_agent.lambd=1.0 \
    default_agent.gamma=1.0 \
    default_agent.zero_stage=3 \
    default_agent.bf16=True \
    default_agent.actor_learning_rate=1e-6 \
    default_agent.critic_learning_rate=9e-6 \
    default_agent.init_kl_coef=0.00 \
    default_agent.use_kl_loss=True \
    default_agent.max_ckpt_num=50 \
    default_agent.normalize_reward=True \
    default_agent.adam_offload=True \
    default_agent.gradient_checkpointing=True \
    default_agent.ckpt_path="${CKPT_PATH}" \
    mask_truncated_completions=True \
    verify_task="scibench" \
    verify_task_eval="scibench" \
    packing_samples=True \
    prompt_data="${PROMPT_DATA}" \
    input_key="prompt" \
    label_key="answer" \
    metadata_key="metadata" \
    apply_chat_template=True \
    use_wandb="${WANDB_KEY}" \
    wandb_project="MARTI" \
    wandb_run_name="${EXP}" \
    use_tensorboard="${TENSORBOARD}" 2>&1 | tee "${ROOT_DIR}/logs/${DATE}-${EXP}.log"

echo "SciBench Training Finished. Shutting down..."

