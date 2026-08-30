#!/bin/bash
set -e

MODEL_DIR=${1:-"meta-llama/Llama-3.1-8B"}
TASK_NAME=${2:-"GSM-Hard"}
MAX_SAMPLES=${3:-100000000}
WANDB_KEY=${WANDB_KEY:-""}

HF_OFFLINE=0
if [[ -d "${MODEL_DIR}" ]]; then
  HF_OFFLINE=1
fi

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. && pwd)

DATE=$(date +%m%d)
ADVANTAGE="rloo"
SHORT_NAME="Llama-3.1-8B"
TASK="MAS"
ALGO="mas-dual-lora"
EXP="${DATE}-${TASK}-${SHORT_NAME}-${ADVANTAGE}-${ALGO}"

SAVE_PATH="${ROOT_DIR}/outputs/${ADVANTAGE}-${ALGO}/${DATE}/${SHORT_NAME}/model"

DATA_ROOT="${ROOT_DIR}/../puppeteer/data"
case "$(echo "${TASK_NAME}" | tr '[:upper:]' '[:lower:]')" in
  "gsm-hard"|"gsm8k"|"gsm")
    PROMPT_DATA="${DATA_ROOT}/GSM-Hard/train.jsonl"
    VERIFY_TASK="math"
    ;;
  "mmlu-pro"|"mmlu")
    PROMPT_DATA="${DATA_ROOT}/MMLU-Pro/train.jsonl"
    VERIFY_TASK="math"
    ;;
  "cw"|"creative_writing")
    PROMPT_DATA="${DATA_ROOT}/CW/creative_writing.jsonl"
    VERIFY_TASK="math"
    ;;
  "srdd")
    PROMPT_DATA="${DATA_ROOT}/SRDD/train.jsonl"
    VERIFY_TASK="math"
    ;;
  "scibench"|"sci"|"science")
    PROMPT_DATA="${DATA_ROOT}/scibench/train_marti.jsonl"
    VERIFY_TASK="scibench"
    ;;
  *)
    echo "[Error] Unknown TASK_NAME: ${TASK_NAME}. 支持：GSM-Hard / MMLU-Pro / CW / SRDD / SciBench"
    exit 1
    ;;
esac

# workflow max steps (can override by exporting MAS_MAX_STEPS)
MAS_MAX_STEPS=${MAS_MAX_STEPS:-12}

TENSORBOARD="${ROOT_DIR}/logs/tensorboard/${ADVANTAGE}-${ALGO}-${DATE}-${SHORT_NAME}"
CKPT_PATH="${ROOT_DIR}/outputs/${ADVANTAGE}-${ALGO}/${DATE}/${SHORT_NAME}/ckpt"

PROMPT_MAX_LEN=4096
GENERATE_MAX_LEN=64

export PYTORCH_NVML_BASED_CUDA_CHECK=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN

#ray start --head --port=6375 --dashboard-host=0.0.0.0 --dashboard-port=8266 --include-dashboard=true

ENV_JSON=$(cat <<EOF
{
  "working_dir": "${ROOT_DIR}",
  "excludes": ["data/", "outputs/", ".git/", "local/", "logs/"],
  "pip": ["hydra-core", "antlr4-python3-runtime==4.9.3", "shortuuid", "class_registry", "json5", "mcp[cli]"],
  "env_vars": {
    "PYTHONPATH": "${ROOT_DIR}/..:${ROOT_DIR}/../puppeteer:${ROOT_DIR}/puppeteer:${PYTHONPATH:-}",
    "CUDA_VISIBLE_DEVICES": "${CUDA_VISIBLE_DEVICES:-0,1}",
    "HF_HUB_OFFLINE": "${HF_OFFLINE}",
    "TRANSFORMERS_OFFLINE": "${HF_OFFLINE}"
  }
}
EOF
)

ray job submit --address="http://localhost:8265" \
    --runtime-env-json="${ENV_JSON}" \
    -- python -m marti.cli.commands.train --config-name "ma_dual_lora" \
    default_agent.pretrain="${MODEL_DIR}" \
    default_agent.save_path="${SAVE_PATH}" \
    default_agent.training_mode="rl" \
    default_agent.advantage_estimator=${ADVANTAGE} \
    default_agent.n_samples_per_prompt=16 \
    default_agent.prompt_max_len=${PROMPT_MAX_LEN} \
    default_agent.generate_max_len=${GENERATE_MAX_LEN} \
    default_agent.max_samples=${MAX_SAMPLES} \
    default_agent.num_episodes=2 \
    default_agent.bf16=True \
    default_agent.gradient_checkpointing=True \
    default_agent.actor_learning_rate=1e-5 \
    default_agent.critic_learning_rate=9e-6 \
    default_agent.init_kl_coef=0.001 \
    default_agent.use_kl_loss=True \
    default_agent.max_ckpt_num=1 \
    default_agent.save_steps=-1 \
    default_agent.ckpt_path="${CKPT_PATH}" \
    default_agent.lora_rank=16 \
    default_agent.lora_alpha=32 \
    default_agent.lora_dropout=0.05 \
    +default_agent.dual_lora=True \
    default_agent.load_in_4bit=False \
    default_agent.reward_clip_range="[-2,2]" \
    credit_ref_num_gpus_per_node=1 \
    credit_num_gpus_per_node=1 \
    default_agent.ref_num_gpus_per_node=1 \
    default_agent.reward_num_gpus_per_node=1 \
    default_agent.actor_num_gpus_per_node=1 \
    default_agent.critic_num_gpus_per_node=1 \
    default_agent.vllm_num_engines=1 \
    +default_agent.vllm_lora_server_url="${VLLM_LORA_SERVER_URL:-http://127.0.0.1:8000}" \
    +default_agent.vllm_lora_update_every=1 \
    +default_agent.vllm_lora_server_timeout=10.0 \
    +default_agent.vllm_lora_server_retries=2 \
    +default_agent.vllm_scheduler_lora_name="mas_scheduler" \
    +default_agent.vllm_router_lora_name="mas_router" \
    default_agent.vllm_gpu_memory_utilization=0.23 \
    default_agent.colocate_all_models=True \
    default_agent.colocate_actor_ref=True \
    default_agent.colocate_critic_reward=True \
    default_agent.rollout_batch_size=12 \
    default_agent.micro_rollout_batch_size=4 \
    default_agent.train_batch_size=8 \
    default_agent.micro_train_batch_size=2 \
    +default_agent.accumulation_steps=4 \
    +default_agent.progress_log_interval=10.0 \
    +workflow_args.personas_path="${ROOT_DIR}/../puppeteer/personas/personas.jsonl" \
    +workflow_args.global_config="${ROOT_DIR}/../puppeteer/config/global.yaml" \
    +workflow_args.top_m=7 \
    +workflow_args.top_n=3 \
    +workflow_args.task_type="${TASK_NAME}" \
    +workflow_args.router_alpha=0.25 \
    +workflow_args.router_eta=1.0 \
    +workflow_args.reward_clip=2.0 \
    +workflow_args.lambda_tok=0.00005 \
    workflow_args.max_steps=${MAS_MAX_STEPS} \
    +workflow_args.use_vllm_server=True \
    +workflow_args.vllm_server_url="${VLLM_LORA_SERVER_URL:-http://127.0.0.1:8000}" \
    +workflow_args.vllm_model="${MODEL_DIR}" \
    +workflow_args.vllm_scheduler_lora_name="mas_scheduler" \
    +workflow_args.vllm_router_lora_name="mas_router" \
    +workflow_args.vllm_server_timeout=60.0 \
    +workflow_args.train_start_step=3 \
    +workflow_args.interleave_by_step=True \
    +workflow_args.enable_posthoc_feedback=True \
    +workflow_args.enable_mmlu_prompt_injection=False \
    +workflow_args.enable_role_based_routing=False \
    +workflow_args.over_budget_penalty=0.0 \
    +workflow_func_path="marti/worlds/workflows/mas_dual_lora_workflow.py" \
    +processor_func_path="marti/worlds/workflows/default_processor.py" \
    eval_before_training=False \
    eval_only=False \
    eval_workers=-1 \
    mask_truncated_completions=True \
    shared_agents=False \
    packing_samples=True \
    prompt_data="${PROMPT_DATA}" \
    input_key="prompt" \
    label_key="answer" \
    verify_task="${VERIFY_TASK}" \
    verify_task_eval="${VERIFY_TASK}" \
    wandb_project="MARTI" \
    wandb_run_name="${EXP}" \
    use_wandb="${WANDB_KEY}"

echo "Training submitted. Check Ray dashboard (8265) and logs under ${ROOT_DIR}/logs."
echo "Ray shutdown when finished:"
echo "  ray stop"

