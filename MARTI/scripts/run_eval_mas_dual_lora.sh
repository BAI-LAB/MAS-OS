#!/bin/bash
set -e

BASE_MODEL=${1:?"用法: $0 <BASE_MODEL> <CKPT_DIR> <TASK> [DATA_LIMIT] [MAX_STEPS] [RESUME_FROM]"}
CKPT_DIR=${2:?"用法: $0 <BASE_MODEL> <CKPT_DIR> <TASK> [DATA_LIMIT] [MAX_STEPS] [RESUME_FROM]"}
TASK=${3:?"用法: $0 <BASE_MODEL> <CKPT_DIR> <TASK> [DATA_LIMIT] [MAX_STEPS] [RESUME_FROM]"}
DATA_LIMIT=${4:-""}
MAX_STEPS=${5:-""}
RESUME_FROM=${6:-""}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MARTI_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
PUPPETEER_DIR="${MARTI_ROOT}/../puppeteer"

if [[ ! "${CKPT_DIR}" = /* ]]; then
    if [[ -d "${CKPT_DIR}" ]]; then
        CKPT_DIR="$(cd -- "${CKPT_DIR}" && pwd)"
    elif [[ -d "${MARTI_ROOT}/${CKPT_DIR}" ]]; then
        CKPT_DIR="$(cd -- "${MARTI_ROOT}/${CKPT_DIR}" && pwd)"
    fi
fi

SCHEDULER_LORA="${CKPT_DIR}/scheduler_lora"
ROUTER_LORA="${CKPT_DIR}/router_lora"

echo "============================================================"
echo "[Eval] MAS Dual LoRA Evaluation"
echo "============================================================"
echo "  Base Model:      ${BASE_MODEL}"
echo "  Scheduler LoRA:  ${SCHEDULER_LORA}"
echo "  Router LoRA:     ${ROUTER_LORA}"
echo "  Task:            ${TASK}"
echo "  Data Limit:      ${DATA_LIMIT:-'all'}"
echo "  Max Steps:       ${MAX_STEPS:-'default (12)'}"
echo "  Resume From:     ${RESUME_FROM:-'none (fresh start)'}"
echo "============================================================"

# 构建参数
EVAL_ARGS=(
    "${TASK}" test
    --base_model "${BASE_MODEL}"
    --scheduler_lora "${SCHEDULER_LORA}"
    --router_lora "${ROUTER_LORA}"
)

if [[ -n "${DATA_LIMIT}" ]]; then
    EVAL_ARGS+=(--data_limit "${DATA_LIMIT}")
fi

if [[ -n "${MAX_STEPS}" ]]; then
    EVAL_ARGS+=(--max_steps "${MAX_STEPS}")
fi

if [[ -n "${RESUME_FROM}" ]]; then
    EVAL_ARGS+=(--resume_from "${RESUME_FROM}")
fi

# 运行评测
cd "${PUPPETEER_DIR}"
python eval_mas_with_lora.py "${EVAL_ARGS[@]}"
