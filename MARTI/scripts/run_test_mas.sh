#!/bin/bash
# set -x


MODEL_DIR=${1}
ROOT_DIR=$(pwd)

task=AMC
model=Qwen2.5-7B-Instruct

MODEL_PATH="${MODEL_DIR}/$model"

if [ -z "$1" ]; then
    cfg_list=("ma_chain" "ma_mad" "ma_moa")
    echo "No config provided. Using default config list: ${cfg_list[*]}"
else
    cfg_list=("$@")
    echo "Using provided config: ${cfg_list[*]}"
fi

for task in AIME AMC MATH
do
    for config in "${cfg_list[@]}"
    do
        DATA_PATH="json@${ROOT_DIR}/data/$task"
        SAVE_PATH="${ROOT_DIR}/results/$config/$model/$task"
        mkdir -p "${SAVE_PATH}"

        python -m marti.cli.commands.test_new --config-name "$config" \
            default_agent.pretrain="$MODEL_PATH" \
            default_agent.vllm_num_engines=1 \
            default_agent.prompt_max_len=8192 \
            default_agent.generate_max_len=8192 \
            default_agent.temperature=0.6 \
            default_agent.rollout_batch_size=64 \
            default_agent.micro_rollout_batch_size=1 \
            default_agent.save_path="$SAVE_PATH" \
            prompt_data="$DATA_PATH" \
            input_key="problem" \
            label_key="answer" \
            add_prompt_suffix="" \

    done
done
