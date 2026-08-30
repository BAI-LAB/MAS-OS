# MAS-OS Training & Evaluation

## Train

1) Start Ray head:

```bash
ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265
```

2) Start vLLM LoRA server (in another terminal):

```bash
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export VLLM_LORA_SERVER_URL="http://127.0.0.1:8000"

vllm serve /root/autodl-tmp/Llama-3.1-8B \
  --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --enable-lora \
  --max-lora-rank 16 \
  --max-loras 2
```

3) Run training:

```bash
bash MARTI/scripts/run_train_mas_dual_lora_grpo.sh meta-llama/Llama-3.1-8B gsm-hard
```

## Evaluate (LoRA, GSM-Hard)

This repo uses the provided evaluation wrapper:

```bash
bash MARTI/scripts/run_eval_mas_dual_lora.sh meta-llama/Llama-3.1-8B <CKPT_DIR> gsm-hard
```

Notes:
- You need a running vLLM OpenAI-compatible server before evaluation (see `MARTI/scripts/run_eval_mas_dual_lora.sh`).
