#!/bin/bash
# 监控MMLU-Pro训练进度和奖励

echo "=========================================="
echo "📊 MMLU-Pro 训练监控"
echo "=========================================="
echo ""

# 1. 检查训练checkpoint
echo "🔍 检查训练checkpoint..."
CKPT_DIR="outputs/rloo-mas-dual-lora/$(date +%m%d)/Llama-3.1-8B-Instruct/ckpt"

if [ -d "$CKPT_DIR" ]; then
    echo "✅ 找到checkpoint目录: $CKPT_DIR"
    echo ""
    
    # 列出所有global steps
    echo "📁 已保存的checkpoints:"
    for dir in "$CKPT_DIR"/_actor_scheduler/global_step*/; do
        if [ -d "$dir" ]; then
            step=$(basename $(dirname "$dir"))
            step_num=$(echo $step | grep -o '[0-9]*')
            size=$(du -sh "$dir" 2>/dev/null | cut -f1)
            echo "  - Global Step $step_num (大小: $size)"
        fi
    done
    echo ""
    
    for dir in "$CKPT_DIR"/_runtime_lora/step_*/; do
        if [ -d "$dir" ]; then
            step=$(basename "$dir")
            step_num=$(echo $step | grep -o '[0-9]*')
            echo "  - Runtime LoRA Step $step_num"
        fi
    done
else
    echo "⚠️  未找到checkpoint目录"
fi

echo ""
echo "=========================================="
echo "💡 查看实时训练日志的方法:"
echo "=========================================="
echo ""
echo "1. 重新连接到训练tmux session:"
echo "   tmux a -t train"
echo ""
echo "2. 在tmux中滚动查看历史日志:"
echo "   按 Ctrl+B 然后按 ["
echo "   使用方向键或PageUp/PageDown滚动"
echo "   按 q 退出滚动模式"
echo ""
echo "3. 查看Ray Dashboard (浏览器):"
echo "   http://localhost:8266"
echo ""
echo "4. 搜索日志中的关键信息:"
echo "   grep -r 'Task Reward\\|episode\\|global_step' /tmp/ray/session_*/logs/*.out | tail -50"
echo ""
echo "=========================================="
echo "📈 训练进度估算"
echo "=========================================="
echo ""

# 统计已完成的steps
if [ -d "$CKPT_DIR/_actor_scheduler" ]; then
    num_steps=$(ls -d "$CKPT_DIR"/_actor_scheduler/global_step*/ 2>/dev/null | wc -l)
    echo "✅ 已完成 $num_steps 个 global steps"
    
    # 估算训练进度（假设总共要训练的steps）
    # MMLU-Pro 200样本，每个episode处理所有样本，共3个episodes
    # 每个样本可能产生多个training steps
    echo ""
    echo "💡 提示: 训练配置为 3 episodes，每个episode处理200个样本"
fi

echo ""
echo "=========================================="
