#!/bin/bash
# 实时监控训练奖励
# 用法: bash scripts/watch_rewards.sh

echo "=========================================="
echo "📊 实时监控MMLU-Pro训练奖励"
echo "=========================================="
echo ""
echo "提示: 按 Ctrl+C 停止监控"
echo ""

# 找到最新的Ray session
RAY_SESSION=$(ls -td /tmp/ray/session_* 2>/dev/null | head -1)

if [ -z "$RAY_SESSION" ]; then
    echo "❌ 未找到Ray session"
    exit 1
fi

echo "📁 Ray Session: $RAY_SESSION"
echo ""

# 监控所有包含reward关键词的日志
echo "🔍 监控训练奖励输出..."
echo "=========================================="
echo ""

# 实时tail所有worker日志，过滤reward相关信息
tail -f $RAY_SESSION/logs/worker-*.out 2>/dev/null | grep --line-buffered -i "reward\|episode\|global.*step\|accuracy\|correct" | while read line; do
    # 添加时间戳
    echo "[$(date '+%H:%M:%S')] $line"
done
