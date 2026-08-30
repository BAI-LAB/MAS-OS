#!/bin/bash
# 实时监控训练过程中的Scheduler和Router reward

echo "🔍 监控训练 reward 输出..."
echo "按 Ctrl+C 停止监控"
echo ""

# 找到最新的Ray session
SESSION=$(ls -td /tmp/ray/session_* 2>/dev/null | head -1)

if [ -z "$SESSION" ]; then
    echo "❌ 未找到Ray session"
    exit 1
fi

echo "📁 Ray Session: $SESSION"
echo ""

# 监控所有worker日志文件中的reward相关信息
tail -f $SESSION/logs/worker-*.out 2>/dev/null | grep --line-buffered -E "(Scheduler reward|Router reward|Global step|episode_reward|scheduler_mean_reward|router_mean_reward|REWARD|Step [0-9]+/[0-9]+)"
