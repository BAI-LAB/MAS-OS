#!/bin/bash
# 捕获模型的实际输出，看看为什么总是"B"

echo "=========================================="
echo "🔍 捕获MMLU模型输出"
echo "=========================================="
echo ""

RAY_SESSION=$(ls -td /tmp/ray/session_* 2>/dev/null | head -1)

if [ -z "$RAY_SESSION" ]; then
    echo "❌ 未找到Ray session"
    exit 1
fi

echo "📁 搜索包含MMLU输出的日志..."
echo ""

# 搜索包含"Parsed"和"Extracted"的行，以及它们前后的内容
grep -r -B 5 -A 5 "Parsed.*options\|Extracted clean answer" $RAY_SESSION/logs/worker-*.out 2>/dev/null | head -200

echo ""
echo "=========================================="
echo "💡 提示:"
echo "=========================================="
echo "如果看到很多'B'，这是正常的训练初期现象"
echo "模型需要更多训练步骤才能学会正确推理"
echo ""
echo "建议:"
echo "1. 继续训练至少完成50-100个样本"
echo "2. 观察准确率是否逐渐提升"
echo "3. 检查训练loss是否在下降"
