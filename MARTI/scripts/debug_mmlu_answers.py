#!/usr/bin/env python3
"""
调试MMLU-Pro答案提取问题
分析为什么模型总是输出"B"
"""

import os
import re
import glob
from collections import Counter

def find_latest_ray_session():
    """找到最新的Ray session目录"""
    sessions = glob.glob("/tmp/ray/session_*")
    if not sessions:
        return None
    sessions.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return sessions[0]

def analyze_mmlu_outputs(log_file):
    """分析MMLU输出"""
    
    answers_extracted = []
    raw_outputs = []
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        
        # 查找所有"Extracted clean answer"的模式
        extracted_pattern = r'\[MMLU\] Extracted clean answer: ([A-J]) from raw output'
        extracted_matches = re.findall(extracted_pattern, content)
        answers_extracted.extend(extracted_matches)
        
        # 查找"Ground Truth"
        gt_pattern = r'\[DEBUG\] Ground Truth: ([A-J])'
        gt_matches = re.findall(gt_pattern, content)
        
        # 查找Task Reward
        reward_pattern = r'★ Task Reward: ([-+]?\d*\.?\d+)'
        reward_matches = re.findall(reward_pattern, content)
        
    return {
        'extracted_answers': answers_extracted,
        'ground_truths': gt_matches,
        'rewards': [float(r) for r in reward_matches]
    }

def main():
    session_dir = find_latest_ray_session()
    
    if not session_dir:
        print("❌ 未找到Ray session")
        return
    
    print(f"📁 Ray Session: {session_dir}\n")
    
    # 查找所有worker日志
    log_pattern = os.path.join(session_dir, "logs", "worker-*.out")
    log_files = glob.glob(log_pattern)
    
    if not log_files:
        print("❌ 未找到worker日志")
        return
    
    print(f"📄 找到 {len(log_files)} 个日志文件\n")
    
    all_extracted = []
    all_gts = []
    all_rewards = []
    
    # 分析所有日志
    for log_file in log_files:
        try:
            data = analyze_mmlu_outputs(log_file)
            all_extracted.extend(data['extracted_answers'])
            all_gts.extend(data['ground_truths'])
            all_rewards.extend(data['rewards'])
        except Exception as e:
            continue
    
    print("="*60)
    print("📊 MMLU答案分析")
    print("="*60)
    print()
    
    if all_extracted:
        print(f"✅ 找到 {len(all_extracted)} 个提取的答案")
        print()
        
        # 统计答案分布
        answer_counts = Counter(all_extracted)
        print("📈 提取答案的分布:")
        for letter, count in sorted(answer_counts.items()):
            percentage = count / len(all_extracted) * 100
            bar = "█" * int(percentage / 2)
            print(f"  {letter}: {count:3d} ({percentage:5.1f}%) {bar}")
        print()
        
        # 如果有ground truth，计算准确率
        if all_gts and len(all_gts) == len(all_extracted):
            correct = sum(1 for e, g in zip(all_extracted, all_gts) if e == g)
            accuracy = correct / len(all_extracted) * 100
            print(f"🎯 准确率: {correct}/{len(all_extracted)} ({accuracy:.1f}%)")
            print()
    else:
        print("⚠️  未找到提取的答案数据")
        print()
    
    if all_rewards:
        print(f"💰 奖励统计 ({len(all_rewards)} 个样本):")
        avg_reward = sum(all_rewards) / len(all_rewards)
        positive = sum(1 for r in all_rewards if r > 0)
        negative = sum(1 for r in all_rewards if r < 0)
        zero = sum(1 for r in all_rewards if r == 0)
        
        print(f"  平均奖励: {avg_reward:.4f}")
        print(f"  分布: ✅ {positive} 正 | ❌ {negative} 负 | ⚪ {zero} 零")
        print()
    
    # 分析问题
    print("="*60)
    print("🔍 问题诊断")
    print("="*60)
    print()
    
    if all_extracted:
        # 检查是否所有答案都是同一个
        unique_answers = set(all_extracted)
        if len(unique_answers) == 1:
            print(f"⚠️  **所有答案都是 '{list(unique_answers)[0]}'**")
            print()
            print("可能的原因:")
            print("  1. 模型还在训练初期，尚未学会推理")
            print("  2. 答案提取逻辑有问题（总是提取到第一个字母）")
            print("  3. 模型输出格式不符合预期")
            print("  4. 提示词不够强，模型没理解任务")
            print()
            print("建议:")
            print("  - 继续训练，观察是否会改善")
            print("  - 检查模型的实际输出内容")
            print("  - 查看训练loss是否在下降")
        elif len(unique_answers) <= 3:
            print(f"⚠️  答案多样性很低，只有 {len(unique_answers)} 种: {unique_answers}")
            print()
            print("模型可能还在学习阶段，建议继续训练")
        else:
            print(f"✅ 答案多样性正常，有 {len(unique_answers)} 种不同答案")
    
    print()
    print("="*60)

if __name__ == "__main__":
    main()
