"""
SciBench Reward Function for MARTI Training.

SciBench 数据集的训练奖励计算逻辑。
与 puppeteer/tasks/evaluator.py 中的 check_scibench 保持一致。

评测逻辑:
1. 将 gold 转为 float
2. 从 pred 文本中抽取数值，转为 float
3. 容差判断: tol = max(1e-3, abs(gold) * 1e-3)

满足任一条件即算正确:
- abs(pred - gold) <= tol (绝对+相对容差)
- round(pred) == round(gold) (四舍五入相等)
- 两者都大于 100 时 int(pred) == int(gold) (大数鲁棒)
"""
import re
import math
from typing import Union, Optional


def extract_number(text: str) -> Optional[float]:
    r"""
    从文本中抽取数值，支持科学计数法。
    
    支持格式:
    - 普通数值: 50.7, -1368, +65.49
    - 科学计数法: 2.88e-10, 2.88E10
    - LaTeX科学计数法: 2.88 \times 10^{-10}, 2.88 × 10^{-10}
    - boxed格式: \boxed{50.7}
    """
    if text is None:
        return None
    text = str(text)
    
    # 1. 先尝试匹配 LaTeX 科学计数法: a \times 10^{b} 或 a × 10^{b}
    latex_sci_pattern = r'(-?\d+\.?\d*)\s*(?:\\times|×)\s*10\^?\{?(-?\d+)\}?'
    latex_match = re.search(latex_sci_pattern, text)
    if latex_match:
        base = float(latex_match.group(1))
        exp = int(latex_match.group(2))
        return base * (10 ** exp)
    
    # 2. 尝试匹配标准科学计数法: aEb 或 aeb
    std_sci_pattern = r'-?\d+\.?\d*[eE][+-]?\d+'
    std_match = re.search(std_sci_pattern, text)
    if std_match:
        return float(std_match.group())
    
    # 3. 匹配普通数值 (带符号)
    num_pattern = r'[+-]?\d+\.\d+|[+-]?\d+'
    matches = re.findall(num_pattern, text)
    return float(matches[0]) if matches else None


def coerce_to_text(value) -> str:
    """
    将模型输出转换为纯文本字符串。
    
    MAS runner 可能返回:
    - 纯文本字符串
    - dict/tuple（某些工具链返回结构化信息）
    - 代码执行输出
    """
    if value is None:
        return ""
    
    # 如果是 tuple 或 list，取第一个元素
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return ""
        value = value[0]
    
    # 如果是 dict，尝试提取常见字段
    if isinstance(value, dict):
        for key in ['answer', 'final_answer', 'Answer', 'text', 'content', 'result', 'output']:
            if key in value and value[key] is not None:
                return str(value[key])
        return str(value)
    
    return str(value)


def check_scibench(pred: Union[str, float, dict, list], gold: Union[str, float]) -> bool:
    """
    SciBench 判分逻辑。
    
    Args:
        pred: 模型预测输出（可能是字符串、数值、dict 或 list）
        gold: 标准答案 (answer_number 字符串或数值)
    
    Returns:
        是否正确
    """
    if pred is None or gold is None:
        return False
    
    # 1. 将 gold 转为 float
    try:
        gold_num = float(str(gold).strip())
    except (ValueError, TypeError):
        return False
    
    # 2. 从预测文本中抽取数值
    pred_text = coerce_to_text(pred)
    pred_num = extract_number(pred_text)
    
    if pred_num is None:
        return False
    
    # 检查是否为有效数值
    if not (math.isfinite(pred_num) and math.isfinite(gold_num)):
        return False
    
    # 3. 容差判断
    # tol = max(1e-3, abs(gold) * 1e-3) - 绝对 + 相对容差
    tol = max(1e-3, abs(gold_num) * 1e-3)
    
    # 条件1: 绝对+相对容差
    if abs(pred_num - gold_num) <= tol:
        return True
    
    # 条件2: 四舍五入相等
    if round(pred_num) == round(gold_num):
        return True
    
    # 条件3: 大数鲁棒 - 两者都大于 100 时比较整数部分
    if abs(pred_num) > 100 and abs(gold_num) > 100:
        if int(pred_num) == int(gold_num):
            return True
    
    return False


def scibench_reward_fn(solution_str: str, ground_truth: Union[str, float]) -> float:
    """
    SciBench 训练奖励函数。
    
    用于 MARTI 训练时的奖励计算。
    
    Args:
        solution_str: 模型生成的完整输出
        ground_truth: 标准答案 (answer_number)
    
    Returns:
        奖励值: 1.0 (正确) 或 0.0 (错误)
    """
    is_correct = check_scibench(solution_str, ground_truth)
    return 1.0 if is_correct else 0.0


def scibench_batch_reward_fn(solutions: list, ground_truths: list) -> list:
    """
    批量计算 SciBench 奖励。
    
    Args:
        solutions: 模型输出列表
        ground_truths: 标准答案列表
    
    Returns:
        奖励值列表
    """
    assert len(solutions) == len(ground_truths), \
        f"Length mismatch: {len(solutions)} solutions vs {len(ground_truths)} ground_truths"
    
    rewards = []
    for sol, gt in zip(solutions, ground_truths):
        reward = scibench_reward_fn(sol, gt)
        rewards.append(reward)
    
    return rewards


# 为了兼容 MARTI 的 auto_verify 接口
def compute_score_scibench(solution: str, ground_truth: str) -> float:
    """
    兼容 MARTI auto_verify 接口的奖励函数。
    
    Args:
        solution: 模型输出
        ground_truth: 标准答案
    
    Returns:
        奖励值 (0.0 或 1.0)
    """
    return scibench_reward_fn(solution, ground_truth)

