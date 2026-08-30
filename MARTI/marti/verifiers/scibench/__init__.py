"""
SciBench Verifier Module for MARTI Training.

SciBench 是一个科学问题数据集，包含物理、化学等领域的数值计算问题。
此模块提供训练时的奖励计算函数。
"""
from .scibench_reward import scibench_reward_fn, check_scibench

__all__ = ['scibench_reward_fn', 'check_scibench']

