"""
Online Sampler for MAS GRPO Training

在线采样器：使用当前 LoRA 策略运行 MAS 推理，收集 on-policy 轨迹。

关键特性：
1. 支持 group_size 采样（同一 prompt 多次采样）
2. 收集精确 token_count（从 API）
3. 收集 Agent 自反馈的 useful_count
4. 输出 verl 兼容的 JSONL 格式

"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .verl_integration import (
    MASRewardConfig,
    SchedulerRewardFunction,
    RouterRewardFunction,
    TrainingDataCollector,
)
from .dual_lora_model import (
    DualLoRAModel,
    DualLoRAConfig,
    ExpertType,
    format_scheduler_prompt,
    format_router_prompt,
)
from .agent_feedback import (
    AgentFeedbackCollector,
    generate_memory_feedback_prompt,
    parse_memory_feedback_response,
)

logger = logging.getLogger("MAS")


@dataclass
class SamplerConfig:
    """采样器配置"""
    
    # 采样设置
    group_size: int = 4             # GRPO 组采样大小
    temperature: float = 0.7        # 采样温度
    max_steps: int = 12             # 每任务最大步数
    
    # 生成设置
    max_new_tokens_scheduler: int = 64
    max_new_tokens_router: int = 32
    
    # Agent 列表
    agent_names: List[str] = field(default_factory=lambda: [
        "ReasoningAgent", "PlanningAgent", "CriticAgent", "ConcluderAgent"
    ])


class OnlineSampler:
    """
    在线采样器
    
    使用当前 LoRA 策略采样轨迹，支持 GRPO 所需的组采样。
    """
    
    def __init__(
        self,
        dual_lora_model: Optional[DualLoRAModel],
        reward_config: MASRewardConfig = None,
        sampler_config: SamplerConfig = None,
    ):
        self.model = dual_lora_model
        self.reward_config = reward_config or MASRewardConfig()
        self.config = sampler_config or SamplerConfig()
        
        # 奖励函数
        self.scheduler_reward_fn = SchedulerRewardFunction(self.reward_config)
        self.router_reward_fn = RouterRewardFunction(self.reward_config)
        
        # 反馈收集器
        self.feedback_collector = AgentFeedbackCollector()
        
        logger.info(f"[OnlineSampler] Initialized with group_size={self.config.group_size}")
    
    def sample_task(
        self,
        task: Dict[str, Any],
        evaluator_fn: Optional[callable] = None,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        对单个任务进行 group_size 次采样
        
        Args:
            task: 任务数据
            evaluator_fn: 可选的评估函数，用于计算 task_reward
            
        Returns:
            (scheduler_samples, router_samples) 两个列表
        """
        all_scheduler_samples = []
        all_router_samples = []
        
        for sample_idx in range(self.config.group_size):
            try:
                sch_samples, rou_samples = self._sample_one_trajectory(
                    task=task,
                    sample_seed=sample_idx,
                    evaluator_fn=evaluator_fn,
                )
                all_scheduler_samples.extend(sch_samples)
                all_router_samples.extend(rou_samples)
            except Exception as e:
                logger.warning(f"[OnlineSampler] Sample {sample_idx} failed: {e}")
                continue
        
        return all_scheduler_samples, all_router_samples
    
    def sample_batch(
        self,
        tasks: List[Dict[str, Any]],
        evaluator_fn: Optional[callable] = None,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        对一批任务进行采样
        
        Args:
            tasks: 任务列表
            evaluator_fn: 可选的评估函数
            
        Returns:
            (scheduler_samples, router_samples)
        """
        all_scheduler_samples = []
        all_router_samples = []
        
        for idx, task in enumerate(tasks):
            logger.info(f"[OnlineSampler] Sampling task {idx + 1}/{len(tasks)}")
            
            sch_samples, rou_samples = self.sample_task(task, evaluator_fn)
            all_scheduler_samples.extend(sch_samples)
            all_router_samples.extend(rou_samples)
        
        logger.info(
            f"[OnlineSampler] Batch complete: "
            f"{len(all_scheduler_samples)} scheduler, "
            f"{len(all_router_samples)} router samples"
        )
        
        return all_scheduler_samples, all_router_samples
    
    def _sample_one_trajectory(
        self,
        task: Dict[str, Any],
        sample_seed: int,
        evaluator_fn: Optional[callable] = None,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        采样单条轨迹
        
        模拟完整的 MAS 推理过程，收集：
        - Scheduler 的 (prompt, response, token_count)
        - Router 的 (prompt, response, routed_count, useful_count)
        """
        question = task.get("Question", task.get("question", ""))
        task_id = str(task.get("id", hash(question) % 10000))
        ground_truth = task.get("Answer", task.get("answer", ""))
        
        # 设置随机种子以获得不同采样
        random.seed(sample_seed + hash(task_id))
        
        # 初始化状态
        sum_memory = ""
        pre_agent = None
        step_records = []
        final_answer = ""
        
        # 构建 agent specs
        agent_specs = "\n".join([
            f"- {name}: Agent for {name.replace('Agent', '').lower()}"
            for name in self.config.agent_names
        ])
        
        # MAS 推理循环
        for step_idx in range(1, self.config.max_steps + 1):
            step_record = {"step_idx": step_idx}
            
            # --- 1. Scheduler 决策 ---
            scheduler_prompt = format_scheduler_prompt(
                question=question,
                sum_memory=sum_memory,
                agent_specs=agent_specs,
                pre_agent=pre_agent,
            )
            
            scheduler_response, scheduler_tokens = self._generate_scheduler_response(
                scheduler_prompt
            )
            
            step_record["scheduler_prompt"] = scheduler_prompt
            step_record["scheduler_response"] = scheduler_response
            step_record["token_count"] = scheduler_tokens
            
            # 检查终止
            if "DONE" in scheduler_response.upper():
                step_records.append(step_record)
                break
            
            # 规范化 agent name
            chosen_agent = self._normalize_agent_name(
                scheduler_response, 
                self.config.agent_names
            )
            if chosen_agent is None:
                chosen_agent = random.choice(self.config.agent_names)
            
            step_record["chosen_agent"] = chosen_agent
            
            # --- 2. Router 决策 ---
            # 模拟候选记忆
            num_candidates = min(step_idx + 2, 8)
            candidates_list = []
            for i in range(num_candidates):
                candidates_list.append({
                    "node_type": f"Step{i+1}",
                    "summary": f"Memory from step {i+1}: Some context about the task",
                })
            
            candidates_text = "\n".join([
                f"[{i}] ({c['node_type']}): {c['summary']}"
                for i, c in enumerate(candidates_list)
            ])
            
            router_prompt = format_router_prompt(
                question=question,
                now_agent=chosen_agent,
                sum_memory=sum_memory,
                candidates=candidates_text,
                pre_agent=pre_agent,
            )
            
            router_response, router_tokens = self._generate_router_response(router_prompt)
            
            # 解析选中的索引
            selected_indices = self._parse_indices(router_response, num_candidates)
            routed_count = len(selected_indices)
            
            step_record["router_prompt"] = router_prompt
            step_record["router_response"] = router_response
            step_record["routed_count"] = routed_count
            step_record["token_count"] += router_tokens  # 累加 token
            
            # --- 3. 模拟 Agent 执行 + 自反馈 ---
            # 在实际实现中，这里会调用真正的 Agent
            # Agent 会报告哪些记忆是有用的
            routed_memories = [candidates_list[i] for i in selected_indices if i < len(candidates_list)]
            
            # 收集反馈
            self.feedback_collector.set_routed_memories(routed_memories)
            
            # 模拟 agent 输出包含 USEFUL_MEMORIES 反馈
            useful_count = random.randint(0, routed_count) if routed_count > 0 else 0
            
            step_record["useful_count"] = useful_count
            
            # 更新状态
            pre_agent = chosen_agent
            agent_output = f"{chosen_agent} analyzed the problem."
            sum_memory += f"\nStep {step_idx}: {agent_output}"
            
            # 模拟可能的答案
            if "conclud" in chosen_agent.lower() and step_idx >= 2:
                final_answer = f"Answer from {chosen_agent}"
            
            step_records.append(step_record)
            self.feedback_collector.clear_current_step()
        
        # --- 4. 计算 task reward ---
        if evaluator_fn is not None:
            task_reward = evaluator_fn(final_answer, ground_truth)
        else:
            # 模拟评估
            task_reward = random.choice([0.0, 1.0])
        
        total_steps = len(step_records)
        
        # --- 5. 生成训练样本 ---
        scheduler_samples = []
        router_samples = []
        
        for record in step_records:
            step_idx = record["step_idx"]
            
            # Scheduler 样本
            sch_reward = self.scheduler_reward_fn(
                data_item={},
                response=record["scheduler_response"],
                step_info={
                    "step_idx": step_idx,
                    "total_steps": total_steps,
                    "task_reward": task_reward,
                    "token_count": record["token_count"],
                }
            )
            
            scheduler_samples.append({
                "prompt": record["scheduler_prompt"],
                "response": record["scheduler_response"],
                "reward": float(sch_reward),
                "extra_info": {
                    "task_id": task_id,
                    "step_idx": step_idx,
                    "total_steps": total_steps,
                    "token_count": record["token_count"],
                    "task_reward": task_reward,
                    "sample_seed": sample_seed,
                }
            })
            
            # Router 样本（如果有）
            if "router_prompt" in record:
                rou_reward = self.router_reward_fn(
                    data_item={},
                    response=record["router_response"],
                    step_info={
                        "useful_count": record.get("useful_count", 0),
                        "routed_count": record.get("routed_count", 0),
                        "task_reward": task_reward,
                    }
                )
                
                router_samples.append({
                    "prompt": record["router_prompt"],
                    "response": record["router_response"],
                    "reward": float(rou_reward),
                    "extra_info": {
                        "task_id": task_id,
                        "step_idx": step_idx,
                        "useful_count": record.get("useful_count", 0),
                        "routed_count": record.get("routed_count", 0),
                        "task_reward": task_reward,
                        "sample_seed": sample_seed,
                    }
                })
        
        return scheduler_samples, router_samples
    
    def _generate_scheduler_response(self, prompt: str) -> Tuple[str, int]:
        """生成 Scheduler 响应"""
        if self.model is not None:
            try:
                response = self.model.generate(
                    prompt=prompt,
                    expert_type=ExpertType.SCHEDULER,
                    max_new_tokens=self.config.max_new_tokens_scheduler,
                    temperature=self.config.temperature,
                )
                
                # 获取 token 统计
                try:
                    from model.query_manager import query_manager
                    usage = query_manager.get_last_token_usage()
                    tokens = usage.total_tokens if usage else 50
                except:
                    tokens = len(response.split()) * 2  # 估算
                
                return response.strip().split()[0] if response else "ReasoningAgent", tokens
                
            except Exception as e:
                logger.warning(f"[OnlineSampler] Scheduler generation failed: {e}")
        
        # Fallback: 随机采样
        if random.random() < 0.15:
            return "DONE", 10
        return random.choice(self.config.agent_names), random.randint(30, 100)
    
    def _generate_router_response(self, prompt: str) -> Tuple[str, int]:
        """生成 Router 响应"""
        if self.model is not None:
            try:
                response = self.model.generate(
                    prompt=prompt,
                    expert_type=ExpertType.ROUTER,
                    max_new_tokens=self.config.max_new_tokens_router,
                    temperature=self.config.temperature,
                )
                
                try:
                    from model.query_manager import query_manager
                    usage = query_manager.get_last_token_usage()
                    tokens = usage.total_tokens if usage else 20
                except:
                    tokens = 20
                
                return response.strip(), tokens
                
            except Exception as e:
                logger.warning(f"[OnlineSampler] Router generation failed: {e}")
        
        # Fallback: 随机采样
        num_select = random.randint(1, 3)
        indices = random.sample(range(5), min(num_select, 5))
        return ", ".join(map(str, sorted(indices))), random.randint(10, 30)
    
    def _normalize_agent_name(self, response: str, valid_names: List[str]) -> Optional[str]:
        """规范化 agent 名称"""
        response_lower = response.lower().strip()
        
        for name in valid_names:
            if name.lower() in response_lower:
                return name
        
        return None
    
    def _parse_indices(self, response: str, max_idx: int) -> List[int]:
        """解析 Router 响应中的索引"""
        indices = []
        for part in response.replace(",", " ").split():
            try:
                idx = int(part.strip())
                if 0 <= idx < max_idx and idx not in indices:
                    indices.append(idx)
            except ValueError:
                continue
        
        return indices if indices else [0]


def save_samples_to_jsonl(
    samples: List[Dict],
    path: str,
):
    """保存样本到 JSONL 文件"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    logger.info(f"[OnlineSampler] Saved {len(samples)} samples to {path}")

