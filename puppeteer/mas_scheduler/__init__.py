"""MAS-style Multi-Agent Scheduler for Puppeteer.

This module implements a hierarchical multi-agent scheduling framework
based on the MAS (Multi-Agent System) design with:
- Scheduler: Dynamically decides which agent to activate next
- Router: Selects which memories the current agent can see
- Memer: Manages shared graph memory with summaries and retrieval

Extended for RL training with verl:
- DualLoRAModel: Single base model with Scheduler + Router LoRA experts
- verl Integration: Training using GRPO algorithm via verl framework
- Reward functions: Scheduler (time-weighted + token cost) and Router (hit rate)

References:
- verl: https://github.com/volcengine/verl
- GRPO: https://verl.readthedocs.io/en/latest/algo/grpo.html
"""

from .config import (
    MASConfig, 
    DEFAULT_MAS_CONFIG,
    DualLoRAConfig,
    RewardConfig,
    TrainingConfig,
)
from .llm import MASLLMClient
from .memer import Memer, MemoryNode
from .scheduler import MASScheduler, AgentSpec, AgentRegistry
from .router import MASRouter
from .task_state import (
    MASTaskState, 
    AgentResult, 
    HistoryEntry,
    MemoryFeedback,
    StepMetrics,
)
from .mas_reasoning import MASReasoning

# LoRA-based components
from .dual_lora_model import (
    DualLoRAModel,
    DualLoRAConfig as ModelDualLoRAConfig,
    ExpertType,
    format_scheduler_prompt,
    format_router_prompt,
)
from .lora_scheduler import LoRAScheduler
from .lora_router import LoRARouter

# verl integration components
from .verl_integration import (
    MASRewardConfig,
    SchedulerRewardFunction,
    RouterRewardFunction,
    MASDatasetForVerl,
    MASRewardManager,
    TrainingDataCollector,
    StepData,
    VerlGRPOConfig,
    generate_verl_grpo_config,
    save_verl_config,
    generate_verl_training_script,
    compute_task_reward,
)

# Agent feedback for router reward
from .agent_feedback import (
    AgentFeedbackCollector,
    MemoryUsefulnessFeedback,
    extract_agent_output_text,
    generate_memory_feedback_prompt,
    parse_memory_feedback_response,
    run_agent_with_parallel_feedback,
)

# Online sampler for GRPO training
from .online_sampler import (
    OnlineSampler,
    SamplerConfig,
    save_samples_to_jsonl,
)

# Legacy reward module (for backward compatibility)
from .reward import (
    RewardConfig as RewardConfigClass,
    StepRecord,
    EpisodeTrajectory,
    RewardCalculator,
    TrajectoryBuffer,
    compute_task_reward_from_evaluator,
)

__all__ = [
    # Config
    "MASConfig",
    "DEFAULT_MAS_CONFIG",
    "DualLoRAConfig",
    "RewardConfig",
    "TrainingConfig",
    
    # Core components
    "MASLLMClient",
    "Memer",
    "MemoryNode",
    "MASScheduler",
    "AgentSpec",
    "AgentRegistry",
    "MASRouter",
    "MASTaskState",
    "AgentResult",
    "HistoryEntry",
    "MemoryFeedback",
    "StepMetrics",
    "MASReasoning",
    
    # Dual LoRA
    "DualLoRAModel",
    "ModelDualLoRAConfig",
    "ExpertType",
    "format_scheduler_prompt",
    "format_router_prompt",
    "LoRAScheduler",
    "LoRARouter",
    
    # verl integration
    "MASRewardConfig",
    "SchedulerRewardFunction",
    "RouterRewardFunction",
    "MASDatasetForVerl",
    "MASRewardManager",
    "TrainingDataCollector",
    "StepData",
    "VerlGRPOConfig",
    "generate_verl_grpo_config",
    "save_verl_config",
    "generate_verl_training_script",
    "compute_task_reward",
    
    # Agent feedback for context allocator reward
    "AgentFeedbackCollector",
    "MemoryUsefulnessFeedback",
    "extract_agent_output_text",
    "generate_memory_feedback_prompt",
    "parse_memory_feedback_response",
    "run_agent_with_parallel_feedback",
    
    # Online sampler for GRPO
    "OnlineSampler",
    "SamplerConfig",
    "save_samples_to_jsonl",
    
    # Legacy reward (backward compat)
    "RewardConfigClass",
    "StepRecord",
    "EpisodeTrajectory",
    "RewardCalculator",
    "TrajectoryBuffer",
    "compute_task_reward_from_evaluator",
]

