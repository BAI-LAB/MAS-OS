"""verl Integration for Dual LoRA (Scheduler + Router) GRPO Training.

This module integrates with verl (Volcano Engine Reinforcement Learning) framework
to train Scheduler and Router using the built-in GRPO algorithm.

Key components:
- MASRewardFunction: Custom reward function for verl (verifiable reward)
- TrainingDataCollector: Collects data with precise token counts and agent feedback
- verl config generation for dual LoRA GRPO training

verl Data Format:
- JSONL/Parquet with fields: prompt, response, reward (for offline)
- For GRPO: needs group_size (rollout.n) samples per prompt

verl GRPO Key Settings:
- algorithm.adv_estimator: "grpo"
- actor_rollout_ref.rollout.n: group_size for sampling
- actor_rollout_ref.actor.use_kl_loss: True
- actor_rollout_ref.actor.kl_loss_coef: typically 0.001

References:
- verl: https://github.com/volcengine/verl
- GRPO docs: https://verl.readthedocs.io/en/latest/algo/grpo.html
"""
from __future__ import annotations

import logging
import os
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime
from pathlib import Path

try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    np = None

logger = logging.getLogger("MAS")


# ============================================================================
# Reward Functions for verl integration
# ============================================================================

@dataclass
class MASRewardConfig:
    """Configuration for MAS reward computation.

    Defaults follow the MAS-OS paper (Appendix A).
    """

    # Context Allocator reward (Eq. 12): R_context = α * R_agent + η * R_task
    alpha: float = 0.25       # Weight for the agent hit-rate term
    eta: float = 1.0          # Weight for the terminal task reward

    # Agent Scheduler reward (Eq. 13): R_scheduler = R_task - λ * c_t
    lambda_tok: float = 5e-5  # Per-step token cost penalty

    # Rewards are clipped to [-2, 2]
    reward_clip: float = 2.0

    # Non-paper extension: penalty applied when the allocator returns more than
    # K memories. 0.0 disables it (paper behaviour); set > 0 to re-enable.
    over_budget_penalty: float = 0.0


class SchedulerRewardFunction:
    """
    Scheduler reward function for verl integration.

    Formula (paper Eq. 13): r^sch_{i,t} = R^task_i - λ_tok * c_{i,t}

    Note there is no time-weight term: every step of a trajectory shares the
    same terminal task reward and is charged only for the tokens it consumed.

    This is designed to be used with verl's reward function interface.
    """

    def __init__(self, config: MASRewardConfig = None):
        self.config = config or MASRewardConfig()

    def __call__(
        self,
        data_item: Dict[str, Any],
        response: str,
        step_info: Dict[str, Any],
    ) -> float:
        """
        Compute reward for a scheduler decision.

        Args:
            data_item: The task data (contains task_reward after evaluation)
            response: The scheduler's output (agent name)
            step_info: Additional info (token_count, task_reward). step_idx and
                total_steps may be present but do not affect the reward.

        Returns:
            Reward value
        """
        # Extract from step_info
        c_t = step_info.get("token_count", 0)
        R_task = step_info.get("task_reward", 0.0)

        # r^sch = R^task - λ * c
        reward = R_task - self.config.lambda_tok * c_t

        # Clip
        return max(-self.config.reward_clip, min(self.config.reward_clip, reward))

    def compute_batch_rewards(
        self,
        task_rewards: List[float],
        token_counts: List[int],
    ) -> List[float]:
        """Compute rewards for a batch of scheduler steps."""
        rewards = []
        for R_task, c in zip(task_rewards, token_counts):
            r = R_task - self.config.lambda_tok * c
            r = max(-self.config.reward_clip, min(self.config.reward_clip, r))
            rewards.append(r)
        return rewards


class RouterRewardFunction:
    """
    Context Allocator reward function for verl integration.

    Formula (paper Eq. 12): r^rou_{i,t} = α * R_agent + η * R^task_i
    where R_agent = (1 / |M^context_t|) * Σ_i u_i is the hit rate reported by
    the active agent after it finished its work (paper Eq. 11).

    This is designed to be used with verl's reward function interface.
    """

    def __init__(self, config: MASRewardConfig = None):
        self.config = config or MASRewardConfig()

    def __call__(
        self,
        data_item: Dict[str, Any],
        response: str,
        step_info: Dict[str, Any],
    ) -> float:
        """
        Compute reward for a router decision.

        Args:
            data_item: The task data
            response: The router's output (selected indices)
            step_info: Additional info (useful_count, routed_count, task_reward, top_n)

        Returns:
            Reward value
        """
        # Extract from step_info
        useful_count = step_info.get("useful_count", 0)
        routed_count = step_info.get("routed_count", 1)
        R_task = step_info.get("task_reward", 0.0)
        top_n = step_info.get("top_n", None)  # Maximum allowed memories

        # Hit rate: R_agent = u / |M^context|
        h_it = useful_count / max(routed_count, 1)

        # r^rou = α * R_agent + η * R^task
        reward = self.config.alpha * h_it + self.config.eta * R_task

        # Non-paper extension, off by default (over_budget_penalty == 0.0).
        penalty = self.config.over_budget_penalty
        if penalty > 0.0 and top_n is not None and routed_count > top_n:
            reward -= penalty
            logger.debug(f"[Router Reward] Penalty applied: routed_count={routed_count} > top_n={top_n}")

        # Clip
        return max(-self.config.reward_clip, min(self.config.reward_clip, reward))
    
    def compute_batch_rewards(
        self,
        task_rewards: List[float],
        useful_counts: List[int],
        routed_counts: List[int],
    ) -> List[float]:
        """Compute rewards for a batch of router steps."""
        rewards = []
        for R_task, u, k in zip(task_rewards, useful_counts, routed_counts):
            h = u / max(k, 1)
            r = self.config.alpha * h + self.config.eta * R_task
            r = max(-self.config.reward_clip, min(self.config.reward_clip, r))
            rewards.append(r)
        return rewards


# ============================================================================
# verl Config Generator for GRPO
# ============================================================================

@dataclass
class VerlGRPOConfig:
    """Configuration for verl GRPO training.
    
    Key GRPO settings (from verl docs):
    - algorithm.adv_estimator: "grpo" (not GAE)
    - rollout.n: group_size for sampling multiple responses per prompt
    - actor.use_kl_loss: True (required for GRPO)
    - actor.kl_loss_coef: typically 0.001
    - actor.kl_loss_type: "low_var_kl" (k3 approximation)
    """
    # Model
    base_model: str = "meta-llama/Llama-3.1-8B"
    output_dir: str = "./verl_output"
    
    # LoRA settings
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])
    
    # GRPO settings
    group_size: int = 4  # rollout.n - samples per prompt
    clip_ratio: float = 0.2
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"  # k3 approximation
    norm_adv_by_std: bool = True
    
    # Training settings
    train_batch_size: int = 8
    ppo_mini_batch_size: int = 4
    ppo_epochs: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_steps: int = 1000
    save_steps: int = 100
    log_steps: int = 10
    
    # Generation settings
    max_new_tokens_scheduler: int = 64   # Scheduler outputs short agent names
    max_new_tokens_router: int = 32      # Router outputs comma-separated indices
    max_prompt_length: int = 4096
    temperature: float = 0.7
    
    # Data settings
    train_data_path: str = ""
    
    # Compute settings
    n_gpus: int = 1
    strategy: str = "fsdp2"


def generate_verl_grpo_config(
    config: VerlGRPOConfig,
    expert_type: str = "scheduler",
    data_path: str = "",
) -> Dict[str, Any]:
    """
    Generate verl config for GRPO training.
    
    This config follows verl's GRPO documentation:
    https://verl.readthedocs.io/en/latest/algo/grpo.html
    
    Key differences from PPO:
    1. algorithm.adv_estimator = "grpo" (not "gae")
    2. No critic network needed
    3. rollout.n > 1 for group sampling
    4. use_kl_loss = True required
    
    Args:
        config: VerlGRPOConfig with training parameters
        expert_type: "scheduler" or "router"
        data_path: Path to training JSONL file
        
    Returns:
        Config dict for verl
    """
    max_new_tokens = (
        config.max_new_tokens_scheduler if expert_type == "scheduler" 
        else config.max_new_tokens_router
    )
    
    verl_config = {
        # Algorithm settings - GRPO specific
        "algorithm": {
            "adv_estimator": "grpo",  # Critical: use GRPO not GAE
            "norm_adv_by_std_in_grpo": config.norm_adv_by_std,
        },
        
        # Data settings
        "data": {
            "train_batch_size": config.train_batch_size,
            "train_files": data_path if data_path else config.train_data_path,
            "max_prompt_length": config.max_prompt_length,
            "max_response_length": max_new_tokens,
        },
        
        # Actor/Rollout/Reference model settings
        "actor_rollout_ref": {
            # Rollout settings - GRPO samples n times per prompt
            "rollout": {
                "n": config.group_size,  # Key for GRPO: group sampling
                "temperature": config.temperature,
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
            },
            
            # Actor (policy) settings
            "actor": {
                "ppo_mini_batch_size": config.ppo_mini_batch_size,
                "ppo_epochs": config.ppo_epochs,
                "clip_ratio": config.clip_ratio,
                # GRPO requires KL loss
                "use_kl_loss": True,
                "kl_loss_coef": config.kl_loss_coef,
                "kl_loss_type": config.kl_loss_type,
                "loss_agg_mode": "token-mean",  # More stable
                
                # Optimizer
                "optim": {
                    "lr": config.learning_rate,
                    "weight_decay": config.weight_decay,
                },
                
                # Strategy
                "strategy": config.strategy,
            },
            
            # Reference model settings
            "ref": {
                "strategy": config.strategy,
            },
        },
        
        # Model settings
        "model": {
            "path": config.base_model,
            "torch_dtype": "bfloat16",
        },
        
        # Trainer settings
        "trainer": {
            "max_steps": config.max_steps,
            "save_steps": config.save_steps,
            "log_steps": config.log_steps,
            "output_dir": os.path.join(config.output_dir, f"{expert_type}_lora"),
        },
    }
    
    # Add LoRA configuration if enabled
    if config.use_lora:
        verl_config["actor_rollout_ref"]["actor"]["use_lora"] = True
        verl_config["actor_rollout_ref"]["actor"]["lora_config"] = {
            "r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "target_modules": config.lora_target_modules,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
    
    return verl_config


def save_verl_config(config: Dict[str, Any], path: str):
    """Save verl config to YAML file."""
    import yaml
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info(f"[VerlConfig] Saved config to {path}")


def generate_verl_training_script(
    config: VerlGRPOConfig,
    expert_type: str,
    config_path: str,
    output_path: str,
) -> str:
    """
    Generate a shell script to run verl GRPO training.
    
    Args:
        config: Training configuration
        expert_type: "scheduler" or "router"
        config_path: Path to verl config YAML
        output_path: Where to save the script
        
    Returns:
        Path to generated script
    """
    script = f'''#!/bin/bash
# verl GRPO Training Script for {expert_type.upper()} LoRA
# Generated: {datetime.now().isoformat()}
# Reference: https://verl.readthedocs.io/en/latest/algo/grpo.html

set -e

# Configuration
CONFIG_PATH="{config_path}"
OUTPUT_DIR="{config.output_dir}/{expert_type}_lora"
N_GPUS={config.n_gpus}

echo "========================================"
echo "verl GRPO Training: {expert_type.upper()} LoRA"
echo "Config: $CONFIG_PATH"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $N_GPUS"
echo "========================================"

# Set environment
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPUS-1)))

# Create output directory
mkdir -p $OUTPUT_DIR

# Run verl GRPO training
# Key GRPO settings are in config, but we can override here:
python -m verl.trainer.main_ppo \\
    --config_path "$CONFIG_PATH" \\
    --algorithm.adv_estimator grpo \\
    --actor_rollout_ref.rollout.n {config.group_size} \\
    --actor_rollout_ref.actor.use_kl_loss True \\
    --actor_rollout_ref.actor.kl_loss_coef {config.kl_loss_coef} \\
    --trainer.output_dir "$OUTPUT_DIR"

echo "Training complete: $OUTPUT_DIR"
'''
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(script)
    os.chmod(output_path, 0o755)
    
    logger.info(f"[VerlConfig] Generated training script: {output_path}")
    return output_path


# ============================================================================
# MAS Dataset for verl
# ============================================================================

class MASDatasetForVerl:
    """
    Dataset wrapper for MAS tasks compatible with verl.
    
    verl expects datasets with prompt/response format.
    This class adapts MAS tasks to that format.
    """
    
    def __init__(
        self,
        tasks: List[Dict[str, Any]],
        expert_type: str = "scheduler",  # "scheduler" or "router"
    ):
        self.tasks = tasks
        self.expert_type = expert_type
        self._prompts = []
        self._prepare_prompts()
    
    def _prepare_prompts(self):
        """Prepare prompts for training."""
        from .dual_lora_model import format_scheduler_prompt, format_router_prompt
        
        for task in self.tasks:
            question = task.get("Question", task.get("question", ""))
            
            if self.expert_type == "scheduler":
                # For scheduler training, we need agent specs
                # This will be filled during actual training
                prompt = format_scheduler_prompt(
                    question=question,
                    sum_memory="",  # Will be dynamic
                    agent_specs="",  # Will be dynamic
                    pre_agent=None,
                )
            else:
                # For router training
                prompt = format_router_prompt(
                    question=question,
                    now_agent="",  # Will be dynamic
                    sum_memory="",
                    candidates="",  # Will be dynamic
                    pre_agent=None,
                )
            
            self._prompts.append({
                "prompt": prompt,
                "task_id": str(task.get("id", len(self._prompts))),
                "task_data": task,
            })
    
    def __len__(self):
        return len(self._prompts)
    
    def __getitem__(self, idx):
        return self._prompts[idx]
    
    def to_verl_format(self) -> List[Dict[str, Any]]:
        """Convert to verl's expected format."""
        return [
            {
                "prompt": item["prompt"],
                "extra_info": {
                    "task_id": item["task_id"],
                    "task_data": item["task_data"],
                }
            }
            for item in self._prompts
        ]


# ============================================================================
# Reward Function Wrapper for verl
# ============================================================================

class MASRewardManager:
    """
    Reward manager that integrates with verl's reward function interface.
    
    verl calls reward functions after rollout to compute rewards.
    This class wraps our scheduler and router reward functions.
    """
    
    def __init__(
        self,
        expert_type: str = "scheduler",
        config: MASRewardConfig = None,
    ):
        self.expert_type = expert_type
        self.config = config or MASRewardConfig()
        
        if expert_type == "scheduler":
            self.reward_fn = SchedulerRewardFunction(config)
        else:
            self.reward_fn = RouterRewardFunction(config)
        
        # Step info storage (to be filled during MAS execution)
        self._step_info: Dict[str, Dict] = {}
    
    def register_step_info(self, task_id: str, step_info: Dict[str, Any]):
        """Register step info for reward computation."""
        self._step_info[task_id] = step_info
    
    def compute_reward(
        self,
        data_item: Dict[str, Any],
        response: str,
    ) -> float:
        """
        Compute reward for verl.
        
        This is called by verl after rollout generation.
        """
        task_id = data_item.get("extra_info", {}).get("task_id", "")
        step_info = self._step_info.get(task_id, {})
        
        return self.reward_fn(data_item, response, step_info)
    
    def __call__(self, *args, **kwargs):
        """Make callable for verl compatibility."""
        return self.compute_reward(*args, **kwargs)


# ============================================================================
# verl Training Script Generator
# ============================================================================

def generate_training_script(
    output_path: str,
    config_path: str,
    expert_type: str = "scheduler",
) -> str:
    """
    Generate a training script that uses verl for GRPO training.
    
    Args:
        output_path: Where to save the script
        config_path: Path to verl config file
        expert_type: "scheduler" or "router"
        
    Returns:
        Path to generated script
    """
    script = f'''#!/bin/bash
# Auto-generated verl GRPO training script for {expert_type} LoRA
# Generated at: {datetime.now().isoformat()}

# Environment setup
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Adjust based on your GPU count

# verl GRPO training command
# Reference: https://verl.readthedocs.io/en/latest/algo/grpo.html

python -m verl.trainer.main_ppo \\
    --config_path {config_path} \\
    --algorithm.adv_estimator grpo \\
    --actor_rollout_ref.rollout.n 4 \\
    --actor_rollout_ref.actor.use_kl_loss True \\
    --actor_rollout_ref.actor.kl_loss_coef 0.001

echo "Training complete for {expert_type} LoRA"
'''
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(script)
    os.chmod(output_path, 0o755)
    
    logger.info(f"Generated training script: {output_path}")
    return output_path


# ============================================================================
# Task Reward Computation (for final evaluation)
# ============================================================================

def compute_task_reward(
    correct: bool,
    task_type: str = "default",
    partial_score: Optional[float] = None,
) -> float:
    """
    Convert evaluator result to numerical task reward R^task.
    
    Args:
        correct: Whether the task was solved correctly
        task_type: Type of task (affects reward scale)
        partial_score: Optional partial credit [0, 1]
        
    Returns:
        Numerical reward value
    """
    if partial_score is not None:
        return 2.0 * partial_score - 1.0
    
    reward_scales = {
        "MMLU-Pro": 1.0,
        "GSM-Hard": 1.0,
        "gsm-hard": 1.0,
        "SRDD": 1.0,
        "CW": 0.5,
        "default": 1.0,
    }
    
    scale = reward_scales.get(task_type, 1.0)
    return scale if correct else -scale


# ============================================================================
# Training Data Collector
# ============================================================================

@dataclass
class StepData:
    """Data collected at each step for verl GRPO training.
    
    Contains all information needed for reward computation:
    - Scheduler: token_count (c_{i,t}), step_idx (t), total_steps (T_i)
    - Router: routed_count (k_{i,t}), useful_count (u_{i,t}) from agent feedback
    """
    step_idx: int
    expert_type: str  # "scheduler" or "router"
    prompt: str
    response: str
    
    # For scheduler reward: r^sch = w_{i,t} * R^task - λ * c_{i,t}
    token_count: int = 0  # c_{i,t}: Precise token consumption
    
    # For router reward: r^rou = α * h_{i,t} + η * R^task
    # where h_{i,t} = u_{i,t} / k_{i,t} (hit rate from agent feedback)
    routed_count: int = 0   # k_{i,t}: Number of memories routed
    useful_count: int = 0   # u_{i,t}: Number marked useful by agent (from self-feedback)
    
    # Filled after episode ends
    task_reward: float = 0.0  # R^task_i: Final task reward
    total_steps: int = 0      # T_i: Total steps in this episode
    
    # Metadata for verl
    task_id: str = ""
    agent_name: str = ""
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class TrainingDataCollector:
    """
    Collects training data during MAS execution for verl GRPO.
    
    This collector:
    1. Records step-level data during MAS reasoning
    2. Captures precise token counts from model API
    3. Stores agent self-feedback for router useful_count
    4. Computes rewards using the defined reward functions
    5. Exports to verl-compatible JSONL format
    
    verl expects JSONL with:
    - prompt: The input text
    - response: The generated text  
    - reward: The computed reward (optional for online GRPO)
    
    For GRPO, group_size samples are generated per prompt during training.
    """
    
    def __init__(self, reward_config: MASRewardConfig = None):
        self.reward_config = reward_config or MASRewardConfig()
        self.scheduler_reward_fn = SchedulerRewardFunction(self.reward_config)
        self.router_reward_fn = RouterRewardFunction(self.reward_config)
        
        # Collected data
        self.scheduler_data: List[StepData] = []
        self.router_data: List[StepData] = []
        
        # Current episode buffers
        self._current_episode_scheduler: List[StepData] = []
        self._current_episode_router: List[StepData] = []
        self._task_id: str = ""
        self._episode_count: int = 0
    
    def start_episode(self, task_id: str):
        """Start collecting for a new episode/task."""
        self._task_id = task_id
        self._current_episode_scheduler = []
        self._current_episode_router = []
        self._episode_count += 1
        logger.debug(f"[DataCollector] Started episode {self._episode_count}: {task_id}")
    
    def record_scheduler_step(
        self,
        step_idx: int,
        prompt: str,
        response: str,
        token_count: int,
        agent_name: str = "",
    ):
        """
        Record a scheduler decision with precise token count.
        
        Args:
            step_idx: Current step number (1-indexed)
            prompt: Full scheduler prompt
            response: Scheduler's output (agent name)
            token_count: Precise token consumption from API
            agent_name: Selected agent name
        """
        step = StepData(
            step_idx=step_idx,
            expert_type="scheduler",
            prompt=prompt,
            response=response,
            token_count=token_count,
            task_id=self._task_id,
            agent_name=agent_name,
            timestamp=datetime.now().isoformat(),
        )
        self._current_episode_scheduler.append(step)
        logger.debug(f"[DataCollector] Scheduler step {step_idx}: {token_count} tokens")
    
    def record_router_step(
        self,
        step_idx: int,
        prompt: str,
        response: str,
        routed_count: int,
        useful_count: int,
        agent_name: str = "",
    ):
        """
        Record a router decision with agent self-feedback.
        
        Args:
            step_idx: Current step number (1-indexed)
            prompt: Full router prompt
            response: Router's output (memory indices)
            routed_count: k_{i,t} - Number of memories routed
            useful_count: u_{i,t} - Number marked useful by agent (from self-feedback)
            agent_name: Target agent name
        """
        step = StepData(
            step_idx=step_idx,
            expert_type="router",
            prompt=prompt,
            response=response,
            routed_count=routed_count,
            useful_count=useful_count,
            task_id=self._task_id,
            agent_name=agent_name,
            timestamp=datetime.now().isoformat(),
        )
        self._current_episode_router.append(step)
        
        hit_rate = useful_count / routed_count if routed_count > 0 else 0.0
        logger.debug(f"[DataCollector] Router step {step_idx}: {useful_count}/{routed_count} useful (hit={hit_rate:.2f})")
    
    def end_episode(self, task_reward: float):
        """
        End episode and finalize data with task reward.
        
        This fills in the episodic information (R^task_i, T_i) that wasn't
        available during step recording.
        """
        total_steps = len(self._current_episode_scheduler)
        
        # Update all scheduler steps with final info
        for step in self._current_episode_scheduler:
            step.task_reward = task_reward
            step.total_steps = total_steps
        
        # Update all router steps with final info
        for step in self._current_episode_router:
            step.task_reward = task_reward
            step.total_steps = total_steps
        
        # Add to main lists
        self.scheduler_data.extend(self._current_episode_scheduler)
        self.router_data.extend(self._current_episode_router)
        
        logger.info(
            f"[DataCollector] Episode {self._episode_count} ended: "
            f"R^task={task_reward:.3f}, T={total_steps}, "
            f"scheduler_steps={len(self._current_episode_scheduler)}, "
            f"router_steps={len(self._current_episode_router)}"
        )
        
        # Clear episode buffers
        self._current_episode_scheduler = []
        self._current_episode_router = []
    
    def compute_rewards(self, expert_type: str = "scheduler") -> List[float]:
        """
        Compute rewards for all collected steps of specified type.

        Uses the reward functions defined in this module:
        - Scheduler: r^sch = R^task - λ * c_{i,t}
        - Router: r^rou = α * R_agent + η * R^task
        """
        if expert_type == "scheduler":
            data = self.scheduler_data
            return [
                self.scheduler_reward_fn.compute_batch_rewards(
                    [step.task_reward],
                    [step.token_count],
                )[0]
                for step in data
            ]
        else:
            data = self.router_data
            return [
                self.router_reward_fn.compute_batch_rewards(
                    [step.task_reward],
                    [step.useful_count],
                    [step.routed_count],
                )[0]
                for step in data
            ]
    
    def to_verl_format(self, expert_type: str = "scheduler") -> List[Dict]:
        """
        Convert collected data to verl's expected JSONL format.
        
        verl expects each line to have:
        - prompt: Input text
        - response: Generated text
        - reward: Computed reward value
        
        Additional metadata is included for debugging.
        """
        rewards = self.compute_rewards(expert_type)
        data = self.scheduler_data if expert_type == "scheduler" else self.router_data
        
        verl_data = []
        for step, reward in zip(data, rewards):
            verl_data.append({
                "prompt": step.prompt,
                "response": step.response,
                "reward": float(reward),
                # Metadata (verl ignores these but useful for debugging)
                "extra_info": {
                    "task_id": step.task_id,
                    "step_idx": step.step_idx,
                    "total_steps": step.total_steps,
                    "token_count": step.token_count,
                    "routed_count": step.routed_count,
                    "useful_count": step.useful_count,
                    "task_reward": step.task_reward,
                    "agent_name": step.agent_name,
                }
            })
        
        return verl_data
    
    def save_to_jsonl(self, path: str, expert_type: str = "scheduler"):
        """Save collected data to JSONL file for verl training."""
        data = self.to_verl_format(expert_type)
        
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        # Log summary statistics
        rewards = [item["reward"] for item in data]
        if rewards:
            import statistics
            logger.info(
                f"[DataCollector] Saved {len(data)} {expert_type} samples to {path}\n"
                f"  Reward stats: mean={statistics.mean(rewards):.4f}, "
                f"std={statistics.stdev(rewards) if len(rewards) > 1 else 0:.4f}, "
                f"min={min(rewards):.4f}, max={max(rewards):.4f}"
            )
        else:
            logger.info(f"[DataCollector] Saved 0 {expert_type} samples to {path}")
    
    def save_all(self, output_dir: str):
        """Save both scheduler and router data to separate files."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        scheduler_path = os.path.join(output_dir, "scheduler_train.jsonl")
        router_path = os.path.join(output_dir, "router_train.jsonl")
        
        self.save_to_jsonl(scheduler_path, "scheduler")
        self.save_to_jsonl(router_path, "router")
        
        # Also save a summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_episodes": self._episode_count,
            "scheduler_samples": len(self.scheduler_data),
            "router_samples": len(self.router_data),
            "reward_config": asdict(self.reward_config),
        }
        
        summary_path = os.path.join(output_dir, "collection_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"[DataCollector] Saved all data to {output_dir}")
    
    def clear(self):
        """Clear all collected data."""
        self.scheduler_data.clear()
        self.router_data.clear()
        self._current_episode_scheduler.clear()
        self._current_episode_router.clear()
        self._episode_count = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics."""
        scheduler_rewards = self.compute_rewards("scheduler")
        router_rewards = self.compute_rewards("router")
        
        def compute_stats(values: List[float]) -> Dict[str, float]:
            if not values:
                return {"count": 0}
            import statistics
            return {
                "count": len(values),
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0,
                "min": min(values),
                "max": max(values),
            }
        
        return {
            "episodes": self._episode_count,
            "scheduler": compute_stats(scheduler_rewards),
            "router": compute_stats(router_rewards),
            "total_tokens": sum(s.token_count for s in self.scheduler_data),
        }

