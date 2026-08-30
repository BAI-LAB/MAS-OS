"""Reward calculation for Scheduler and Router.

Reward functions (MAS-OS paper Eq. 11-13):
- Context Allocator: r_rou_{i,t} = α * R_agent + η * R^task_i
  - R_agent = useful_count / |M^context_t| (hit rate from agent self-report)
  - R^task_i = final task reward from evaluator

- Agent Scheduler: r_sch_{i,t} = R^task_i - λ_tok * c_{i,t}
  - c_{i,t} = token consumption of the activated agent at step t
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math

import torch

logger = logging.getLogger("MAS")


@dataclass
class RewardConfig:
    """Configuration for reward computation.

    NOTE: this class is NOT the one used during training. The live reward
    hyperparameters live in `verl_integration.MASRewardConfig`, which the MARTI
    workflow instantiates. Defaults here are kept in sync with the paper
    (Appendix A) so the two cannot drift.
    """

    # Context Allocator reward hyperparameters (Eq. 12)
    alpha: float = 0.25       # Weight for the agent hit-rate term R_agent
    eta: float = 1.0          # Weight for the terminal task reward R^task

    # Agent Scheduler reward hyperparameters (Eq. 13)
    lambda_tok: float = 5e-5  # Per-step token cost penalty

    # Discount factor for return calculation
    gamma_discount: float = 1.0  # 1.0 = no discounting (episodic)

    # Reward normalization
    normalize_rewards: bool = True
    reward_clip: float = 2.0  # Clip rewards to [-clip, clip]


@dataclass
class StepRecord:
    """Record of a single step in the trajectory."""
    
    step_idx: int                     # t (1-indexed for formulas)
    
    # Scheduler decision
    scheduler_action: str             # Agent name chosen
    scheduler_log_prob: float = 0.0   # log π(a|s) for scheduler
    scheduler_prompt: str = ""        # Full prompt used
    
    # Router decision
    router_action: List[int] = field(default_factory=list)  # Indices selected
    router_log_prob: float = 0.0      # log π(a|s) for router
    router_prompt: str = ""           # Full prompt used
    
    # Agent feedback (for router hit rate)
    routed_count: int = 0             # k_{i,t}: number of memories routed
    useful_count: int = 0             # u_{i,t}: number marked useful by agent
    
    # Token consumption (for scheduler cost)
    token_count: int = 0              # c_{i,t}: tokens consumed this step
    
    # Computed rewards (filled after episode ends)
    router_reward: float = 0.0
    scheduler_reward: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Compute h_{i,t} = u_{i,t} / k_{i,t}."""
        if self.routed_count == 0:
            return 0.0
        return self.useful_count / self.routed_count


@dataclass  
class EpisodeTrajectory:
    """Complete trajectory for one episode (task)."""
    
    task_id: str
    steps: List[StepRecord] = field(default_factory=list)
    task_reward: float = 0.0          # R^task_i from evaluator
    total_tokens: int = 0             # Sum of all token costs
    
    @property
    def total_steps(self) -> int:
        """T_i: total number of steps."""
        return len(self.steps)
    
    def add_step(self, step: StepRecord):
        """Add a step record."""
        self.steps.append(step)
        self.total_tokens += step.token_count
    
    def set_task_reward(self, reward: float):
        """Set the final task reward from evaluator."""
        self.task_reward = reward


class RewardCalculator:
    """
    Computes rewards for Scheduler and Router based on episode trajectory.

    Key formulas (paper Eq. 12-13):
    - Context Allocator: r^rou_{i,t} = α * R_agent + η * R^task_i
    - Agent Scheduler:   r^sch_{i,t} = R^task_i - λ_tok * c_{i,t}
    """
    
    def __init__(self, config: RewardConfig = None):
        self.config = config or RewardConfig()
    
    def compute_rewards(self, trajectory: EpisodeTrajectory) -> EpisodeTrajectory:
        """
        Compute all step rewards for an episode trajectory.
        
        This should be called AFTER the episode ends and task_reward is known.
        
        Args:
            trajectory: The episode trajectory with steps recorded
            
        Returns:
            The same trajectory with rewards filled in
        """
        T_i = trajectory.total_steps
        R_task = trajectory.task_reward
        
        if T_i == 0:
            logger.warning("[Reward] Empty trajectory, nothing to compute")
            return trajectory
        
        logger.info(f"[Reward] Computing rewards for {T_i} steps, R^task = {R_task:.4f}")
        
        for step in trajectory.steps:
            t = step.step_idx  # 1-indexed

            # Context Allocator reward (Eq. 12): r^rou = α * R_agent + η * R^task
            h_it = step.hit_rate
            r_rou = self.config.alpha * h_it + self.config.eta * R_task
            step.router_reward = self._clip_reward(r_rou)

            # Agent Scheduler reward (Eq. 13): r^sch = R^task - λ_tok * c_{i,t}
            c_it = step.token_count
            r_sch = R_task - self.config.lambda_tok * c_it
            step.scheduler_reward = self._clip_reward(r_sch)

            logger.debug(
                f"[Reward] Step {t}: h={h_it:.3f}, "
                f"r_rou={step.router_reward:.4f}, r_sch={step.scheduler_reward:.4f}"
            )
        
        return trajectory
    
    def compute_returns(
        self,
        trajectory: EpisodeTrajectory,
    ) -> Tuple[List[float], List[float]]:
        """
        Compute returns G_t for each step (for policy gradient).
        
        G_t = sum_{t'=t}^{T} γ^{t'-t} * r_{t'}
        
        Args:
            trajectory: Trajectory with rewards computed
            
        Returns:
            (scheduler_returns, router_returns) lists
        """
        T = trajectory.total_steps
        gamma = self.config.gamma_discount
        
        # Collect rewards in order
        sch_rewards = [s.scheduler_reward for s in trajectory.steps]
        rou_rewards = [s.router_reward for s in trajectory.steps]
        
        # Compute returns (backward cumulative sum with discount)
        sch_returns = self._compute_discounted_returns(sch_rewards, gamma)
        rou_returns = self._compute_discounted_returns(rou_rewards, gamma)
        
        # Optional normalization
        if self.config.normalize_rewards:
            sch_returns = self._normalize(sch_returns)
            rou_returns = self._normalize(rou_returns)
        
        return sch_returns, rou_returns
    
    def _compute_discounted_returns(
        self,
        rewards: List[float],
        gamma: float,
    ) -> List[float]:
        """Compute discounted cumulative returns from the end."""
        T = len(rewards)
        returns = [0.0] * T
        
        # G_T = r_T
        # G_t = r_t + γ * G_{t+1}
        running_return = 0.0
        for t in range(T - 1, -1, -1):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return
        
        return returns
    
    def _normalize(self, values: List[float]) -> List[float]:
        """Normalize values to have zero mean and unit variance."""
        if len(values) == 0:
            return values
        
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / max(len(values), 1)
        std = math.sqrt(var) + 1e-8
        
        return [(v - mean) / std for v in values]
    
    def _clip_reward(self, reward: float) -> float:
        """Clip reward to configured range."""
        return max(-self.config.reward_clip, min(self.config.reward_clip, reward))


def compute_task_reward_from_evaluator(
    correct: bool,
    task_type: str = "default",
    partial_score: Optional[float] = None,
) -> float:
    """
    Convert evaluator result to numerical task reward R^task.

    Per the paper (Appendix A), R^task lies in [0, 1]: a binary correctness
    indicator for the reasoning benchmarks (GSM-hard, SciBench, MMLU-Pro), and
    the arithmetic mean of the three component scores for SRDD (passed in via
    `partial_score`).

    Args:
        correct: Whether the task was solved correctly
        task_type: Type of task (unused; kept for call-site compatibility)
        partial_score: Optional partial credit already in [0, 1]

    Returns:
        Numerical reward value in [0, 1]
    """
    if partial_score is not None:
        # Already in [0, 1] (e.g. SRDD's mean of completeness/executability/consistency)
        return max(0.0, min(1.0, float(partial_score)))

    return 1.0 if correct else 0.0


class TrajectoryBuffer:
    """
    Buffer to store episode trajectories for batch training.
    
    Supports:
    - Collecting multiple episodes
    - Converting to training batches
    - Separating scheduler and router data
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.trajectories: List[EpisodeTrajectory] = []
        self.reward_calculator = RewardCalculator()
    
    def add(self, trajectory: EpisodeTrajectory):
        """Add a completed trajectory."""
        if len(self.trajectories) >= self.max_size:
            self.trajectories.pop(0)  # Remove oldest
        self.trajectories.append(trajectory)
    
    def clear(self):
        """Clear all trajectories."""
        self.trajectories.clear()
    
    def get_training_batch(
        self,
        compute_rewards: bool = True,
    ) -> Dict[str, List]:
        """
        Convert trajectories to training batch format.
        
        Returns dict with:
        - scheduler_prompts: List of prompts
        - scheduler_actions: List of action strings
        - scheduler_returns: List of returns G_t
        - scheduler_log_probs: List of original log probs
        - router_prompts: List of prompts
        - router_actions: List of action index lists
        - router_returns: List of returns G_t
        - router_log_probs: List of original log probs
        """
        batch = {
            "scheduler_prompts": [],
            "scheduler_actions": [],
            "scheduler_returns": [],
            "scheduler_log_probs": [],
            "router_prompts": [],
            "router_actions": [],
            "router_returns": [],
            "router_log_probs": [],
        }
        
        for traj in self.trajectories:
            if compute_rewards:
                traj = self.reward_calculator.compute_rewards(traj)
            
            sch_returns, rou_returns = self.reward_calculator.compute_returns(traj)
            
            for i, step in enumerate(traj.steps):
                # Scheduler data
                batch["scheduler_prompts"].append(step.scheduler_prompt)
                batch["scheduler_actions"].append(step.scheduler_action)
                batch["scheduler_returns"].append(sch_returns[i])
                batch["scheduler_log_probs"].append(step.scheduler_log_prob)
                
                # Router data (only if routing happened)
                if step.router_prompt:
                    batch["router_prompts"].append(step.router_prompt)
                    batch["router_actions"].append(step.router_action)
                    batch["router_returns"].append(rou_returns[i])
                    batch["router_log_probs"].append(step.router_log_prob)
        
        return batch
    
    def __len__(self) -> int:
        return len(self.trajectories)
    
    def total_steps(self) -> int:
        """Total number of steps across all trajectories."""
        return sum(t.total_steps for t in self.trajectories)

