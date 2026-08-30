"""LoRA-based Scheduler that uses the Dual LoRA model for agent selection.

This replaces the original MASScheduler's LLM calls with the trainable
Scheduler LoRA expert.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .config import MASConfig, DEFAULT_MAS_CONFIG
from .scheduler import AgentRegistry, AgentSpec

logger = logging.getLogger("MAS")


class LoRAScheduler:
    """
    Scheduler that uses the Scheduler LoRA expert for agent selection.
    
    Key differences from MASScheduler:
    - Uses local Llama + LoRA instead of OpenAI API
    - Returns log_prob for training
    - Integrates with DualLoRAModel
    """
    
    def __init__(
        self,
        registry: AgentRegistry,
        dual_lora_model = None,  # DualLoRAModel instance
        config: MASConfig = DEFAULT_MAS_CONFIG,
    ):
        self.registry = registry
        self.config = config
        self._model = dual_lora_model
        self._expert_type = None
        
        if dual_lora_model is not None:
            from .dual_lora_model import ExpertType
            self._expert_type = ExpertType.SCHEDULER
    
    def set_model(self, dual_lora_model):
        """Set or update the dual LoRA model."""
        from .dual_lora_model import ExpertType
        self._model = dual_lora_model
        self._expert_type = ExpertType.SCHEDULER
    
    def choose(
        self,
        scheduler_view: Dict[str, Any],
        return_log_prob: bool = False,
    ) -> Tuple[Optional[str], str, float]:
        """
        Decide which agent should work next.
        
        Args:
            scheduler_view: Dict containing task, pre_mem, pre_agent, sum_memory, agent_specs
            return_log_prob: Whether to return log probability (for training)
            
        Returns:
            (agent_name, prompt, log_prob)
            - agent_name: Name of agent to activate, or None to terminate
            - prompt: The prompt used (for training)
            - log_prob: Log probability of the action (for training)
        """
        task = scheduler_view["task"]
        pre_agent = scheduler_view.get("pre_agent")
        sum_memory = scheduler_view.get("sum_memory", "")
        agent_specs = scheduler_view.get("agent_specs", {})
        
        agent_names = list(agent_specs.keys()) if agent_specs else self.registry.names()
        
        # Get task question
        question = task.get("Question", task.get("question", str(task)))
        
        logger.info(f"[LoRAScheduler] Deciding next agent. Pre-agent: {pre_agent}")
        
        # Build prompt
        from .dual_lora_model import format_scheduler_prompt
        prompt = format_scheduler_prompt(
            question=question,
            sum_memory=sum_memory,
            agent_specs=self.registry.to_prompt(),
            pre_agent=pre_agent,
        )
        
        # Generate with LoRA model
        if self._model is None:
            logger.warning("[LoRAScheduler] No model set, using fallback")
            chosen = self._fallback_choose(sum_memory, agent_names, pre_agent)
            return chosen, prompt, 0.0
        
        try:
            if return_log_prob:
                response, log_probs = self._model.generate(
                    prompt=prompt,
                    expert_type=self._expert_type,
                    max_new_tokens=32,
                    temperature=0.0,
                    return_logprobs=True,
                )
                # Sum log probs for the response
                total_log_prob = log_probs.sum().item() if len(log_probs) > 0 else 0.0
            else:
                response = self._model.generate(
                    prompt=prompt,
                    expert_type=self._expert_type,
                    max_new_tokens=32,
                    temperature=0.0,
                    return_logprobs=False,
                )
                total_log_prob = 0.0
            
            logger.debug(f"[LoRAScheduler] Model response: '{response}'")
            
            # Parse response
            chosen = response.strip().split()[0] if response.strip() else ""
            
            if "DONE" in chosen.upper():
                return None, prompt, total_log_prob
            
            # Normalize agent name
            normalized = self._normalize_agent_name(chosen, agent_names)
            if normalized:
                logger.info(f"[LoRAScheduler] Chose: {normalized}")
                return normalized, prompt, total_log_prob
            else:
                logger.warning(f"[LoRAScheduler] Invalid response '{chosen}', using fallback")
                fallback = self._fallback_choose(sum_memory, agent_names, pre_agent)
                return fallback, prompt, 0.0
                
        except Exception as e:
            logger.error(f"[LoRAScheduler] Error: {e}")
            chosen = self._fallback_choose(sum_memory, agent_names, pre_agent)
            return chosen, prompt, 0.0
    
    def _normalize_agent_name(self, text: str, agent_names: List[str]) -> Optional[str]:
        """Normalize LLM output to a valid agent name."""
        text_upper = text.upper().strip()
        
        for name in agent_names:
            if name.upper() == text_upper:
                return name
            if name.upper() in text_upper:
                return name
        
        # Try partial match
        for name in agent_names:
            name_parts = name.replace("_", " ").replace("-", " ").upper().split()
            if any(part in text_upper for part in name_parts if len(part) > 3):
                return name
        
        return None
    
    def _fallback_choose(
        self,
        sum_memory: str,
        agent_names: List[str],
        pre_agent: Optional[str],
    ) -> Optional[str]:
        """Rule-based fallback when model fails."""
        if not sum_memory:
            for name in agent_names:
                if "planner" in name.lower() or "planning" in name.lower():
                    return name
            for name in agent_names:
                if "reasoning" in name.lower():
                    return name
            return agent_names[0] if agent_names else None
        
        text = sum_memory.lower()
        
        # Check for completion
        if "task complete" in text or "final answer" in text:
            return None
        
        # Check if answer ready
        if "answer is" in text or "result is" in text:
            for name in agent_names:
                if "terminator" in name.lower():
                    return name
        
        # Default: cycle
        if pre_agent and pre_agent in agent_names:
            idx = agent_names.index(pre_agent)
            return agent_names[(idx + 1) % len(agent_names)]
        
        return agent_names[0] if agent_names else None
    
    def describe_agents(self) -> Dict[str, AgentSpec]:
        """Return all agent specifications."""
        return self.registry.all_specs()

