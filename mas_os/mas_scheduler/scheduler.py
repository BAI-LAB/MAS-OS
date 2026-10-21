"""Scheduler: Dynamically decides which agent to activate next.

The Scheduler is a meta-agent that:
1. Reads the global progress summary (sum_memory)
2. Understands what has been accomplished and what's missing
3. Decides which worker agent should act next
4. Can also decide to terminate (DONE)

Key principle: Decision is based on semantic understanding of progress,
not hardcoded workflow sequences.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .config import MASConfig, DEFAULT_MAS_CONFIG
from .llm import MASLLMClient

logger = logging.getLogger("MAS")


@dataclass
class AgentSpec:
    """Specification for a worker agent."""
    name: str
    description: str
    capabilities: List[str]
    actions: List[str]  # The actions this agent can perform
    
    def to_prompt_line(self) -> str:
        """Format for inclusion in LLM prompt - 完整描述，使用第三人称"""
        actions = ", ".join(self.actions) if self.actions else "general"
        caps = ", ".join(self.capabilities) if self.capabilities else "general"
        return f"- {self.name}: {self.description} (actions: {actions}, capabilities: {caps})"


class AgentRegistry:
    """Registry of available worker agents."""
    
    def __init__(self, specs: List[AgentSpec]):
        self._specs: Dict[str, AgentSpec] = {spec.name: spec for spec in specs}
    
    def get(self, name: str) -> Optional[AgentSpec]:
        return self._specs.get(name)
    
    def names(self) -> List[str]:
        return list(self._specs.keys())
    
    def to_prompt(self) -> str:
        """Generate agent descriptions for LLM prompt."""
        lines = [spec.to_prompt_line() for spec in self._specs.values()]
        return "\n".join(lines)
    
    def add(self, spec: AgentSpec) -> None:
        self._specs[spec.name] = spec
    
    def all_specs(self) -> Dict[str, AgentSpec]:
        return self._specs.copy()


def build_scheduler_prompt(
    question: str,
    agent_specs_prompt: str,
    sum_memory: str,
    pre_agent: Optional[str] = None,
    pre_mem: str = "",
) -> tuple[str, str]:
    """Build the Agent Scheduler prompt as a (system, user) pair.

    This is the single source of truth for the scheduler prompt: it is both what
    `MASScheduler._llm_choose` sends to the policy and what callers record as the
    training prompt. Keeping one builder prevents the recorded prompt from
    drifting away from the prompt that actually produced the action.

    Observation follows paper Eq. 7:
        o^sch_t = [x, M^global_{t-1}, s_{t-1}, a_{sigma_{t-1}}, A]
    """
    system_prompt = f"""You are a scheduler for a multi-agent reasoning system.

AVAILABLE AGENTS:
{agent_specs_prompt}

YOUR TASK:
Read the task progress description and decide which agent should work next.

DECISION GUIDELINES:
- If no work has been done → start with a planning/reasoning agent
- If there's a plan but no execution → choose an execution agent (tool, search, code)
- If execution produced results → choose a reasoning agent to analyze
- If analysis suggests problems → choose appropriate agent to fix
- If reasoning concluded with answer → choose TerminatorAgent to finalize
- If task is clearly complete → return DONE

CRITICAL:
- Read the progress description carefully
- Understand the semantic meaning, not just keywords
- Consider what's been accomplished vs what's still needed
- DO NOT select the same agent consecutively (avoid repeating PREVIOUS AGENT)
- Vary agent selection to make progress; repetition wastes steps

Reply with EXACTLY one word: one of the agent names, or DONE"""

    progress_text = sum_memory if sum_memory else "Task just started. No agents have worked yet."

    # Include pre_mem (last step's memory summary s_{t-1}) for better context
    last_step_info = ""
    if pre_agent and pre_mem:
        last_step_info = f"\nLAST STEP OUTPUT ({pre_agent}): {pre_mem[:300]}"

    user_prompt = f"""QUESTION: {question}

PREVIOUS AGENT: {pre_agent if pre_agent else 'None (first step)'}{last_step_info}

CURRENT PROGRESS:
{progress_text}

Which agent should work next? Reply with one word only."""

    return system_prompt, user_prompt


class MASScheduler:
    """
    Scheduler that dynamically selects the next agent via LLM.
    
    Core principles:
    1. Decisions based on sum_memory's natural language description
    2. No hardcoded workflow sequences
    3. Semantic understanding of task progress
    
    Inputs (from scheduler_view):
    - task: The full task object
    - pre_mem: Last step's memory summary
    - pre_agent: Last agent that worked
    - sum_memory: Global progress summary
    - agent_specs: Available agents and their capabilities
    
    Output: Agent name to activate next, or None to terminate
    """
    
    def __init__(
        self,
        registry: AgentRegistry,
        llm_client: Optional[MASLLMClient] = None,
        config: MASConfig = DEFAULT_MAS_CONFIG,
    ):
        self.registry = registry
        self.config = config
        self._llm = llm_client
    
    def choose(self, scheduler_view: Dict[str, Any]) -> Optional[str]:
        """
        Decide which agent should work next based on the current progress.
        
        Args:
            scheduler_view: Dict containing task, pre_mem, pre_agent, sum_memory, agent_specs
            
        Returns:
            Agent name to activate, or None if task should terminate
        """
        task = scheduler_view["task"]
        pre_mem = scheduler_view.get("pre_mem", "")
        pre_agent = scheduler_view.get("pre_agent")
        sum_memory = scheduler_view.get("sum_memory", "")
        agent_specs = scheduler_view.get("agent_specs", {})
        
        agent_names = list(agent_specs.keys()) if agent_specs else self.registry.names()
        
        # Get task question
        question = task.get("Question", task.get("question", str(task)))
        
        logger.info(f"[Scheduler] Deciding next agent. Pre-agent: {pre_agent}")
        logger.debug(f"[Scheduler] pre_mem: {pre_mem[:200] if pre_mem else 'Empty'}")
        logger.debug(f"[Scheduler] sum_memory: {sum_memory[:300] if sum_memory else 'Empty'}")
        
        # Try LLM-based decision (the paper's policy pi_theta)
        if self.config.use_llm_scheduler and getattr(self._llm, "is_available", False):
            chosen = self._llm_choose(question, sum_memory, agent_names, pre_agent, pre_mem)
            if chosen is not None:
                logger.info(f"[Scheduler] LLM chose: {chosen}")
                return chosen
        
        # Fallback to rule-based
        chosen = self._fallback_choose(sum_memory, agent_names, pre_agent)
        logger.info(f"[Scheduler] Fallback chose: {chosen}")
        return chosen
    
    def _llm_choose(
        self,
        question: str,
        sum_memory: str,
        agent_names: List[str],
        pre_agent: Optional[str],
        pre_mem: str = "",
    ) -> Optional[str]:
        """Use LLM to decide the next agent."""
        try:
            system_prompt, user_prompt = build_scheduler_prompt(
                question=question,
                agent_specs_prompt=self.registry.to_prompt(),
                sum_memory=sum_memory,
                pre_agent=pre_agent,
                pre_mem=pre_mem,
            )

            response = self._llm.chat(system_prompt, user_prompt, temperature=0.0)
            raw_response = response.strip()
            logger.debug(f"[Scheduler] LLM raw response: '{raw_response}'")
            
            # Parse response
            chosen = raw_response.split()[0] if raw_response else ""
            
            if "DONE" in chosen.upper():
                return None
            
            # Normalize agent name
            return self._normalize_agent_name(chosen, agent_names)
            
        except Exception as e:
            logger.error(f"[Scheduler] LLM error: {e}")
            return None
    
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
            # Match by key parts (e.g., "Reasoning" matches "ReasoningAgent_gpt4o")
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
        """
        Rule-based fallback when LLM is unavailable.
        
        Uses keyword matching on sum_memory to understand current state.
        """
        if not sum_memory:
            # First step: start with planning or reasoning
            for name in agent_names:
                if "planner" in name.lower() or "planning" in name.lower():
                    return name
            for name in agent_names:
                if "reasoning" in name.lower():
                    return name
            return agent_names[0] if agent_names else None
        
        text = sum_memory.lower()
        
        # Check for completion signals
        completion_signals = [
            "task complete",
            "final answer generated",
            "answer has been generated",
            "successfully answered",
            "task is done",
        ]
        if any(signal in text for signal in completion_signals):
            return None
        
        # Check if we need to terminate
        if "terminate" in text and "should terminate" in text:
            for name in agent_names:
                if "terminator" in name.lower():
                    return name
        
        # Check if we have an answer and need to conclude
        answer_ready_signals = [
            "final answer:",
            "concluded that",
            "answer is",
            "result is",
        ]
        if any(signal in text for signal in answer_ready_signals):
            for name in agent_names:
                if "terminator" in name.lower():
                    return name
            for name in agent_names:
                if "conclud" in name.lower():
                    return name
        
        # Check if we need more reasoning
        reasoning_signals = [
            "need to analyze",
            "need to reason",
            "should think",
            "requires reasoning",
        ]
        if any(signal in text for signal in reasoning_signals):
            for name in agent_names:
                if "reasoning" in name.lower():
                    return name
        
        # Check if we need tool/search
        tool_signals = [
            "need to search",
            "need to run",
            "need to execute",
            "should search",
            "requires execution",
        ]
        if any(signal in text for signal in tool_signals):
            for name in agent_names:
                if "search" in name.lower() or "python" in name.lower():
                    return name
        
        # Check if we need critique/reflection
        reflection_signals = [
            "should review",
            "need to verify",
            "check the answer",
            "validate",
        ]
        if any(signal in text for signal in reflection_signals):
            for name in agent_names:
                if "critic" in name.lower() or "reflect" in name.lower() or "verif" in name.lower():
                    return name
        
        # Default: cycle through agents, avoiding immediate repeat
        if pre_agent and pre_agent in agent_names:
            idx = agent_names.index(pre_agent)
            next_idx = (idx + 1) % len(agent_names)
            return agent_names[next_idx]
        
        return agent_names[0] if agent_names else None
    
    def describe_agents(self) -> Dict[str, AgentSpec]:
        """Return all agent specifications."""
        return self.registry.all_specs()

