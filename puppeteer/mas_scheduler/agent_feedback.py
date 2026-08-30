"""Agent Feedback Module for Context Allocator Reward Computation.

This module enables work agents to provide explicit feedback on the usefulness
of routed memories. This feedback is used to compute the hit rate R_agent for
the Context Allocator reward function:

    r^rou_{i,t} = α * R_agent + η * R^task_i

where R_agent = u_{i,t} / |M^context_t| (paper Eq. 11)
- u_{i,t}: Number of memories the agent actually used
- |M^context_t|: Number of memories routed to the agent

Paper semantics (Eq. 11): the active agent returns u_i **after completing its
work**, so the judgement is made in retrospect, with the agent's output in hand.

Collection mode:
- `collect_feedback_posthoc` — the paper-faithful path. Runs serially AFTER the
  agent finishes and is given the agent's actual output.
- `collect_feedback_parallel` / `run_agent_with_parallel_feedback` — deprecated.
  These fire concurrently with the agent and therefore cannot see its output,
  making u_i a prior prediction rather than a post-hoc report.
"""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("MAS")


@dataclass
class MemoryUsefulnessFeedback:
    """Feedback from agent about which routed memories were useful.
    
    Attributes:
        routed_count: Total number of memories routed (k_{i,t})
        useful_count: Number agent marked as useful (u_{i,t})
        useful_indices: Specific indices of useful memories
        feedback_text: Raw feedback from agent
        confidence: Agent's confidence in this feedback [0, 1]
    """
    routed_count: int = 0
    useful_count: int = 0
    useful_indices: List[int] = None
    feedback_text: str = ""
    confidence: float = 1.0
    
    def __post_init__(self):
        if self.useful_indices is None:
            self.useful_indices = []
    
    @property
    def hit_rate(self) -> float:
        """Compute h_{i,t} = u_{i,t} / k_{i,t}."""
        if self.routed_count <= 0:
            return 0.0
        return min(1.0, self.useful_count / self.routed_count)


def extract_agent_output_text(action: Any) -> str:
    """Render a worker agent's action into the text shown to the post-hoc judge.

    Accepts a puppeteer `Action` (see puppeteer/agent/agent_info/workflow.py) or
    anything dict-like. Prefers the action's own `to_str()`, then falls back to
    the informative fields of `to_dict()`, then to `str()`.
    """
    if action is None:
        return ""

    # puppeteer Action exposes to_str() -> "Agent: .. Action: .. Result: .."
    to_str = getattr(action, "to_str", None)
    if callable(to_str):
        try:
            text = to_str()
            if text:
                return str(text)
        except Exception:  # pragma: no cover - defensive
            pass

    to_dict = getattr(action, "to_dict", None)
    if callable(to_dict):
        try:
            payload = to_dict()
        except Exception:  # pragma: no cover - defensive
            payload = None
        if isinstance(payload, dict):
            parts = []
            result = payload.get("result")
            if isinstance(result, dict):
                for key in ("step_data", "answer", "code"):
                    value = result.get(key)
                    if value:
                        parts.append(f"{key}: {value}")
            elif result:
                parts.append(f"result: {result}")
            act = payload.get("action")
            if act:
                parts.append(f"action: {act}")
            if parts:
                return "\n".join(parts)

    return str(action)


def generate_memory_feedback_prompt(routed_memories: List[Dict]) -> str:
    """Generate prompt for agent to evaluate memory usefulness.
    
    This prompt is appended to the agent's context after task completion,
    asking which of the provided memories were actually useful.
    
    Args:
        routed_memories: List of memories that were routed to the agent
        
    Returns:
        Prompt string for memory feedback
    """
    if not routed_memories:
        return ""
    
    memory_list = []
    for idx, mem in enumerate(routed_memories):
        source = mem.get("node_type", mem.get("source", "unknown"))
        summary = mem.get("summary", "")[:150]
        memory_list.append(f"[{idx}] ({source}): {summary}")
    
    memories_text = "\n".join(memory_list)
    
    prompt = f"""
---
[Memory Usefulness Self-Assessment]
You were provided with the following {len(routed_memories)} memories to help with your task:

{memories_text}

Based on your work, which memories were ACTUALLY USEFUL for completing your task?
Reply in this format: USEFUL_MEMORIES: <comma-separated indices or "none">

Examples:
- If memory 0 and 2 were useful: USEFUL_MEMORIES: 0, 2
- If only memory 1 was useful: USEFUL_MEMORIES: 1  
- If no memories were useful: USEFUL_MEMORIES: none
- If all memories were useful: USEFUL_MEMORIES: 0, 1, 2

Your assessment (one line only):
"""
    return prompt


def parse_memory_feedback_response(
    response: str,
    routed_count: int,
) -> MemoryUsefulnessFeedback:
    """Parse agent's response to extract memory usefulness feedback.
    
    Args:
        response: Agent's response containing USEFUL_MEMORIES
        routed_count: Number of memories that were routed
        
    Returns:
        MemoryUsefulnessFeedback with extracted information
    """
    feedback = MemoryUsefulnessFeedback(routed_count=routed_count)
    feedback.feedback_text = response
    
    # Pattern to match USEFUL_MEMORIES: <indices>
    pattern = r"USEFUL_MEMORIES:\s*(.+?)(?:\n|$)"
    match = re.search(pattern, response, re.IGNORECASE)
    
    if not match:
        # Try fallback patterns
        fallback_patterns = [
            r"useful.*?:\s*(.+?)(?:\n|$)",
            r"(\d+(?:\s*,\s*\d+)*)",  # Just comma-separated numbers
        ]
        for pat in fallback_patterns:
            match = re.search(pat, response, re.IGNORECASE)
            if match:
                break
    
    if match:
        indices_text = match.group(1).strip().lower()
        
        if "none" in indices_text or "no " in indices_text or indices_text == "":
            feedback.useful_count = 0
            feedback.useful_indices = []
        elif "all" in indices_text:
            feedback.useful_count = routed_count
            feedback.useful_indices = list(range(routed_count))
        else:
            # Parse comma-separated indices
            indices = []
            for part in indices_text.replace(",", " ").split():
                try:
                    idx = int(part.strip())
                    if 0 <= idx < routed_count:
                        indices.append(idx)
                except ValueError:
                    continue
            
            feedback.useful_indices = list(set(indices))  # Deduplicate
            feedback.useful_count = len(feedback.useful_indices)
    else:
        # If parsing fails, assume at least minimal usefulness if agent succeeded
        feedback.useful_count = 0
        feedback.useful_indices = []
        feedback.confidence = 0.5  # Lower confidence for failed parsing
        logger.warning(f"[AgentFeedback] Failed to parse feedback: {response[:100]}...")
    
    logger.debug(f"[AgentFeedback] Parsed: {feedback.useful_count}/{feedback.routed_count} useful")
    return feedback


class AgentFeedbackCollector:
    """Collects memory usefulness feedback from work agents.
    
    This class manages the feedback collection process:
    1. Stores routed memories for each agent step
    2. Generates feedback prompts after agent execution
    3. Parses agent responses to extract usefulness feedback
    4. Provides feedback data for Router reward computation
    """
    
    def __init__(self, llm_client=None):
        self._llm = llm_client
        # Store routed memories per step
        self._current_step_memories: List[Dict] = []
        self._step_feedbacks: Dict[str, MemoryUsefulnessFeedback] = {}
    
    def set_routed_memories(self, memories: List[Dict]) -> None:
        """Store the memories routed to current agent for later feedback."""
        self._current_step_memories = memories.copy() if memories else []
        logger.debug(f"[AgentFeedback] Stored {len(self._current_step_memories)} routed memories")
    
    def get_routed_count(self) -> int:
        """Get count of routed memories for current step."""
        return len(self._current_step_memories)
    
    def generate_feedback_prompt(self) -> str:
        """Generate the feedback prompt for current step."""
        if not self._current_step_memories:
            return ""
        return generate_memory_feedback_prompt(self._current_step_memories)
    
    def collect_feedback_from_llm(
        self,
        agent_name: str,
        step_idx: int,
    ) -> MemoryUsefulnessFeedback:
        """
        Use LLM to generate memory usefulness feedback.
        
        This is called when the agent's own response doesn't contain feedback,
        asking the LLM to evaluate which memories were useful based on the
        agent's output.
        """
        if not self._current_step_memories:
            return MemoryUsefulnessFeedback()
        
        if self._llm is None:
            logger.warning("[AgentFeedback] No LLM client for feedback collection")
            return MemoryUsefulnessFeedback(
                routed_count=len(self._current_step_memories),
                useful_count=0,
            )
        
        prompt = self.generate_feedback_prompt()
        
        try:
            response = self._llm.chat(
                system_prompt="You are evaluating memory usefulness for a task agent.",
                user_prompt=prompt,
                temperature=0.0,
            )
            
            feedback = parse_memory_feedback_response(
                response,
                len(self._current_step_memories),
            )
            
            # Store feedback
            step_key = f"{agent_name}_{step_idx}"
            self._step_feedbacks[step_key] = feedback
            
            return feedback
            
        except Exception as e:
            logger.error(f"[AgentFeedback] LLM feedback error: {e}")
            return MemoryUsefulnessFeedback(
                routed_count=len(self._current_step_memories),
                useful_count=0,
            )
    
    def parse_agent_response(
        self,
        agent_response: str,
        agent_name: str,
        step_idx: int,
    ) -> MemoryUsefulnessFeedback:
        """
        Parse feedback from agent's own response.
        
        If the agent included memory feedback in its response, extract it.
        Otherwise, return empty feedback (can call collect_feedback_from_llm later).
        """
        if not self._current_step_memories:
            return MemoryUsefulnessFeedback()
        
        routed_count = len(self._current_step_memories)
        
        # Try to extract feedback from agent's response
        if "USEFUL_MEMORIES" in agent_response.upper():
            feedback = parse_memory_feedback_response(agent_response, routed_count)
        else:
            # No explicit feedback in response
            feedback = MemoryUsefulnessFeedback(
                routed_count=routed_count,
                useful_count=0,
                confidence=0.0,  # Mark as no explicit feedback
            )
        
        # Store feedback
        step_key = f"{agent_name}_{step_idx}"
        self._step_feedbacks[step_key] = feedback
        
        return feedback
    
    def get_step_feedback(
        self,
        agent_name: str,
        step_idx: int,
    ) -> Optional[MemoryUsefulnessFeedback]:
        """Get stored feedback for a specific step."""
        step_key = f"{agent_name}_{step_idx}"
        return self._step_feedbacks.get(step_key)
    
    def clear_current_step(self) -> None:
        """Clear current step's routed memories."""
        self._current_step_memories = []
    
    def clear_all(self) -> None:
        """Clear all stored data."""
        self._current_step_memories = []
        self._step_feedbacks = {}

    def collect_feedback_posthoc(
        self,
        agent_name: str,
        step_idx: int,
        question: str,
        agent_output: str,
        task_context: str = "",
    ) -> MemoryUsefulnessFeedback:
        """
        事后自评（论文 Eq. 11）：active agent 完成工作后，判断每条被分配的记忆是否
        真的被用上了，返回 u_i ∈ {0, 1}。

        与已废弃的 collect_feedback_parallel 的关键区别：本方法**在 agent 执行结束
        之后**串行调用，并把 agent 的实际输出喂给判定器。并行版本按构造看不到 agent
        的输出，其 u_i 实际是"这些记忆看起来有没有用"的事前预测，而非论文要求的事后
        回报。

        Args:
            agent_name: 当前工作 agent 名称
            step_idx: 当前步骤索引
            question: 原始问题
            agent_output: agent 本步的实际输出（判定的核心依据）
            task_context: 任务上下文（全局摘要）

        Returns:
            MemoryUsefulnessFeedback with useful_count
        """
        if not self._current_step_memories:
            return MemoryUsefulnessFeedback()

        routed_count = len(self._current_step_memories)

        if self._llm is None:
            logger.warning("[AgentFeedback] No LLM client for post-hoc feedback")
            return MemoryUsefulnessFeedback(
                routed_count=routed_count,
                useful_count=0,
            )

        # 构建记忆列表
        memory_list = []
        for idx, mem in enumerate(self._current_step_memories):
            source = mem.get("node_type", mem.get("source", "unknown"))
            summary = mem.get("summary", "")[:200]
            memory_list.append(f"[{idx}] ({source}): {summary}")

        memories_text = "\n".join(memory_list)
        output_text = str(agent_output)[:1500] if agent_output else "(empty output)"

        system_prompt = f"""You are auditing which memories agent "{agent_name}" ACTUALLY USED.

The agent has already FINISHED its work. You are shown the {routed_count} memories it was given and the output it produced. Decide, in retrospect, which memories were actually used or relied upon to produce that output.

Criteria for "actually used":
- The output reuses information, facts, or intermediate results from the memory
- The output builds on, corrects, or explicitly references the memory
- Do NOT mark a memory useful merely because it looks topically related

Output ONLY in this format: USEFUL_MEMORIES: <comma-separated indices or "none">
Examples:
- USEFUL_MEMORIES: 0, 2
- USEFUL_MEMORIES: 1
- USEFUL_MEMORIES: none"""

        user_prompt = f"""Task Question: {question}

Global progress before this step: {task_context[:500] if task_context else "No prior context"}

Memories given to {agent_name}:
{memories_text}

Output that {agent_name} actually produced:
{output_text}

Which of those memories were actually used to produce this output?
Reply with USEFUL_MEMORIES: <indices or none>"""

        try:
            response = self._llm.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
            )

            feedback = parse_memory_feedback_response(response, routed_count)

            # Store feedback
            step_key = f"{agent_name}_{step_idx}"
            self._step_feedbacks[step_key] = feedback

            logger.info(
                f"[AgentFeedback] Post-hoc feedback: {feedback.useful_count}/{routed_count} useful"
            )
            return feedback

        except Exception as e:
            logger.error(f"[AgentFeedback] Post-hoc feedback error: {e}")
            return MemoryUsefulnessFeedback(
                routed_count=routed_count,
                useful_count=0,
            )

    def collect_feedback_parallel(
        self,
        agent_name: str,
        step_idx: int,
        task_context: str,
        question: str,
    ) -> MemoryUsefulnessFeedback:
        """
        并行反馈收集（**已废弃，与论文 Eq. 11 不符**）。

        本方法设计为与 agent 执行并行调用，因此按构造看不到 agent 的输出，判定的是
        "这些记忆看起来有没有用"（事前预测），而论文要求的是 active agent 完成工作
        **之后**返回的 u_i（事后回报）。请改用 collect_feedback_posthoc。

        保留仅为向后兼容。

        Args:
            agent_name: 当前工作 agent 名称
            step_idx: 当前步骤索引
            task_context: 任务上下文（问题 + 历史摘要）
            question: 原始问题

        Returns:
            MemoryUsefulnessFeedback with useful_count
        """
        if not self._current_step_memories:
            return MemoryUsefulnessFeedback()
        
        if self._llm is None:
            logger.warning("[AgentFeedback] No LLM client for parallel feedback")
            return MemoryUsefulnessFeedback(
                routed_count=len(self._current_step_memories),
                useful_count=0,
            )
        
        routed_count = len(self._current_step_memories)
        
        # 构建记忆列表
        memory_list = []
        for idx, mem in enumerate(self._current_step_memories):
            source = mem.get("node_type", mem.get("source", "unknown"))
            summary = mem.get("summary", "")[:200]
            memory_list.append(f"[{idx}] ({source}): {summary}")
        
        memories_text = "\n".join(memory_list)
        
        system_prompt = f"""You are evaluating which memories are useful for agent "{agent_name}" to complete a task.

The agent was provided with {routed_count} memories. Your job is to determine which ones would be ACTUALLY USEFUL for completing the task.

Criteria for "useful":
- Memory provides relevant information for answering the question
- Memory contains reasoning, facts, or conclusions that help the agent
- Memory is not redundant or off-topic

Output ONLY in this format: USEFUL_MEMORIES: <comma-separated indices or "none">
Examples:
- USEFUL_MEMORIES: 0, 2
- USEFUL_MEMORIES: 1
- USEFUL_MEMORIES: none"""

        user_prompt = f"""Task Question: {question}

Current Context: {task_context[:500] if task_context else "No prior context"}

Memories provided to {agent_name}:
{memories_text}

Which memories would be useful for {agent_name} to complete this task?
Reply with USEFUL_MEMORIES: <indices or none>"""

        try:
            response = self._llm.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
            )
            
            feedback = parse_memory_feedback_response(response, routed_count)
            
            # Store feedback
            step_key = f"{agent_name}_{step_idx}"
            self._step_feedbacks[step_key] = feedback
            
            logger.info(f"[AgentFeedback] Parallel feedback: {feedback.useful_count}/{routed_count} useful")
            return feedback
            
        except Exception as e:
            logger.error(f"[AgentFeedback] Parallel feedback error: {e}")
            return MemoryUsefulnessFeedback(
                routed_count=routed_count,
                useful_count=0,
            )


def run_agent_with_parallel_feedback(
    agent_action_fn: Callable,
    feedback_fn: Callable,
    timeout: float = 120.0,
) -> Tuple[Any, MemoryUsefulnessFeedback]:
    """
    并行执行 agent 工作和反馈收集。

    **已废弃，与论文 Eq. 11 不符。** 反馈请求与 agent 并发发出，因此看不到 agent 的
    输出，得到的是事前预测而非事后回报。正确做法：先执行 agent，再调用
    AgentFeedbackCollector.collect_feedback_posthoc(agent_output=...)。

    保留仅为向后兼容。

    - agent_action_fn: 正常的 agent 执行函数（返回 action, terminated）
    - feedback_fn: 反馈收集函数（返回 MemoryUsefulnessFeedback）

    Args:
        agent_action_fn: 无参函数，执行 agent.take_action()，返回 (action, terminated)
        feedback_fn: 无参函数，执行反馈收集，返回 MemoryUsefulnessFeedback
        timeout: 超时时间（秒）

    Returns:
        (agent_result, feedback) 其中 agent_result = (action, terminated)
    """
    agent_result = None
    feedback_result = MemoryUsefulnessFeedback()
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        # 并行提交两个任务
        future_agent = executor.submit(agent_action_fn)
        future_feedback = executor.submit(feedback_fn)
        
        # 等待 agent 完成（主任务）
        try:
            agent_result = future_agent.result(timeout=timeout)
        except Exception as e:
            logger.error(f"[ParallelFeedback] Agent execution failed: {e}")
            agent_result = (None, True)  # 失败时标记 terminated
        
        # 等待 feedback 完成（不阻塞主流程太久）
        try:
            feedback_result = future_feedback.result(timeout=30.0)
        except Exception as e:
            logger.warning(f"[ParallelFeedback] Feedback collection failed: {e}")
            feedback_result = MemoryUsefulnessFeedback()
    
    return agent_result, feedback_result


def integrate_feedback_into_agent_prompt(
    base_prompt: str,
    routed_memories: List[Dict],
) -> str:
    """
    Integrate memory context and feedback request into agent prompt.
    
    This modifies the agent's prompt to:
    1. Include the routed memories as context
    2. Request explicit feedback on memory usefulness
    
    Args:
        base_prompt: Original agent prompt
        routed_memories: Memories routed by the Router
        
    Returns:
        Modified prompt with memory context and feedback request
    """
    if not routed_memories:
        return base_prompt
    
    # Build memory context section
    memory_context = "\n[Relevant Memories from Previous Steps]\n"
    for idx, mem in enumerate(routed_memories):
        source = mem.get("node_type", mem.get("source", "unknown"))
        summary = mem.get("summary", "")
        content = mem.get("content", {})
        
        memory_context += f"\nMemory [{idx}] from {source}:\n"
        memory_context += f"  Summary: {summary}\n"
        
        if isinstance(content, dict):
            if "answer" in content:
                memory_context += f"  Answer: {str(content['answer'])[:200]}\n"
            if "reasoning" in content:
                memory_context += f"  Reasoning: {str(content['reasoning'])[:200]}\n"
    
    # Add feedback request
    feedback_request = """
---
[IMPORTANT: Memory Usefulness Feedback]
After completing your task, include this line in your response:
USEFUL_MEMORIES: <comma-separated indices of memories that were actually useful, or "none">
Example: USEFUL_MEMORIES: 0, 2
---
"""
    
    # Insert memory context at appropriate position
    # Try to insert after system prompt section
    if "<|start_header_id|>user<|end_header_id|>" in base_prompt:
        parts = base_prompt.split("<|start_header_id|>user<|end_header_id|>", 1)
        return parts[0] + "<|start_header_id|>user<|end_header_id|>" + memory_context + feedback_request + parts[1]
    else:
        # Prepend to prompt
        return memory_context + feedback_request + base_prompt

