"""Router: Selects which memories the current agent can see.

The Router is responsible for:
1. Receiving TopM candidate memories from Memer
2. Selecting TopN most relevant memories for the current agent
3. Making the agent's context window focused and relevant

Key principle: Different agents may need different memories.
Router understands each agent's role and selects accordingly.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .config import MASConfig, DEFAULT_MAS_CONFIG
from .llm import MASLLMClient
from .memer import MemoryNode

logger = logging.getLogger("MAS")


def format_memory_candidates(topm_nodes: List[MemoryNode]) -> str:
    """Render M^candidate_t as the indexed list shown to the allocator policy.

    Single source of truth for the candidate line format, so the prompt used at
    inference matches the one recorded for training byte-for-byte.
    """
    return "\n".join(
        f"[{idx}] ({node.node_type}): {node.summary}"
        for idx, node in enumerate(topm_nodes)
    )


def build_router_prompt(
    question: str,
    now_agent: str,
    sum_memory: str,
    candidates: str,
    pre_agent: Optional[str] = None,
    pre_mem: str = "",
    top_n: int = 3,
    agent_description: str = "",
) -> tuple[str, str]:
    """Build the Context Allocator prompt as a (system, user) pair.

    This is the single source of truth for the allocator prompt: it is both what
    `MASRouter._llm_route` sends to the policy and what callers record as the
    training prompt.

    Observation follows paper Eq. 9:
        o^con_t = [x, M^candidate_t, M^global_{t-1}, a_{sigma_t}]
    """
    # Agent description 上下文（第三人称）
    agent_desc_text = ""
    if agent_description:
        agent_desc_text = f"\nABOUT {now_agent}: {agent_description}\n"

    system_prompt = f"""You are a memory router for a multi-agent reasoning system.
Your job is to select ONLY the truly relevant memories for the current agent.
{agent_desc_text}
RULES:
- Select between 1 and {top_n} memories (inclusive)
- You MUST select at least 1 memory
- Do NOT select more than {top_n} memories
- Only select memories that genuinely help {now_agent} do its job
- Do NOT output duplicate indices (each index may appear at most once)
- Do NOT pad the list - if only 1 or 2 memories are useful, that's fine
- Quality over quantity

GUIDELINES:
- Consider what specific information this agent type needs based on its description above
- Prioritize memories directly relevant to the agent's task
- Previous agent's output is often highly relevant
- Avoid redundant or tangential memories

OUTPUT FORMAT:
Reply with comma-separated indices only. Indices must be unique (no repeats like "0,0" or "1,1"). Examples:
- If only 1 useful: "2"
- If 2 useful: "0, 3"
- If 3 useful: "1, 2, 4" """

    # Build previous step context
    prev_context = ""
    if pre_agent:
        prev_context = f"Previous agent: {pre_agent}\n"
        if pre_mem:
            prev_context += (
                f"Previous step summary: {pre_mem[:200]}...\n"
                if len(pre_mem) > 200
                else f"Previous step summary: {pre_mem}\n"
            )

    user_prompt = f"""Task question: {question}
Current agent: {now_agent}
{prev_context}Global progress: {sum_memory if sum_memory else 'None'}

Available memories (select 1-{top_n} that are truly useful):
{candidates}

Which memories should {now_agent} see? Reply with indices only:"""

    return system_prompt, user_prompt


class MASRouter:
    """
    Router that selects memories for the current agent via LLM.
    
    Inputs:
    - task: The full task object
    - now_agent: Currently selected agent name
    - sum_memory: Global progress summary
    - pre_agent: Last agent that worked (for recent context)
    - pre_mem: Last step's memory summary (for recent context)
    - topm_nodes: TopM candidate memory nodes from Memer
    
    Output: TopN memory items (dict format) for the current agent

    Special rules (only when config.enable_role_based_routing is True — these
    bypass the learned policy and are not part of the paper's method):
    - For planning agents: prioritize failed verification results
    - For termination agents: prioritize answers and conclusions
    """
    
    def __init__(
        self,
        llm_client: Optional[MASLLMClient] = None,
        config: MASConfig = DEFAULT_MAS_CONFIG,
    ):
        self.config = config
        self._llm = llm_client
    
    def route(
        self,
        task: Dict[str, Any],
        now_agent: str,
        sum_memory: str,
        topm_nodes: List[MemoryNode],
        pre_agent: Optional[str] = None,
        pre_mem: str = "",
        agent_description: str = "",
    ) -> List[Dict]:
        """
        Select TopN memories from TopM candidates for the current agent.
        
        Args:
            task: The task object
            now_agent: Name of the agent that will receive these memories
            sum_memory: Global progress summary
            topm_nodes: Candidate memory nodes from Memer
            pre_agent: Last agent that worked (for recent context)
            pre_mem: Last step's memory summary (for recent context)
            agent_description: Description of the current agent (third-person)
            
        Returns:
            List of dicts with node_id, summary, content, metadata
        """
        if not topm_nodes:
            return []
        
        question = task.get("Question", task.get("question", str(task)))

        logger.debug(f"[Router] Routing for {now_agent}, {len(topm_nodes)} candidates")
        logger.debug(f"[Router] Previous agent: {pre_agent}, pre_mem: {pre_mem[:100] if pre_mem else 'None'}...")

        # Non-paper extension, off by default: hand-written rules that bypass the
        # learned allocation policy pi_phi entirely for planner/terminator agents.
        # Enable via config.enable_role_based_routing only to reproduce the
        # originally reported numbers.
        if getattr(self.config, "enable_role_based_routing", False):
            agent_lower = now_agent.lower()

            if "planner" in agent_lower or "planning" in agent_lower:
                return self._route_for_planner(topm_nodes, pre_agent, pre_mem)

            if "terminator" in agent_lower:
                return self._route_for_terminator(topm_nodes)
        
        # Try LLM-based routing (the paper's allocation policy pi_phi)
        if self.config.use_llm_router and getattr(self._llm, "is_available", False):
            routed = self._llm_route(question, now_agent, sum_memory, topm_nodes, pre_agent, pre_mem, agent_description)
            if routed:
                logger.debug(f"[Router] LLM routed {len(routed)} memories")
                return routed
        
        # Fallback to recency-based
        return self._fallback_route(topm_nodes)
    
    def _route_for_planner(
        self, 
        topm_nodes: List[MemoryNode],
        pre_agent: Optional[str] = None,
        pre_mem: str = "",
    ) -> List[Dict]:
        """
        Special routing for planning agents.
        
        Priority:
        1. If pre_agent was a verifier/critic, ensure that memory is included first
        2. Previous verification failures (so planner knows what went wrong)
        3. Previous plans (to avoid repetition)
        4. Recent results
        """
        payload = []
        seen_ids = set()
        
        # 0. If pre_agent was a verifier/critic, prioritize finding that specific memory
        if pre_agent:
            pre_agent_lower = pre_agent.lower()
            if "critic" in pre_agent_lower or "verif" in pre_agent_lower or "reflect" in pre_agent_lower:
                # Find the most recent node from pre_agent
                for node in topm_nodes:
                    if node.node_type == pre_agent and node.node_id not in seen_ids:
                        payload.append(self._node_to_dict(node))
                        seen_ids.add(node.node_id)
                        logger.info(f"[Router] Prioritized pre_agent ({pre_agent}) memory for planner: {node.summary[:60]}...")
                        break
        
        # 1. Add critic/verifier outputs first
        for node in topm_nodes:
            node_type_lower = node.node_type.lower()
            if "critic" in node_type_lower or "verif" in node_type_lower or "reflect" in node_type_lower:
                if node.node_id not in seen_ids:
                    payload.append(self._node_to_dict(node))
                    seen_ids.add(node.node_id)
                    logger.debug(f"[Router] Added critic/verifier memory for planner: {node.summary[:60]}...")
        
        # 2. Add previous planner outputs
        for node in topm_nodes:
            if "planner" in node.node_type.lower() or "planning" in node.node_type.lower():
                if node.node_id not in seen_ids and len(payload) < self.config.top_n:
                    payload.append(self._node_to_dict(node))
                    seen_ids.add(node.node_id)
        
        # 3. Fill with other recent memories
        for node in topm_nodes:
            if node.node_id not in seen_ids and len(payload) < self.config.top_n:
                payload.append(self._node_to_dict(node))
                seen_ids.add(node.node_id)
        
        return payload[:self.config.top_n]
    
    def _route_for_terminator(self, topm_nodes: List[MemoryNode]) -> List[Dict]:
        """
        Special routing for terminator agent.
        
        Priority:
        1. Concluder/summarizer outputs (final answers)
        2. Recent reasoning results
        3. Any answers found
        """
        payload = []
        seen_ids = set()
        
        # 1. Add concluder/summarizer outputs
        for node in topm_nodes:
            node_type_lower = node.node_type.lower()
            if "conclud" in node_type_lower or "summar" in node_type_lower or "respond" in node_type_lower:
                if node.node_id not in seen_ids:
                    payload.append(self._node_to_dict(node))
                    seen_ids.add(node.node_id)
        
        # 2. Add reasoning outputs
        for node in topm_nodes:
            if "reasoning" in node.node_type.lower():
                if node.node_id not in seen_ids and len(payload) < self.config.top_n:
                    payload.append(self._node_to_dict(node))
                    seen_ids.add(node.node_id)
        
        # 3. Add any nodes with answers
        for node in topm_nodes:
            if node.node_id not in seen_ids and len(payload) < self.config.top_n:
                content = node.content
                if content.get("answer") or content.get("final_answer"):
                    payload.append(self._node_to_dict(node))
                    seen_ids.add(node.node_id)
        
        # 4. Fill remaining
        for node in topm_nodes:
            if node.node_id not in seen_ids and len(payload) < self.config.top_n:
                payload.append(self._node_to_dict(node))
                seen_ids.add(node.node_id)
        
        return payload[:self.config.top_n]
    
    def _llm_route(
        self,
        question: str,
        now_agent: str,
        sum_memory: str,
        topm_nodes: List[MemoryNode],
        pre_agent: Optional[str] = None,
        pre_mem: str = "",
        agent_description: str = "",
    ) -> Optional[List[Dict]]:
        """Use LLM to select the most relevant memories."""
        try:
            system_prompt, user_prompt = build_router_prompt(
                question=question,
                now_agent=now_agent,
                sum_memory=sum_memory,
                candidates=format_memory_candidates(topm_nodes),
                pre_agent=pre_agent,
                pre_mem=pre_mem,
                top_n=self.config.top_n,
                agent_description=agent_description,
            )

            response = self._llm.chat(system_prompt, user_prompt, temperature=0.0)
            selected_indices = self._parse_indices(response, len(topm_nodes))
            
            # Ensure at least 1, at most top_n
            if not selected_indices:
                # Fallback: select the first one
                selected_indices = [0]
            
            payload = []
            for idx in selected_indices[:self.config.top_n]:
                node = topm_nodes[idx]
                payload.append(self._node_to_dict(node))
            
            logger.info(f"[Router] LLM selected {len(payload)} memories (max {self.config.top_n})")
            return payload
            
        except Exception as e:
            logger.error(f"[Router] LLM error: {e}")
            return None
    
    def _parse_indices(self, response: str, max_idx: int) -> List[int]:
        """Parse LLM response into list of indices."""
        indices = []
        # Try comma-separated
        parts = response.replace(" ", "").split(",")
        for part in parts:
            try:
                idx = int(part.strip())
                if 0 <= idx < max_idx and idx not in indices:
                    indices.append(idx)
            except ValueError:
                continue
        return indices
    
    def _fallback_route(self, topm_nodes: List[MemoryNode]) -> List[Dict]:
        """Simple fallback: return most recent memories (at least 1, up to top_n)."""
        # Ensure at least 1 memory is returned
        count = min(len(topm_nodes), self.config.top_n)
        count = max(count, 1) if topm_nodes else 0
        
        payload = []
        for node in topm_nodes[:count]:
            payload.append(self._node_to_dict(node))
        
        logger.debug(f"[Router] Fallback selected {len(payload)} memories")
        return payload
    
    def _node_to_dict(self, node: MemoryNode) -> Dict:
        """Convert MemoryNode to dict format for agents."""
        return {
            # Do not expose internal node_id/metadata to worker agents
            "node_type": node.node_type,
            "summary": node.summary,
            "content": node.content,
        }

