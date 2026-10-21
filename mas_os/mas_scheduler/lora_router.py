"""LoRA-based Router that uses the Dual LoRA model for memory selection.

This replaces the original MASRouter's LLM calls with the trainable
Router LoRA expert.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .config import MASConfig, DEFAULT_MAS_CONFIG
from .memer import MemoryNode

logger = logging.getLogger("MAS")


class LoRARouter:
    """
    Router that uses the Router LoRA expert for memory selection.
    
    Key differences from MASRouter:
    - Uses local Llama + LoRA instead of OpenAI API
    - Returns log_prob for training
    - Integrates with DualLoRAModel
    """
    
    def __init__(
        self,
        dual_lora_model = None,  # DualLoRAModel instance
        config: MASConfig = DEFAULT_MAS_CONFIG,
    ):
        self.config = config
        self._model = dual_lora_model
        self._expert_type = None
        
        if dual_lora_model is not None:
            from .dual_lora_model import ExpertType
            self._expert_type = ExpertType.ROUTER
    
    def set_model(self, dual_lora_model):
        """Set or update the dual LoRA model."""
        from .dual_lora_model import ExpertType
        self._model = dual_lora_model
        self._expert_type = ExpertType.ROUTER
    
    def route(
        self,
        task: Dict[str, Any],
        now_agent: str,
        sum_memory: str,
        topm_nodes: List[MemoryNode],
        pre_agent: Optional[str] = None,
        pre_mem: str = "",
        return_log_prob: bool = False,
    ) -> Tuple[List[Dict], str, float, int]:
        """
        Select memories for the current agent.
        
        Args:
            task: The task object
            now_agent: Name of the agent that will receive these memories
            sum_memory: Global progress summary
            topm_nodes: Candidate memory nodes from Memer
            pre_agent: Last agent that worked
            pre_mem: Last step's memory summary
            return_log_prob: Whether to return log probability
            
        Returns:
            (routed_memories, prompt, log_prob, routed_count)
            - routed_memories: List of dicts with node_type, summary, content
            - prompt: The prompt used (for training)
            - log_prob: Log probability of the action
            - routed_count: Number of memories routed (k_{i,t})
        """
        if not topm_nodes:
            return [], "", 0.0, 0
        
        question = task.get("Question", task.get("question", str(task)))
        
        logger.debug(f"[LoRARouter] Routing for {now_agent}, {len(topm_nodes)} candidates")
        
        # Build candidate list for prompt
        candidates = []
        for idx, node in enumerate(topm_nodes):
            candidates.append(f"[{idx}] ({node.node_type}) {node.summary}")
        candidates_text = "\n".join(candidates)
        
        # Build prompt
        from .dual_lora_model import format_router_prompt
        prompt = format_router_prompt(
            question=question,
            now_agent=now_agent,
            sum_memory=sum_memory,
            candidates=candidates_text,
            pre_agent=pre_agent,
            top_n=self.config.top_n,
        )
        
        # Generate with LoRA model
        if self._model is None:
            logger.warning("[LoRARouter] No model set, using fallback")
            payload = self._fallback_route(topm_nodes)
            return payload, prompt, 0.0, len(payload)
        
        try:
            if return_log_prob:
                response, log_probs = self._model.generate(
                    prompt=prompt,
                    expert_type=self._expert_type,
                    max_new_tokens=32,
                    temperature=0.0,
                    return_logprobs=True,
                )
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
            
            logger.debug(f"[LoRARouter] Model response: '{response}'")
            
            # Parse indices from response
            selected_indices = self._parse_indices(response, len(topm_nodes))
            
            # Ensure at least 1, at most top_n
            if not selected_indices:
                selected_indices = [0]  # Fallback: select first
            
            # Build payload
            payload = []
            for idx in selected_indices[:self.config.top_n]:
                node = topm_nodes[idx]
                payload.append(self._node_to_dict(node))
            
            logger.info(f"[LoRARouter] Selected {len(payload)} memories")
            return payload, prompt, total_log_prob, len(payload)
            
        except Exception as e:
            logger.error(f"[LoRARouter] Error: {e}")
            payload = self._fallback_route(topm_nodes)
            return payload, prompt, 0.0, len(payload)
    
    def _parse_indices(self, response: str, max_idx: int) -> List[int]:
        """Parse LLM response into list of indices."""
        indices = []
        # Try comma-separated
        parts = response.replace(" ", "").split(",")
        for part in parts:
            try:
                # Handle cases like "0" or "[0]"
                clean = part.strip("[]")
                idx = int(clean)
                if 0 <= idx < max_idx and idx not in indices:
                    indices.append(idx)
            except ValueError:
                continue
        return indices
    
    def _fallback_route(self, topm_nodes: List[MemoryNode]) -> List[Dict]:
        """Simple fallback: return most recent memories."""
        count = min(len(topm_nodes), self.config.top_n)
        count = max(count, 1) if topm_nodes else 0
        
        payload = []
        for node in topm_nodes[:count]:
            payload.append(self._node_to_dict(node))
        
        logger.debug(f"[LoRARouter] Fallback selected {len(payload)} memories")
        return payload
    
    def _node_to_dict(self, node: MemoryNode) -> Dict:
        """Convert MemoryNode to dict format for agents."""
        return {
            "node_type": node.node_type,
            "summary": node.summary,
            "content": node.content,
        }

