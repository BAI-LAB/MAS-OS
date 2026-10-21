"""Memer: Memory Manager for MAS-OS.

Memer is responsible for:
1. Receiving agent outputs and converting them to memory nodes (o_t, s_t)
2. Building the memory graph G_t = (V_t, E^T_t, E^S_t) with temporal and
   τ-thresholded semantic edges
3. Maintaining the rolling global summary M^global_t
4. Building the candidate pool M^candidate_t (capacity M) for the Context
   Allocator by traversing the graph backward from v_{t-1}
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None

from .config import MASConfig, DEFAULT_MAS_CONFIG
from .llm import MASLLMClient
from .task_state import AgentResult

logger = logging.getLogger("MAS")


@dataclass
class MemoryNode:
    """A node in the shared graph memory."""
    node_id: str
    node_type: str  # Agent name that created this node
    summary: str    # Semantic summary of the content
    content: Dict[str, Any]  # Full content
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    embedding: Optional[List[float]] = None
    
    # Edges
    temporal_prev: Optional[str] = None  # Previous node in time
    semantic_neighbors: List[str] = field(default_factory=list)


class Memer:
    """Memory manager that maintains the shared memory graph.

    Key responsibilities:
    1. ingest(): Convert agent output to a MemoryNode with a semantic summary
    2. retrieve(): Build the candidate pool M^candidate_t for the allocator
    3. provide_summary(): Expose the rolling global summary for the Scheduler
    """
    
    def __init__(
        self,
        config: MASConfig = DEFAULT_MAS_CONFIG,
        llm_client: Optional[MASLLMClient] = None,
    ):
        self.config = config
        self._llm = llm_client
        self._embedder = None
        
        # Graph memory storage
        self._nodes: Dict[str, MemoryNode] = {}
        self._node_order: List[str] = []  # Temporal order
        
        # Per-task state
        self._current_task_id: Optional[str] = None
        self._current_question: str = ""
        self._global_summary: str = ""
    
    def start_task(self, task_id: str, task_question: str = "") -> None:
        """Initialize memory for a new task."""
        self._current_task_id = task_id
        self._current_question = task_question
        self._nodes.clear()
        self._node_order.clear()
        self._global_summary = ""
        logger.info(f"[Memer] Started task: {task_id}")
    
    def ingest(self, result: AgentResult, task_question: str = "") -> MemoryNode:
        """Convert agent output to a MemoryNode and add to graph.
        
        This method:
        1. Generates a semantic summary of the agent's output
        2. Creates a MemoryNode with the summary and full content
        3. Links it temporally to the previous node
        4. Finds and links semantic neighbors
        5. Updates the global summary
        
        Args:
            result: The agent's output
            task_question: The task question for context
            
        Returns:
            The created MemoryNode
        """
        # Generate semantic summary
        summary = self._generate_summary(result, task_question)
        
        # Create node
        node_id = str(uuid.uuid4())
        embedding = self._compute_embedding(summary)
        node = MemoryNode(
            node_id=node_id,
            node_type=result.name,
            summary=summary,
            content=result.raw_output,
            metadata={
                "control": result.control,
                "step": len(self._node_order),
            },
            embedding=embedding.tolist() if embedding is not None else None,
        )
        
        # Link to previous node (temporal edge)
        if self._node_order:
            prev_id = self._node_order[-1]
            node.temporal_prev = prev_id
        
        # Find semantic neighbors
        node.semantic_neighbors = self._find_semantic_neighbors(node)
        # Make edges bidirectional for new neighbors
        for neighbor_id in node.semantic_neighbors:
            neighbor = self._nodes.get(neighbor_id)
            if neighbor is not None and node.node_id not in neighbor.semantic_neighbors:
                neighbor.semantic_neighbors.append(node.node_id)
        
        # Store node
        self._nodes[node_id] = node
        self._node_order.append(node_id)
        
        # Update global summary
        self._update_global_summary(node)
        
        logger.info(f"[Memer] Ingested {result.name} output -> {summary[:80]}...")
        return node
    
    def retrieve(self, query: str = "", top_m: Optional[int] = None) -> List[MemoryNode]:
        """Build the candidate memory pool M^candidate_t for the Context Allocator.

        Implements the paper's traversal (§3.2.3): starting from the memory node
        at step t-1, walk BACKWARD along both temporal and semantic edges,
        collecting at each step the current node plus its 1-hop neighbours, until
        the candidate set reaches the capacity M (|M^candidate_t| <= M).

        Realised as a breadth-first walk over the memory graph: the frontier is
        seeded with v_{t-1}; each dequeued node contributes itself and its 1-hop
        neighbours (semantic neighbours + temporal predecessor), and those
        neighbours are enqueued so the walk keeps moving backward through the
        graph. If the graph is disconnected, the walk falls back to plain
        reverse-chronological order so the pool is still filled up to M.
        """
        if not self._nodes:
            return []

        limit = top_m if top_m is not None else self.config.top_m
        if limit <= 0:
            return []

        payload: List[MemoryNode] = []
        seen: set[str] = set()

        # Seed with the most recent node, i.e. the node produced at step t-1.
        frontier: List[str] = [self._node_order[-1]]
        expanded: set[str] = set()

        while frontier and len(payload) < limit:
            node_id = frontier.pop(0)
            if node_id in expanded:
                continue
            expanded.add(node_id)

            node = self._nodes.get(node_id)
            if node is None:
                continue

            # Collect the node itself...
            self._add_node_to_payload(node_id, payload, seen, limit)

            # ...then its 1-hop neighbours along both edge types.
            neighbors = list(node.semantic_neighbors)
            if node.temporal_prev:
                neighbors.append(node.temporal_prev)

            for nb_id in neighbors:
                if nb_id not in self._nodes:
                    continue
                self._add_node_to_payload(nb_id, payload, seen, limit)
                if nb_id not in expanded:
                    frontier.append(nb_id)

        # The graph may be disconnected (e.g. no semantic edges and a missing
        # temporal link). Top up from the remaining history, most recent first,
        # so the allocator still sees up to M candidates.
        if len(payload) < limit:
            for node_id in reversed(self._node_order):
                if len(payload) >= limit:
                    break
                self._add_node_to_payload(node_id, payload, seen, limit)

        return payload[:limit]
    def provide_summary(self) -> str:
        """Provide the current global progress summary."""
        return self._global_summary
    
    def _generate_summary(self, result: AgentResult, task_question: str) -> str:
        """Generate a one-sentence summary of what the agent did this step.
        
        The summary should be:
        - One sentence only
        - Describe what the agent did and key output
        - Concise and focused
        """
        if not self.config.use_llm_memer or not getattr(self._llm, "is_available", False):
            return self._fallback_summary(result)
        
        try:
            system_prompt = """You are a memory summarizer. Generate ONE sentence summarizing what this agent did.

Rules:
- Output exactly ONE sentence
- Describe what action was taken and key result/output
- Be concise (under 50 words)
- Include key content, not just action name"""

            # Prepare content for summarization
            content_str = self._format_content_for_summary(result.raw_output)
            
            user_prompt = f"""Agent: {result.name}
Output: {content_str}

One-sentence summary of what {result.name} did:"""

            summary = self._llm.chat(system_prompt, user_prompt, temperature=0.0)
            # Ensure it's really one sentence
            summary = summary.strip().split('\n')[0]
            return summary
        except Exception as e:
            logger.warning(f"[Memer] LLM summary failed: {e}, using fallback")
            return self._fallback_summary(result)
    
    def _fallback_summary(self, result: AgentResult) -> str:
        """Generate a rule-based summary when LLM is unavailable."""
        parts = [f"{result.name}:"]
        
        raw = result.raw_output
        
        # Extract key information based on common output patterns
        if "action" in raw:
            parts.append(f"executed {raw['action']}")
        
        if "reasoning" in raw:
            reasoning = raw["reasoning"]
            if reasoning:
                reasoning_str = str(reasoning)
                if len(reasoning_str) > 100:
                    reasoning_str = reasoning_str[:100] + "..."
                parts.append(f"reasoning: {reasoning_str}")
        
        if "answer" in raw and raw["answer"]:
            answer = str(raw["answer"])
            if len(answer) > 50:
                answer = answer[:50] + "..."
            parts.append(f"answer: {answer}")
        
        if "step_data" in raw and raw["step_data"]:
            data = str(raw["step_data"])
            if len(data) > 100:
                data = data[:100] + "..."
            parts.append(f"result: {data}")
        
        # Control signals
        control = result.control
        if control.get("task_complete"):
            parts.append("[TASK COMPLETE]")
        if control.get("terminated"):
            parts.append("[TERMINATED]")
        
        return " ".join(parts)
    
    def _format_content_for_summary(self, raw_output: Dict[str, Any]) -> str:
        """Format raw output for LLM summarization."""
        parts = []
        
        for key, value in raw_output.items():
            if value is None:
                continue
            val_str = str(value)
            if len(val_str) > 500:
                val_str = val_str[:500] + "..."
            parts.append(f"{key}: {val_str}")
        
        return "\n".join(parts) if parts else "No output"

    def _get_embedder(self):
        """Lazy-load embedding model."""
        if not getattr(self.config, "use_embeddings", False):
            # Explicitly disabled: semantic edges fall back to keyword overlap.
            return None
        if self._embedder is not None:
            return self._embedder
        if SentenceTransformer is None:
            logger.error(
                "use_embeddings=True but sentence-transformers is not installed. "
                "Semantic edges will degrade to keyword overlap, which does NOT "
                "match the paper's memory graph (τ-thresholded cosine similarity). "
                "Install sentence-transformers, or set use_embeddings=False to "
                "acknowledge the degraded mode."
            )
            return None
        try:
            self._embedder = SentenceTransformer(
                self.config.embedding_model,
                device=self.config.embedding_device,
            )
            logger.info(f"[Memer] Loaded embedding model: {self.config.embedding_model}")
        except Exception as e:  # pragma: no cover - external dependency
            logger.error(
                f"Failed to load embedding model {self.config.embedding_model}: {e}. "
                "Semantic edges will degrade to keyword overlap, which does NOT "
                "match the paper's memory graph."
            )
            self._embedder = None
        return self._embedder

    def _compute_embedding(self, text: str) -> Optional[np.ndarray]:
        """Encode text to normalized embedding."""
        embedder = self._get_embedder()
        if embedder is None:
            return None
        try:
            vec = embedder.encode(text, normalize_embeddings=True)
            return np.asarray(vec, dtype=np.float32)
        except Exception as e:  # pragma: no cover - external dependency
            logger.error(f"Embedding encode failed: {e}; falling back to keyword overlap")
            return None

    def _find_semantic_neighbors(self, node: MemoryNode) -> List[str]:
        """Find semantically similar nodes via embedding cosine; fallback to keyword overlap.

        Implements the paper's E^S_t rule: an edge is added between v_t and a
        previous node v_i iff sim(s_i, s_t) >= τ, for i < t.

        The `i < t` constraint holds implicitly: this is called from `ingest()`
        BEFORE the new node is inserted into `self._nodes`, so only earlier nodes
        are visible here. Do not move the call after insertion.
        """
        if node.embedding is None:
            return self._fallback_semantic_neighbors(node)

        neighbors = []
        node_vec = np.array(node.embedding)

        for nid, other in self._nodes.items():
            if nid == node.node_id or other.embedding is None:
                continue

            other_vec = np.array(other.embedding)
            # Embeddings are normalized; dot product = cosine similarity
            sim = float(np.dot(node_vec, other_vec))
            if sim >= self.config.similarity_threshold:
                neighbors.append(nid)

        return neighbors

    def _fallback_semantic_neighbors(self, node: MemoryNode) -> List[str]:
        """Fallback: keyword overlap when embeddings are unavailable.

        Uses `fallback_overlap_threshold`, NOT the embedding threshold τ. Jaccard
        overlap between two word sets rarely exceeds ~0.4 even for near-duplicate
        text, so applying τ = 0.80 here would yield no semantic edges at all.
        """
        neighbors = []
        node_words = set(node.summary.lower().split())
        threshold = getattr(self.config, "fallback_overlap_threshold", 0.30)

        for nid, other in self._nodes.items():
            if nid == node.node_id:
                continue

            other_words = set(other.summary.lower().split())
            overlap = len(node_words & other_words) / max(len(node_words | other_words), 1)

            if overlap >= threshold:
                neighbors.append(nid)

        return neighbors

    def _add_node_to_payload(
        self,
        node_id: str,
        payload: List[MemoryNode],
        seen: set,
        limit: int,
    ) -> None:
        """Append node to payload if not seen and under limit."""
        if node_id in seen or len(payload) >= limit:
            return
        node = self._nodes.get(node_id)
        if node is None:
            return
        payload.append(node)
        seen.add(node_id)
    
    def _update_global_summary(self, new_node: MemoryNode) -> None:
        """Incrementally update global summary based on previous summary + current step.
        
        Incremental update style:
        - Input: previous global summary + this step's summary
        - Output: updated global summary
        
        This avoids re-summarizing all history and keeps context rolling forward.
        """
        if not self.config.use_llm_memer or not getattr(self._llm, "is_available", False):
            self._global_summary = self._fallback_global_summary()
            return
        
        try:
            # Get previous summary (empty string if first step)
            prev_summary = self._global_summary if self._global_summary else "No progress yet."
            
            # Current step info
            current_step = f"{new_node.node_type}: {new_node.summary}"
            
            system_prompt = """You are a progress tracker. Update the global summary by incorporating the new step.

Rules:
- Merge the previous summary with the new step info
- Keep it concise (under 80 words)
- Describe what has been DONE, not the final answer
- Focus on: which agents worked, what actions were taken, what intermediate results were found
- The summary should help decide WHAT TO DO NEXT, not conclude the task"""

            user_prompt = f"""Task: {self._current_question}

Previous summary: {prev_summary}

New step: {current_step}

Generate updated global summary:"""

            self._global_summary = self._llm.chat(system_prompt, user_prompt, temperature=0.0).strip()
        except Exception as e:
            logger.warning(f"[Memer] Global summary LLM failed: {e}")
            self._global_summary = self._fallback_global_summary()
    
    def _fallback_global_summary(self) -> str:
        """Generate a rule-based incremental global summary."""
        if not self._node_order:
            return "Task just started. No agents have worked yet."
        
        # Get last node's summary
        last_node = self._nodes[self._node_order[-1]]
        last_step = f"{last_node.node_type}: {last_node.summary}"
        
        # Build incremental summary
        prev = self._global_summary if self._global_summary else ""
        
        # Simple append with length control
        if prev:
            new_summary = f"{prev} | Step {len(self._node_order)}: {last_step}"
        else:
            new_summary = f"Step 1: {last_step}"
        
        # Keep it under ~200 chars by trimming old content
        if len(new_summary) > 300:
            # Keep the last part
            new_summary = "..." + new_summary[-280:]
        
        # Check for completion signals
        if last_node.metadata.get("control", {}).get("task_complete"):
            new_summary += " [TASK COMPLETE]"
        
        return new_summary
    
    def get_all_nodes(self) -> List[MemoryNode]:
        """Get all nodes in temporal order."""
        return [self._nodes[nid] for nid in self._node_order]
    
    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        """Get a specific node by ID."""
        return self._nodes.get(node_id)

