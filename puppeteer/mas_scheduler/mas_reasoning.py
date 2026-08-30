"""MAS-style reasoning loop that replaces the original REINFORCE-based policy.

This module implements the main reasoning loop using:
- Scheduler: To decide which agent acts next
- Router: To select which memories the agent sees
- Memer: To manage and summarize execution history

The key difference from the original GraphReasoning:
- No policy network / reinforcement learning
- Pure LLM-based dynamic scheduling
- Explicit memory management with summaries

Extended for RL training:
- Collects step-level data for GRPO training
- Integrates agent self-feedback for router reward (useful_count)
- Tracks precise token consumption for scheduler reward
"""
from __future__ import annotations

import copy
import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from .config import MASConfig, DEFAULT_MAS_CONFIG
from .llm import MASLLMClient
from .memer import Memer
from .scheduler import MASScheduler, AgentRegistry, AgentSpec, build_scheduler_prompt
from .router import MASRouter, build_router_prompt, format_memory_candidates
from .task_state import MASTaskState, AgentResult
from .agent_feedback import (
    AgentFeedbackCollector,
    MemoryUsefulnessFeedback,
    extract_agent_output_text,
    generate_memory_feedback_prompt,
    parse_memory_feedback_response,
)

logger = logging.getLogger("MAS")


class MASReasoning:
    """
    MAS-style multi-agent reasoning system.
    
    Replaces the original GraphReasoning + ContinuousREINFORCE with:
    - LLM-based Scheduler for agent selection
    - LLM-based Router for memory routing
    - Memer for memory management
    
    Main loop:
    1. Scheduler chooses next agent based on sum_memory
    2. Router selects TopN memories from TopM candidates
    3. Agent executes with routed memories as context
    4. Agent self-reports memory usefulness AFTER finishing (paper Eq. 11)
    5. Memer ingests output, updates summaries
    6. Repeat until termination

    Extended for RL training:
    - Collects (prompt, response, step_info) for GRPO training
    - Agent post-hoc self-feedback provides useful_count for allocator reward
    - Precise token counting for scheduler reward
    """
    
    def __init__(
        self,
        task: Dict[str, Any],
        agents: Dict[str, Any],  # name -> agent instance
        config: MASConfig = DEFAULT_MAS_CONFIG,
        workspace_path: str = "./logs",
        log_manager: Any = None,
        enable_training_data_collection: bool = False,
    ):
        self.task = task
        self.agents = agents
        self.config = config
        self.workspace_path = workspace_path
        self.log_manager = log_manager
        self.enable_training_data_collection = enable_training_data_collection
        
        # Initialize LLM client (shared by all components)
        self._llm = MASLLMClient(config)
        
        # Build agent registry from provided agents
        self._registry = self._build_registry(agents)
        
        # Initialize MAS components
        self.memer = Memer(config, self._llm)
        self.scheduler = MASScheduler(self._registry, self._llm, config)
        self.router = MASRouter(self._llm, config)
        
        # Task state
        self.task_state = MASTaskState(task)
        
        # Agent feedback collector for router reward
        self.feedback_collector = AgentFeedbackCollector(self._llm)
        
        # Current step's routed memories (for feedback)
        self._current_routed_memories: List[Dict] = []
        
        # Final answer
        self.final_answer = ""
        self.answers: List[str] = []
        
        logger.info("=" * 60)
        logger.info("[MASReasoning] Initialized")
        logger.info(f"Task: {task.get('Question', task.get('question', 'N/A'))[:100]}...")
        logger.info(f"Available agents: {list(agents.keys())}")
        logger.info(f"Training data collection: {enable_training_data_collection}")
        logger.info("=" * 60)
    
    def _build_registry(self, agents: Dict[str, Any]) -> AgentRegistry:
        """Build AgentRegistry from puppeteer agent instances."""
        specs = []
        for name, agent in agents.items():
            # Extract agent info
            role_prompt = getattr(agent, 'role_prompt', '')
            actions = getattr(agent, 'actions', [])
            
            # Build capabilities description
            capabilities = []
            for action in actions:
                if action in ['reasoning', 'critique', 'reflect', 'conclude', 'summarize']:
                    capabilities.append('reasoning')
                elif action in ['run_python']:
                    capabilities.append('code execution')
                elif action in ['search_bing', 'search_arxiv', 'access_website']:
                    capabilities.append('web search')
                elif action in ['read_file']:
                    capabilities.append('file reading')
                elif action in ['terminate']:
                    capabilities.append('termination')
                elif action in ['planning']:
                    capabilities.append('planning')
                elif action in ['question']:
                    capabilities.append('questioning')
                elif action in ['modify']:
                    capabilities.append('error correction')
            
            spec = AgentSpec(
                name=name,
                description=role_prompt[:200] if role_prompt else f"Agent for {actions}",
                capabilities=capabilities or ['general'],
                actions=actions,
            )
            specs.append(spec)
        
        return AgentRegistry(specs)
    
    def start(self, global_info: Any) -> None:
        """Initialize the reasoning process."""
        task_id = self.task.get('id', str(uuid.uuid4()))
        question = self.task.get('Question', self.task.get('question', ''))
        
        self.memer.start_task(task_id, question)
        self.global_info = global_info
        
        logger.info(f"[MASReasoning] Started task {task_id}")
    
    def n_step(self, n: int) -> tuple:
        """
        Run up to n reasoning steps.
        
        Returns:
            (final_answer, ground_truth_answer)
        """
        naturally_finished = False
        
        for step in range(n):
            logger.info(f"\n{'='*20} Step {step + 1}/{n} {'='*20}")
            
            # Check termination conditions
            if self.task_state.finished:
                logger.info("[MASReasoning] Task marked as finished")
                naturally_finished = True
                break
            
            if self.task_state.steps >= self.config.max_steps:
                logger.info(f"[MASReasoning] Reached max steps ({self.config.max_steps})")
                break
            
            # Run one step
            terminated = self._step()
            
            if terminated:
                logger.info("[MASReasoning] Terminated by scheduler or agent")
                naturally_finished = True
                break
        
        # Finalize and return answer
        return self._finalize()
    
    def _step(self) -> bool:
        """
        Execute one reasoning step.
        
        Returns:
            True if should terminate, False otherwise
        """
        # Begin step metrics tracking
        step_metrics = self.task_state.begin_step()
        step_token_count = 0
        
        # 1. Scheduler decides which agent to activate
        scheduler_desc = self.scheduler.describe_agents()
        scheduler_view = self.task_state.to_scheduler_view(scheduler_desc)

        # Construct scheduler prompt for training. Uses the SAME builder that
        # MASScheduler._llm_choose sends to the policy, so the recorded prompt is
        # exactly the one that produced the action (required for on-policy RL).
        try:
            question = self.task.get("Question", self.task.get("question", ""))
            agent_specs_prompt = ""
            if isinstance(scheduler_desc, dict) and "agent_specs_prompt" in scheduler_desc:
                agent_specs_prompt = scheduler_desc["agent_specs_prompt"]
            else:
                # fallback: join agent specs from registry
                agent_specs_prompt = self._registry.to_prompt()
            sch_sys, sch_user = build_scheduler_prompt(
                question=question,
                agent_specs_prompt=agent_specs_prompt,
                sum_memory=self.task_state.sum_memory,
                pre_agent=self.task_state.pre_agent,
                pre_mem=self.task_state.pre_mem,
            )
            scheduler_prompt_text = f"{sch_sys}\n\n{sch_user}"
        except Exception as e:
            logger.debug(f"[MASReasoning] Failed to format scheduler prompt: {e}")
            scheduler_prompt_text = ""

        agent_name = self.scheduler.choose(scheduler_view)
        # Record scheduler decision (prompt/response) for training
        self.task_state.record_scheduler_decision(
            action=agent_name if agent_name else "",
            log_prob=0.0,
            prompt=scheduler_prompt_text,
        )
        
        if agent_name is None:
            logger.info("[MASReasoning] Scheduler returned DONE")
            return True
        
        if agent_name not in self.agents:
            logger.warning(f"[MASReasoning] Unknown agent: {agent_name}, using fallback")
            agent_name = list(self.agents.keys())[0]
        
        logger.info(f"[MASReasoning] Step {self.task_state.steps + 1}: Scheduler selected -> {agent_name}")
        
        agent = self.agents[agent_name]
        
        # 2. Router selects memories for this agent
        # 获取 agent 描述（第三人称）供 router 参考
        agent_spec = self._registry.get(agent_name)
        agent_description = agent_spec.description if agent_spec else ""
        
        topm_nodes = self.memer.retrieve()
        routed_memories = self.router.route(
            task=self.task,
            now_agent=agent_name,
            sum_memory=self.task_state.sum_memory,
            topm_nodes=topm_nodes,
            pre_agent=self.task_state.pre_agent,
            pre_mem=self.task_state.pre_mem,
            agent_description=agent_description,
        )
        
        routed_count = len(routed_memories)
        logger.info(f"[MASReasoning] Router provided {routed_count} memories")
        
        # Store routed memories for feedback collection
        self._current_routed_memories = routed_memories
        self.feedback_collector.set_routed_memories(routed_memories)
        
        # Build router prompt string for training record. Uses the SAME builder
        # and the SAME candidate list (topm_nodes = M^candidate_t) that the
        # allocator policy saw. Recording the post-selection `routed_memories`
        # here would leak the answer into the prompt.
        try:
            router_prompt_text = "{}\n\n{}".format(
                *build_router_prompt(
                    question=self.task.get("Question", self.task.get("question", "")),
                    now_agent=agent_name,
                    sum_memory=self.task_state.sum_memory,
                    candidates=format_memory_candidates(topm_nodes),
                    pre_agent=self.task_state.pre_agent,
                    pre_mem=self.task_state.pre_mem,
                    top_n=self.config.top_n,
                    agent_description=agent_description,
                )
            )
        except Exception as e:
            logger.debug(f"[MASReasoning] Failed to format router prompt: {e}")
            router_prompt_text = ""

        # Recover which candidate indices the allocator actually selected, so the
        # recorded action matches the recorded prompt's indexing.
        selected_indices: List[int] = []
        for mem in routed_memories:
            for idx, node in enumerate(topm_nodes):
                if idx in selected_indices:
                    continue
                if node.node_type == mem.get("node_type") and node.summary == mem.get("summary"):
                    selected_indices.append(idx)
                    break

        # Record router decision
        self.task_state.record_router_decision(
            action=selected_indices,
            log_prob=0.0,
            prompt=router_prompt_text,
            routed_count=routed_count,
        )
        
        # 3. Prepare context and execute agent
        # Inject routed memories into global_info for the agent
        self._inject_routed_memories(routed_memories)
        
        # 针对 MMLU-Pro 任务，为特定 agent 增强提示词
        self._enhance_agent_prompt_for_mmlu(agent, agent_name)
        
        # Activate agent
        agent.activate(self.global_info, initial_dialog_history=agent.initial_dialog_history)
        
        # 4. 执行 agent，然后事后收集记忆有用性反馈（论文 Eq. 11）
        question = self.task.get('Question', self.task.get('question', ''))

        # 检查是否需要收集 feedback（评测时可关闭）
        should_collect_feedback = getattr(self.config, 'collect_feedback', True)

        current_action, terminated = agent.take_action(
            self.global_info,
            external_tools_enabled=True,
        )

        if self.enable_training_data_collection and routed_count > 0 and should_collect_feedback:
            # 论文 Eq. 11：active agent 完成工作**之后**，对每条被分配的记忆返回 u_i。
            # 判定器必须看到 agent 的实际输出，因此这一步只能串行。
            feedback_result = self.feedback_collector.collect_feedback_posthoc(
                agent_name=agent_name,
                step_idx=self.task_state.steps + 1,
                question=question,
                agent_output=extract_agent_output_text(current_action),
                task_context=self.task_state.sum_memory,
            )
            useful_count = feedback_result.useful_count
            logger.info(f"[MASReasoning] Post-hoc feedback: {useful_count}/{routed_count} memories useful")
        else:
            useful_count = 0
        
        # 5. Extract token count from agent action
        if hasattr(current_action, 'to_dict'):
            action_dict = current_action.to_dict()
            step_token_count = action_dict.get('tokens', 0)
            if step_token_count == 0:
                # Try to get from cost
                cost = action_dict.get('cost', {})
                if isinstance(cost, dict):
                    step_token_count = cost.get('tokens', 0)
        
        # Try to get precise token count from query manager
        try:
            from model.query_manager import query_manager
            last_usage = query_manager.get_last_token_usage()
            if last_usage and last_usage.total_tokens > 0:
                step_token_count = last_usage.total_tokens
                logger.debug(f"[MASReasoning] Precise token count: {step_token_count}")
        except Exception as e:
            logger.debug(f"[MASReasoning] Could not get precise token count: {e}")
        
        # Record token count
        self.task_state.record_token_count(step_token_count)
        
        # 6. Convert agent output to AgentResult
        result = self._convert_to_agent_result(agent_name, current_action, terminated)
        
        logger.info(f"[MASReasoning] {agent_name} output summary: {result.summary[:100]}...")
        
        # Record agent feedback for router reward (only if collecting feedback)
        if should_collect_feedback:
            self.task_state.record_agent_feedback(useful_count)
            logger.info(f"[MASReasoning] Agent feedback: {useful_count}/{routed_count} memories useful")
        
        # 7. Memer ingests output and updates memory
        node = self.memer.ingest(result, task_question=question)
        summary = node.summary
        sum_memory = self.memer.provide_summary()
        
        # 8. Update task state
        self.task_state.register_step(result, summary, sum_memory)
        
        # Update global_info
        self.global_info.update(current_action)
        
        # Collect answers
        if result.raw_output.get("answer"):
            self.answers.append(result.raw_output["answer"])
        
        # Clear feedback collector for next step
        self.feedback_collector.clear_current_step()
        
        return terminated
    
    def _collect_agent_feedback(
        self,
        agent_name: str,
        agent_response: str,
        routed_count: int,
    ) -> int:
        """
        Collect agent's self-feedback on memory usefulness.

        NOTE: currently unused. `_step` calls
        AgentFeedbackCollector.collect_feedback_posthoc directly, which is the
        paper-faithful path (Eq. 11). Kept for the case where the worker agent
        itself emits a USEFUL_MEMORIES line in its response.

        This provides the u_{i,t} value for the allocator reward:
        R_agent = u_{i,t} / |M^context_t|

        Strategy:
        1. Try to parse feedback from agent's response (if it included USEFUL_MEMORIES)
        2. If no explicit feedback, use LLM to evaluate based on agent output
        3. Fallback to heuristic if both fail

        Returns:
            Number of memories the agent found useful
        """
        if routed_count == 0:
            return 0
        
        step_idx = self.task_state.steps + 1
        
        # First, try to parse from agent's response
        feedback = self.feedback_collector.parse_agent_response(
            agent_response=agent_response,
            agent_name=agent_name,
            step_idx=step_idx,
        )
        
        # If agent provided explicit feedback (confidence > 0), use it
        if feedback.confidence > 0.5:
            return feedback.useful_count
        
        # Otherwise, use LLM to evaluate (if enabled for training)
        if self.enable_training_data_collection and self._llm is not None:
            try:
                feedback = self.feedback_collector.collect_feedback_from_llm(
                    agent_name=agent_name,
                    step_idx=step_idx,
                )
                return feedback.useful_count
            except Exception as e:
                logger.warning(f"[MASReasoning] LLM feedback collection failed: {e}")
        
        # Fallback: heuristic based on agent success
        # If agent succeeded, assume at least 1 memory was useful
        return 1 if routed_count > 0 else 0
    
    def _inject_routed_memories(self, routed_memories: List[Dict]) -> None:
        """Inject routed memories into the agent's context.
        
        This makes the routed memories available to the agent through global_info.
        """
        # Store routed memories in a format the agent can use
        memory_context = []
        for mem in routed_memories:
            memory_context.append({
                "source": mem.get("node_type", "unknown"),
                "summary": mem.get("summary", ""),
                "content": mem.get("content", {}),
            })
        
        # Add to workflow's valid_reasoning_results for agents to see
        if hasattr(self.global_info, 'workflow'):
            # Prepend memory context to workflow state
            if memory_context:
                context_str = "\n[Previous Agent Memories]\n"
                for mc in memory_context[-3:]:  # Last 3 memories
                    context_str += f"- {mc['source']}: {mc['summary']}\n"
                
                # This will be included in agent prompts
                if not hasattr(self.global_info.workflow, '_mas_context'):
                    self.global_info.workflow._mas_context = ""
                self.global_info.workflow._mas_context = context_str
    
    def _finalize(self) -> tuple:
        """Finalize reasoning and return the answer."""
        logger.info("\n" + "=" * 60)
        logger.info("[MASReasoning] Finalizing")
        
        # Collect all answers
        all_answers = self.answers + self.task_state.answers
        
        if all_answers:
            # Use the last answer
            self.final_answer = all_answers[-1]
        elif self.global_info.answers:
            self.final_answer = self.global_info.answers[-1]
        else:
            self.final_answer = ""
        
        # Get ground truth
        ground_truth = self.task.get("Answer", self.task.get("answer", ""))
        task_type = self.task.get("type", "").lower()
        
        # 对于 MMLU-Pro，尝试提取干净的选项字母
        if task_type in ["mmlu-pro", "mmlu_pro", "mmlu"] and self.final_answer:
            from tasks.evaluator import BenchmarkEvaluator
            extracted = BenchmarkEvaluator.extract_mmlu_answer(self.final_answer)
            if extracted:
                logger.info(f"[MMLU] Extracted clean answer: {extracted} from raw output")
                self.final_answer = extracted
        
        # Safe preview: convert to string to avoid slicing non-string objects (e.g., dict)
        fa_preview = str(self.final_answer)
        gt_preview = str(ground_truth)
        logger.info(f"Final Answer: {fa_preview[:200] if fa_preview else 'None'}...")
        logger.info(f"Ground Truth: {gt_preview[:200] if gt_preview else 'None'}...")
        logger.info(f"Total Steps: {self.task_state.steps}")
        logger.info(f"Agent Sequence: {' -> '.join(self.task_state.get_agent_sequence())}")
        logger.info("=" * 60)
        
        return self.final_answer, ground_truth
    
    def get_execution_history(self) -> List[Dict]:
        """Get the full execution history."""
        return [
            {
                "agent": h.agent,
                "summary": h.summary,
                "control": h.control,
                "raw_output": h.raw_output,
            }
            for h in self.task_state.history
        ]
    
    def visualize_path(self) -> None:
        """Visualize the reasoning path (compatibility method)."""
        logger.info("\n[Reasoning Path Visualization]")
        for i, entry in enumerate(self.task_state.history):
            logger.info(f"  Step {i+1}: {entry.agent} -> {entry.summary[:80]}...")
    
    def visualize_graph(self) -> None:
        """Visualize the memory graph (compatibility method)."""
        logger.info("\n[Memory Graph Visualization]")
        nodes = self.memer.get_all_nodes()
        for node in nodes:
            logger.info(f"  [{node.node_type}] {node.summary[:80]}...")
            if node.temporal_prev:
                logger.info(f"    <- temporal: {node.temporal_prev[:8]}...")
            if node.semantic_neighbors:
                logger.info(f"    <-> semantic: {len(node.semantic_neighbors)} neighbors")
    
    def _enhance_agent_prompt_for_mmlu(self, agent, agent_name: str) -> None:
        """
        针对 MMLU-Pro 任务，为特定 agent 增强提示词。

        **论文未描述此机制，默认关闭**（config.enable_mmlu_prompt_injection=False）。
        仅在需要复现原先报告的 MMLU-Pro 数字时开启。

        对于选择题任务，需要强调：
        1. 只输出选项字母（A-J）
        2. 不要输出数字、金额或其他格式
        3. 确保最终答案是单个大写字母

        Args:
            agent: 要增强的 agent 实例
            agent_name: agent 名称
        """
        if not getattr(self.config, "enable_mmlu_prompt_injection", False):
            return

        task_type = self.task.get("type", "").lower()
        
        # 只对 MMLU-Pro 任务进行增强
        if task_type not in ["mmlu-pro", "mmlu_pro", "mmlu"]:
            return
        
        # 针对负责得出结论的 agent 进行增强
        # 只对明确负责输出最终答案的 agent 进行增强
        target_agents = [
            "concluderagent", "concluder",
            "summarizeragent", "summarizer",
            "terminatoragent", "terminator"
        ]
        
        agent_name_lower = agent_name.lower().replace("_", "").replace("-", "")
        
        if any(target in agent_name_lower for target in target_agents):
            # 增强提示词
            mmlu_instruction = (
                "\n\n**CRITICAL INSTRUCTION FOR MULTIPLE CHOICE QUESTIONS:**\n"
                "This is a multiple choice question. Your final answer MUST be:\n"
                "1. A SINGLE UPPERCASE LETTER from the available options (A, B, C, D, E, F, G, H, I, or J)\n"
                "2. You MUST choose one of the given options - DO NOT create your own answer\n"
                "3. DO NOT output numbers, dollar amounts, calculations, or explanations as the final answer\n"
                "4. Even if you calculated a value, you MUST match it to the closest given option and output that option's letter\n"
                "5. Format your response as: 'The answer is: X' where X is the option letter\n"
                "6. Example: If the correct option is 'B: $32.50', output 'The answer is: B' (NOT '$32.50')\n"
            )
            
            # 将增强指令添加到 agent 的 role_prompt
            if hasattr(agent, 'role_prompt'):
                original_prompt = agent.role_prompt
                if mmlu_instruction not in original_prompt:
                    agent.role_prompt = original_prompt + mmlu_instruction
                    logger.info(f"[MMLU Enhancement] Enhanced prompt for {agent_name}")

