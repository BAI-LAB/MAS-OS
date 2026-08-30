"""
使用训练好的 Scheduler/Router LoRA 适配器进行 MAS 评测。

核心原则：**和训练时使用完全一致的 prompt 模板和处理逻辑**
- Prompt 模板：复制自 mas_dual_lora_workflow.py
- 第三人称转换：复制自 mas_dual_lora_workflow.py
- vLLM 调用：使用 chat/completions 接口，分离 system/user 消息

用法:
    # 先启动 vLLM 服务器（见下方说明）
    python eval_mas_with_lora.py gsm-hard test \
        --base_model /path/to/Llama-3.1-8B \
        --scheduler_lora /path/to/scheduler_lora \
        --router_lora /path/to/router_lora \
        --data_limit 10

启动 vLLM 服务器（单独终端）:
    VLLM_ALLOW_RUNTIME_LORA_UPDATING=True python -m vllm.entrypoints.openai.api_server \
        --model /path/to/Llama-3.1-8B \
        --enable-lora --max-lora-rank 64 \
        --port 8000 --trust-remote-code
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
import yaml
import requests
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger("MAS-LoRA-Eval")

# Import puppeteer components
from agent.register.register import agent_global_registry
from agent.agent_info.global_info import GlobalInfo
from utils.log_manager import LogManager

from mas_scheduler import MASConfig, DEFAULT_MAS_CONFIG
from mas_scheduler.scheduler import AgentRegistry, AgentSpec
from mas_scheduler.memer import Memer, MemoryNode
from mas_scheduler.task_state import MASTaskState, AgentResult
from mas_scheduler.llm import MASLLMClient
from tasks.evaluator import BenchmarkEvaluator


# =============================================================================
# 以下函数完全复制自 MARTI/marti/worlds/workflows/mas_dual_lora_workflow.py
# 确保训练和评测使用完全一致的 prompt 模板
# =============================================================================

def format_scheduler_prompt(
    question: str,
    agent_specs_prompt: str,
    sum_memory: str,
    pre_agent: Optional[str] = None,
    pre_mem: str = "",
) -> Tuple[str, str]:
    """
    和 mas_dual_lora_workflow.py 的 format_scheduler_prompt 完全一致。
    
    返回 (system_prompt, user_prompt) 元组，以便正确使用 chat messages 格式。
    这对于 Llama-3 等模型的 chat template 非常重要。
    """
    progress_text = sum_memory if sum_memory else "Task just started. No agents have worked yet."
    pre_agent_text = pre_agent if pre_agent else "None (first step)"
    
    # System prompt - 和 mas_dual_lora_workflow.py 一致
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

    # Include pre_mem (last step's memory summary) for better context
    last_step_info = ""
    if pre_agent and pre_mem:
        last_step_info = f"\nLAST STEP OUTPUT ({pre_agent}): {pre_mem[:300]}"

    # User prompt - 和 mas_dual_lora_workflow.py 一致
    user_prompt = f"""QUESTION: {question}

PREVIOUS AGENT: {pre_agent_text}{last_step_info}

CURRENT PROGRESS:
{progress_text}

Which agent should work next? Reply with one word only."""

    return system_prompt, user_prompt


def format_router_prompt(
    question: str,
    now_agent: str,
    sum_memory: str,
    candidates: str,
    pre_agent: Optional[str] = None,
    pre_mem: str = "",
    top_n: int = 5,
    agent_description: str = "",
) -> Tuple[str, str]:
    """
    和 mas_dual_lora_workflow.py 的 format_router_prompt 完全一致。
    
    返回 (system_prompt, user_prompt) 元组，以便正确使用 chat messages 格式。
    """
    # Build previous step context (和 mas_dual_lora_workflow.py 一致)
    prev_context = ""
    if pre_agent:
        prev_context = f"Previous agent: {pre_agent}\n"
        if pre_mem:
            prev_context += f"Previous step summary: {pre_mem[:200]}...\n" if len(pre_mem) > 200 else f"Previous step summary: {pre_mem}\n"

    # Agent description context (第三人称)
    agent_desc_text = ""
    if agent_description:
        agent_desc_text = f"\nABOUT {now_agent}: {agent_description}\n"

    # System prompt
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

    # User prompt
    user_prompt = f"""Task question: {question}
Current agent: {now_agent}
{prev_context}Global progress: {sum_memory if sum_memory else 'None'}

Available memories (select 1-{top_n} that are truly useful):
{candidates}

Which memories should {now_agent} see? Reply with indices only:"""

    return system_prompt, user_prompt


def convert_to_third_person(text: str) -> str:
    """
    将第二人称描述转换为第三人称，和 mas_dual_lora_workflow.py 的 build_agent_registry 一致。
    """
    result = text
    result = result.replace("You are", "It is").replace("you are", "it is")
    result = result.replace("Your task", "Its task").replace("your task", "its task")
    result = result.replace("Your ", "Its ").replace("your ", "its ")
    result = result.replace("You ", "It ").replace("you ", "it ")
    return result


def build_agent_registry(agents_dict: Dict[str, Any]) -> AgentRegistry:
    """
    从 puppeteer agent 实例构建 AgentRegistry，和 mas_dual_lora_workflow.py 完全一致。
    包含第三人称转换。
    """
    specs = []
    for name, agent in agents_dict.items():
        role_prompt = getattr(agent, 'role_prompt', '')
        actions = getattr(agent, 'actions', [])
        
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
        
        if not capabilities:
            capabilities = ['general']
        
        # 将第二人称改为第三人称，避免模型误解（和 mas_dual_lora_workflow.py 一致）
        desc = role_prompt if role_prompt else f"{name} agent"
        desc = convert_to_third_person(desc)
        # 截取前 200 字符
        desc = desc[:200] if len(desc) > 200 else desc
        
        spec = AgentSpec(
            name=name,
            description=desc,
            capabilities=list(set(capabilities)),
            actions=actions if actions else ['general'],
        )
        specs.append(spec)
    
    return AgentRegistry(specs)


def _enhance_agent_prompt_for_mmlu(
    agent, agent_name: str, task_type: str, enabled: bool = False
) -> None:
    """
    针对 MMLU-Pro 任务，为特定 agent 增强提示词（和 MASReasoning 一致）。

    **论文未描述此机制，默认关闭。** 仅在需要复现原先报告的 MMLU-Pro 数字时，
    通过 MASConfig.enable_mmlu_prompt_injection=True 开启。
    """
    if not enabled:
        return

    # 只对 MMLU-Pro 任务进行增强
    if task_type.lower() not in ["mmlu-pro", "mmlu_pro", "mmlu"]:
        return
    
    # 针对负责得出结论的 agent 进行增强
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


def parse_scheduler_response(response: str, agent_names: List[str]) -> Optional[str]:
    """解析 scheduler 响应，和 mas_dual_lora_workflow.py 一致。"""
    response = response.strip()
    
    if "DONE" in response.upper():
        return None
    
    # Exact match
    for name in agent_names:
        if name.upper() == response.upper():
            return name
        if name.upper() in response.upper():
            return name
    
    # Partial match
    for name in agent_names:
        name_parts = name.replace("_", " ").replace("-", " ").upper().split()
        if any(part in response.upper() for part in name_parts if len(part) > 3):
            return name
    
    return None


def parse_router_response(response: str, max_idx: int) -> List[int]:
    """解析 router 响应，和 mas_dual_lora_workflow.py 一致。"""
    indices = []
    parts = response.replace(" ", "").split(",")
    for part in parts:
        try:
            idx = int(part.strip())
            if 0 <= idx < max_idx and idx not in indices:
                indices.append(idx)
        except ValueError:
            continue
    return indices


# =============================================================================
# vLLM 调用相关
# =============================================================================

class LoRALLMClient:
    """
    LLM Client that calls vLLM server with LoRA adapters.
    
    使用 chat/completions 接口，分离 system/user 消息，和训练时完全一致。
    """
    
    def __init__(
        self,
        vllm_url: str,
        base_model: str,
        lora_name: str,  # "mas_scheduler" or "mas_router"
        temperature: float = 0.0,
    ):
        self.vllm_url = vllm_url.rstrip("/")
        self.base_model = base_model
        self.lora_name = lora_name
        self.temperature = temperature
    
    @property
    def is_available(self) -> bool:
        return True
    
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: int = 64,
        stop: Optional[List[str]] = None,
    ) -> str:
        """
        发送请求到 vLLM 服务器，使用指定的 LoRA adapter。
        
        和 mas_dual_lora_workflow.py 的 _server_generate 一致：
        - 使用 chat/completions 接口
        - 分离 system/user 消息
        """
        temp = temperature if temperature is not None else self.temperature
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        # 使用 lora_name 参数（和训练时一致）
        payload = {
            "model": self.base_model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": max_tokens,
            "lora_name": self.lora_name,
        }
        if stop:
            payload["stop"] = stop
        
        url = f"{self.vllm_url}/v1/chat/completions"
        
        try:
            resp = requests.post(url, json=payload, timeout=60)
            if resp.status_code >= 400:
                # 如果 lora_name 不支持，尝试用 model 字段指定 LoRA
                payload_alt = {
                    "model": self.lora_name,
                    "messages": messages,
                    "temperature": temp,
                    "max_tokens": max_tokens,
                }
                if stop:
                    payload_alt["stop"] = stop
                resp = requests.post(url, json=payload_alt, timeout=60)
            
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"[LoRALLMClient] Error calling vLLM: {e}")
            raise


def load_lora_adapters(vllm_url: str, scheduler_lora: str, router_lora: str):
    """加载 LoRA 适配器到 vLLM 服务器。"""
    base_url = vllm_url.rstrip("/")
    
    for lora_name, lora_path in [("mas_scheduler", scheduler_lora), ("mas_router", router_lora)]:
        logger.info(f"[vLLM] Loading {lora_name} from {lora_path}")
        
        # 检查文件存在（兼容 safetensors 格式）
        adapter_bin = os.path.join(lora_path, "adapter_model.bin")
        adapter_safetensors = os.path.join(lora_path, "adapter_model.safetensors")
        if not os.path.exists(adapter_bin) and not os.path.exists(adapter_safetensors):
            raise FileNotFoundError(f"LoRA adapter not found: {lora_path}/adapter_model.bin or .safetensors")
        
        try:
            resp = requests.post(
                f"{base_url}/v1/load_lora_adapter",
                json={"lora_name": lora_name, "lora_path": lora_path},
                timeout=30,
            )
            if resp.status_code >= 400 and "already" not in resp.text.lower():
                logger.warning(f"[vLLM] Load {lora_name} returned: {resp.text[:200]}")
            else:
                logger.info(f"[vLLM] {lora_name} loaded successfully")
        except Exception as e:
            logger.warning(f"[vLLM] Failed to load {lora_name}: {e}")


# =============================================================================
# MAS 推理循环 - 使用和训练完全一致的逻辑
# =============================================================================

class LoRAMASReasoning:
    """
    使用 LoRA 的 MAS 推理系统。
    
    不再继承 MASReasoning，而是完全复制训练时的推理循环逻辑，
    确保 prompt 模板、第三人称转换等完全一致。
    """
    
    def __init__(
        self,
        task: Dict[str, Any],
        agents: Dict[str, Any],
        config: MASConfig,
        workspace_path: str,
        log_manager: Any,
        vllm_url: str,
        base_model: str,
    ):
        self.task = task
        self.agents = agents
        self.config = config
        self.workspace_path = workspace_path
        self.log_manager = log_manager
        
        # 构建 agent registry（使用第三人称转换）
        self._registry = build_agent_registry(agents)
        self.agent_names = self._registry.names()
        self.agent_specs_prompt = self._registry.to_prompt()
        
        # 初始化 LoRA LLM clients
        self.scheduler_llm = LoRALLMClient(vllm_url, base_model, "mas_scheduler")
        self.router_llm = LoRALLMClient(vllm_url, base_model, "mas_router")
        
        # 初始化 Memer（使用 puppeteer 的 MASLLMClient）
        try:
            llm_client = MASLLMClient(config)
        except Exception as e:
            logger.warning(f"[LoRAMASReasoning] LLM client for Memer not available: {e}")
            llm_client = None
        self.memer = Memer(config, llm_client)
        
        # Task state
        self.task_state = MASTaskState(task)
        
        # Answers
        self.answers: List[str] = []
        self.final_answer = ""
        
        logger.info("=" * 60)
        logger.info("[LoRAMASReasoning] Initialized (aligned with training)")
        logger.info(f"  Available agents: {self.agent_names}")
        logger.info(f"  Third-person conversion: ENABLED")
        logger.info("=" * 60)
    
    def start(self, global_info: Any) -> None:
        """初始化推理过程。"""
        task_id = self.task.get('id', str(uuid.uuid4()))
        question = self.task.get('Question', self.task.get('question', ''))
        
        self.memer.start_task(str(task_id), question)
        self.global_info = global_info
        
        logger.info(f"[LoRAMASReasoning] Started task {task_id}")
    
    def n_step(self, n: int) -> Tuple[str, str]:
        """
        运行最多 n 步推理。
        
        逻辑和 mas_dual_lora_workflow.py 完全一致。
        """
        question = self.task.get('Question', self.task.get('question', ''))
        naturally_finished = False
        
        for step in range(n):
            step_1idx = step + 1
            logger.info(f"\n{'='*20} Step {step_1idx}/{n} {'='*20}")
            
            # Check termination
            if self.task_state.finished:
                logger.info("[LoRAMASReasoning] Task marked as finished")
                naturally_finished = True
                break
            
            # ===== 1. SCHEDULER: 决定下一个 agent =====
            scheduler_sys_prompt, scheduler_user_prompt = format_scheduler_prompt(
                question=question,
                agent_specs_prompt=self.agent_specs_prompt,
                sum_memory=self.task_state.sum_memory,
                pre_agent=self.task_state.pre_agent,
                pre_mem=self.task_state.pre_mem,
            )
            
            try:
                scheduler_output = self.scheduler_llm.chat(
                    scheduler_sys_prompt,
                    scheduler_user_prompt,
                    temperature=0.0,
                    max_tokens=64,
                )
            except Exception as e:
                logger.error(f"[Scheduler] Generation failed: {e}")
                break
            
            logger.info(f"[Scheduler] Output: '{scheduler_output[:80]}...'")
            
            # Parse scheduler decision
            chosen_agent = parse_scheduler_response(scheduler_output, self.agent_names)
            
            if chosen_agent is None:
                logger.info("[Scheduler] → DONE")
                naturally_finished = True
                break
            
            if chosen_agent not in self.agents:
                logger.warning(f"[Scheduler] Unknown agent: {chosen_agent}, using first available")
                chosen_agent = self.agent_names[0] if self.agent_names else None
                if not chosen_agent:
                    break
            
            logger.info(f"[Scheduler] → {chosen_agent}")
            
            # ===== 2. ROUTER: 选择记忆 =====
            topm_nodes = self.memer.retrieve()
            routed_memories: List[MemoryNode] = []
            
            # 获取 agent 描述（第三人称）
            chosen_agent_spec = self._registry.get(chosen_agent)
            chosen_agent_desc = chosen_agent_spec.description if chosen_agent_spec else ""
            
            if topm_nodes:
                candidates_text = "\n".join([
                    f"[{idx}] ({node.node_type}): {node.summary}"
                    for idx, node in enumerate(topm_nodes)
                ])
                
                router_sys_prompt, router_user_prompt = format_router_prompt(
                    question=question,
                    now_agent=chosen_agent,
                    sum_memory=self.task_state.sum_memory,
                    candidates=candidates_text,
                    pre_agent=self.task_state.pre_agent,
                    pre_mem=self.task_state.pre_mem,
                    top_n=self.config.top_n,
                    agent_description=chosen_agent_desc,
                )
                
                try:
                    router_output = self.router_llm.chat(
                        router_sys_prompt,
                        router_user_prompt,
                        temperature=0.0,
                        max_tokens=64,
                    )
                except Exception as e:
                    logger.error(f"[Router] Generation failed: {e}")
                    router_output = "0"
                
                logger.info(f"[Router] Output: '{router_output[:30]}...'")
                
                selected_indices = parse_router_response(router_output, len(topm_nodes))
                routed_memories = [topm_nodes[i] for i in selected_indices if i < len(topm_nodes)]
                
                logger.info(f"[Router] Routed {len(routed_memories)} memories")
            
            # ===== 3. WORKER AGENT: 执行 =====
            worker = self.agents[chosen_agent]
            
            # Inject routed memories into global_info
            if routed_memories:
                memory_context = "\n[Previous Agent Memories]\n"
                for mem in routed_memories[-3:]:
                    memory_context += f"- {mem.node_type}: {mem.summary}\n"
                self.global_info.memory_context = memory_context
            
            # 针对 MMLU-Pro 任务，为特定 agent 增强提示词（和 MASReasoning 一致）
            _enhance_agent_prompt_for_mmlu(
                worker,
                chosen_agent,
                self.task.get("type", ""),
                getattr(self.config, "enable_mmlu_prompt_injection", False),
            )
            
            logger.info(f"[Worker] → {chosen_agent}")
            worker.activate(self.global_info, initial_dialog_history=worker.initial_dialog_history)
            
            try:
                current_action, terminated = worker.take_action(
                    self.global_info,
                    external_tools_enabled=True,
                )
                
                # Convert to AgentResult
                action_dict = current_action.to_dict() if hasattr(current_action, 'to_dict') else {}
                result_dict = action_dict.get('result', {})
                
                raw_output = {
                    "action": action_dict.get('action', {}),
                    "answer": result_dict.get('answer'),
                    "reasoning": result_dict.get('step_data'),
                }
                
                if self.global_info.answers:
                    raw_output["answer"] = self.global_info.answers[-1]
                
                summary = f"{chosen_agent}: {action_dict.get('action', {}).get('action', 'executed')}"
                
                agent_result = AgentResult(
                    name=chosen_agent,
                    summary=summary,
                    raw_output=raw_output,
                    control={"terminated": terminated},
                )
                
                # Collect answer
                if raw_output.get("answer"):
                    self.answers.append(raw_output["answer"])
                
            except Exception as e:
                logger.error(f"[Worker] {chosen_agent} failed: {e}")
                agent_result = AgentResult(
                    name=chosen_agent,
                    summary=f"{chosen_agent}: failed with error",
                    raw_output={"error": str(e)},
                    control={"terminated": False},
                )
                terminated = False
            
            # ===== 4. Update Memory =====
            node = self.memer.ingest(agent_result, task_question=question)
            sum_memory = self.memer.provide_summary()
            self.task_state.register_step(agent_result, node.summary, sum_memory)
            self.global_info.update(current_action if 'current_action' in dir() else None)
            
            if terminated:
                logger.info("[Worker] Agent signaled termination")
                naturally_finished = True
                break

        return self._finalize()

    def _finalize(self) -> Tuple[str, str]:
        """完成推理并返回答案。"""
        logger.info("\n" + "=" * 60)
        logger.info("[LoRAMASReasoning] Finalizing")

        all_answers = self.answers + self.task_state.answers

        if all_answers:
            self.final_answer = all_answers[-1]
        elif self.global_info.answers:
            self.final_answer = self.global_info.answers[-1]
        else:
            self.final_answer = ""

        ground_truth = self.task.get("Answer", self.task.get("answer", ""))
        task_type = self.task.get("type", "").lower()

        # 对于 MMLU-Pro，尝试提取干净的选项字母
        if task_type in ["mmlu-pro", "mmlu_pro", "mmlu"] and self.final_answer:
            from tasks.evaluator import BenchmarkEvaluator
            extracted = BenchmarkEvaluator.extract_mmlu_answer(self.final_answer)
            if extracted:
                logger.info(f"[MMLU] Extracted clean answer: {extracted} from raw output")
                self.final_answer = extracted

        fa_preview = str(self.final_answer)[:200] if self.final_answer else "None"
        gt_preview = str(ground_truth)[:200] if ground_truth else "None"
        logger.info(f"Final Answer: {fa_preview}...")
        logger.info(f"Ground Truth: {gt_preview}...")
        logger.info(f"Total Steps: {self.task_state.steps}")
        logger.info("=" * 60)

        return self.final_answer, ground_truth

    def visualize_path(self) -> None:
        """打印推理路径（agent 调用序列）。"""
        logger.info("\n[Reasoning Path]")
        for i, entry in enumerate(self.task_state.history):
            logger.info(f"  Step {i+1}: {entry.agent} -> {str(entry.summary)[:80]}...")


# =============================================================================
# Benchmark Runner
# =============================================================================

class LoRAMASBenchmarkRunner:
    """
    使用训练好的 Scheduler/Router LoRA 适配器跑评测。

    和 main_mas_vllm.MASBenchmarkRunner 结构一致，区别在于：
    - 每个任务构造 LoRAMASReasoning（走 vLLM 的 LoRA 适配器），而非 MASReasoning
    - 需要额外的 vllm_url / base_model 定位 LoRA 服务
    """

    def __init__(
        self,
        personas_path: str,
        global_config: Dict[str, Any],
        mas_config: Optional[MASConfig] = None,
        vllm_url: str = "http://127.0.0.1:8000",
        base_model: str = "",
    ):
        self.personas_path = personas_path
        self.global_config = global_config
        self.mas_config = mas_config or DEFAULT_MAS_CONFIG
        self.vllm_url = vllm_url.rstrip("/")
        self.base_model = base_model

        self.max_step_num = self.global_config.get("graph", {}).get("max_step_num", 12)
        self.mas_config.max_steps = self.max_step_num

        # 只注册一次 agents（避免内存泄漏），每个任务只 reset
        agent_global_registry.register_all_agents(self.personas_path)

        logger.info("=" * 60)
        logger.info("LoRA MAS Benchmark Runner Initialized")
        logger.info(f"  Personas: {personas_path}")
        logger.info(f"  Max steps: {self.max_step_num}")
        logger.info(f"  vLLM URL: {self.vllm_url}")
        logger.info(f"  Base model: {self.base_model}")
        logger.info(f"  Registered {agent_global_registry.agent_num} agents")
        logger.info("=" * 60)

    def setup_reasoning(self, data_item: Dict[str, Any]):
        """为单个任务准备 LoRAMASReasoning。"""
        agent_global_registry.reset_all_agents()

        agents = {}
        for name in agent_global_registry.agent_names:
            agent = agent_global_registry.get_agent_from_name(name)
            if agent:
                agents[name] = agent

        logger.debug(f"Using {len(agents)} agents: {list(agents.keys())}")

        log_manager = LogManager("./config/global.yaml", data_item.get("type", "unknown"))
        workspace_path = log_manager.folder_path

        reasoning = LoRAMASReasoning(
            task=data_item,
            agents=agents,
            config=self.mas_config,
            workspace_path=workspace_path,
            log_manager=log_manager,
            vllm_url=self.vllm_url,
            base_model=self.base_model,
        )

        return reasoning, workspace_path

    def run_reasoning(self, data_item: Dict[str, Any]) -> str:
        """跑完一个任务，返回最终答案。"""
        reasoning, workspace_path = self.setup_reasoning(data_item)

        global_info = GlobalInfo(
            path_id=0,
            workpath=workspace_path,
            task=data_item,
        )
        global_info.logger = logger

        reasoning.start(global_info)

        final_ans, _ = reasoning.n_step(self.max_step_num)

        reasoning.visualize_path()

        return final_ans


# ========== 评测函数 ==========


def run_mmlu_pro(runner, evaluator, results_dir, mode, data_limit, resume_from=None):
    """评测 MMLU-Pro 数据集。"""
    import pandas as pd
    import string
    from tqdm import tqdm

    path = os.path.join("data", "MMLU-Pro", f"{mode}.parquet")
    data = pd.read_parquet(path)
    if data_limit:
        data = data[:data_limit]

    result_path = os.path.join(results_dir, f"MMLU-Pro_{mode}_lora.jsonl")

    # 断点续跑：读取已有结果
    done_ids = set()
    acc = 0
    if os.path.exists(result_path):
        with open(result_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    done_ids.add(record["id"])
                    if record.get("correct", False):
                        acc += 1
                except Exception:
                    pass
        logger.info(f"[Resume] Found {len(done_ids)} existing results. Continuing...")

    total = len(data)
    logger.info(f"Running MMLU-Pro ({mode}), {total} samples total, {len(done_ids)} already done")

    with open(result_path, "a", encoding="utf-8") as fd:
        for idx, row in tqdm(data.iterrows(), total=total):
            if row["question_id"] in done_ids:
                continue

            options = [
                f"{letter}: {op}"
                for letter, op in zip(string.ascii_uppercase, row["options"])
            ]
            prompt = (
                f"The following are multiple choice questions (with answers) "
                f"about {row['category']}."
            )
            question = prompt + "\n" + row["question"] + "\n" + " ".join(options)

            task = {
                "type": "MMLU-Pro",
                "Question": question,
                "Answer": row["answer"],
                "id": row["question_id"],
            }

            final_ans = runner.run_reasoning(task)
            flag = evaluator.check_mmlu(final_ans, task["Answer"], options=options)

            if flag:
                acc += 1

            record = {
                "id": task["id"],
                "pred": final_ans,
                "gold": task["Answer"],
                "correct": flag,
            }
            fd.write(json.dumps(record, ensure_ascii=False) + "\n")
            fd.flush()

            logger.info(f"Sample {idx + 1}/{total}: {'✓' if flag else '✗'}")

    final_acc = acc / total if total > 0 else 0
    logger.info(f"\n{'='*40}")
    logger.info(f"MMLU-Pro Final Accuracy: {final_acc:.4f} ({acc}/{total})")
    logger.info(f"Results saved to: {result_path}")


def run_gsm_hard(runner, evaluator, results_dir, mode, data_limit):
    """评测 GSM-Hard 数据集。"""
    import pandas as pd
    from tqdm import tqdm

    split = "train" if mode in ["train", "validation"] else "test"
    path = os.path.join("data", "GSM-Hard", f"{split}.parquet")
    if not os.path.exists(path):
        path = os.path.join("data", "GSM-Hard", "test.parquet")
    data = pd.read_parquet(path)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    if data_limit:
        data = data[:data_limit]

    result_path = os.path.join(results_dir, "gsm-hard_lora.jsonl")
    acc = 0
    total = len(data)

    logger.info(f"Running GSM-Hard, {total} samples")

    with open(result_path, "w", encoding="utf-8") as fd:
        for idx, row in enumerate(tqdm(data.iterrows(), total=total)):
            task = {
                "type": "GSM-Hard",
                "Question": "You need to write python program to solve math problems:\n" + row[1]["input"],
                "Answer": row[1]["target"],
                "id": idx,
            }

            final_ans = runner.run_reasoning(task)
            flag = evaluator.check_gsm8k(final_ans, task["Answer"])

            if flag:
                acc += 1

            record = {"id": idx, "pred": final_ans, "correct": flag}
            fd.write(json.dumps(record, ensure_ascii=False) + "\n")

            logger.info(f"Sample {idx + 1}/{total}: {'✓' if flag else '✗'} (Acc: {acc/(idx+1):.3f})")

    logger.info(f"\n{'='*40}")
    logger.info(f"GSM-Hard Final Accuracy: {acc/total:.4f} ({acc}/{total})")
    logger.info(f"Results saved to: {result_path}")


def run_srdd(runner, evaluator, results_dir, mode, data_limit):
    """评测 SRDD 数据集。"""
    import pandas as pd
    from tqdm import tqdm

    split = "train" if mode in ["train", "validation"] else "test"
    path = os.path.join("data", "SRDD", f"{split}.csv")
    if not os.path.exists(path):
        path = os.path.join("data", "SRDD", "SRDD.csv")
    data = pd.read_csv(path)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    if data_limit:
        data = data[:data_limit]

    result_path = os.path.join(results_dir, "srdd_lora.jsonl")
    total = len(data)

    logger.info(f"Running SRDD, {total} samples")

    
    with open(result_path, "w", encoding="utf-8") as fd:
        for idx, row in tqdm(data.iterrows(), total=total):
            task = {
                "type": "SRDD",
                "req": "code",
                "Question": "Develop a pythonic software following description:\n" + row["Description"],
                "id": idx,
            }
            
            final_ans = runner.run_reasoning(task)
            record = {"id": idx, "pred": final_ans}
            fd.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    logger.info(f"\nSRDD completed. Results saved to: {result_path}")


def run_cw(runner, evaluator, results_dir, mode, data_limit):
    """评测 Creative Writing 数据集。"""
    from tqdm import tqdm
    
    with open("./data/CW/creative_writing.jsonl", "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    
    if data_limit:
        data = data[:data_limit]
    
    result_path = os.path.join(results_dir, "cw_lora.jsonl")
    total = len(data)
    
    logger.info(f"Running Creative Writing, {total} samples")
    
    with open(result_path, "w", encoding="utf-8") as fd:
        for idx, q in enumerate(tqdm(data)):
            question = "Concepts: " + ", ".join(q["concepts"]) + \
                       "\nGenerate a sentence including all key concepts, grammatically correct and coherent."
            
            task = {
                "type": "CW",
                "req": "text",
                "Question": question,
                "id": idx,
                "concepts": q["concepts"],
            }
            
            final_ans = runner.run_reasoning(task)
            record = {"id": idx, "pred": final_ans}
            fd.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    logger.info(f"\nCreative Writing completed. Results saved to: {result_path}")


def run_scibench(runner, evaluator, results_dir, mode, data_limit, resume_from=None):
    """
    评测 SciBench 数据集。
    
    SciBench 数据集包含科学领域数值计算问题。
    评测逻辑与 GSM-Hard 类似，使用数值容差判断。
    
    Args:
        resume_from: 断点续跑的结果目录路径（可选）
    """
    from tqdm import tqdm
    from tasks import scibench
    
    # 加载数据
    dataset = scibench.load_dataset(mode, data_limit)
    
    result_path = os.path.join(results_dir, f"scibench_{mode}_lora.jsonl")
    acc = 0
    total = len(dataset)
    
    # 断点续跑：读取已完成的样本 ID
    completed_ids = set()
    if resume_from:
        resume_file = os.path.join(resume_from, f"scibench_{mode}_lora.jsonl")
        if os.path.exists(resume_file):
            logger.info(f"[Resume] Loading completed samples from: {resume_file}")
            with open(resume_file, "r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    completed_ids.add(record["id"])
                    if record.get("correct", False):
                        acc += 1
            logger.info(f"[Resume] Found {len(completed_ids)} completed samples, {acc} correct")
            
            # 复制已有结果到新目录
            import shutil
            shutil.copy2(resume_file, result_path)
            logger.info(f"[Resume] Copied existing results to: {result_path}")
        else:
            logger.warning(f"[Resume] Result file not found: {resume_file}")
    
    logger.info(f"Running SciBench ({mode}), {total} samples (skipping {len(completed_ids)} completed)")
    
    # 追加模式打开文件
    file_mode = "a" if resume_from and completed_ids else "w"
    with open(result_path, file_mode, encoding="utf-8") as fd:
        for idx, item in enumerate(tqdm(dataset, total=total)):
            task = scibench.format_question(item, idx)
            
            # 跳过已完成的样本
            if task["id"] in completed_ids:
                continue
            
            # 运行推理
            final_ans = runner.run_reasoning(task)
            
            # 评测
            flag = evaluator.check_scibench(final_ans, task["Answer"])
            
            if flag:
                acc += 1
            
            # 输出抽取
            pred_text = evaluator._coerce_to_text(final_ans)
            pred_num = evaluator.extract_number(pred_text)
            
            # 归一化 -0, -0.0, 0.000 -> "0"
            pred_str = str(pred_num) if pred_num is not None else pred_text
            if pred_num is not None and pred_num == 0:
                pred_str = "0"
            
            record = {
                "id": task["id"],
                "pred": pred_str,
                "pred_raw": str(final_ans)[:500] if final_ans else "",
                "gold": task["Answer"],
                "correct": flag,
                "unit": task.get("unit", ""),
                "source": task.get("source", ""),
            }
            fd.write(json.dumps(record, ensure_ascii=False) + "\n")
            fd.flush()  # 立即写入磁盘，防止中断丢失数据
            
            completed_count = len(completed_ids) + (idx + 1 - len([i for i in range(idx+1) if dataset[i] and scibench.format_question(dataset[i], i)["id"] in completed_ids]))
            logger.info(f"Sample {completed_count}/{total}: {'✓' if flag else '✗'} (Acc: {acc/completed_count:.3f})")
    
    logger.info(f"\n{'='*40}")
    logger.info(f"SciBench Final Accuracy: {acc/total:.4f} ({acc}/{total})")
    logger.info(f"Results saved to: {result_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate MAS with trained LoRA adapters (aligned with training)")
    parser.add_argument("task", choices=["gsm-hard", "MMLU-Pro", "SRDD", "CW", "scibench"],
                       help="Benchmark task")
    parser.add_argument("mode", choices=["validation", "test"], default="test",
                       help="Dataset split")
    parser.add_argument("--base_model", required=True,
                       help="Path to base model (e.g., Llama-3.1-8B-Instruct)")
    parser.add_argument("--scheduler_lora", required=True,
                       help="Path to scheduler LoRA adapter directory")
    parser.add_argument("--router_lora", required=True,
                       help="Path to router LoRA adapter directory")
    parser.add_argument("--vllm_url", default="http://127.0.0.1:8000",
                       help="vLLM server URL (default: http://127.0.0.1:8000)")
    parser.add_argument("--data_limit", type=int, default=None,
                       help="Limit number of samples")
    parser.add_argument("--personas", default="personas/personas.jsonl",
                       help="Path to personas file")
    parser.add_argument("--max_steps", type=int, default=None,
                       help="Max reasoning steps (default: from global.yaml, typically 12)")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Resume from existing results directory (e.g., results_lora/scibench_test_20251221_124950)")
    
    args = parser.parse_args()
    
    # 加载 LoRA 适配器到 vLLM
    load_lora_adapters(args.vllm_url, args.scheduler_lora, args.router_lora)
    
    # 加载 global config
    with open("config/global.yaml", "r") as f:
        global_config = yaml.safe_load(f)
    
    # 如果命令行指定了 max_steps，覆盖 global config
    if args.max_steps is not None:
        if 'graph' not in global_config:
            global_config['graph'] = {}
        global_config['graph']['max_step_num'] = args.max_steps
        logger.info(f"[Config] Max steps overridden to: {args.max_steps}")
    
    # 创建 MAS config
    mas_config = DEFAULT_MAS_CONFIG
    # 评测时关闭 feedback 收集（节省 API 调用）
    mas_config.collect_feedback = False
    
    # 创建 runner
    runner = LoRAMASBenchmarkRunner(
        personas_path=args.personas,
        global_config=global_config,
        mas_config=mas_config,
        vllm_url=args.vllm_url,
        base_model=args.base_model,
    )
    evaluator = BenchmarkEvaluator()
    
    # 创建结果目录
    if args.resume_from:
        # 断点续跑：直接使用原目录，追加写入
        results_dir = args.resume_from
        logger.info(f"[Resume] Resuming from: {results_dir}")
        logger.info(f"[Resume] Results will be appended to existing files")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(os.getcwd(), "results_lora", f"{args.task}_{args.mode}_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        logger.info(f"Results will be saved to: {results_dir}")
    
    # 运行评测
    task_runners = {
        "gsm-hard": run_gsm_hard,
        "MMLU-Pro": run_mmlu_pro,
        "SRDD": run_srdd,
        "CW": run_cw,
        "scibench": run_scibench,
    }
    
    if args.task in task_runners:
        if args.task in ["scibench", "MMLU-Pro"]:
            task_runners[args.task](runner, evaluator, results_dir, args.mode, args.data_limit, args.resume_from)
        else:
            task_runners[args.task](runner, evaluator, results_dir, args.mode, args.data_limit)
    else:
        logger.error(f"Unknown task: {args.task}")
        sys.exit(1)


if __name__ == "__main__":
    main()
