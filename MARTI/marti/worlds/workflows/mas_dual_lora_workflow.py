"""
MARTI workflow for MAS dual LoRA training (Scheduler + Router).

核心逻辑：
1. Scheduler LoRA (agent_index=0): 使用 MARTI vLLM 决定下一个 worker agent
2. Router LoRA (agent_index=1): 使用 MARTI vLLM 选择相关记忆
3. Worker agents: 调用 puppeteer 的外部 API（和 main_mas.py 一样）

提示词格式和 puppeteer/mas_scheduler/scheduler.py、router.py 完全一致。
"""

from __future__ import annotations

import os
import sys
import glob
import json
import logging
from typing import Any, Dict, List, Optional
import ray

# Ensure puppeteer is on path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# MARTI repo root (this file is marti/worlds/workflows/...)
MARTI_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../"))
# Project root that contains both MARTI and puppeteer
PROJECT_ROOT = os.path.abspath(os.path.join(MARTI_ROOT, ".."))
PUPPETEER_DIR = os.path.join(PROJECT_ROOT, "puppeteer")

# For importing `puppeteer` as a package, we need PROJECT_ROOT on sys.path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger("MAS")
logging.basicConfig(level=logging.INFO, format='[%(name)s] %(message)s')

# Import puppeteer components
try:
    from puppeteer.mas_scheduler.verl_integration import (
        MASRewardConfig,
        SchedulerRewardFunction,
        RouterRewardFunction,
    )
    from puppeteer.mas_scheduler.config import MASConfig, DEFAULT_MAS_CONFIG
    from puppeteer.mas_scheduler.memer import Memer, MemoryNode
    from puppeteer.mas_scheduler.scheduler import AgentRegistry, AgentSpec
    from puppeteer.mas_scheduler.task_state import MASTaskState, AgentResult
    from puppeteer.mas_scheduler.agent_feedback import (
        AgentFeedbackCollector,
        extract_agent_output_text,
        MemoryUsefulnessFeedback,
    )
    from puppeteer.agent.register.register import agent_global_registry
    from puppeteer.agent.agent_info.global_info import GlobalInfo
    from puppeteer.utils.log_manager import LogManager
    from puppeteer.tasks.evaluator import BenchmarkEvaluator

    PUPPETEER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"[MAS Workflow] Puppeteer import failed: {e}")
    PUPPETEER_AVAILABLE = False


# ========== Prompt builders ==========
# The real builders live in puppeteer (single source of truth) so the prompts
# used during training are byte-for-byte identical to the ones the inference
# path sends. This file previously kept its own copies, which had already
# drifted. Imported lazily because puppeteer may not be importable at module
# load time in a Ray worker.


def format_scheduler_prompt(
    question: str,
    agent_specs_prompt: str,
    sum_memory: str,
    pre_agent: Optional[str] = None,
    pre_mem: str = "",
) -> tuple[str, str]:
    """Thin wrapper over puppeteer's build_scheduler_prompt (kept for the
    existing call sites below). Returns (system_prompt, user_prompt)."""
    from puppeteer.mas_scheduler.scheduler import build_scheduler_prompt

    return build_scheduler_prompt(
        question=question,
        agent_specs_prompt=agent_specs_prompt,
        sum_memory=sum_memory,
        pre_agent=pre_agent,
        pre_mem=pre_mem,
    )


def format_memory_candidates(topm_nodes) -> str:
    """Thin wrapper over puppeteer's format_memory_candidates."""
    from puppeteer.mas_scheduler.router import format_memory_candidates as _fmt

    return _fmt(topm_nodes)


def format_router_prompt(
    question: str,
    now_agent: str,
    sum_memory: str,
    candidates: str,
    pre_agent: Optional[str] = None,
    pre_mem: str = "",
    top_n: int = 3,
    agent_description: str = "",
) -> tuple[str, str]:
    """Thin wrapper over puppeteer's build_router_prompt (kept for the existing
    call sites below). Returns (system_prompt, user_prompt)."""
    from puppeteer.mas_scheduler.router import build_router_prompt

    return build_router_prompt(
        question=question,
        now_agent=now_agent,
        sum_memory=sum_memory,
        candidates=candidates,
        pre_agent=pre_agent,
        pre_mem=pre_mem,
        top_n=top_n,
        agent_description=agent_description,
    )


def build_agent_registry(agents_dict: Dict[str, Any]) -> AgentRegistry:
    """Build AgentRegistry from puppeteer agent instances."""
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
        
        if not capabilities:
            capabilities = ['general']
        
        # 将第二人称改为第三人称，避免模型误解
        # You are -> It is, Your task -> Its task, you -> it
        desc = role_prompt if role_prompt else f"{name} agent"
        desc = desc.replace("You are", "It is").replace("you are", "it is")
        desc = desc.replace("Your task", "Its task").replace("your task", "its task")
        desc = desc.replace("Your ", "Its ").replace("your ", "its ")
        desc = desc.replace("You ", "It ").replace("you ", "it ")
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


def parse_scheduler_response(response: str, agent_names: List[str]) -> Optional[str]:
    """Parse scheduler response to get agent name. No fallback - returns None if no match."""
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
    
    # No fallback - return None so caller sees the failure
    return None


def parse_router_response(response: str, max_idx: int) -> List[int]:
    """Parse router response to get memory indices. No fallback - returns empty list if no valid indices."""
    indices = []
    parts = response.replace(" ", "").split(",")
    for part in parts:
        try:
            idx = int(part.strip())
            if 0 <= idx < max_idx and idx not in indices:
                indices.append(idx)
        except ValueError:
            continue
    # No fallback - return whatever we parsed (may be empty)
    return indices


def _enhance_agent_prompt_for_mmlu(
    agent: Any,
    agent_name: str,
    task_type: str,
    enabled: bool = False,
) -> None:
    """
    针对 MMLU-Pro 任务，为特定 agent 增强提示词（和 MASReasoning._enhance_agent_prompt_for_mmlu 一致）。

    **论文未描述此机制，默认关闭。** 仅在需要复现原先报告的 MMLU-Pro 数字时，通过
    workflow_args.enable_mmlu_prompt_injection=True 开启。

    对于选择题任务，需要强调：
    1. 只输出选项字母（A-J）
    2. 不要输出数字、金额或其他格式
    3. 确保最终答案是单个大写字母

    Args:
        agent: 要增强的 agent 实例
        agent_name: agent 名称
        task_type: 任务类型
        enabled: 是否启用（默认 False = 论文行为）
    """
    if not enabled:
        return

    # 只对 MMLU-Pro 任务进行增强
    if task_type.lower() not in ["mmlu-pro", "mmlu_pro", "mmlu"]:
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


def extract_mmlu_answer(text: str) -> Optional[str]:
    """
    增强版 MMLU-Pro 答案提取（和 evaluator.py 保持一致）。
    
    从可能很乱的模型输出中提取单个选项字母 (A-J)。
    """
    import re
    
    if text is None:
        return None
    
    text = str(text)
    
    # 清理转义字符
    text = text.replace('\\n', '\n').replace("\\'", "'")
    
    # 有效选项字母
    valid_letters = set('ABCDEFGHIJ')
    
    # Pattern 1: FINAL ANSWER: X 或 FINAL ANSWER: (X)
    patterns_priority = [
        r'FINAL\s*ANSWER[:\s]+\(?([A-J])\)?',
        r'final\s*answer[:\s]+\(?([A-Ja-j])\)?',
        r'[Tt]he\s+answer\s+is\s*[:\s]?\s*\(?([A-Ja-j])\)?',
        r'[Aa]nswer\s+is\s*[:\s]?\s*\(?([A-Ja-j])\)?',
        r'[Cc]orrect\s+answer\s+is\s*[:\s]?\s*\(?([A-Ja-j])\)?',
        r'[Bb]oxed\{([A-Ja-j])\}',
        r'\\boxed\{([A-Ja-j])\}',
    ]
    
    for pattern in patterns_priority:
        match = re.search(pattern, text)
        if match:
            letter = match.group(1).upper()
            if letter in valid_letters:
                return letter
    
    # Pattern 2: 括号中的字母，如 (A) 或 [A]
    paren_patterns = [
        r'\(([A-J])\)',
        r'\[([A-J])\]',
    ]
    for pattern in paren_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # 取最后一个（通常是最终答案）
            letter = matches[-1].upper()
            if letter in valid_letters:
                return letter
    
    # Pattern 3: 冒号后的字母，如 "Answer: A" 或 ": A"
    colon_pattern = r'[:\s]\s*([A-J])(?:[\s,.:;\)]|$)'
    matches = re.findall(colon_pattern, text, re.IGNORECASE)
    if matches:
        letter = matches[-1].upper()
        if letter in valid_letters:
            return letter
    
    # Pattern 4: 加粗的字母 **A** 或 *A*
    bold_pattern = r'\*+([A-J])\*+'
    match = re.search(bold_pattern, text, re.IGNORECASE)
    if match:
        letter = match.group(1).upper()
        if letter in valid_letters:
            return letter
    
    # Pattern 5: 开头或结尾独立的字母
    # 检查开头
    start_pattern = r'^\s*([A-J])(?:[\s,.:;\)]|$)'
    match = re.search(start_pattern, text.strip(), re.IGNORECASE)
    if match:
        letter = match.group(1).upper()
        if letter in valid_letters:
            return letter
    
    # 检查结尾（取最后一行）
    lines = text.strip().split('\n')
    last_line = lines[-1].strip() if lines else ""
    end_pattern = r'(?:^|[\s,.:;\(])([A-J])\s*[.!)?]*\s*$'
    match = re.search(end_pattern, last_line, re.IGNORECASE)
    if match:
        letter = match.group(1).upper()
        if letter in valid_letters:
            return letter
    
    # Pattern 6: 任何独立的单字母（被空格或标点包围）
    standalone_pattern = r'(?:^|[\s,.:;\(\[])([A-J])(?:[\s,.:;\)\]]|$)'
    matches = re.findall(standalone_pattern, text, re.IGNORECASE)
    if matches:
        # 取最后一个
        letter = matches[-1].upper()
        if letter in valid_letters:
            return letter
    
    # Pattern 7: 文本长度为1且是字母
    text_stripped = text.strip()
    if len(text_stripped) == 1 and text_stripped.upper() in valid_letters:
        return text_stripped.upper()
    
    # Pattern 8: 取文本中第一个出现的有效字母（fallback）
    for char in text:
        if char.upper() in valid_letters:
            return char.upper()
    
    return None


def parse_mmlu_options_from_prompt(prompt: str) -> Optional[List[str]]:
    """
    从MMLU-Pro的prompt中解析选项列表。
    
    MMLU-Pro的prompt格式通常为:
    "The following are multiple choice questions ... about {category}.
    {question}
    A: {option_a} B: {option_b} C: {option_c} ..."
    
    返回格式: ["A: xxx", "B: xxx", "C: xxx", ...]
    如果解析失败返回None
    """
    import re
    
    if not prompt:
        return None
    
    # 尝试匹配 "A: xxx B: xxx C: xxx ..." 格式
    # 选项通常在问题之后，用空格分隔
    options = []
    
    # Pattern: 匹配 "X: content" 格式，其中X是A-J
    # 注意：选项内容可能包含空格，所以我们需要找到下一个选项的开始位置
    option_pattern = r'([A-J]):\s*'
    matches = list(re.finditer(option_pattern, prompt))
    
    if len(matches) < 2:
        return None
    
    for i, match in enumerate(matches):
        letter = match.group(1)
        start_pos = match.end()
        
        # 找到下一个选项的开始位置，或者字符串结束
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(prompt)
        
        content = prompt[start_pos:end_pos].strip()
        options.append(f"{letter}: {content}")
    
    return options if len(options) >= 2 else None


async def workflow(
    prompt: str,
    label: str,
    agents: List[Dict[str, Any]],  # MARTI's vLLM engines for Scheduler/Router LoRA
    tool_manager,
    task: str,
    metadata: Optional[Dict] = None,
    workflow_args: Optional[Dict] = None,
    max_length: int = None,
    prompt_id: int = 0,
    is_eval: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    MAS dual LoRA workflow - 核心逻辑和 main_mas.py 完全一致！
    
    - agents[0]: Scheduler LoRA (vLLM engine) - 决定下一个 worker agent
    - agents[1]: Router LoRA (vLLM engine) - 选择相关记忆
    - Worker agents: puppeteer agents (外部 API 调用)
    """
    import yaml
    
    # Ensure puppeteer is on path (in case sys.path was reset in Ray worker)
    global PROJECT_ROOT, PUPPETEER_DIR
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    
    # Re-import puppeteer if needed (in case import failed at module level).
    # IMPORTANT: declare these as globals so the conditional imports below do NOT create
    # function-local variables (which would cause UnboundLocalError when the branch isn't taken).
    global PUPPETEER_AVAILABLE
    global MASRewardConfig, SchedulerRewardFunction, RouterRewardFunction
    global MASConfig, DEFAULT_MAS_CONFIG
    global Memer, MemoryNode
    global AgentRegistry, AgentSpec
    global MASTaskState, AgentResult
    global AgentFeedbackCollector, extract_agent_output_text, MemoryUsefulnessFeedback
    global agent_global_registry, GlobalInfo, LogManager, BenchmarkEvaluator
    if not PUPPETEER_AVAILABLE:
        try:
            from puppeteer.mas_scheduler.verl_integration import (
                MASRewardConfig,
                SchedulerRewardFunction,
                RouterRewardFunction,
            )
            from puppeteer.mas_scheduler.config import MASConfig, DEFAULT_MAS_CONFIG
            from puppeteer.mas_scheduler.memer import Memer, MemoryNode
            from puppeteer.mas_scheduler.scheduler import AgentRegistry, AgentSpec
            from puppeteer.mas_scheduler.task_state import MASTaskState, AgentResult
            from puppeteer.mas_scheduler.agent_feedback import (
                AgentFeedbackCollector,
                extract_agent_output_text,
                MemoryUsefulnessFeedback,
            )
            from puppeteer.agent.register.register import agent_global_registry
            from puppeteer.agent.agent_info.global_info import GlobalInfo
            from puppeteer.utils.log_manager import LogManager
            from puppeteer.tasks.evaluator import BenchmarkEvaluator
            PUPPETEER_AVAILABLE = True
            logger.info(f"[MAS Workflow] Successfully imported puppeteer from {PUPPETEER_DIR}")
        except ImportError as e:
            logger.error(f"[MAS Workflow] Puppeteer import failed: {e}")
            logger.error(f"[MAS Workflow] PROJECT_ROOT: {PROJECT_ROOT}, exists: {os.path.exists(PROJECT_ROOT)}")
            logger.error(f"[MAS Workflow] PUPPETEER_DIR: {PUPPETEER_DIR}, exists: {os.path.exists(PUPPETEER_DIR)}")
            logger.error(f"[MAS Workflow] sys.path (head): {sys.path[:5]}...")
    
    if not PUPPETEER_AVAILABLE:
        logger.error("[MAS Workflow] Puppeteer not available!")
        return {"prompt": prompt, "label": label, "trajectory": [], "final_reward": -1.0}
    
    workflow_args = workflow_args or {}
    # NOTE: upstream may pass non-dict metadata (e.g., str/int). Make it robust.
    if metadata is None:
        metadata = {}
    task_type = "GSM-Hard"
    task_id = prompt_id
    if isinstance(metadata, dict):
        task_type = metadata.get("task_type", workflow_args.get("task_type", task_type))
        task_id = metadata.get("id", task_id)
    else:
        # If metadata is a scalar, treat it as an id (and keep default task_type)
        task_id = metadata
        metadata = {"id": task_id}
    personas_path = workflow_args.get(
        "personas_path",
        os.path.join(PUPPETEER_DIR, "personas", "personas.jsonl"),
    )
    global_config_path = workflow_args.get(
        "global_config",
        os.path.join(PUPPETEER_DIR, "config", "global.yaml"),
    )
    max_steps = workflow_args.get("max_steps", 8)

    # Build task dict (和 main_mas.py 一样)
    task_dict = {
        "type": task_type,
        "Question": prompt,
        "Answer": label,
        "id": task_id,
    }
    question = prompt

    # Load global config
    try:
        with open(global_config_path, "r") as f:
            global_config = yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"[MAS Workflow] Cannot load global config: {e}")
        global_config = {}
    
    # ---------- Setup Worker Agents (puppeteer, 外部 API) ----------
    agent_global_registry.register_all_agents(personas_path)
    agent_global_registry.reset_all_agents()
    worker_agents: Dict[str, Any] = {}
    for name in agent_global_registry.agent_names:
        agent = agent_global_registry.get_agent_from_name(name)
        if agent:
            worker_agents[name] = agent
    
    agent_registry = build_agent_registry(worker_agents)
    agent_names = agent_registry.names()
    agent_specs_prompt = agent_registry.to_prompt()
    
    logger.info(f"[MAS Workflow] Worker agents (API): {agent_names}")
    
    # ---------- Setup Memory (Memer) ----------
    mas_config = DEFAULT_MAS_CONFIG
    mas_config.max_steps = max_steps
    # Allow overriding retrieval budget from workflow_args without editing puppeteer code
    try:
        if "top_m" in workflow_args and workflow_args["top_m"] is not None:
            mas_config.top_m = int(workflow_args["top_m"])
    except Exception:
        pass
    try:
        if "top_n" in workflow_args and workflow_args["top_n"] is not None:
            mas_config.top_n = int(workflow_args["top_n"])
    except Exception:
        pass
    # Non-paper extensions, off by default. Enable only to reproduce the
    # originally reported numbers.
    enable_mmlu_injection = bool(
        workflow_args.get("enable_mmlu_prompt_injection", False)
    )
    try:
        mas_config.enable_mmlu_prompt_injection = enable_mmlu_injection
        mas_config.enable_role_based_routing = bool(
            workflow_args.get("enable_role_based_routing", False)
        )
    except Exception:
        pass
    
    from puppeteer.mas_scheduler.llm import MASLLMClient
    try:
        llm_client = MASLLMClient(mas_config)
    except Exception as e:
        logger.warning(f"[MAS Workflow] LLM client for Memer not available: {e}")
        llm_client = None
    
    memer = Memer(mas_config, llm_client)
    memer.start_task(str(prompt_id), question)

    # ---------- Agent feedback collector (parallel usefulness feedback for router) ----------
    # Only enable parallel feedback when an OpenAI client is actually available.
    feedback_collector = (
        AgentFeedbackCollector(llm_client)
        if (llm_client is not None and getattr(llm_client, "is_available", False))
        else None
    )

    # ---------- Setup Logging ----------
    log_manager = LogManager(global_config_path, task_dict.get("type", "unknown"))
    workspace_path = log_manager.folder_path

    global_info = GlobalInfo(
        path_id=0,
        workpath=workspace_path,
        task=task_dict,
    )
    global_info.logger = logger
    
    # ---------- Logging directories ----------
    # Root for debug/result logs. Defaults to <MARTI_ROOT>/logs; override with
    # workflow_args.log_dir. Never hardcode an absolute path here: this runs on
    # the live training path and a non-writable directory would abort the run.
    log_root = str(workflow_args.get("log_dir", "") or "").strip()
    if not log_root:
        log_root = os.path.join(MARTI_ROOT, "logs")

    debug_log_path = None
    results_log_path = None
    try:
        debug_log_dir = os.path.join(log_root, "scheduler_debug")
        os.makedirs(debug_log_dir, exist_ok=True)
        debug_log_path = os.path.join(debug_log_dir, f"prompt_{prompt_id}.txt")
        logger.info(f"[MAS] Scheduler debug log will be saved to: {debug_log_path}")

        results_log_dir = os.path.join(log_root, "training_results")
        os.makedirs(results_log_dir, exist_ok=True)
        results_log_path = os.path.join(results_log_dir, f"{task_type}_results.jsonl")
        logger.info(f"[MAS] Training results will be saved to: {results_log_path}")
    except OSError as e:
        # Diagnostics must never take down training.
        logger.warning(f"[MAS] Could not create log dirs under {log_root}: {e}. Logging disabled.")

    def _log_scheduler_input(step_idx: int, sys_prompt: str, user_prompt: str, output: str):
        """将 scheduler 输入和输出写入调试文件"""
        if not debug_log_path:
            return
        try:
            with open(debug_log_path, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"STEP {step_idx}\n")
                f.write(f"{'='*60}\n")
                f.write(f"\n--- SYSTEM PROMPT ---\n{sys_prompt}\n")
                f.write(f"\n--- USER PROMPT ---\n{user_prompt}\n")
                f.write(f"\n--- OUTPUT ---\n{output}\n")
                f.flush()  # 确保立即写入
        except Exception as e:
            logger.warning(f"Failed to write scheduler debug log: {e}")
    
    # ---------- Reward functions ----------
    # Allow overriding reward hyperparams from workflow_args (Hydra-friendly).
    # These control ONLY the Scheduler/Router step-level rewards computed below.
    try:
        reward_cfg = MASRewardConfig(
            alpha=float(workflow_args.get("router_alpha", MASRewardConfig().alpha)),
            eta=float(workflow_args.get("router_eta", MASRewardConfig().eta)),
            lambda_tok=float(workflow_args.get("lambda_tok", MASRewardConfig().lambda_tok)),
            reward_clip=float(workflow_args.get("reward_clip", MASRewardConfig().reward_clip)),
            over_budget_penalty=float(
                workflow_args.get("over_budget_penalty", MASRewardConfig().over_budget_penalty)
            ),
        )
    except Exception:
        reward_cfg = MASRewardConfig()
    sch_reward_fn = SchedulerRewardFunction(reward_cfg)
    rou_reward_fn = RouterRewardFunction(reward_cfg)
    
    # ---------- Get vLLM engines for Scheduler/Router LoRA ----------
    scheduler_llm = agents[0]["llm"] if len(agents) > 0 and agents[0] else None
    router_llm = agents[1]["llm"] if len(agents) > 1 and agents[1] else scheduler_llm
    
    if scheduler_llm is None:
        logger.error("[MAS Workflow] No vLLM engine for Scheduler LoRA!")
        return {"prompt": prompt, "label": label, "trajectory": [], "final_reward": -1.0}
    
    logger.info(f"[MAS Workflow] Scheduler LoRA: agent_index=0, Router LoRA: agent_index=1")

    # ---------- Optional: use vLLM OpenAI server for inference with runtime LoRA updates ----------
    use_vllm_server = bool(workflow_args.get("use_vllm_server", False))
    vllm_server_url = str(workflow_args.get("vllm_server_url", "") or "").strip()
    vllm_model = str(workflow_args.get("vllm_model", "") or "").strip()
    vllm_scheduler_lora_name = str(workflow_args.get("vllm_scheduler_lora_name", "scheduler") or "scheduler")
    vllm_router_lora_name = str(workflow_args.get("vllm_router_lora_name", "router") or "router")

    if use_vllm_server and not vllm_server_url:
        logger.warning("[MAS Workflow] use_vllm_server=True but vllm_server_url is empty. Falling back to local vLLM engines.")
        use_vllm_server = False

    if use_vllm_server and not vllm_model:
        # default to the training base model if not provided
        vllm_model = str(kwargs.get("pretrain", "")) or ""
        if not vllm_model:
            logger.warning("[MAS Workflow] vllm_model not provided; please set workflow_args.vllm_model to your base model name.")

    def _server_generate(
        system_prompt: str,
        user_prompt: str,
        *,
        lora_name: str,
        stop: list[str],
        max_tokens: int,
    ) -> str:
        """
        Generate using vLLM server with chat/completions.
        
        正确使用 system + user 消息格式，这对于 Llama-3 等模型的 chat template 非常重要。
        vLLM 会自动应用模型的 chat template（Llama-3 的特殊 tokens、Qwen 格式等）。
        """
        from marti.models.vllm.lora_server_client import chat_completions
        # 正确分离 system 和 user 消息，匹配原来 MASLLMClient.chat() 的行为
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return chat_completions(
            vllm_server_url,
            model=vllm_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
            stop=stop,
            lora_name=lora_name,
            timeout=float(workflow_args.get("vllm_server_timeout", 60.0) or 60.0),
        )
    
    # ---------- Main Loop (和 main_mas.py 的 MASReasoning.n_step 一样) ----------
    trajectory: List[Dict[str, Any]] = []
    task_state = MASTaskState(task_dict)
    answers: List[str] = []
    total_token_count = 0
    
    # workflow args
    train_start_step = 1
    try:
        train_start_step = int(workflow_args.get("train_start_step", 1))
    except Exception:
        train_start_step = 1
    # Post-hoc memory-usefulness feedback (paper Eq. 11). `enable_parallel_feedback`
    # is the legacy key for the same switch and is still honoured.
    enable_posthoc_feedback = bool(
        workflow_args.get(
            "enable_posthoc_feedback",
            workflow_args.get("enable_parallel_feedback", True),
        )
    )
    
    naturally_finished = False  # Track if task finished naturally (和 puppeteer 一致)
    
    for step in range(max_steps):
        step_1idx = step + 1
        logger.info(f"\n{'='*20} Step {step_1idx}/{max_steps} {'='*20}")
        
        # ===== 1. SCHEDULER LoRA: 决定下一个 worker agent =====
        scheduler_sys_prompt, scheduler_user_prompt = format_scheduler_prompt(
            question=question,
            agent_specs_prompt=agent_specs_prompt,
            sum_memory=task_state.sum_memory,
            pre_agent=task_state.pre_agent,
            pre_mem=task_state.pre_mem,
        )
        # 完整 prompt 用于记录和本地 vLLM（不支持 chat messages）
        scheduler_prompt_full = f"{scheduler_sys_prompt}\n\n{scheduler_user_prompt}"
        
        # Generate with Scheduler LoRA (vLLM) - agent_index=0
        # No stop tokens, no fallback - raw output for debugging
        try:
            if use_vllm_server:
                # 使用正确分离的 system + user 消息格式
                scheduler_output = _server_generate(
                    scheduler_sys_prompt,
                    scheduler_user_prompt,
                    lora_name=vllm_scheduler_lora_name,
                    stop=[],  # No stop - let model output freely
                    max_tokens=64,
                ).strip()
            else:
                from vllm import SamplingParams
                scheduler_sampling = SamplingParams(
                    temperature=0.0,
                    max_tokens=64,
                    stop=[],  # No stop
                )
                # 本地 vLLM generate 不支持 chat messages，使用合并的 prompt
                scheduler_result = ray.get(scheduler_llm.generate.remote(
                    [scheduler_prompt_full], scheduler_sampling
                ))[0]
                scheduler_output = scheduler_result.outputs[0].text.strip()
        except Exception as e:
            logger.error(f"[Scheduler LoRA] Generation failed: {e}")
            # No fallback - propagate the error
            raise
        
        # Log scheduler output with more details for debugging
        _sched_out = scheduler_output[:80] + "..." if len(scheduler_output) > 80 else scheduler_output
        logger.info(f"[MAS]   Scheduler → '{_sched_out}'")
        
        # 写入调试文件（完整的 scheduler 输入和输出）
        _log_scheduler_input(step_1idx, scheduler_sys_prompt, scheduler_user_prompt, scheduler_output)
        
        # Parse scheduler decision - no fallback
        chosen_agent = parse_scheduler_response(scheduler_output, agent_names)
        
        # No safety guard - show raw behavior
        
        # Record scheduler trajectory (agent_index=0)
        trajectory.append({
            "turn_id": len(trajectory),
            "agent_index": 0,  # Scheduler LoRA
            "agent_name": "Scheduler",
            "agent_role": "scheduler",
            "agent_input": scheduler_prompt_full,
            "agent_output": scheduler_output,
            "agent_reward": 0.0,
            # Use 1-indexed step_idx to match reward formula conventions
            "metadata": {"step_idx": step_1idx, "chosen_agent": chosen_agent, "token_count": 0},
        })
        
        if chosen_agent is None:
            logger.info("  Scheduler → DONE")
            naturally_finished = True
            break
        
        # ===== 2. ROUTER LoRA: 选择相关记忆 =====
        topm_nodes = memer.retrieve()
        routed_memories = []
        
        # 获取当前 agent 的描述（第三人称）给 Router 参考
        chosen_agent_spec = agent_registry.get(chosen_agent)
        chosen_agent_desc = chosen_agent_spec.description if chosen_agent_spec else ""
        
        if topm_nodes:
            candidates_text = format_memory_candidates(topm_nodes)

            router_sys_prompt, router_user_prompt = format_router_prompt(
                question=question,
                now_agent=chosen_agent,
                sum_memory=task_state.sum_memory,
                candidates=candidates_text,
                pre_agent=task_state.pre_agent,
                pre_mem=task_state.pre_mem,
                top_n=mas_config.top_n,
                agent_description=chosen_agent_desc,
            )
            # 完整 prompt 用于记录和本地 vLLM
            router_prompt_full = f"{router_sys_prompt}\n\n{router_user_prompt}"
            
            # Generate with Router LoRA (vLLM) - agent_index=1
            # No stop tokens, no fallback - raw output for debugging
            try:
                if use_vllm_server:
                    # 使用正确分离的 system + user 消息格式
                    router_output = _server_generate(
                        router_sys_prompt,
                        router_user_prompt,
                        lora_name=vllm_router_lora_name,
                        stop=[],  # No stop
                        max_tokens=64,
                    ).strip()
                else:
                    from vllm import SamplingParams
                    router_sampling = SamplingParams(
                        temperature=0.0,
                        max_tokens=64,
                        stop=[],  # No stop
                    )
                    # 本地 vLLM generate 不支持 chat messages，使用合并的 prompt
                    router_result = ray.get(router_llm.generate.remote(
                        [router_prompt_full], router_sampling
                    ))[0]
                    router_output = router_result.outputs[0].text.strip()
            except Exception as e:
                logger.error(f"[Router LoRA] Generation failed: {e}")
                # No fallback - propagate the error
                raise
            
            # Log router output (truncated)
            _router_out = router_output[:30] + "..." if len(router_output) > 30 else router_output
            logger.info(f"  Router → '{_router_out}'")
            
            # Parse router decision
            selected_indices = parse_router_response(router_output, len(topm_nodes))
            routed_memories = [topm_nodes[i] for i in selected_indices if i < len(topm_nodes)]
            
            # Record router trajectory (agent_index=1)
            trajectory.append({
                "turn_id": len(trajectory),
                "agent_index": 1,  # Router LoRA
                "agent_name": "Router",
                "agent_role": "router",
                "agent_input": router_prompt_full,
                "agent_output": router_output,
                "agent_reward": 0.0,
                "metadata": {
                    "step_idx": step_1idx,
                    "routed_count": len(routed_memories),
                    "selected_indices": selected_indices,
                    "useful_count": 0,
                },
            })
            
            logger.info(f"  Routed: {len(routed_memories)} memories")
        else:
            # No retrievable memories yet (common at early steps). Still record a Router turn so the
            # router adapter always has training samples; the correct behavior is typically "none".
            router_sys_prompt, router_user_prompt = format_router_prompt(
                question=question,
                now_agent=chosen_agent,
                sum_memory=task_state.sum_memory,
                candidates="(no memories available)",
                pre_agent=task_state.pre_agent,
                pre_mem=task_state.pre_mem,
                top_n=mas_config.top_n,
                agent_description=chosen_agent_desc,
            )
            # 完整 prompt 用于记录和本地 vLLM
            router_prompt_full = f"{router_sys_prompt}\n\n{router_user_prompt}"

            try:
                if use_vllm_server:
                    # 使用正确分离的 system + user 消息格式
                    router_output = _server_generate(
                        router_sys_prompt,
                        router_user_prompt,
                        lora_name=vllm_router_lora_name,
                        stop=["\n"],
                        max_tokens=16,
                    ).strip()
                else:
                    from vllm import SamplingParams
                    router_sampling = SamplingParams(
                        temperature=0.0,
                        max_tokens=16,
                        stop=["\n"],
                    )
                    # 本地 vLLM generate 不支持 chat messages，使用合并的 prompt
                    router_result = ray.get(router_llm.generate.remote(
                        [router_prompt_full], router_sampling
                    ))[0]
                    router_output = router_result.outputs[0].text.strip()
                    pass  # No memories to route, skip logging
            except Exception as e:
                logger.error(f"[Router LoRA] Generation failed (no memories): {e}")
                router_output = "none"

            selected_indices = parse_router_response(router_output, 0)
            routed_memories = []

            trajectory.append({
                "turn_id": len(trajectory),
                "agent_index": 1,  # Router LoRA
                "agent_name": "Router",
                "agent_role": "router",
                "agent_input": router_prompt_full,
                "agent_output": router_output,
                "agent_reward": 0.0,
                "metadata": {
                    "step_idx": step_1idx,
                    "routed_count": 0,
                    "selected_indices": selected_indices,
                    "useful_count": 0,
                },
            })
        
        # ===== 3. WORKER AGENT: 调用外部 API (和 main_mas.py 一样) =====
        if chosen_agent not in worker_agents:
            logger.warning(f"  ⚠ Unknown agent: {chosen_agent}")
            chosen_agent = agent_names[0] if agent_names else None
            if not chosen_agent:
                break
        
        worker = worker_agents[chosen_agent]
        
        # Inject routed memories into global_info
        if routed_memories:
            memory_context = "\n[Previous Agent Memories]\n"
            for mem in routed_memories[-3:]:
                memory_context += f"- {mem.node_type}: {mem.summary}\n"
            global_info.memory_context = memory_context
        
        # 针对 MMLU-Pro 任务，为特定 agent 增强提示词（和 MASReasoning._step 一致）
        _enhance_agent_prompt_for_mmlu(worker, chosen_agent, task_type, enable_mmlu_injection)
        
        # ===== 调用 Worker Agent (外部 API) + 并行自反馈（useful_count）=====
        logger.info(f"  → Agent: {chosen_agent}")
        worker.activate(global_info, initial_dialog_history=worker.initial_dialog_history)
        
        # Prepare feedback collector with routed memories (as dicts)
        routed_memories_payload = []
        for node in routed_memories:
            if isinstance(node, MemoryNode):
                routed_memories_payload.append(
                    {"node_type": node.node_type, "summary": node.summary, "content": node.content}
                )
        if feedback_collector is not None:
            feedback_collector.set_routed_memories(routed_memories_payload)

        # helper: build task_context for feedback
        task_context = task_state.sum_memory if getattr(task_state, "sum_memory", "") else ""

        try:
            current_action, terminated = worker.take_action(
                global_info,
                external_tools_enabled=True,
            )

            if (
                enable_posthoc_feedback
                and feedback_collector is not None
                and len(routed_memories_payload) > 0
            ):
                # 论文 Eq. 11：active agent 完成工作**之后**，对每条被分配的记忆返回
                # u_i。判定器必须看到 agent 的实际输出，因此这一步只能串行。
                feedback_result = feedback_collector.collect_feedback_posthoc(
                    agent_name=chosen_agent,
                    step_idx=step_1idx,
                    question=question,
                    agent_output=extract_agent_output_text(current_action),
                    task_context=task_context,
                )
            else:
                feedback_result = MemoryUsefulnessFeedback(
                    routed_count=len(routed_memories_payload),
                    useful_count=0,
                )

            # Simplified log - just show completion status
            if terminated:
                logger.info(f"  ✓ {chosen_agent} → DONE")
            
            # Extract per-step token count (prefer per-step, not cumulative)
            step_tokens = 0
            if hasattr(current_action, "to_dict"):
                action_dict = current_action.to_dict()
                step_tokens = int(action_dict.get("tokens", 0) or 0)
            total_token_count += step_tokens

            # Update metadata for this step:
            # - Scheduler token_count (for scheduler reward)
            # - Router useful_count (for router hit-rate reward)
            try:
                # Scheduler turn for this step is the last appended agent_index=0 with step_idx==step_1idx
                for turn in reversed(trajectory):
                    if turn.get("agent_index") == 0 and turn.get("metadata", {}).get("step_idx") == step_1idx:
                        turn["metadata"]["token_count"] = step_tokens
                        break
            except Exception:
                pass

            try:
                for turn in reversed(trajectory):
                    if turn.get("agent_index") == 1 and turn.get("metadata", {}).get("step_idx") == step_1idx:
                        turn["metadata"]["useful_count"] = int(getattr(feedback_result, "useful_count", 0) or 0)
                        turn["metadata"]["routed_count"] = int(getattr(feedback_result, "routed_count", turn.get("metadata", {}).get("routed_count", 0)) or 0)
                        break
            except Exception:
                pass
            
            # Convert to AgentResult
            action_dict = current_action.to_dict() if hasattr(current_action, 'to_dict') else {}
            result_dict = action_dict.get('result', {})
            
            raw_output = {
                "action": action_dict.get('action', {}),
                "answer": result_dict.get('answer'),
                "reasoning": result_dict.get('step_data'),
            }
            
            if global_info.answers:
                raw_output["answer"] = global_info.answers[-1]
            
            summary = f"{chosen_agent}: {action_dict.get('action', {}).get('action', 'executed')}"
            
            agent_result = AgentResult(
                name=chosen_agent,
                summary=summary,
                raw_output=raw_output,
                control={"terminated": terminated},
            )
            
            # Don't log full output, just minimal info
            
            # Collect answer
            if raw_output.get("answer"):
                answers.append(raw_output["answer"])
                
        except Exception as e:
            logger.error(f"[Worker Agent] {chosen_agent} failed: {e}")
            agent_result = AgentResult(
                name=chosen_agent,
                summary=f"{chosen_agent}: failed with error",
                raw_output={"error": str(e)},
                control={"terminated": False},
            )
            terminated = False
        
        # ===== 4. Update Memory (Memer) =====
        node = memer.ingest(agent_result, task_question=question)
        sum_memory = memer.provide_summary()
        
        # Update task state
        task_state.register_step(agent_result, node.summary, sum_memory)
        global_info.update(current_action if 'current_action' in dir() else None)
        
        if terminated:
            logger.info("  Agent signaled termination")
            naturally_finished = True
            break
    else:
        # Loop completed without break (max_steps reached without natural finish)
        naturally_finished = False

    # ===== Force final answer if no answers collected (不管是 DONE 还是 max_steps) =====
    # 核心逻辑：只要结束时 answers 为空，就强制调用 Concluder 生成答案
    # 这保证了 task_reward 不会因为"Scheduler 提前 DONE 但没选过 Concluder"而变成 -1
    need_force_answer = len(answers) == 0 and len(global_info.answers) == 0
    if need_force_answer:
        logger.info("\n" + "=" * 60)
        logger.info("[MAS] No answer collected, forcing Concluder to generate final answer...")
        
        # Find a concluder-type agent (和 puppeteer/_force_final_answer 一致)
        concluder_agent = None
        concluder_name = None
        for name, agent in worker_agents.items():
            name_lower = name.lower()
            if "conclud" in name_lower or "summar" in name_lower:
                concluder_agent = agent
                concluder_name = name
                break
        
        if concluder_agent is None:
            # Fallback: use ReasoningAgent if no concluder
            for name, agent in worker_agents.items():
                if "reasoning" in name.lower():
                    concluder_agent = agent
                    concluder_name = name
                    break
        
        if concluder_agent is None and worker_agents:
            # Last resort: use any available agent
            concluder_name = list(worker_agents.keys())[0]
            concluder_agent = worker_agents[concluder_name]
        
        if concluder_agent:
            logger.info(f"[MAS] Using {concluder_name} to generate final answer")
            
            # Router: 为 concluder 选择记忆
            topm_nodes = memer.retrieve()
            concluder_spec = agent_registry.get(concluder_name)
            concluder_desc = concluder_spec.description if concluder_spec else ""
            
            if topm_nodes:
                candidates_text = format_memory_candidates(topm_nodes)

                router_sys_prompt, router_user_prompt = format_router_prompt(
                    question=question,
                    now_agent=concluder_name,
                    sum_memory=task_state.sum_memory,
                    candidates=candidates_text,
                    pre_agent=task_state.pre_agent,
                    pre_mem=task_state.pre_mem,
                    top_n=mas_config.top_n,
                    agent_description=concluder_desc,
                )
                router_prompt_full = f"{router_sys_prompt}\n\n{router_user_prompt}"
                
                try:
                    if use_vllm_server:
                        router_output = _server_generate(
                            router_sys_prompt,
                            router_user_prompt,
                            lora_name=vllm_router_lora_name,
                            stop=[],
                            max_tokens=32,
                        ).strip()
                    else:
                        from vllm import SamplingParams
                        router_sampling = SamplingParams(temperature=0.0, max_tokens=32, stop=[])
                        router_result = ray.get(router_llm.generate.remote([router_prompt_full], router_sampling))[0]
                        router_output = router_result.outputs[0].text.strip()
                    
                    # Parse indices from router output (简化版，和 router.py 一致)
                    def _parse_router_indices(response: str, max_idx: int):
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
                    
                    selected_indices = _parse_router_indices(router_output, len(topm_nodes))
                    if not selected_indices:
                        selected_indices = [0]
                    
                    routed_memories_payload = []
                    for idx in selected_indices[:mas_config.top_n]:
                        node = topm_nodes[idx]
                        routed_memories_payload.append({
                            "summary": node.summary,
                            "content": node.content,
                            "node_type": node.node_type,
                        })
                    
                    # Inject memories via global_info (和主循环一致)
                    if routed_memories_payload:
                        memory_context = "\n[Previous Agent Memories]\n"
                        for mem in routed_memories_payload[-3:]:
                            memory_context += f"- {mem['node_type']}: {mem['summary']}\n"
                        global_info.memory_context = memory_context
                    
                except Exception as e:
                    logger.warning(f"[MAS] Router failed for force concluder: {e}")
            
            # 针对 MMLU-Pro 任务增强提示词（和 MASReasoning._force_final_answer 一致）
            _enhance_agent_prompt_for_mmlu(
                concluder_agent, concluder_name, task_type, enable_mmlu_injection
            )

            # 对于MMLU任务，在强制生成最终答案时添加额外的强制要求
            # （论文未描述，随 enable_mmlu_prompt_injection 一同开关）
            if (
                enable_mmlu_injection
                and task_type.lower() in ["mmlu-pro", "mmlu_pro", "mmlu"]
                and hasattr(concluder_agent, 'role_prompt')
            ):
                force_answer_instruction = (
                    "\n\n**URGENT: YOU MUST OUTPUT A FINAL ANSWER NOW!**\n"
                    "This is the LAST step. You MUST provide a definitive answer.\n"
                    "- Review all previous reasoning and information\n"
                    "- Choose the BEST option from the available choices (A-J)\n"
                    "- Output ONLY the option letter in format: 'The answer is: X'\n"
                    "- DO NOT say 'I need more information' or 'cannot determine'\n"
                    "- You MUST make a choice even if uncertain - pick the most likely option\n"
                )
                if force_answer_instruction not in concluder_agent.role_prompt:
                    concluder_agent.role_prompt = concluder_agent.role_prompt + force_answer_instruction
                    logger.info(f"[MMLU Force Answer] Added mandatory answer requirement for {concluder_name}")
            
            # Execute concluder
            try:
                concluder_agent.activate(global_info, initial_dialog_history=concluder_agent.initial_dialog_history)
                current_action, _ = concluder_agent.take_action(global_info, external_tools_enabled=True)
                
                # Extract answer
                action_dict = current_action.to_dict() if hasattr(current_action, 'to_dict') else {}
                result_dict = action_dict.get('result', {})
                final_ans = result_dict.get('answer')
                if not final_ans and global_info.answers:
                    final_ans = global_info.answers[-1]
                if final_ans:
                    answers.append(final_ans)
                    logger.info(f"  ✓ Forced {concluder_name} → answer obtained")
            except Exception as e:
                logger.error(f"[MAS] Forced concluder failed: {e}")

    # ---------- Evaluate task reward ----------
    evaluator = BenchmarkEvaluator()
    task_reward = 0.0
    
    # 综合收集答案（和 MASReasoning._finalize() 一致）
    all_answers = answers + (global_info.answers if global_info.answers else [])
    if all_answers:
        final_answer = all_answers[-1]
    else:
        final_answer = ""
    
    # 对于 MMLU-Pro，尝试提取干净的选项字母
    # 同时解析选项列表，支持选项内容匹配（宽松判分）
    mmlu_options = None
    pred_letter = None
    if task_type.lower() in ["mmlu-pro", "mmlu_pro", "mmlu"]:
        # 从prompt中解析选项列表
        mmlu_options = parse_mmlu_options_from_prompt(prompt)
        if mmlu_options:
            logger.info(f"[MMLU] Parsed {len(mmlu_options)} options from prompt")
        
        # DEBUG: 打印原始答案内容，用于调试
        _raw_answer_preview = str(final_answer)[:200] if final_answer else "None"
        logger.info(f"[MMLU DEBUG] Raw final_answer: {_raw_answer_preview}")
        
        # 使用增强版答案提取，传入选项列表以支持内容匹配（和 main_mas.py 一致）
        pred_letter = evaluator.extract_mmlu_answer(final_answer, options=mmlu_options)
        if pred_letter:
            logger.info(f"[MMLU] Extracted answer: {pred_letter}")
            final_answer = pred_letter  # 和 MASReasoning._finalize() 一致，用提取后的字母替换
        else:
            logger.warning(f"[MMLU] Failed to extract answer from: {_raw_answer_preview}")
    
    # DEBUG: 打印标准答案，用于调试
    _label_preview = (str(label)[:40] + "...") if label and len(str(label)) > 40 else (str(label) or "None")
    logger.info(f"[DEBUG] Ground Truth: {_label_preview}")
    
    if task_type.lower() in ["mmlu-pro", "mmlu_pro", "mmlu"]:
        # 使用宽松判分，传入options支持选项内容匹配
        task_reward = 1.0 if evaluator.check_mmlu(final_answer, label, options=mmlu_options) else 0.0
        
        # 保存结果到文件（类似 main_mas.py 的格式）
        try:
            import json
            result_record = {
                "prompt_id": prompt_id,
                "pred": pred_letter if pred_letter else "",
                "pred_raw": str(final_answer)[:500] if final_answer else "",
                "gold": label,
                "correct": task_reward > 0,
                "reward": task_reward,
            }
            with open(results_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result_record, ensure_ascii=False) + "\n")
                f.flush()
            logger.info(f"[MMLU] Result saved: pred={pred_letter}, gold={label}, correct={task_reward > 0}")
        except Exception as e:
            logger.warning(f"[MMLU] Failed to save result: {e}")
    elif task_type.lower() in ["gsm-hard", "gsm8k", "gsm"]:
        task_reward = 1.0 if evaluator.check_gsm8k(final_answer, label) else 0.0
    elif task_type.lower() in ["scibench", "sci", "science"]:
        is_correct = evaluator.check_scibench(final_answer, label)
        task_reward = 1.0 if is_correct else 0.0
        # DEBUG: 打印详细的判分信息
        logger.info(f"[DEBUG] SciBench Check: pred='{final_answer[:50]}...' | gold='{label}' | correct={is_correct}")
    elif task_type.lower() in ["srdd"]:
        # 论文 Appendix A：SRDD 的 R_task 是 Completeness / Executability /
        # Consistency 三分量的算术平均，evaluator.check_srdd 已实现该定义。
        #
        # 注意 check_srdd(code_path, text) 要的是**代码文件**路径（内部 read_code
        # 用 os.path.isfile，executability 直接 `python {path}`），不是工作目录。
        # 优先用 agent 落盘时记录的 global_info.code_path；若为空则回退到扫描
        # workpath 下的 agent-main_*.py（见 utils/file_utils.write_file 的命名）。
        #
        # check_srdd 会起子进程跑 executability，且 consistency 依赖 embedding
        # API——两者都不能让训练崩掉，故整体包 try/except。
        try:
            code_path = getattr(global_info, "code_path", "") or ""
            if not code_path or not os.path.isfile(code_path):
                candidates = sorted(
                    glob.glob(os.path.join(global_info.workpath, "agent-main*.py"))
                )
                code_path = candidates[-1] if candidates else ""

            if not code_path:
                logger.warning(
                    f"[SRDD] No generated code file found under {global_info.workpath}; reward 0.0"
                )
                task_reward = 0.0
            else:
                reward_t, srdd_metrics = evaluator.check_srdd(code_path, question)
                task_reward = float(reward_t.item()) if hasattr(reward_t, "item") else float(reward_t)
                _fmt = {
                    k: round(float(v.item()) if hasattr(v, "item") else float(v), 4)
                    for k, v in (srdd_metrics or {}).items()
                }
                logger.info(f"[SRDD] reward={task_reward:.4f} components={_fmt} path={code_path}")
        except Exception as e:
            logger.warning(f"[SRDD] check_srdd failed ({e}); falling back to reward 0.0")
            task_reward = 0.0
    else:
        logger.warning(
            f"[MAS] No R_task defined for task_type='{task_type}'; reward will be 0.0. "
            "Training with an all-zero reward signal provides no learning signal."
        )
        task_reward = 0.0

    # Summarize result
    _ans_preview = (final_answer[:60] + "...") if final_answer and len(final_answer) > 60 else (final_answer or "None")
    logger.info(f"{'='*20} Result {'='*20}")
    logger.info(f"  Answer: {_ans_preview}")
    logger.info(f"  ★ Task Reward: {task_reward:.4f}")
    
    # ---------- Compute rewards for trajectory ----------
    total_steps = len([t for t in trajectory if t["agent_index"] == 0])
    
    for turn in trajectory:
        if turn["agent_index"] == 0:  # Scheduler LoRA
            step_idx = turn["metadata"]["step_idx"]
            sch_reward = sch_reward_fn(
                data_item={},
                response=turn["agent_output"],
                step_info={
                    "step_idx": step_idx,
                    "total_steps": total_steps,
                    "task_reward": task_reward,
                    # Use per-step token_count (cleaner credit assignment than total tokens)
                    "token_count": int(turn["metadata"].get("token_count", 0) or 0),
                },
            )
            turn["agent_reward"] = float(sch_reward)
            logger.info(f"  Reward[Sch@{step_idx}]: {sch_reward:.4f}")
        
        elif turn["agent_index"] == 1:  # Router LoRA
            routed_count = turn["metadata"].get("routed_count", 1)
            useful_count = turn["metadata"].get("useful_count", 0)
            
            rou_reward = rou_reward_fn(
                data_item={},
                response=turn["agent_output"],
                step_info={
                    "useful_count": useful_count,
                    "routed_count": routed_count,
                    "task_reward": task_reward,
                    "top_n": mas_config.top_n,  # Pass top_n for penalty check
                },
            )
            turn["agent_reward"] = float(rou_reward)
            # Log with penalty info if exceeded
            penalty_info = f" ⚠{routed_count}>{mas_config.top_n}" if routed_count > mas_config.top_n else ""
            logger.info(f"  Reward[Router]: {rou_reward:.4f}{penalty_info}")

    return {
        "prompt": prompt,
        "label": label,
        "trajectory": trajectory,
        "final_reward": task_reward,
    }
