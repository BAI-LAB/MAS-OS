"""
MAS-style Multi-Agent Scheduler entry point using local vLLM deployment.

This script uses a locally deployed vLLM server (Llama-3.1-8B-Instruct) for all agents
instead of external API calls. All components (Scheduler, Router, Memer, Workers) use
the same local model.

Usage:
    python main_mas_vllm.py <task> <mode> [--data_limit LIMIT] [--max_steps STEPS]

Example:
    python main_mas_vllm.py gsm-hard test --data_limit 10 --max_steps 12
    python main_mas_vllm.py MMLU-Pro validation --data_limit 5

Prerequisites:
    Start vLLM server first:
    python -m vllm.entrypoints.openai.api_server \
        --model <YOUR_MODEL_PATH> \
        --port 8000 --trust-remote-code \
        --max-model-len 8192 --max-num-seqs 8

    Then pass the same model id via --model (or export MAS_VLLM_MODEL).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import yaml
from datetime import datetime
from typing import Dict, List, Optional, Any

# Setup logging before imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger("MAS-vLLM")

# Import puppeteer components
from agent.register.register import agent_global_registry
from agent.agent_info.global_info import GlobalInfo
from utils.log_manager import LogManager

# Import MAS scheduler
from mas_scheduler import (
    MASConfig,
    DEFAULT_MAS_CONFIG,
    MASLLMClient,
)
from mas_scheduler.mas_reasoning import MASReasoning

# Import task modules
from tasks import mmlu_pro, gsm_hard, srdd, creative_writing, scibench
from tasks.evaluator import BenchmarkEvaluator


def create_vllm_config(
    max_steps: int = 12,
    model: str = "",
    base_url: str = "",
) -> MASConfig:
    """
    Create MAS config that uses local vLLM deployment.

    All agents will use the same local model served by vLLM.

    Args:
        max_steps: Maximum reasoning steps
        model: Model id as registered in vLLM (falls back to $MAS_VLLM_MODEL)
        base_url: vLLM OpenAI-compatible endpoint (falls back to $MAS_VLLM_BASE_URL)
    """
    config = MASConfig()

    # Configure to use local vLLM server
    config.openai_api_key = "EMPTY"  # vLLM doesn't require real API key
    config.openai_base_url = (
        base_url or os.environ.get("MAS_VLLM_BASE_URL") or "http://localhost:8000/v1"
    )
    config.llm_model = model or os.environ.get("MAS_VLLM_MODEL", "")
    if not config.llm_model:
        raise SystemExit(
            "No vLLM model specified. Pass --model, or set $MAS_VLLM_MODEL to the "
            "model id registered with your vLLM server (e.g. the path you passed "
            "to `vllm serve`)."
        )

    # Set max steps
    config.max_steps = max_steps

    # Enable LLM-based components
    config.use_llm_scheduler = True
    config.use_llm_router = True
    config.use_llm_memer = True

    # Disable feedback collection during evaluation
    config.collect_feedback = False

    # Temperature for generation
    config.llm_temperature = 0.0

    logger.info("=" * 60)
    logger.info("vLLM Configuration")
    logger.info(f"  Base URL: {config.openai_base_url}")
    logger.info(f"  Model: {config.llm_model}")
    logger.info(f"  Max steps: {config.max_steps}")
    logger.info(f"  Temperature: {config.llm_temperature}")
    logger.info("=" * 60)

    return config


def update_global_config_for_vllm(
    global_config: Dict[str, Any],
    mas_config: Optional[MASConfig] = None,
) -> Dict[str, Any]:
    """
    Update global config to use local vLLM for all agent API calls.

    This ensures that worker agents also use the local vLLM deployment, with the
    same endpoint/model that create_vllm_config resolved.
    """
    if "api_keys" not in global_config:
        global_config["api_keys"] = {}

    base_url = (
        getattr(mas_config, "openai_base_url", "")
        or os.environ.get("MAS_VLLM_BASE_URL")
        or "http://localhost:8000/v1"
    )
    model = getattr(mas_config, "llm_model", "") or os.environ.get("MAS_VLLM_MODEL", "")

    # Configure all agents to use local vLLM
    global_config["api_keys"]["openai_api_key"] = "EMPTY"
    global_config["api_keys"]["openai_base_url"] = base_url
    global_config["api_keys"]["openai_model"] = model

    logger.info("Updated global config to use local vLLM for all agents")

    return global_config


class MASBenchmarkRunner:
    """
    Runner that uses MAS-style scheduling with local vLLM deployment.
    
    All agents (Scheduler, Router, Memer, Workers) use the same local model.
    """
    
    def __init__(
        self,
        personas_path: str,
        global_config: Dict[str, Any],
        mas_config: Optional[MASConfig] = None,
    ):
        self.personas_path = personas_path
        self.mas_config = mas_config or create_vllm_config()
        self.global_config = update_global_config_for_vllm(global_config, self.mas_config)
        self.max_step_num = self.mas_config.max_steps
        
        logger.info("=" * 60)
        logger.info("MAS Benchmark Runner (vLLM) Initialized")
        logger.info(f"  Personas: {personas_path}")
        logger.info(f"  Max steps: {self.max_step_num}")
        logger.info(f"  LLM model: {self.mas_config.llm_model}")
        logger.info(f"  LLM base URL: {self.mas_config.openai_base_url}")
        logger.info("=" * 60)
    
    def setup_reasoning(self, data_item: Dict[str, Any]) -> MASReasoning:
        """Setup MAS-style reasoning for a task."""
        # Register all agents from personas
        agent_global_registry.register_all_agents(self.personas_path)
        agent_global_registry.reset_all_agents()
        
        # Get all registered agents
        agents = {}
        for name in agent_global_registry.agent_names:
            agent = agent_global_registry.get_agent_from_name(name)
            if agent:
                agents[name] = agent
        
        logger.info(f"Registered {len(agents)} agents: {list(agents.keys())}")
        
        # Create log manager
        log_manager = LogManager("./config/global.yaml", data_item.get("type", "unknown"))
        workspace_path = log_manager.folder_path
        
        # Create MAS reasoning
        reasoning = MASReasoning(
            task=data_item,
            agents=agents,
            config=self.mas_config,
            workspace_path=workspace_path,
            log_manager=log_manager,
        )
        
        return reasoning, workspace_path
    
    def run_reasoning(self, data_item: Dict[str, Any]) -> str:
        """Run MAS-style reasoning on a single task."""
        reasoning, workspace_path = self.setup_reasoning(data_item)
        
        # Create global_info for the task
        global_info = GlobalInfo(
            path_id=0,
            workpath=workspace_path,
            task=data_item,
        )
        # Ensure logger is available for agents
        global_info.logger = logger
        
        # Start reasoning
        reasoning.start(global_info)
        
        # Run for max_step_num steps
        final_ans, _ = reasoning.n_step(self.max_step_num)
        
        # Visualize (optional)
        reasoning.visualize_path()
        
        return final_ans


def run_mmlu_pro(runner: MASBenchmarkRunner, evaluator: BenchmarkEvaluator, 
                 results_dir: str, mode: str, data_limit: Optional[int] = None):
    """Run MAS on MMLU-Pro dataset."""
    import pandas as pd
    import string
    from tqdm import tqdm
    
    path = os.path.join("data", "MMLU-Pro", f"{mode}.parquet")
    data = pd.read_parquet(path)
    if data_limit:
        data = data[:data_limit]
    
    result_path = os.path.join(results_dir, f"MMLU-Pro_{mode}_vllm.jsonl")
    
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
                except:
                    pass
        logger.info(f"[Resume] Found {len(done_ids)} existing results. Continuing...")
    
    total = len(data)
    logger.info(f"Running MMLU-Pro ({mode}), {total} samples total, {len(done_ids)} already done")
    
    with open(result_path, "a", encoding="utf-8") as fd:
        for idx, row in tqdm(data.iterrows(), total=total):
            if row["question_id"] in done_ids:
                continue
            
            # Format question
            options = [f"{letter}: {op}" for letter, op in zip(string.ascii_uppercase, row["options"])]
            prompt = f"The following are multiple choice questions (with answers) about {row['category']}."
            question = prompt + "\n" + row["question"] + "\n" + " ".join(options)
            
            task = {
                "type": "MMLU-Pro",
                "Question": question,
                "Answer": row["answer"],
                "id": row["question_id"],
            }
            
            final_ans = runner.run_reasoning(task)
            flag = evaluator.check_mmlu(final_ans, task["Answer"])
            
            if flag:
                acc += 1
            
            record = {
                "id": task["id"],
                "pred": final_ans,
                "correct": flag,
            }
            fd.write(json.dumps(record, ensure_ascii=False) + "\n")
            fd.flush()
            
            done_count = len(done_ids) + idx + 1 - len([i for i in range(idx + 1) if data.iloc[i]["question_id"] in done_ids])
            current_acc = acc / done_count if done_count > 0 else 0
            logger.info(f"Sample {idx + 1}/{total}: {'✓' if flag else '✗'} (Running acc: {current_acc:.3f})")
    
    final_count = total
    final_acc = acc / final_count if final_count > 0 else 0
    logger.info(f"\n{'='*40}")
    logger.info(f"MMLU-Pro Final Accuracy: {final_acc:.4f} ({acc}/{final_count})")
    logger.info(f"Results saved to: {result_path}")


def run_gsm_hard(runner: MASBenchmarkRunner, evaluator: BenchmarkEvaluator,
                 results_dir: str, mode: str, data_limit: Optional[int] = None):
    """Run MAS on GSM-Hard dataset."""
    import pandas as pd
    from tqdm import tqdm
    
    path = os.path.join("data", "GSM-Hard", "test.parquet")
    data = pd.read_parquet(path)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    if data_limit:
        data = data[:data_limit]
    
    result_path = os.path.join(results_dir, "gsm-hard_vllm.jsonl")
    
    # 断点续跑：读取已有结果，跳过已处理的样本
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
                except:
                    pass
        logger.info(f"[Resume] Found {len(done_ids)} existing results, {acc} correct. Continuing...")
    
    total = len(data)
    
    logger.info(f"Running GSM-Hard, {total} samples total, {len(done_ids)} already done")
    
    # 追加模式写入
    with open(result_path, "a", encoding="utf-8") as fd:
        for idx, row in enumerate(tqdm(data.iterrows(), total=total)):
            # 跳过已处理的样本
            if idx in done_ids:
                continue
            
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
            
            record = {
                "id": task["id"],
                "pred": final_ans,
                "correct": flag,
            }
            fd.write(json.dumps(record, ensure_ascii=False) + "\n")
            fd.flush()  # 实时写入，避免中断丢失
            
            done_count = len(done_ids) + idx - len([i for i in range(idx) if i in done_ids]) + 1
            current_acc = acc / done_count if done_count > 0 else 0
            logger.info(f"Sample {idx + 1}/{total}: {'✓' if flag else '✗'} (Running acc: {current_acc:.3f})")
    
    final_count = len(done_ids) + sum(1 for i in range(total) if i not in done_ids)
    final_acc = acc / final_count if final_count > 0 else 0
    logger.info(f"\n{'='*40}")
    logger.info(f"GSM-Hard Final Accuracy: {final_acc:.4f} ({acc}/{final_count})")
    logger.info(f"Results saved to: {result_path}")


def run_srdd(runner: MASBenchmarkRunner, evaluator: BenchmarkEvaluator,
             results_dir: str, mode: str, data_limit: Optional[int] = None):
    """Run MAS on SRDD dataset."""
    import pandas as pd
    from tqdm import tqdm
    
    data = pd.read_csv("./data/SRDD/SRDD.csv")
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    if data_limit:
        data = data[:data_limit]
    
    result_path = os.path.join(results_dir, "srdd_vllm.jsonl")
    
    # 断点续跑
    done_ids = set()
    if os.path.exists(result_path):
        with open(result_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    done_ids.add(record["id"])
                except:
                    pass
        logger.info(f"[Resume] Found {len(done_ids)} existing results. Continuing...")
    
    total = len(data)
    logger.info(f"Running SRDD, {total} samples total, {len(done_ids)} already done")
    
    with open(result_path, "a", encoding="utf-8") as fd:
        for idx, row in tqdm(data.iterrows(), total=total):
            if idx in done_ids:
                continue
            
            task = {
                "type": "SRDD",
                "req": "code",
                "Question": "Develop a pythonic software following description:\n" + row["Description"],
                "id": idx,
            }
            
            final_ans = runner.run_reasoning(task)
            
            record = {
                "id": task["id"],
                "pred": final_ans,
            }
            fd.write(json.dumps(record, ensure_ascii=False) + "\n")
            fd.flush()
    
    logger.info(f"\nSRDD completed. Results saved to: {result_path}")


def run_creative_writing(runner: MASBenchmarkRunner, evaluator: BenchmarkEvaluator,
                         results_dir: str, mode: str, data_limit: Optional[int] = None):
    """Run MAS on Creative Writing dataset."""
    from tqdm import tqdm
    
    path = "./data/CW/creative_writing.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    
    if data_limit:
        data = data[:data_limit]
    
    result_path = os.path.join(results_dir, "cw_vllm.jsonl")
    
    # 断点续跑
    done_ids = set()
    if os.path.exists(result_path):
        with open(result_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    done_ids.add(record["id"])
                except:
                    pass
        logger.info(f"[Resume] Found {len(done_ids)} existing results. Continuing...")
    
    total = len(data)
    logger.info(f"Running Creative Writing, {total} samples total, {len(done_ids)} already done")
    
    with open(result_path, "a", encoding="utf-8") as fd:
        for idx, q in enumerate(tqdm(data)):
            if idx in done_ids:
                continue
            
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
            
            record = {
                "id": task["id"],
                "pred": final_ans,
            }
            fd.write(json.dumps(record, ensure_ascii=False) + "\n")
            fd.flush()
    
    logger.info(f"\nCreative Writing completed. Results saved to: {result_path}")


def run_scibench(runner: MASBenchmarkRunner, evaluator: BenchmarkEvaluator,
                 results_dir: str, mode: str, data_limit: Optional[int] = None):
    """
    Run MAS on SciBench dataset.
    
    SciBench 数据集包含科学领域数值计算问题。
    评测逻辑与 GSM-Hard 类似，使用数值容差判断。
    """
    from tqdm import tqdm
    
    # 加载数据
    dataset = scibench.load_dataset(mode, data_limit)
    
    result_path = os.path.join(results_dir, f"scibench_{mode}_vllm.jsonl")
    
    # 断点续跑
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
                except:
                    pass
        logger.info(f"[Resume] Found {len(done_ids)} existing results, {acc} correct. Continuing...")
    
    total = len(dataset)
    logger.info(f"Running SciBench ({mode}), {total} samples total, {len(done_ids)} already done")
    
    with open(result_path, "a", encoding="utf-8") as fd:
        for idx, item in enumerate(tqdm(dataset, total=total)):
            task = scibench.format_question(item, idx)
            
            # 跳过已处理的样本
            if task["id"] in done_ids:
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
            fd.flush()
            
            done_count = len(done_ids) + idx + 1 - len([i for i in range(idx + 1) if scibench.format_question(dataset[i], i)["id"] in done_ids])
            current_acc = acc / done_count if done_count > 0 else 0
            logger.info(f"Sample {idx + 1}/{total}: {'✓' if flag else '✗'} (Running acc: {current_acc:.3f})")
    
    final_count = total
    final_acc = acc / final_count if final_count > 0 else 0
    logger.info(f"\n{'='*40}")
    logger.info(f"SciBench Final Accuracy: {final_acc:.4f} ({acc}/{final_count})")
    logger.info(f"Results saved to: {result_path}")


def main():
    parser = argparse.ArgumentParser(description="Run MAS with local vLLM deployment")
    parser.add_argument("task", choices=["MMLU-Pro", "gsm-hard", "SRDD", "CW", "scibench"],
                       help="Benchmark task to run")
    parser.add_argument("mode", choices=["validation", "test"],
                       help="Dataset split (validation or test)")
    parser.add_argument("--level", type=int, default=1,
                       help="Task level (if applicable)")
    parser.add_argument("--index", type=int, default=-1,
                       help="Specific task index (if applicable)")
    parser.add_argument("--data_limit", type=int, default=None,
                       help="Limit number of samples to process (default: process all)")
    parser.add_argument("--max_steps", type=int, default=12,
                       help="Maximum reasoning steps (default: 12)")
    parser.add_argument("--personas", type=str, default="personas/personas.jsonl",
                       help="Path to personas file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from existing results directory")
    parser.add_argument("--model", type=str, default="",
                       help="Model id as registered in vLLM (default: $MAS_VLLM_MODEL)")
    parser.add_argument("--base_url", type=str, default="",
                       help="vLLM OpenAI-compatible endpoint "
                            "(default: $MAS_VLLM_BASE_URL or http://localhost:8000/v1)")

    args = parser.parse_args()

    # Load global config
    with open("config/global.yaml", "r") as f:
        global_config = yaml.safe_load(f)

    # Create vLLM-based MAS config
    mas_config = create_vllm_config(
        max_steps=args.max_steps,
        model=args.model,
        base_url=args.base_url,
    )
    
    # Create runner
    runner = MASBenchmarkRunner(args.personas, global_config, mas_config)
    evaluator = BenchmarkEvaluator()
    
    # Setup results directory
    if args.resume:
        # 断点续跑：使用已有目录
        results_dir = args.resume
        if not os.path.isabs(results_dir):
            results_dir = os.path.join(os.getcwd(), results_dir)
        if not os.path.exists(results_dir):
            logger.error(f"Resume directory not found: {results_dir}")
            sys.exit(1)
        logger.info(f"Resuming from: {results_dir}")
    else:
        # 新建目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(os.getcwd(), "results_vllm", f"{args.task}_{args.mode}_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        logger.info(f"Results will be saved to: {results_dir}")
    
    # Run the selected task
    task_runners = {
        "MMLU-Pro": run_mmlu_pro,
        "gsm-hard": run_gsm_hard,
        "SRDD": run_srdd,
        "CW": run_creative_writing,
        "scibench": run_scibench,
    }
    
    if args.task in task_runners:
        task_runners[args.task](runner, evaluator, results_dir, args.mode, args.data_limit)
    else:
        logger.error(f"Unknown task: {args.task}")
        sys.exit(1)


if __name__ == "__main__":
    main()
