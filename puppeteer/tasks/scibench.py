"""
SciBench Task Module - 科学问题数据集的加载和评测

SciBench 数据集包含科学领域（物理、化学等）的数值计算问题。
每条样本字段：
- problem_text: 题干（包含 LaTeX）
- answer_number: 数值答案（字符串形式，如 "+65.49", "0", "50.7"）
- unit: 单位（评测时不强制要求输出单位）
- source: 来源（如 atkins）
- problemid: 题目 id
- answer_latex/comment/solution: 可选字段

评测逻辑与 GSM-Hard 类似，使用数值容差判断。
"""
import os
import json
from typing import Dict, List, Optional, Any
from tqdm import tqdm


def load_dataset(mode: str = "train", data_limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    加载 SciBench 数据集。
    
    数据路径优先级：
    1. 本地: ./data/scibench/train.jsonl
    2. 如果本地不存在，尝试从 HuggingFace 下载 (xw27/scibench)
    
    Args:
        mode: 数据集模式 (train/test/validation)，SciBench 只有 train split
        data_limit: 限制样本数量
    
    Returns:
        样本列表
    """
    split = "train" if mode in ["train", "validation"] else "test"
    local_path = os.path.join("data", "scibench", f"{split}.jsonl")
    
    data = []
    if os.path.exists(local_path):
        # 从本地加载
        with open(local_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError:
                        continue
    else:
        # 尝试从 HuggingFace 下载（仅当 train.jsonl 不存在时）
        if split != "train":
            fallback_train_path = os.path.join("data", "scibench", "train.jsonl")
            if os.path.exists(fallback_train_path):
                with open(fallback_train_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                item = json.loads(line)
                                data.append(item)
                            except json.JSONDecodeError:
                                continue
                if data_limit:
                    data = data[:data_limit]
                return data

        try:
            from datasets import load_dataset as hf_load_dataset
            print("[SciBench] Local data not found, downloading from HuggingFace...")
            
            ds = hf_load_dataset("xw27/scibench", split="train")
            # 确保目录存在
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
             
            # 保存到本地
            with open(local_path, "w", encoding="utf-8") as f:
                for item in ds:
                    data.append(dict(item))
                    f.write(json.dumps(dict(item), ensure_ascii=False) + "\n")
            
            print(f"[SciBench] Downloaded and saved {len(data)} samples to {local_path}")
        except Exception as e:
            raise FileNotFoundError(
                f"SciBench data not found at {local_path} and failed to download: {e}\n"
                "Please ensure the data file exists or install 'datasets' package."
            )
    
    if data_limit:
        data = data[:data_limit]
    
    return data


def format_question(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """
    将 SciBench 样本格式化为 MAS 任务结构。
    
    拼接逻辑：
    - 固定前缀 instruction
    - 加上 problem_text
    - Answer 设为 answer_number
    
    Args:
        item: 原始样本
        idx: 样本索引
    
    Returns:
        MAS 任务字典
    """
    problem_text = item.get("problem_text", "")
    answer_number = item.get("answer_number", "")
    unit = item.get("unit", "")
    source = item.get("source", "")
    problemid = item.get("problemid", str(idx))
    
    # 构建 instruction + problem_text
    instruction = (
        "Solve the following science problem.\n"
        "Return only the final numeric answer (do not include units unless explicitly asked).\n\n"
    )
    question = instruction + problem_text
    
    return {
        "type": "SciBench",
        "Question": question,
        "Answer": answer_number,  # gold answer
        "id": problemid if problemid else str(idx),
        "unit": unit,
        "source": source,
        "idx": idx,  # 保留原始索引
    }


def run(runner, evaluator, results_dir: str, mode: str, data_limit: Optional[int] = None):
    """
    运行 SciBench 评测。
    
    Args:
        runner: MAS 推理 runner
        evaluator: BenchmarkEvaluator 实例
        results_dir: 结果保存目录
        mode: 数据集模式
        data_limit: 限制样本数量
    """
    dataset = load_dataset(mode, data_limit)
    result_path = os.path.join(results_dir, f"scibench_{mode}_mas.jsonl")
    
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
        print(f"[SciBench] Resume: Found {len(done_ids)} existing results, {acc} correct")
    
    total = len(dataset)
    print(f"[SciBench] Running {total} samples, {len(done_ids)} already done")
    
    with open(result_path, "a", encoding="utf-8") as fd:
        for idx, item in enumerate(tqdm(dataset, total=total, desc="SciBench")):
            task = format_question(item, idx)
            
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
            
            done_count = len(done_ids) + idx + 1 - len([i for i in range(idx + 1) if format_question(dataset[i], i)["id"] in done_ids])
            current_acc = acc / done_count if done_count > 0 else 0
            
    final_acc = acc / total if total > 0 else 0
    print(f"\n{'='*40}")
    print(f"SciBench Final Accuracy: {final_acc:.4f} ({acc}/{total})")
    print(f"Results saved to: {result_path}")
    
    return final_acc

