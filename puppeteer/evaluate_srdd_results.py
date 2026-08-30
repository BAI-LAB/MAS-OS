"""
SRDD Results Evaluation Script

This script evaluates SRDD benchmark results based on three metrics:
1. Executability: Whether the code can run without errors
2. Consistency: Semantic similarity between requirements and code (via embeddings)
3. Completeness: Whether the code contains placeholder statements (pass/todo)

Final Reward = consistency * completeness (if executable), else -1.0

Usage:
    python evaluate_srdd_results.py <path_to_jsonl_file>

Example:
    python evaluate_srdd_results.py results_mas/SRDD_validation_20251207_104536/srdd_mas.jsonl
"""

import json
import os
import sys
import re
import subprocess
import signal
import time
import numpy as np
import yaml
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass

# ==================== Configuration ====================
# Conda environment path for executing code
CONDA_ENV_PYTHON = "/root/miniconda3/envs/mas/bin/python"
# Embedding model aligned with ChatDev SRDD official script
EMBEDDING_MODEL = "text-embedding-ada-002"

# Load OpenAI config from global.yaml
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "global.yaml")

def load_config():
    """Load configuration from global.yaml"""
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    return {}

CONFIG = load_config()
OPENAI_API_KEY = CONFIG.get("api_keys", {}).get("openai_api_key", "")
OPENAI_BASE_URL = CONFIG.get("api_keys", {}).get("openai_base_url", "")

# Initialize OpenAI client for embeddings
try:
    from openai import OpenAI
    if OPENAI_API_KEY and OPENAI_BASE_URL:
        openai_client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL
        )
        HAS_EMBEDDING = True
        print(f"✓ OpenAI client initialized with base_url: {OPENAI_BASE_URL}")
    else:
        openai_client = None
        HAS_EMBEDDING = False
        print("Warning: OpenAI API key or base URL not configured.")
except ImportError:
    openai_client = None
    HAS_EMBEDDING = False
    print("Warning: OpenAI library not available. Run: pip install openai")


@dataclass
class SRDDMetrics:
    executability: float
    consistency: float
    completeness: float
    reward: float
    error_msg: str = ""


def read_code(path: str) -> str:
    """Read code from file."""
    if not os.path.exists(path):
        return ""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return ""


def remove_comments(code: str) -> str:
    """Remove comments from Python code."""
    def remove_comments_by_regex(string, regex):
        lines = string.split("\n")
        lines = [line for line in lines if not line.strip().startswith("#")]
        string = "\n".join(lines)
        comments = []
        matches = re.finditer(regex, string, re.DOTALL)
        for match in matches:
            group1 = match.group(1)
            comments.append(group1)
        for comment in comments + ["''''''\n"]:
            string = string.replace(comment, "")
        return string

    code = remove_comments_by_regex(code, r"'''(.*?)'''")
    code = remove_comments_by_regex(code, r"\"\"\"(.*?)\"\"\"")
    return code


def check_executability(code_path: str, timeout: int = 10) -> Tuple[bool, str]:
    """Check if code is executable using the specified conda environment."""
    def robust_kill(process):
        if process.poll() is None:
            if os.name == 'nt':
                os.kill(process.pid, signal.SIGTERM)
                time.sleep(1)
                if process.poll() is None:
                    os.kill(process.pid, signal.CTRL_BREAK_EVENT)
            else:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    time.sleep(1)
                    if process.poll() is None:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except:
                    pass

    try:
        if not os.path.exists(code_path):
            return False, "File does not exist"

        # Use the specified conda environment Python
        python_executable = CONDA_ENV_PYTHON if os.path.exists(CONDA_ENV_PYTHON) else "python3"
        
        if os.name == 'nt':
            command = f"{python_executable} {code_path}"
            process = subprocess.Popen(
                command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            command = f"{python_executable} {code_path}"
            process = subprocess.Popen(
                command, shell=True, preexec_fn=os.setsid,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

        try:
            out, err = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            robust_kill(process)
            return True, "Timeout (considered executable)"

        return_code = process.returncode
        output = out.decode('utf-8', errors='ignore')
        error_output = err.decode('utf-8', errors='ignore')

        if process.poll() is None:
            robust_kill(process)

        if return_code == 0:
            return True, output
        else:
            if error_output and "Traceback".lower() in error_output.lower():
                return False, error_output
            return False, error_output

    except Exception as ex:
        return False, f"Error: {str(ex)}"


def check_completeness(code: str) -> float:
    """Check if code contains placeholder statements (pass/todo)."""
    lines = code.split("\n")
    # Filter out lines with password, passenger, etc.
    lines = [line for line in lines if
             "password" not in line.lower() and 
             "passenger" not in line.lower() and 
             "passed" not in line.lower() and 
             "passes" not in line.lower()]
    # Check for pass or todo
    placeholder_lines = [line for line in lines if "pass" in line.lower() or "todo" in line.lower()]
    if len(placeholder_lines) > 0:
        return 0.0
    return 1.0


def get_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings."""
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2).T
    cos_sim = embedding1.dot(embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return float(cos_sim)


def get_openai_embedding(text: str, model: str = EMBEDDING_MODEL) -> List[float]:
    """Get embedding from OpenAI API."""
    if not openai_client:
        return None
    try:
        # Truncate text if too long (max ~8000 tokens for embedding models)
        max_chars = 8000 * 4  # rough estimate
        if len(text) > max_chars:
            text = text[:max_chars]
        
        response = openai_client.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None


def check_consistency(text: str, code: str) -> float:
    """Calculate semantic similarity between requirements and code."""
    if not HAS_EMBEDDING or not openai_client:
        # Fallback: simple heuristic based on keyword overlap
        text_words = set(re.findall(r'\b\w+\b', text.lower()))
        code_words = set(re.findall(r'\b\w+\b', code.lower()))
        if len(text_words) == 0:
            return 0.0
        overlap = len(text_words & code_words) / len(text_words)
        return min(overlap, 1.0)
    
    try:
        code_clean = remove_comments(code)
        text_clean = re.sub(r'^[^\n]*\n', '', text)  # Remove first line
        
        text_embedding = get_openai_embedding(text_clean)
        code_embedding = get_openai_embedding(code_clean)
        
        if text_embedding is None or code_embedding is None:
            # Fallback to keyword overlap
            text_words = set(re.findall(r'\b\w+\b', text.lower()))
            code_words = set(re.findall(r'\b\w+\b', code.lower()))
            if len(text_words) == 0:
                return 0.0
            return min(len(text_words & code_words) / len(text_words), 1.0)
        
        return get_cosine_similarity(text_embedding, code_embedding)
    except Exception as e:
        print(f"Error computing consistency: {e}")
        return 0.0


def evaluate_single_sample(pred: Dict, task_description: str = "") -> SRDDMetrics:
    """Evaluate a single SRDD prediction."""
    # Handle different prediction formats
    if isinstance(pred, dict):
        code_path = pred.get("code_path", "")
        code = pred.get("code", "")
    elif isinstance(pred, str):
        # Try to parse as JSON
        try:
            pred_dict = json.loads(pred)
            code_path = pred_dict.get("code_path", "")
            code = pred_dict.get("code", "")
        except:
            # Assume it's just code
            code_path = ""
            code = pred
    else:
        return SRDDMetrics(0, 0, 0, 0.0, "Invalid prediction format")

    # Get code content
    if code_path and os.path.exists(code_path):
        code = read_code(code_path)
    
    if not code or len(code.strip()) == 0:
        return SRDDMetrics(0, 0, 0, 0.0, "No code found")

    # 1. Check executability
    if code_path and os.path.exists(code_path):
        executable, error_msg = check_executability(code_path)
    else:
        # If no file path, assume not executable (can't test)
        executable = False
        error_msg = "No executable file path"
    
    executability = 1.0 if executable else 0.0

    # 2. Check completeness
    completeness = check_completeness(code)

    # 3. Check consistency (if description available)
    if task_description:
        consistency = check_consistency(task_description, code)
    else:
        consistency = 0.5  # Default value if no description

    # Calculate final reward
    reward = (executability + consistency + completeness) / 3

    return SRDDMetrics(
        executability=executability,
        consistency=consistency,
        completeness=completeness,
        reward=reward,
        error_msg=error_msg if not executable else ""
    )


def load_srdd_dataset() -> Dict[int, str]:
    """Load SRDD dataset to get original descriptions."""
    import pandas as pd
    srdd_path = os.path.join(os.path.dirname(__file__), "data", "SRDD", "SRDD.csv")
    
    if not os.path.exists(srdd_path):
        print(f"Warning: SRDD dataset not found at {srdd_path}")
        return {}
    
    try:
        df = pd.read_csv(srdd_path)
        # Shuffle with same random state as in main_mas.py
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        # Create mapping from index to description
        descriptions = {idx: row["Description"] for idx, row in df.iterrows()}
        print(f"✓ Loaded {len(descriptions)} SRDD descriptions")
        return descriptions
    except Exception as e:
        print(f"Error loading SRDD dataset: {e}")
        return {}


def evaluate_results_file(jsonl_path: str, use_embeddings: bool = True) -> Dict:
    """Evaluate all results in a JSONL file."""
    results = []
    
    # Load original descriptions for consistency calculation
    descriptions = load_srdd_dataset() if use_embeddings else {}
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total = len(lines)
    print(f"Evaluating {total} samples...")
    
    for line_num, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            pred = data.get("pred", {})
            task_id = data.get("id", line_num)
            
            # Get original description for this task
            task_description = descriptions.get(task_id, "")
            
            # Evaluate
            metrics = evaluate_single_sample(pred, task_description)
            
            results.append({
                "id": task_id,
                "executability": metrics.executability,
                "consistency": metrics.consistency,
                "completeness": metrics.completeness,
                "reward": metrics.reward,
                "error": metrics.error_msg
            })
            
            # Progress indicator
            if (line_num + 1) % 10 == 0 or line_num == total - 1:
                print(f"  Progress: {line_num + 1}/{total} ({(line_num + 1) / total * 100:.1f}%)")
            
        except json.JSONDecodeError as e:
            print(f"Error parsing line {line_num}: {e}")
            continue
    
    return results


def compute_statistics(results: List[Dict]) -> Dict:
    """Compute aggregate statistics from evaluation results."""
    n = len(results)
    if n == 0:
        return {"error": "No results to evaluate"}
    
    # Basic metrics
    executabilities = [r["executability"] for r in results]
    consistencies = [r["consistency"] for r in results]
    completenesses = [r["completeness"] for r in results]
    rewards = [r["reward"] for r in results]
    
    # Positive rewards only (for meaningful average)
    positive_rewards = [r for r in rewards if r > 0]
    # Overall score: mean of three averaged metrics (exec, completeness, consistency)
    avg_exec = sum(executabilities) / n
    avg_complete = sum(completenesses) / n
    avg_consistency = np.mean(consistencies)
    overall_score = (avg_exec + avg_complete + avg_consistency) / 3

    stats = {
        "total_samples": n,
        
        # Executability
        "executability_rate": avg_exec,
        "executable_count": int(sum(executabilities)),
        
        # Completeness
        "completeness_rate": avg_complete,
        "complete_count": int(sum(completenesses)),
        
        # Consistency (only for valid samples)
        "avg_consistency": avg_consistency,
        "std_consistency": np.std(consistencies),
        
        # Reward
        "avg_reward_all": np.mean(rewards),
        "avg_reward_positive": np.mean(positive_rewards) if positive_rewards else 0,
        "positive_reward_count": len(positive_rewards),
        "positive_reward_rate": len(positive_rewards) / n,
        
        # Success rate (executable AND complete)
        "success_rate": sum(1 for r in results if r["executability"] > 0 and r["completeness"] > 0) / n,

        # Overall score (mean of the three core metrics)
        "overall_score": overall_score,
    }
    
    return stats


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nUsage: python evaluate_srdd_results.py <path_to_jsonl_file> [--no-embeddings]")
        sys.exit(1)
    
    jsonl_path = sys.argv[1]
    use_embeddings = "--no-embeddings" not in sys.argv
    
    if not os.path.exists(jsonl_path):
        print(f"Error: File not found: {jsonl_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("SRDD Results Evaluation")
    print("=" * 60)
    print(f"📁 File: {jsonl_path}")
    print(f"🐍 Execution Python: {CONDA_ENV_PYTHON}")
    print(f"🔑 OpenAI API: {'Configured ✓' if HAS_EMBEDDING else 'Not available ✗'}")
    print(f"📊 Use embeddings: {'Yes' if use_embeddings and HAS_EMBEDDING else 'No (keyword overlap)'}")
    print()
    
    # Evaluate
    print("Evaluating results...")
    results = evaluate_results_file(jsonl_path, use_embeddings=use_embeddings and HAS_EMBEDDING)
    
    # Compute statistics
    stats = compute_statistics(results)
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\n📊 Total Samples: {stats['total_samples']}")
    
    print("\n📈 Executability:")
    print(f"   - Rate: {stats['executability_rate']:.2%} ({stats['executable_count']}/{stats['total_samples']})")
    
    print("\n📈 Completeness:")
    print(f"   - Rate: {stats['completeness_rate']:.2%} ({stats['complete_count']}/{stats['total_samples']})")
    
    print("\n📈 Consistency:")
    print(f"   - Average: {stats['avg_consistency']:.4f}")
    print(f"   - Std Dev: {stats['std_consistency']:.4f}")
    
    print("\n📈 Reward (Alignment):")
    print(f"   - Average (all samples): {stats['avg_reward_all']:.4f}")
    print(f"   - Average (positive only): {stats['avg_reward_positive']:.4f}")
    print(f"   - Positive count: {stats['positive_reward_count']}/{stats['total_samples']} ({stats['positive_reward_rate']:.2%})")
    
    print("\n🎯 Success Rate (Executable & Complete):")
    print(f"   - Rate: {stats['success_rate']:.2%}")
    
    print("\n" + "=" * 60)
    
    # Save detailed results
    output_path = jsonl_path.replace(".jsonl", "_evaluated.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "statistics": stats,
            "detailed_results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Detailed results saved to: {output_path}")


if __name__ == "__main__":
    main()

