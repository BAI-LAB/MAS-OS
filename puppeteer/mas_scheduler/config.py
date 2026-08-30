"""Configuration for MAS-style scheduler."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import yaml
import os


@dataclass
class MASConfig:
    """Configuration for MAS scheduler components."""
    
    # Maximum steps before forcing termination
    max_steps: int = 12
    
    # Memory retrieval parameters (paper Appendix A: M=7, K=3)
    top_m: int = 7  # M: candidate memory pool size from Memer
    top_n: int = 3   # K: context budget actually allocated to the agent

    # Semantic edge threshold τ for embedding cosine similarity (paper: τ = 0.80)
    similarity_threshold: float = 0.8
    # Separate, much looser threshold for the keyword-overlap (Jaccard) fallback
    # used when no embedding model is available. Reusing τ = 0.80 here would be
    # unreachable for word sets, silently producing zero semantic edges.
    fallback_overlap_threshold: float = 0.30
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: Optional[str] = None
    # Embedding-based semantic edges. Required for the paper's memory graph;
    # turning this off degrades semantic edges to keyword overlap.
    use_embeddings: bool = True
    
    # LLM settings
    # OpenAI-compatible model name used for auxiliary LLM calls
    # (memory summarization / usefulness feedback / fallback reasoning)
    llm_model: str = ""
    llm_temperature: float = 0.0
    
    # Whether to use the LLM policies for the OS kernel components.
    # The paper's Scheduler (pi_theta), Context Allocator (pi_phi) and Memory
    # Manager are all LLM-driven, so these default to True. Setting any to False
    # swaps in a rule-based fallback (keyword matching / string concatenation),
    # which is a degraded mode, NOT the paper's method.
    use_llm_scheduler: bool = True
    use_llm_router: bool = True
    use_llm_memer: bool = True

    # --- Non-paper extensions. All default to the paper-faithful setting. ---
    # Inject MMLU-Pro-specific answer-format instructions into worker agents'
    # role prompts. Not described in the paper; enable only to reproduce the
    # originally reported MMLU-Pro numbers.
    enable_mmlu_prompt_injection: bool = False
    # Bypass the learned allocation policy pi_phi with hand-written rules for
    # planner/terminator agents. Not described in the paper, and it removes
    # those steps from the RL gradient. Enable only to reproduce old numbers.
    enable_role_based_routing: bool = False
    
    # Logging
    log_level: str = "INFO"
    
    # Feedback collection (for router reward during training)
    # Set to False during evaluation to skip feedback API calls
    collect_feedback: bool = False
    
    # OpenAI API settings (loaded from global config)
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None


@dataclass
class DualLoRAConfig:
    """Configuration for Dual LoRA model (Scheduler + Router experts)."""
    
    # Base model
    base_model_name: str = "meta-llama/Llama-3.1-8B"
    
    # LoRA hyperparameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = None
    
    # Quantization for memory efficiency
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = True
    
    # Device settings
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    
    # Generation settings
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 0.9
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]


@dataclass
class RewardConfig:
    """Configuration for reward computation.

    NOTE: this class is NOT the one used during training. The live reward
    hyperparameters live in `verl_integration.MASRewardConfig`, which the MARTI
    workflow instantiates. Defaults here are kept in sync with the paper
    (Appendix A) so the two cannot drift.
    """

    # Context Allocator reward (Eq. 12): R_context = α * R_agent + η * R_task
    alpha: float = 0.25       # Weight for the agent hit-rate term
    eta: float = 1.0          # Weight for the terminal task reward

    # Agent Scheduler reward (Eq. 13): R_scheduler = R_task - λ * c_t
    lambda_tok: float = 5e-5  # Per-step token cost penalty

    # Return calculation
    gamma_discount: float = 1.0  # Discount factor (1.0 = no discounting)

    # Normalization
    normalize_rewards: bool = True
    reward_clip: float = 2.0


@dataclass
class TrainingConfig:
    """Configuration for RL training."""
    
    # Dual LoRA model config
    lora: DualLoRAConfig = field(default_factory=DualLoRAConfig)
    
    # Reward config
    reward: RewardConfig = field(default_factory=RewardConfig)
    
    # Training hyperparameters
    learning_rate: float = 1e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    num_epochs: int = 3
    
    # PPO/GRPO specific
    ppo_epochs: int = 4
    clip_range: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    
    # Buffer settings
    buffer_size: int = 1000
    min_batch_size: int = 16  # Minimum samples before training
    
    # Checkpointing
    save_steps: int = 100
    checkpoint_dir: str = "./checkpoints/dual_lora"
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 50
    
    # verl specific settings
    use_verl: bool = True
    verl_algorithm: str = "grpo"  # "ppo", "grpo", "reinforce"


def load_mas_config() -> MASConfig:
    """Load MAS config, merging with puppeteer's global config for API keys."""
    config = MASConfig()
    
    # 使用绝对路径加载配置，避免依赖当前工作目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    global_config_path = os.path.join(project_root, "config", "global.yaml")
    
    if os.path.exists(global_config_path):
        with open(global_config_path, "r", encoding="utf-8") as f:
            global_config = yaml.safe_load(f)
        
        api_keys = global_config.get("api_keys", {})
        config.openai_api_key = api_keys.get("openai_api_key")
        config.openai_base_url = api_keys.get("openai_base_url")
        # Optional: allow specifying model name in global config
        # (supports both "llm_model" and "openai_model" keys)
        config.llm_model = api_keys.get("llm_model") or api_keys.get("openai_model") or config.llm_model
        
        # Use graph settings if available
        graph_config = global_config.get("graph", {})
        if "max_step_num" in graph_config:
            config.max_steps = graph_config["max_step_num"]
    
    return config


DEFAULT_MAS_CONFIG = load_mas_config()

