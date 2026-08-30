"""Dual LoRA Expert Model for Scheduler and Router.

This module implements a single base model (Llama-3.1-8B) with two LoRA experts:
- Scheduler LoRA: Decides which agent to activate next
- Router LoRA: Selects which memories the current agent can see

The routing between experts is based on the prompt prefix/type.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

try:
    from peft import (
        LoraConfig,
        PeftModel,
        get_peft_model,
        prepare_model_for_kbit_training,
        TaskType,
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    LoraConfig = None

logger = logging.getLogger("MAS")


class ExpertType(Enum):
    """Type of LoRA expert to use."""
    SCHEDULER = "scheduler"
    ROUTER = "router"


@dataclass
class DualLoRAConfig:
    """Configuration for Dual LoRA model."""
    
    # Base model
    base_model_name: str = "meta-llama/Llama-3.1-8B"
    
    # LoRA settings (shared)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = None  # Will default to standard attention modules
    
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
            # Default target modules for Llama
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]


class DualLoRAModel:
    """
    A single base model with two LoRA adapters for Scheduler and Router.
    
    Architecture:
    - Base: Llama-3.1-8B-Instruct (frozen)
    - LoRA 1: Scheduler expert (trainable)
    - LoRA 2: Router expert (trainable)
    
    Routing logic:
    - Based on prompt prefix/type, activate the corresponding LoRA adapter
    - Only one adapter is active at a time for inference
    - During training, gradients flow through the active adapter only
    """
    
    def __init__(
        self,
        config: DualLoRAConfig = None,
        scheduler_adapter_path: Optional[str] = None,
        router_adapter_path: Optional[str] = None,
    ):
        if not PEFT_AVAILABLE:
            raise ImportError("peft library required. Install with: pip install peft")
        
        self.config = config or DualLoRAConfig()
        self._model = None
        self._tokenizer = None
        self._current_expert: Optional[ExpertType] = None
        
        # Adapter names
        self.scheduler_adapter_name = "scheduler_lora"
        self.router_adapter_name = "router_lora"
        
        # Load model and adapters
        self._load_base_model()
        self._create_or_load_adapters(scheduler_adapter_path, router_adapter_path)
        
        logger.info(f"[DualLoRA] Model initialized with base: {self.config.base_model_name}")
    
    def _load_base_model(self):
        """Load the base Llama model with quantization."""
        logger.info(f"[DualLoRA] Loading base model: {self.config.base_model_name}")
        
        # Quantization config for memory efficiency
        if self.config.use_4bit:
            compute_dtype = getattr(torch, self.config.bnb_4bit_compute_dtype)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=self.config.use_nested_quant,
            )
        else:
            bnb_config = None
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_name,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"  # For generation
        
        # Load model
        torch_dtype = getattr(torch, self.config.torch_dtype)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name,
            quantization_config=bnb_config,
            device_map=self.config.device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        
        # Prepare for training if using quantization
        if self.config.use_4bit:
            self._model = prepare_model_for_kbit_training(self._model)
        
        logger.info("[DualLoRA] Base model loaded")
    
    def _create_or_load_adapters(
        self,
        scheduler_path: Optional[str] = None,
        router_path: Optional[str] = None,
    ):
        """Create new LoRA adapters or load from checkpoints."""
        
        # Create LoRA config
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # First adapter: Scheduler
        if scheduler_path:
            logger.info(f"[DualLoRA] Loading Scheduler adapter from: {scheduler_path}")
            self._model = PeftModel.from_pretrained(
                self._model,
                scheduler_path,
                adapter_name=self.scheduler_adapter_name,
            )
        else:
            logger.info("[DualLoRA] Creating new Scheduler LoRA adapter")
            self._model = get_peft_model(self._model, lora_config, adapter_name=self.scheduler_adapter_name)
        
        # Second adapter: Router
        if router_path:
            logger.info(f"[DualLoRA] Loading Router adapter from: {router_path}")
            self._model.load_adapter(router_path, adapter_name=self.router_adapter_name)
        else:
            logger.info("[DualLoRA] Creating new Router LoRA adapter")
            self._model.add_adapter(self.router_adapter_name, lora_config)
        
        # Default to scheduler
        self.set_expert(ExpertType.SCHEDULER)
        
        logger.info(f"[DualLoRA] Active adapters: {list(self._model.peft_config.keys())}")
    
    def set_expert(self, expert_type: ExpertType):
        """Switch to the specified LoRA expert."""
        if expert_type == ExpertType.SCHEDULER:
            self._model.set_adapter(self.scheduler_adapter_name)
        else:
            self._model.set_adapter(self.router_adapter_name)
        
        self._current_expert = expert_type
        logger.debug(f"[DualLoRA] Switched to expert: {expert_type.value}")
    
    def get_current_expert(self) -> ExpertType:
        """Get the currently active expert type."""
        return self._current_expert
    
    @property
    def model(self) -> nn.Module:
        """Access the underlying model."""
        return self._model
    
    @property
    def tokenizer(self):
        """Access the tokenizer."""
        return self._tokenizer
    
    def generate(
        self,
        prompt: str,
        expert_type: ExpertType,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_logprobs: bool = False,
    ) -> Union[str, Tuple[str, torch.Tensor]]:
        """
        Generate response using the specified expert.
        
        Args:
            prompt: Input prompt text
            expert_type: Which LoRA expert to use
            max_new_tokens: Override default max tokens
            temperature: Override default temperature
            return_logprobs: Whether to return log probabilities
            
        Returns:
            Generated text, or (text, log_probs) if return_logprobs=True
        """
        # Switch to the correct expert
        self.set_expert(expert_type)
        
        # Tokenize
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(self._model.device)
        
        # Generation config
        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self.config.max_new_tokens,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        
        temp = temperature if temperature is not None else self.config.temperature
        if temp > 0:
            gen_kwargs["temperature"] = temp
            gen_kwargs["do_sample"] = True
            gen_kwargs["top_p"] = self.config.top_p
        else:
            gen_kwargs["do_sample"] = False
        
        if return_logprobs:
            gen_kwargs["output_scores"] = True
            gen_kwargs["return_dict_in_generate"] = True
        
        # Generate
        with torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)
        
        if return_logprobs:
            # Extract generated tokens (excluding input)
            input_len = inputs["input_ids"].shape[1]
            generated_ids = outputs.sequences[0, input_len:]
            
            # Compute log probabilities
            scores = outputs.scores  # List of (batch, vocab) tensors
            log_probs = []
            for i, score in enumerate(scores):
                probs = torch.softmax(score[0], dim=-1)
                token_id = generated_ids[i].item()
                log_probs.append(torch.log(probs[token_id] + 1e-10))
            
            log_probs_tensor = torch.stack(log_probs) if log_probs else torch.tensor([])
            
            # Decode
            response = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
            return response.strip(), log_probs_tensor
        else:
            # Decode only the generated part
            input_len = inputs["input_ids"].shape[1]
            generated_ids = outputs[0, input_len:]
            response = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
            return response.strip()
    
    def forward_with_logprobs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        expert_type: ExpertType,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that returns loss and log probabilities.
        
        Used for training with policy gradient methods.
        
        Args:
            input_ids: Tokenized input
            attention_mask: Attention mask
            labels: Target token IDs
            expert_type: Which LoRA expert to use
            
        Returns:
            (loss, log_probs) tensors
        """
        self.set_expert(expert_type)
        
        outputs = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        
        # Get logits and compute per-token log probs
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Gather log probs for the actual tokens
        # labels contains -100 for positions we don't want to compute loss on
        mask = labels != -100
        selected_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=labels.unsqueeze(-1).clamp(min=0),
        ).squeeze(-1)
        
        # Mask out padding positions
        selected_log_probs = selected_log_probs * mask
        
        return outputs.loss, selected_log_probs
    
    def get_trainable_parameters(self, expert_type: Optional[ExpertType] = None) -> List[nn.Parameter]:
        """Get trainable parameters for the specified expert(s)."""
        if expert_type is None:
            # Return all LoRA parameters
            return [p for n, p in self._model.named_parameters() if p.requires_grad]
        
        # Filter by adapter name
        adapter_name = (
            self.scheduler_adapter_name if expert_type == ExpertType.SCHEDULER
            else self.router_adapter_name
        )
        
        params = []
        for name, param in self._model.named_parameters():
            if param.requires_grad and adapter_name in name:
                params.append(param)
        
        return params
    
    def save_adapters(self, save_dir: str):
        """Save both LoRA adapters to disk."""
        import os
        
        scheduler_dir = os.path.join(save_dir, "scheduler_lora")
        router_dir = os.path.join(save_dir, "router_lora")
        
        self._model.save_pretrained(scheduler_dir, adapter_name=self.scheduler_adapter_name)
        self._model.save_pretrained(router_dir, adapter_name=self.router_adapter_name)
        
        logger.info(f"[DualLoRA] Saved adapters to: {save_dir}")
    
    def print_trainable_parameters(self):
        """Print the number of trainable parameters."""
        trainable = 0
        total = 0
        for name, param in self._model.named_parameters():
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()
        
        pct = 100 * trainable / total
        logger.info(f"[DualLoRA] Trainable: {trainable:,} / {total:,} ({pct:.2f}%)")


# Convenience functions for prompt formatting

def format_scheduler_prompt(
    question: str,
    sum_memory: str,
    agent_specs: str,
    pre_agent: Optional[str] = None,
) -> str:
    """Format prompt for Scheduler LoRA expert."""
    pre_agent_text = pre_agent if pre_agent else "None (first step)"
    progress_text = sum_memory if sum_memory else "Task just started. No agents have worked yet."
    
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a scheduler for a multi-agent reasoning system.

AVAILABLE AGENTS:
{agent_specs}

YOUR TASK:
Read the task progress and decide which agent should work next.
Reply with EXACTLY one word: an agent name, or DONE if task is complete.

CRITICAL: Do NOT repeat the previous agent consecutively.<|eot_id|><|start_header_id|>user<|end_header_id|>
QUESTION: {question}

PREVIOUS AGENT: {pre_agent_text}

CURRENT PROGRESS:
{progress_text}

Which agent should work next?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    return prompt


def format_router_prompt(
    question: str,
    now_agent: str,
    sum_memory: str,
    candidates: str,
    pre_agent: Optional[str] = None,
    top_n: int = 5,
) -> str:
    """Format prompt for Router LoRA expert."""
    prev_context = f"Previous agent: {pre_agent}\n" if pre_agent else ""
    
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a memory router for a multi-agent reasoning system.
Select 1-{top_n} most relevant memories for the current agent.

RULES:
- Select at least 1, at most {top_n} memories
- Only select genuinely useful memories
- Do NOT output duplicate indices (each index may appear at most once)
- Quality over quantity

OUTPUT FORMAT: Comma-separated indices only. Indices must be unique (no repeats like "0,0" or "1,1"). Example: "0, 2, 4"<|eot_id|><|start_header_id|>user<|end_header_id|>
Task: {question}
Current agent: {now_agent}
{prev_context}Progress: {sum_memory if sum_memory else 'None'}

Available memories:
{candidates}

Which memories should {now_agent} see?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    return prompt

