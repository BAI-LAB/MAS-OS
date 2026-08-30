from typing import Dict, Any, Optional, List
from dataclasses import dataclass

@dataclass
class ModelConfig:
    name: str
    function_name: str     
    api_model_name: str    
    provider: str          
    max_tokens: int      
    model_size: int   # for open-source models, this is the number of parameters in millions; but for API models, this is just an estimate
    url: Optional[str] = None      
    temperature: float = 0.1       
    description: str = ""          


MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "gpt-3.5": ModelConfig(
        name = "gpt-3.5",
        function_name="query_gpt",
        api_model_name="",
        provider="openai",
        model_size=175,# which is estimated
        max_tokens=4096,
        description="OpenAI GPT-3.5 Turbo model"
    ),
    
    "gpt-4o": ModelConfig(
        name = "gpt-4o",
        function_name="query_gpt4o",
        api_model_name="",
        provider="openai",
        model_size=200,# which is estimated
        max_tokens=128000,
        description="OpenAI GPT-4o model"
    ),
    "qwen-2.5-14b": ModelConfig(
        name = "qwen-2.5-14b",
        function_name="query_qwen2_5_14b",
        api_model_name="Qwen/Qwen2.5-14B-Instruct",
        provider="local",
        model_size=14,
        max_tokens=8192,    
        url="http://",
        description="Qwen 2.5 14B Instruct model deployed locally"
    ),
    "openai_model_placeholder_1": ModelConfig(
        name = "openai_model_placeholder_1",
        function_name="query_llama_3_1_8b",
        api_model_name="",
        provider="openai",  # 使用 openai 兼容接口，通过 global.yaml 的 base_url 访问
        model_size=8,
        max_tokens=8192,
        description="Fireworks Meta-Llama 3.1 8B Instruct via OpenAI-compatible endpoint"
    ),
    "gpt-4o-mini": ModelConfig(
        name = "gpt-4o-mini",
        function_name="query_gpt4o_mini",
        api_model_name="",
        provider="openai",
        model_size=100,# which is estimated
        max_tokens=128000,
        description="OpenAI GPT-4o Mini model"
    ),
    "openai_model_placeholder_2": ModelConfig(
        name = "openai_model_placeholder_2",
        function_name="query_qwen2_5_7b_turbo",
        api_model_name="",
        provider="openai",  # 使用 openai 兼容接口，通过 global.yaml 的 base_url 访问
        model_size=7,
        max_tokens=8192,
        description="Fireworks Qwen 2.5 7B Instruct Turbo via OpenAI-compatible endpoint"
    ),
}

class ModelRegistry:
    def __init__(self):
        self.registry = MODEL_REGISTRY.copy()
    
    def register_model(self, key: str, config: ModelConfig) -> None:
        self.registry[key] = config
    
    def get_model_config(self, key: str) -> Optional[ModelConfig]:
        return self.registry.get(key)

    def get_model_size(self, key: str) -> Optional[int]:
        config = self.get_model_config(key)
        return config.model_size if config else None    
    
    def get_all_models(self) -> Dict[str, ModelConfig]:
        return self.registry.copy()
    
    def get_models_by_provider(self, provider: str) -> Dict[str, ModelConfig]:
        return {k: v for k, v in self.registry.items() if v.provider == provider}
    
    def get_function_name(self, key: str) -> Optional[str]:
        config = self.get_model_config(key)
        return config.function_name if config else None
    
    def get_api_model_name(self, key: str) -> Optional[str]:
        config = self.get_model_config(key)
        return config.api_model_name if config else None
    
    def list_available_models(self) -> List[str]:
        return list(self.registry.keys())
    
    def search_models(self, keyword: str) -> Dict[str, ModelConfig]:
        keyword = keyword.lower()
        return {
            k: v for k, v in self.registry.items() 
            if keyword in k.lower() or keyword in v.display_name.lower()
        }

model_registry = ModelRegistry()