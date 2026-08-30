import os
import yaml


class APIConfig:
    def __init__(self):
        self._config = self._init_config()
    
    def _init_config(self):
        """
        Load API config from puppeteer project root, independent of CWD.
        This allows importing puppeteer from external projects (e.g., MARTI/Ray workers)
        without relying on './config/global.yaml' relative to the current working dir.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, ".."))
        global_config_path = os.path.join(project_root, "config", "global.yaml")

        try:
            with open(global_config_path, "r", encoding="utf-8") as f:
                global_config = yaml.safe_load(f)
        except FileNotFoundError:
            # Fallback to empty config – callers should handle missing keys gracefully.
            global_config = {}
        api_keys = global_config.get("api_keys") or {}
        key_config = {
            "openai":{
            "openai_api_key": api_keys.get("openai_api_key"),
            "openai_base_url": api_keys.get("openai_base_url", None),
            },
            "retry_times": global_config.get("max_retry_times", 10),
            "weight_path": global_config.get("model_weight_path")
        }
        return key_config

    def get(self, provider: str) -> dict:
        return self._config.get(provider, {})
    
    def global_openai_client(self):
        from openai import OpenAI
        api_key = self._config.get("openai").get("openai_api_key", None)
        base_url = self._config.get("openai").get("openai_base_url", None)
        return OpenAI(api_key=api_key, base_url=base_url)

api_config = APIConfig()