"""LLM client for MAS scheduler, reusing puppeteer's OpenAI configuration."""
from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any

from .config import MASConfig, DEFAULT_MAS_CONFIG

logger = logging.getLogger("MAS")


class MASLLMClient:
    """OpenAI-compatible LLM client for MAS scheduler components."""
    
    def __init__(self, config: MASConfig = DEFAULT_MAS_CONFIG):
        self.config = config
        self._client = None
        self._setup_client()
    
    def _setup_client(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            
            api_key = self.config.openai_api_key
            base_url = self.config.openai_base_url
            
            if not api_key:
                raise RuntimeError("OpenAI API key not configured")
            
            if base_url:
                self._client = OpenAI(api_key=api_key, base_url=base_url)
            else:
                self._client = OpenAI(api_key=api_key)
            
            logger.info("MAS LLM client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            self._client = None
    
    @property
    def is_available(self) -> bool:
        return self._client is not None
    
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: int = 1024,
    ) -> str:
        """Send a chat completion request."""
        if not self.is_available:
            raise RuntimeError("LLM client not available")
        
        temp = temperature if temperature is not None else self.config.llm_temperature
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        try:
            response = self._client.chat.completions.create(
                model=self.config.llm_model,
                messages=messages,
                temperature=temp,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM chat error: {e}")
            raise
    
    def chat_with_history(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: int = 2048,
    ) -> str:
        """Send a chat completion request with full message history."""
        if not self.is_available:
            raise RuntimeError("LLM client not available")
        
        temp = temperature if temperature is not None else self.config.llm_temperature
        
        try:
            response = self._client.chat.completions.create(
                model=self.config.llm_model,
                messages=messages,
                temperature=temp,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM chat error: {e}")
            raise

