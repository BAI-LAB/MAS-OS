from __future__ import annotations

import os
import time
import logging
from typing import Any, Dict, Optional, Tuple

import requests

_logger = logging.getLogger("vLLM-LoRA")


class VLLMServerError(RuntimeError):
    pass


def _post_json(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    """Post JSON and return response. Handles both JSON and plain-text success responses."""
    r = requests.post(url, json=payload, timeout=timeout)
    if r.status_code >= 400:
        raise VLLMServerError(f"HTTP {r.status_code} from {url}: {r.text[:400]}")
    
    try:
        return r.json()
    except Exception:
        body = r.text.strip()
        if "success" in body.lower():
            return {"status": "success", "message": body}
        raise VLLMServerError(f"Invalid response from {url}: {body[:400]}")


def reload_lora_adapter(
    base_url: str,
    lora_name: str,
    lora_path: str,
    *,
    timeout: float = 10.0,
    retries: int = 2,
    raise_on_error: bool = True,
) -> Tuple[bool, Optional[str]]:
    """Load or reload a LoRA adapter on a running vLLM OpenAI server."""
    base_url = base_url.rstrip("/")
    load_url = f"{base_url}/v1/load_lora_adapter"
    unload_url = f"{base_url}/v1/unload_lora_adapter"

    adapter_model_path = os.path.join(lora_path, "adapter_model.bin")
    adapter_config_path = os.path.join(lora_path, "adapter_config.json")
    
    if not os.path.exists(adapter_model_path):
        err = f"Adapter model not found: {adapter_model_path}"
        if raise_on_error:
            raise VLLMServerError(err)
        return False, err
    
    if not os.path.exists(adapter_config_path):
        err = f"Adapter config not found: {adapter_config_path}"
        if raise_on_error:
            raise VLLMServerError(err)
        return False, err

    last_err: Optional[str] = None
    unload_first = False
    
    for i in range(max(1, retries + 1)):
        try:
            if unload_first:
                try:
                    _post_json(unload_url, {"lora_name": lora_name}, timeout=timeout)
                    _logger.debug(f"Unloaded existing adapter: {lora_name}")
                except Exception as ue:
                    _logger.debug(f"Unload failed (OK if first load): {ue}")
            
            result = _post_json(load_url, {"lora_name": lora_name, "lora_path": lora_path}, timeout=timeout)
            _logger.info(f"Successfully loaded adapter '{lora_name}' from {lora_path}")
            return True, None
            
        except Exception as e:
            last_err = str(e)
            err_lower = last_err.lower()
            
            if "already" in err_lower or "exists" in err_lower or "registered" in err_lower:
                unload_first = True
                _logger.debug(f"Adapter exists, will try unload-then-load: {last_err}")
            else:
                _logger.warning(f"Load attempt {i+1} failed: {last_err}")
            
            time.sleep(0.5 * (i + 1))

    full_err = f"Failed to reload LoRA adapter '{lora_name}' from {lora_path} after {retries+1} attempts: {last_err}"
    if raise_on_error:
        raise VLLMServerError(full_err)
    return False, full_err


def completions(
    base_url: str,
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    stop: Optional[list[str]] = None,
    lora_name: Optional[str] = None,
    timeout: float = 60.0,
) -> str:
    """Call vLLM OpenAI-compatible /v1/completions."""
    base_url = base_url.rstrip("/")
    url = f"{base_url}/v1/completions"

    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    if stop:
        payload["stop"] = stop

    if lora_name:
        payload_with_lora = dict(payload)
        payload_with_lora["lora_name"] = lora_name
        try:
            out = _post_json(url, payload_with_lora, timeout=timeout)
            return out["choices"][0].get("text", "").strip()
        except Exception:
            payload_model_alias = dict(payload)
            payload_model_alias["model"] = lora_name
            out = _post_json(url, payload_model_alias, timeout=timeout)
            return out["choices"][0].get("text", "").strip()

    out = _post_json(url, payload, timeout=timeout)
    return out["choices"][0].get("text", "").strip()


def chat_completions(
    base_url: str,
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    stop: Optional[list[str]] = None,
    lora_name: Optional[str] = None,
    timeout: float = 60.0,
) -> str:
    """Call vLLM OpenAI-compatible /v1/chat/completions."""
    base_url = base_url.rstrip("/")
    url = f"{base_url}/v1/chat/completions"

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    if stop:
        payload["stop"] = stop

    if lora_name:
        payload_with_lora = dict(payload)
        payload_with_lora["lora_name"] = lora_name
        try:
            out = _post_json(url, payload_with_lora, timeout=timeout)
            content = out["choices"][0]["message"]["content"].strip()
            _logger.debug(f"[vLLM] LoRA '{lora_name}' response: {content[:60]}...")
            return content
        except Exception as e:
            _logger.debug(f"[vLLM] LoRA '{lora_name}' not available ({e}), using base model")

    _logger.debug(f"[vLLM] Calling base model: {model}")
    out = _post_json(url, payload, timeout=timeout)
    content = out["choices"][0]["message"]["content"].strip()
    _logger.debug(f"[vLLM] Base model response: {content[:60]}...")
    return content
