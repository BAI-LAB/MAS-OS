from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from tenacity import retry
from tenacity.stop import stop_after_attempt
from tenacity.wait import wait_exponential

logger = logging.getLogger("model")


@dataclass
class TokenUsage:
    """Detailed token usage statistics from API calls.
    
    Used for precise token counting in reward computation.
    """
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    @classmethod
    def from_response(cls, usage) -> "TokenUsage":
        """Create from OpenAI response usage object."""
        return cls(
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
            total_tokens=getattr(usage, "total_tokens", 0),
        )
    
    @classmethod
    def estimate(cls, text: str) -> "TokenUsage":
        """Estimate token count from text (fallback)."""
        # Rough estimate: ~4 chars per token for English
        estimated = int(len(text) / 3.5)
        return cls(
            prompt_tokens=0,
            completion_tokens=estimated,
            total_tokens=estimated,
        )


class APIConfig:
    SLOW_FLAG = False 
    TRUNCATE_FACTOR = 0
    
    # Store last token usage for external access
    _last_token_usage: Optional[TokenUsage] = None
    
    @classmethod
    def get_last_token_usage(cls) -> Optional[TokenUsage]:
        """Get the token usage from the last API call."""
        return cls._last_token_usage
    
    @classmethod
    def set_last_token_usage(cls, usage: TokenUsage):
        """Store token usage from an API call."""
        cls._last_token_usage = usage

def model_log_and_print(content):
    if content is not None:
        logger.info(content)
        print(content)

def summarize_messages_brief(messages, max_words: int = 8, max_messages: int = 3) -> str:
    """Create a short one-line summary of chat messages for logging.
    
    - Keep only the latest few messages (system + latest user/assistant) to avoid
      huge, repeated dumps.
    - Flatten newlines/escapes so the log stays on one line.
    """
    if not messages:
        return ""
    
    def _clean_snippet(text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        text = text.replace("\\n", " ").replace("\n", " ")
        words = text.split()
        snippet = " ".join(words[:max_words])
        if len(words) > max_words:
            snippet += " ..."
        return snippet
    
    # Pick last system, last user, last assistant (in that order)
    latest = []
    for role in ("system", "user", "assistant"):
        for msg in reversed(messages):
            if msg.get("role") == role:
                latest.append(msg)
                break
    
    # Respect max_messages while keeping order system->user->assistant
    selected = [m for m in latest if m is not None][-max_messages:]
    
    parts = []
    for msg in selected:
        role = msg.get("role", "?")
        text = msg.get("content", "")
        parts.append(f"{role}: {_clean_snippet(text)}")
    return "; ".join(parts)

def truncate_messages(messages):
    max_length = 0
    max_index = 0
    for i, msg in enumerate(messages):
        if len(msg.get('content', '')) > max_length:
            max_length = len(msg['content'])
            max_index = i

    content = messages[max_index]['content']
    factor = 1/(2**APIConfig.TRUNCATE_FACTOR)
    messages[max_index]['content'] = content[:int(len(content)*factor)]  
    return messages


def calc_max_token(messages, max_tokens):
    string = "\n".join([str(message["content"]) for message in messages])
    num_prompt_tokens = int(len(string)//1.8) # approximation of tokens number 
    gap_between_send_receive = 15 * len(messages)
    num_prompt_tokens += gap_between_send_receive

    num_max_completion_tokens = max_tokens - num_prompt_tokens
    logger.info(f"num_prompt_tokens: {num_prompt_tokens}, num_max_completion_tokens: {num_max_completion_tokens}")
    if num_max_completion_tokens < 0:
        logger.warning(f"num_max_completion_tokens is negative: {num_max_completion_tokens}")
        return 0
    return num_max_completion_tokens


@retry(wait=wait_exponential(min=5, max=10), stop=stop_after_attempt(10))
def chat_completion_request(messages, model, new_client, model_config_dict: Dict = None) -> Tuple:
    """Make a chat completion request and return response with token usage.
    
    Returns:
        Tuple of (response, total_tokens) where total_tokens is an int.
        Also stores detailed TokenUsage in APIConfig for access.
    """
    if model_config_dict is None:
        model_config_dict = {
            "temperature": 0.1,
            "top_p": 1.0,
            "n": 1,
            "stream": False,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "logit_bias": {},
        }

    json_data = {
        "model": model,
        "messages": messages,
        "max_tokens": 4096,
        "temperature": model_config_dict["temperature"],
        "top_p": model_config_dict["top_p"],
        "n": model_config_dict["n"],
        "stream": model_config_dict["stream"],
        "frequency_penalty": model_config_dict["frequency_penalty"],
        "presence_penalty": model_config_dict["presence_penalty"],
        "logit_bias": model_config_dict["logit_bias"],
    }

    try:
        model_log_and_print("[Model Query] {}".format(summarize_messages_brief(messages)))
        if APIConfig.SLOW_FLAG:
            messages = truncate_messages(messages=messages)

        response = new_client.chat.completions.create(**json_data)

        # Extract token usage from response
        if hasattr(response, 'usage') and response.usage is not None:
            token_usage = TokenUsage.from_response(response.usage)
        else:
            # Fallback: estimate from response content
            content = response.choices[0].message.content if response.choices else ""
            token_usage = TokenUsage.estimate(content)
        
        # Store detailed usage for external access
        APIConfig.set_last_token_usage(token_usage)
        
        # Ensure we have a valid total
        total_tokens = token_usage.total_tokens
        if total_tokens == 0:
            total_tokens = token_usage.prompt_tokens + token_usage.completion_tokens
        if total_tokens == 0:
            # Last resort estimate
            content = response.choices[0].message.content if response.choices else ""
            total_tokens = int(len(content) / 3.5)
        
        model_log_and_print(
            f"[Model Query] Token Usage: "
            f"\nCompletion Tokens: {token_usage.completion_tokens} "
            f"\nPrompt Tokens: {token_usage.prompt_tokens} "
            f"\nTotal Tokens: {total_tokens}"
        )
        
        APIConfig.SLOW_FLAG = False
        APIConfig.TRUNCATE_FACTOR = 0
        return response, total_tokens   

    except Exception as e:
        print("Unable to generate ChatCompletion response. " + f"OpenAI calling Exception: {e}")
        APIConfig.SLOW_FLAG = True
        APIConfig.TRUNCATE_FACTOR += 1
        model_log_and_print(f"[Model Query: ChatCompletion] query failed: {str(e)}")
        raise Exception()


def get_last_token_usage() -> Optional[TokenUsage]:
    """Get detailed token usage from the last API call.
    
    This is useful for reward computation that needs precise token counts.
    """
    return APIConfig.get_last_token_usage()