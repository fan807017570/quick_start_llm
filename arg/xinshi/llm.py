import os

from langchain_deepseek import ChatDeepSeek


def _env_int(name: str, default: int, min_value: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return max(min_value, int(raw))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def get_llm():
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    max_tokens = _env_int("DEEPSEEK_MAX_TOKENS", 1024, 128)
    temperature = _env_float("DEEPSEEK_TEMPERATURE", 0.0)
    return ChatDeepSeek(
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=api_key,
        streaming=True,
        stream_usage=True,
        max_tokens=max_tokens,
        temperature=temperature,
    )
