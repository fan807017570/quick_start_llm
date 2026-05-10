import os

from langchain_deepseek import ChatDeepSeek


def get_llm():
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    return ChatDeepSeek(
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=api_key,
        temperature=0.0,
    )
