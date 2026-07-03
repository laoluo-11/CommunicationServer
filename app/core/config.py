from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(slots=True)
class LLMSettings:
    """LLM / 云端大模型配置。兼容 OpenAI API 及其兼容服务（vLLM / Ollama 等）。"""
    api_key: str = ""
    base_url: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 1024

    @classmethod
    def from_env(cls, defaults: "LLMSettings | None" = None) -> "LLMSettings":
        if defaults is None:
            defaults = cls()
        return cls(
            api_key=os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", defaults.api_key)),
            base_url=os.getenv("LLM_BASE_URL", defaults.base_url),
            model=os.getenv("LLM_MODEL", defaults.model),
            temperature=float(os.getenv("LLM_TEMPERATURE", str(defaults.temperature))),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", str(defaults.max_tokens))),
        )


@dataclass(slots=True)
class Settings:
    app_name: str = "Communication Server"
    app_version: str = "0.1.0"
    host: str = "0.0.0.0"
    port: int = 8001
    debug: bool = False
    llm: LLMSettings = field(default_factory=LLMSettings)
    # 对话历史最大保留轮数（每轮 = user + assistant 各一条）
    max_history_turns: int = 20
    # 机器人角色设定
    system_prompt: str = (
        "你是一个名叫 Bumi 的人形机器人助手，性格友好、乐于助人。"
        "你的回答应该简洁、自然，像人类对话一样。"
        "用中文回答问题，除非用户用其他语言。"
        "回答控制在 2-3 句话以内，不要过于冗长。"
    )

    @classmethod
    def from_env(cls) -> "Settings":
        defaults = cls()
        return cls(
            app_name=os.getenv("COMM_APP_NAME", defaults.app_name),
            app_version=os.getenv("COMM_APP_VERSION", defaults.app_version),
            host=os.getenv("COMM_HOST", defaults.host),
            port=int(os.getenv("COMM_PORT", str(defaults.port))),
            debug=os.getenv("COMM_DEBUG", str(defaults.debug)).lower() in ("1", "true", "yes"),
            llm=LLMSettings.from_env(defaults.llm),
            max_history_turns=int(os.getenv("COMM_MAX_HISTORY_TURNS", str(defaults.max_history_turns))),
            system_prompt=os.getenv("COMM_SYSTEM_PROMPT", defaults.system_prompt),
        )
