"""LLM 对话服务 — 调用云端大模型进行智能对话。"""

from __future__ import annotations

import logging
from typing import AsyncIterator

from openai import AsyncOpenAI

from app.core.config import LLMSettings

logger = logging.getLogger("comm.llm")


class LLMService:
    """OpenAI-compatible LLM 对话服务。"""

    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            kwargs = {"api_key": self.settings.api_key}
            if self.settings.base_url:
                kwargs["base_url"] = self.settings.base_url
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def chat(self, messages: list[dict[str, str]]) -> str:
        """发送对话到 LLM，返回完整回复文本。"""
        response = await self.client.chat.completions.create(
            model=self.settings.model,
            messages=messages,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
        )
        return response.choices[0].message.content or ""

    async def chat_stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """流式对话，逐 token 返回文本。"""
        stream = await self.client.chat.completions.create(
            model=self.settings.model,
            messages=messages,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
