"""WebSocket 客户端 — 连接 CommunicationServer，发送文字、接收流式回复。"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

import websockets

logger = logging.getLogger("robot.client")


class ChatClient:
    """与 CommunicationServer 的 WebSocket 对话客户端。"""

    def __init__(self, server_url: str, robot_id: str) -> None:
        self.server_url = server_url.rstrip("/") + f"/{robot_id}"
        self.robot_id = robot_id
        self._ws = None

    async def connect(self) -> None:
        self._ws = await websockets.connect(self.server_url)
        logger.info("Connected to %s", self.server_url)

    async def close(self) -> None:
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def send_text(self, text: str) -> str:
        """发送文字消息，返回 LLM 完整回复。"""
        if not self._ws:
            await self.connect()

        await self._ws.send(json.dumps({"type": "chat", "text": text}))
        full_text = ""

        async for raw in self._ws:
            msg = json.loads(raw)
            if msg["type"] == "token":
                full_text += msg["text"]
            elif msg["type"] == "done":
                return full_text or msg.get("text", "")
            elif msg["type"] == "error":
                logger.error("Server error: %s", msg.get("message", ""))
                return ""

        return full_text

    async def send_text_stream(self, text: str) -> AsyncIterator[str]:
        """发送文字消息，流式逐 token 返回。"""
        if not self._ws:
            await self.connect()

        await self._ws.send(json.dumps({"type": "chat", "text": text}))

        async for raw in self._ws:
            msg = json.loads(raw)
            if msg["type"] == "token":
                yield msg["text"]
            elif msg["type"] == "done":
                return
            elif msg["type"] == "error":
                logger.error("Server error: %s", msg.get("message", ""))
                return

    async def reset(self) -> None:
        if not self._ws:
            await self.connect()
        await self._ws.send(json.dumps({"type": "reset"}))


# Alias for friendlier name
RobotChatClient = ChatClient
