"""对话会话管理 — 维护每个机器人的对话历史。"""

from __future__ import annotations

import time
from collections import defaultdict


class ChatSession:
    __slots__ = ("robot_id", "system_prompt", "messages", "created_at", "last_active")

    def __init__(self, robot_id: str, system_prompt: str) -> None:
        self.robot_id = robot_id
        self.system_prompt = system_prompt
        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
        self.created_at = time.time()
        self.last_active = time.time()

    def add_user_message(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})
        self.last_active = time.time()

    def add_assistant_message(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})
        self.last_active = time.time()

    def trim(self, max_turns: int) -> None:
        """保留最近 N 轮对话（system prompt 始终保留）。"""
        # messages[0] 是 system prompt，之后每 2 条是一轮 (user + assistant)
        if max_turns <= 0:
            return
        max_messages = 1 + max_turns * 2  # system + N 轮
        if len(self.messages) > max_messages:
            # 保留 system + 最近 N 轮
            self.messages = self.messages[:1] + self.messages[-(max_turns * 2):]

    def reset(self) -> None:
        self.messages = self.messages[:1]  # 只保留 system prompt


class ChatSessionManager:
    """管理所有机器人的对话会话。"""

    def __init__(self, system_prompt: str, max_history_turns: int = 20) -> None:
        self.system_prompt = system_prompt
        self.max_history_turns = max_history_turns
        self._sessions: dict[str, ChatSession] = defaultdict(
            lambda: ChatSession("", system_prompt)
        )

    def get_or_create(self, robot_id: str) -> ChatSession:
        session = self._sessions[robot_id]
        # 修复 defaultdict 导致的空 robot_id
        if not session.robot_id:
            session.robot_id = robot_id
        return session

    def reset(self, robot_id: str) -> None:
        if robot_id in self._sessions:
            self._sessions[robot_id].reset()

    def get_history(self, robot_id: str) -> list[dict[str, str]]:
        session = self.get_or_create(robot_id)
        session.trim(self.max_history_turns)
        return list(session.messages)

    def add_turn(self, robot_id: str, user_text: str, assistant_text: str) -> None:
        session = self.get_or_create(robot_id)
        session.add_user_message(user_text)
        session.add_assistant_message(assistant_text)
        session.trim(self.max_history_turns)
