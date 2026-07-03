from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat, system, ws
from app.core.config import Settings
from app.services.chat import ChatSessionManager
from app.services.llm import LLMService

settings = Settings.from_env()
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Communication Server — 机器人与云端大模型之间的实时语音对话中继服务。",
    openapi_tags=[
        {"name": "system", "description": "服务状态与健康检查。"},
        {"name": "chat", "description": "REST 对话接口 — 发送文本获取 LLM 回复。"},
        {"name": "ws", "description": "WebSocket 连接 — 实时流式对话。"},
    ],
)
app.state.settings = settings
app.state.llm_service = LLMService(settings.llm)
app.state.session_manager = ChatSessionManager(
    settings.system_prompt, settings.max_history_turns
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system.router)
app.include_router(chat.router)
app.include_router(ws.router)
