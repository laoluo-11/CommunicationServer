"""REST 对话接口 — POST /chat，返回完整 LLM 回复。"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from app.api.schemas import ChatData, ChatRequest, ChatResponse, ResetData, ResetRequest, ResetResponse
from app.services.chat import ChatSessionManager
from app.services.llm import LLMService

logger = logging.getLogger("comm.api.chat")
router = APIRouter(prefix="/chat", tags=["chat"])


def _get_services(request: Request) -> tuple[ChatSessionManager, LLMService]:
    return request.app.state.session_manager, request.app.state.llm_service


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    """发送消息给机器人，获取 LLM 智能回复。"""
    session_mgr, llm = _get_services(request)

    # 1. 获取或创建会话，添加用户消息
    session = session_mgr.get_or_create(req.robot_id)
    session.add_user_message(req.text)

    # 2. 获取完整对话历史
    messages = session_mgr.get_history(req.robot_id)

    # 3. 调用 LLM
    try:
        reply = await llm.chat(messages)
    except Exception as e:
        logger.exception("LLM chat failed for robot=%s", req.robot_id)
        return ChatResponse(code=-1, message=f"LLM error: {e}")

    # 4. 保存回复
    session.add_assistant_message(reply)

    return ChatResponse(
        data=ChatData(robot_id=req.robot_id, reply=reply, model=llm.settings.model),
    )


@router.post("/reset", response_model=ResetResponse)
async def reset(req: ResetRequest, request: Request) -> ResetResponse:
    """重置某个机器人的对话历史。"""
    session_mgr, _ = _get_services(request)
    session_mgr.reset(req.robot_id)
    logger.info("Session reset for robot=%s", req.robot_id)
    return ResetResponse(data=ResetData(robot_id=req.robot_id))
