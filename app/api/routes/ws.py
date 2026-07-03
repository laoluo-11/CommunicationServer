"""WebSocket 流式对话接口 — 实时逐 token 返回 LLM 回复。"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

from app.services.chat import ChatSessionManager
from app.services.llm import LLMService

logger = logging.getLogger("comm.api.ws")
router = APIRouter()


def _get_services(request_or_ws) -> tuple[ChatSessionManager, LLMService]:
    app = request_or_ws.app if hasattr(request_or_ws, "app") else request_or_ws
    return app.state.session_manager, app.state.llm_service


@router.websocket("/ws/chat/{robot_id}")
async def ws_chat(websocket: WebSocket, robot_id: str) -> None:
    """WebSocket 流式对话。

    客户端发送 JSON:
      {"type": "chat", "text": "你好"}
      {"type": "reset"}

    服务端推送 JSON:
      {"type": "token", "text": "你"}      # 流式 token
      {"type": "done", "text": "你好！"}     # 完整回复
      {"type": "error", "message": "..."}    # 错误
      {"type": "reset_ok"}                    # 重置确认
    """
    await websocket.accept()
    session_mgr, llm = _get_services(websocket)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            msg_type = msg.get("type", "")

            if msg_type == "chat":
                text = msg.get("text", "").strip()
                if not text:
                    await websocket.send_json({"type": "error", "message": "Empty text"})
                    continue

                # 添加用户消息
                session = session_mgr.get_or_create(robot_id)
                session.add_user_message(text)

                # 流式返回
                messages = session_mgr.get_history(robot_id)
                full_reply = ""
                try:
                    async for token in llm.chat_stream(messages):
                        full_reply += token
                        await websocket.send_json({"type": "token", "text": token})
                except Exception as e:
                    logger.exception("LLM stream failed for robot=%s", robot_id)
                    await websocket.send_json({"type": "error", "message": str(e)})
                    continue

                # 保存完整回复
                session.add_assistant_message(full_reply)
                await websocket.send_json({"type": "done", "text": full_reply})

            elif msg_type == "reset":
                session_mgr.reset(robot_id)
                await websocket.send_json({"type": "reset_ok"})

            else:
                await websocket.send_json({"type": "error", "message": f"Unknown type: {msg_type}"})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: robot=%s", robot_id)
    except Exception:
        logger.exception("WebSocket error for robot=%s", robot_id)
