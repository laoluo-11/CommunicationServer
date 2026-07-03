from __future__ import annotations

from pydantic import BaseModel, Field


class ApiResponse(BaseModel):
    code: int = Field(default=0, description="Status code.")
    message: str = Field(default="ok", description="Human-readable message.")


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = ""


class ChatRequest(BaseModel):
    robot_id: str = Field(default="default", description="机器人 ID")
    text: str = Field(..., min_length=1, description="用户语音转文字后的文本")


class ChatData(BaseModel):
    robot_id: str
    reply: str
    model: str = ""


class ChatResponse(ApiResponse):
    data: ChatData | None = None


class ResetRequest(BaseModel):
    robot_id: str = Field(default="default", description="机器人 ID")


class ResetData(BaseModel):
    robot_id: str


class ResetResponse(ApiResponse):
    data: ResetData | None = None
