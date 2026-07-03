from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import get_settings
from app.api.schemas import HealthResponse

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version="0.1.0")
