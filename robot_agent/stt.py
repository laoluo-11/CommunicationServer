"""语音识别 (STT) — 将音频数据转为文字。"""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Protocol

from openai import OpenAI

from robot_agent.config import STTConfig

logger = logging.getLogger("robot.stt")


class STTEngine(Protocol):
    def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        ...


class OpenAIWhisperSTT:
    """使用 OpenAI Whisper API 进行语音识别。"""

    def __init__(self, config: STTConfig, api_key: str = "") -> None:
        self.config = config
        api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL", "")
        client_kw = {"api_key": api_key}
        if base_url:
            client_kw["base_url"] = base_url
        self._client = OpenAI(**client_kw)

    def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        """将 WAV 音频数据转为文字。"""
        import io
        audio_file = io.BytesIO(audio_data)
        audio_file.name = "audio.wav"

        try:
            result = self._client.audio.transcriptions.create(
                model=self.config.model,
                file=audio_file,
                language=self.config.language,
            )
            text = result.text.strip()
            logger.info("STT result: %s", text)
            return text
        except Exception as e:
            logger.exception("STT failed")
            return ""


def build_stt(config: STTConfig) -> STTEngine:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if config.provider == "openai":
        return OpenAIWhisperSTT(config, api_key)
    raise ValueError(f"Unknown STT provider: {config.provider}")
