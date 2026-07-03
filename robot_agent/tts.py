"""语音合成 (TTS) — 将文字转为音频并播放。"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import tempfile
from typing import Protocol

from robot_agent.config import TTSConfig

logger = logging.getLogger("robot.tts")


class TTSEngine(Protocol):
    async def synthesize(self, text: str) -> bytes:
        """将文字转为 WAV 音频字节，返回音频数据。"""
        ...


class EdgeTTS:
    """使用 Microsoft Edge TTS（免费，高质量中文语音）。"""

    def __init__(self, config: TTSConfig) -> None:
        self.config = config

    async def synthesize(self, text: str) -> bytes:
        try:
            import edge_tts
        except ImportError:
            logger.error("edge-tts not installed. Run: pip install edge-tts")
            return b""

        communicate = edge_tts.Communicate(
            text=text,
            voice=self.config.voice,
            rate=self.config.speed,
        )

        # Collect all audio chunks
        audio_chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])

        if not audio_chunks:
            logger.warning("TTS produced no audio")
            return b""

        # edge-tts outputs MP3 chunks, join them
        return b"".join(audio_chunks)


class OpenAITTS:
    """使用 OpenAI TTS API。"""

    def __init__(self, config: TTSConfig) -> None:
        self.config = config
        self._api_key = os.environ.get("OPENAI_API_KEY", "")

    async def synthesize(self, text: str) -> bytes:
        import httpx
        url = "https://api.openai.com/v1/audio/speech"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": "tts-1",
            "input": text,
            "voice": "nova",  # or alloy, echo, fable, onyx, nova, shimmer
            "response_format": "wav",
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            return resp.content


async def _play_audio(audio_data: bytes) -> None:
    """使用系统命令播放音频（跨平台）。"""
    # 写入临时文件
    suffix = ".mp3"  # edge-tts 输出 MP3
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_data)
        tmp_path = f.name

    try:
        # 尝试多种播放方式
        import platform
        system = platform.system()

        if system == "Linux":
            # 优先用 ffplay（如果可用），其次 paplay, aplay
            for cmd in (
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", tmp_path],
                ["paplay", tmp_path],
                ["aplay", tmp_path],
            ):
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                try:
                    await asyncio.wait_for(proc.wait(), timeout=30)
                    break
                except asyncio.TimeoutError:
                    proc.kill()
        elif system == "Darwin":
            proc = await asyncio.create_subprocess_exec("afplay", tmp_path)
            await proc.wait()
        elif system == "Windows":
            import winsound
            # Windows 需要 WAV 格式
            pass
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def build_tts(config: TTSConfig) -> TTSEngine:
    if config.provider == "edge":
        return EdgeTTS(config)
    elif config.provider == "openai":
        return OpenAITTS(config)
    raise ValueError(f"Unknown TTS provider: {config.provider}")


# 暴露播放函数
play_audio = _play_audio
