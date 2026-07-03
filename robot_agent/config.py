"""robot-agent 配置 — 音频、STT、TTS、服务器连接。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    record_seconds: int = 5         # 每次录音时长
    silence_threshold: int = 500    # 低于此值视为静音（用于 VAD）
    device_index: int | None = None # 音频设备索引，None 为默认


@dataclass
class STTConfig:
    provider: str = "openai"        # openai | azure | local
    model: str = "whisper-1"
    language: str = "zh"            # 语音语言

    @classmethod
    def from_env(cls) -> "STTConfig":
        return cls(
            provider=os.getenv("STT_PROVIDER", "openai"),
            model=os.getenv("STT_MODEL", "whisper-1"),
            language=os.getenv("STT_LANGUAGE", "zh"),
        )


@dataclass
class TTSConfig:
    provider: str = "edge"          # edge (免费) | openai | azure
    voice: str = "zh-CN-XiaoxiaoNeural"  # edge-tts 中文女声
    speed: str = "+0%"              # 语速

    @classmethod
    def from_env(cls) -> "TTSConfig":
        return cls(
            provider=os.getenv("TTS_PROVIDER", "edge"),
            voice=os.getenv("TTS_VOICE", "zh-CN-XiaoxiaoNeural"),
            speed=os.getenv("TTS_SPEED", "+0%"),
        )


@dataclass
class RobotAgentConfig:
    robot_id: str = "bumi-01"
    server_url: str = "ws://localhost:8001/ws/chat"
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    # 唤醒词（可选）
    wake_word: str = ""
    # 静音超时秒数，超时自动结束录音
    silence_timeout: float = 1.5

    @classmethod
    def from_env(cls) -> "RobotAgentConfig":
        return cls(
            robot_id=os.getenv("ROBOT_ID", "bumi-01"),
            server_url=os.getenv("COMM_SERVER_URL", "ws://localhost:8001/ws/chat"),
            stt=STTConfig.from_env(),
            tts=TTSConfig.from_env(),
            wake_word=os.getenv("WAKE_WORD", ""),
        )
