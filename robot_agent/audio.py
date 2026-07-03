"""音频采集 — 从麦克风录音，返回 WAV 数据。"""

from __future__ import annotations

import logging
import struct
import wave
from io import BytesIO

from robot_agent.config import AudioConfig

logger = logging.getLogger("robot.audio")


class AudioRecorder:
    """麦克风录音器。按空格键触发录音（类似对讲机），或持续监听。"""

    def __init__(self, config: AudioConfig) -> None:
        self.config = config
        self._pyaudio = None
        self._stream = None

    def _ensure_pyaudio(self):
        if self._pyaudio is None:
            import pyaudio
            self._pyaudio = pyaudio.PyAudio()

    def record(self, duration: float | None = None) -> bytes:
        """录制一段音频，返回 WAV 格式的字节数据。

        Args:
            duration: 录音秒数，None 则使用配置默认值。
        """
        if duration is None:
            duration = self.config.record_seconds

        self._ensure_pyaudio()
        rate = self.config.sample_rate
        channels = self.config.channels
        chunk = self.config.chunk_size

        stream = self._pyaudio.open(
            format=self._pyaudio.paInt16,
            channels=channels,
            rate=rate,
            input=True,
            frames_per_buffer=chunk,
        )

        logger.info("Recording for %.1f seconds...", duration)
        frames = []
        total_chunks = int(rate / chunk * duration)
        for _ in range(total_chunks):
            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(data)

        stream.stop_stream()
        stream.close()

        logger.info("Recording done, %d frames", len(frames))

        # Encode as WAV
        buf = BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(self._pyaudio.get_sample_size(self._pyaudio.paInt16))
            wf.setframerate(rate)
            wf.writeframes(b"".join(frames))

        return buf.getvalue()

    def record_with_vad(
        self,
        silence_timeout: float = 1.5,
        max_duration: float = 15.0,
    ) -> bytes:
        """带 VAD（语音活动检测）的录音。检测到说话开始录制，静音超时后停止。

        Args:
            silence_timeout: 静音多少秒后结束录音。
            max_duration: 最长录音秒数。
        """
        self._ensure_pyaudio()
        rate = self.config.sample_rate
        channels = self.config.channels
        chunk = self.config.chunk_size
        threshold = self.config.silence_threshold

        stream = self._pyaudio.open(
            format=self._pyaudio.paInt16,
            channels=channels,
            rate=rate,
            input=True,
            frames_per_buffer=chunk,
        )

        frames = []
        silent_chunks = 0
        speaking = False
        chunks_per_second = rate / chunk
        max_chunks = int(max_duration * chunks_per_second)

        logger.info("Listening (VAD)...")

        for i in range(max_chunks):
            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(data)

            # RMS 音量检测
            rms = _rms(data)
            is_silent = rms < threshold

            if not speaking and not is_silent:
                speaking = True
                logger.info("Speech detected (RMS=%.0f)", rms)

            if speaking:
                if is_silent:
                    silent_chunks += 1
                    if silent_chunks >= int(silence_timeout * chunks_per_second):
                        logger.info("Silence timeout, stopping")
                        break
                else:
                    silent_chunks = 0

        stream.stop_stream()
        stream.close()

        if not speaking or len(frames) < int(0.5 * chunks_per_second):
            logger.info("No speech detected or too short")
            return b""

        logger.info("Recording done: %.1fs, %d frames", len(frames) / chunks_per_second, len(frames))

        # Encode as WAV
        buf = BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(self._pyaudio.get_sample_size(self._pyaudio.paInt16))
            wf.setframerate(rate)
            wf.writeframes(b"".join(frames))

        return buf.getvalue()

    def play(self, audio_data: bytes) -> None:
        """播放 PCM/WAV 音频数据。"""
        self._ensure_pyaudio()

        # Parse WAV header
        buf = BytesIO(audio_data)
        with wave.open(buf, "rb") as wf:
            rate = wf.getframerate()
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            raw_data = wf.readframes(wf.getnframes())

        format_map = {1: self._pyaudio.paInt8, 2: self._pyaudio.paInt16, 4: self._pyaudio.paInt32}
        pyaudio_format = format_map.get(width, self._pyaudio.paInt16)

        stream = self._pyaudio.open(
            format=pyaudio_format,
            channels=channels,
            rate=rate,
            output=True,
        )
        stream.write(raw_data)
        stream.stop_stream()
        stream.close()

    def list_devices(self) -> list[dict]:
        """列出所有音频设备。"""
        self._ensure_pyaudio()
        devices = []
        for i in range(self._pyaudio.get_device_count()):
            info = self._pyaudio.get_device_info_by_index(i)
            devices.append({
                "index": i,
                "name": info["name"],
                "inputs": info["maxInputChannels"],
                "outputs": info["maxOutputChannels"],
                "default_sample_rate": int(info["defaultSampleRate"]),
            })
        return devices

    def close(self) -> None:
        if self._pyaudio:
            self._pyaudio.terminate()
            self._pyaudio = None


def _rms(data: bytes) -> float:
    """计算音频数据的 RMS（均方根）值。"""
    count = len(data) // 2
    fmt = f"{count}h"
    samples = struct.unpack(fmt, data)
    sum_squares = sum(s * s for s in samples)
    return (sum_squares / count) ** 0.5 if count > 0 else 0
