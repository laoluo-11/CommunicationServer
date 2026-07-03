#!/usr/bin/env python3
"""robot-agent 主程序 — 麦克风 → STT → WebSocket → LLM → TTS → 扬声器。

用法:
    python -m robot_agent.main
    python -m robot_agent.main --robot-id bumi-01 --server-url ws://localhost:8001/ws/chat
    python -m robot_agent.main --list-devices   # 列出音频设备

依赖:
    pip install pyaudio websockets openai edge-tts
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys

from robot_agent.audio import AudioRecorder
from robot_agent.client import ChatClient
from robot_agent.config import AudioConfig, RobotAgentConfig
from robot_agent.stt import build_stt
from robot_agent.tts import build_tts, play_audio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("robot.main")


class VoiceLoop:
    """语音对话主循环。"""

    def __init__(self, config: RobotAgentConfig) -> None:
        self.config = config
        self.recorder = AudioRecorder(config.audio)
        self.stt = build_stt(config.stt)
        self.tts = build_tts(config.tts)
        self.client = ChatClient(config.server_url, config.robot_id)
        self._running = False
        self._awaiting_response = False

    async def start(self) -> None:
        """启动语音对话循环。"""
        logger.info("Bumi 语音助手启动")
        logger.info("  机器人 ID: %s", self.config.robot_id)
        logger.info("  服务器: %s", self.config.server_url)
        logger.info("  STT: %s (%s)", self.config.stt.provider, self.config.stt.model)
        logger.info("  TTS: %s (%s)", self.config.tts.provider, self.config.tts.voice)
        logger.info("  采样率: %d Hz", self.config.audio.sample_rate)

        await self.client.connect()
        self._running = True

        try:
            print("\n🎤 Bumi 语音助手已就绪")
            print("   按 Enter 开始录音，说完后自动识别并回复")
            print("   输入 /reset 重置对话，/quit 退出\n")

            while self._running:
                await self._handle_input()

        finally:
            await self.client.close()
            self.recorder.close()

    async def _handle_input(self) -> None:
        """处理用户输入（控制台模式）。"""
        loop = asyncio.get_event_loop()

        # 在单独线程中等待输入
        user_input = await loop.run_in_executor(None, input, ">>> ")

        if not user_input:
            return
        if user_input.strip() == "/quit":
            self._running = False
            return
        if user_input.strip() == "/reset":
            await self.client.reset()
            print("🔄 对话已重置\n")
            return
        if user_input.strip() == "/devices":
            self._list_devices()
            return
        if user_input.strip() == "/listen":
            await self._listen_and_respond()
            return

        # 直接发送文字
        await self._process_text(user_input.strip())

    async def _listen_and_respond(self) -> None:
        """录音 → STT → 对话 → TTS → 播放。"""
        if self._awaiting_response:
            print("⚠️  正在等待回复，请稍后...")
            return

        print("🎤 正在录音... (说话后自动结束)")
        audio_data = self.recorder.record_with_vad(
            silence_timeout=self.config.silence_timeout,
            max_duration=15.0,
        )

        if not audio_data or len(audio_data) < 1600:  # < 0.1s
            print("🔇 未检测到语音")
            return

        print("📝 正在识别...")
        text = self.stt.transcribe(audio_data, self.config.audio.sample_rate)

        if not text:
            print("❌ 识别失败，请重试")
            return

        print(f"🗣  你说: {text}")
        await self._process_text(text)

    async def _process_text(self, text: str) -> None:
        """发送文字到 LLM，流式播放 TTS 回复。"""
        self._awaiting_response = True

        try:
            print("🤖 Bumi: ", end="", flush=True)

            # 收集文本 token，同时缓冲足够文本后开始 TTS
            full_text = ""
            buffer = ""
            min_chunk = 20  # 最小 TTS 字符数

            async for token in self.client.send_text_stream(text):
                full_text += token
                buffer += token
                print(token, end="", flush=True)

                # 当缓冲足够字符时开始 TTS（流式播放）
                # 这里积累完整文本后一次性 TTS，避免碎片化

            print()  # 换行

            if full_text:
                print("🔊 正在合成语音...")
                audio_data = await self.tts.synthesize(full_text)
                if audio_data:
                    await play_audio(audio_data)
                    print("✅ 播放完成\n")

        except Exception as e:
            logger.exception("对话处理失败")
            print(f"\n❌ 错误: {e}")

        finally:
            self._awaiting_response = False

    def _list_devices(self) -> None:
        """列出音频设备。"""
        devices = self.recorder.list_devices()
        print(f"\n音频设备 ({len(devices)} 个):")
        for d in devices:
            io = []
            if d["inputs"] > 0:
                io.append(f"INx{d['inputs']}")
            if d["outputs"] > 0:
                io.append(f"OUTx{d['outputs']}")
            print(f"  [{d['index']}] {d['name'][:50]}  {','.join(io)}  {d['default_sample_rate']}Hz")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Bumi 语音助手")
    parser.add_argument("--robot-id", type=str, help="机器人 ID")
    parser.add_argument("--server-url", type=str, help="CommunicationServer WebSocket URL")
    parser.add_argument("--list-devices", action="store_true", help="列出音频设备并退出")
    parser.add_argument("--sample-rate", type=int, default=16000, help="采样率 (默认 16000)")
    parser.add_argument("--device-index", type=int, help="音频输入设备索引")
    parser.add_argument("--stt-provider", type=str, default="openai", help="STT 提供商")
    parser.add_argument("--tts-provider", type=str, default="edge", help="TTS 提供商")
    args = parser.parse_args()

    config = RobotAgentConfig.from_env()

    # 命令行覆盖
    if args.robot_id:
        config.robot_id = args.robot_id
    if args.server_url:
        config.server_url = args.server_url
    if args.stt_provider:
        config.stt.provider = args.stt_provider
    if args.tts_provider:
        config.tts.provider = args.tts_provider
    if args.device_index is not None:
        config.audio.device_index = args.device_index
    config.audio.sample_rate = args.sample_rate

    if args.list_devices:
        recorder = AudioRecorder(config.audio)
        devices = recorder.list_devices()
        print(f"音频设备 ({len(devices)} 个):")
        for d in devices:
            io = []
            if d["inputs"] > 0:
                io.append(f"INx{d['inputs']}")
            if d["outputs"] > 0:
                io.append(f"OUTx{d['outputs']}")
            print(f"  [{d['index']}] {d['name'][:60]}  {','.join(io)}  {d['default_sample_rate']}Hz")
        recorder.close()
        return

    loop = VoiceLoop(config)
    try:
        asyncio.run(loop.start())
    except KeyboardInterrupt:
        print("\n👋 再见!")


if __name__ == "__main__":
    main()
