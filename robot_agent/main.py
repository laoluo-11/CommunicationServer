#!/usr/bin/env python3
"""robot-agent — Bumi 语音助手（全自动语音对话模式）。

持续监听麦克风，检测到语音自动识别 → 发送 LLM → 流式 TTS 播放。
支持打断（barge-in）：在机器人说话时再次开口即可打断。

用法:
    python -m robot_agent.main
    python -m robot_agent.main --mode continuous   # 全自动模式（默认）
    python -m robot_agent.main --mode interactive  # 交互模式（键盘控制）
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from robot_agent.audio import AudioRecorder
from robot_agent.client import ChatClient
from robot_agent.config import RobotAgentConfig
from robot_agent.stt import build_stt
from robot_agent.tts import build_tts, play_audio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("robot.main")


# ─────────────────────────────────────────────────────
# 流式 TTS — 边接收 LLM token 边合成播放
# ─────────────────────────────────────────────────────

class StreamingSpeaker:
    """流式语音播放器：缓冲文本 token，攒够一定量就合成播放。"""

    def __init__(self, tts_engine, min_chars: int = 15, max_concurrent: int = 2):
        self.tts = tts_engine
        self.min_chars = min_chars  # 最少积累多少字才开始合成
        self._buffer = ""
        self._tasks: list[asyncio.Task] = []
        self._stop = asyncio.Event()
        self._sentence_endings = {"。", "！", "？", ".", "!", "?", "
"}

    async def feed(self, token: str) -> None:
        """喂入一个 token，自动攒句播放。"""
        self._buffer += token

        # 遇到句末标点，或 buffer 够长 → 合成播放
        if self._should_flush():
            await self._flush()

    async def finish(self) -> None:
        """剩余内容全部播放。"""
        if self._buffer.strip():
            await self._flush(force=True)

        # 等待所有播放任务完成
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    def cancel(self) -> None:
        """打断当前播放。"""
        self._stop.set()
        self._buffer = ""
        for t in self._tasks:
            if not t.done():
                t.cancel()

    def reset(self) -> None:
        self._stop.clear()
        self._buffer = ""

    async def feed_and_speak(self, tokens: AsyncIterator[str]) -> bool:
        """流式接收 tokens 并播放。返回 True 表示完整播放，False 表示被打断。"""
        self.reset()

        try:
            async for token in tokens:
                if self._stop.is_set():
                    return False
                await self.feed(token)
            await self.finish()
            return not self._stop.is_set()
        except asyncio.CancelledError:
            return False

    def _should_flush(self) -> bool:
        if len(self._buffer) < self.min_chars:
            return False
        # 遇到句末标点
        if self._buffer[-1] in self._sentence_endings:
            return True
        # buffer 超过两倍 min_chars 强制 flush
        if len(self._buffer) >= self.min_chars * 3:
            return True
        return False

    async def _flush(self, force: bool = False) -> None:
        if not self._buffer.strip():
            return
        text = self._buffer.strip()
        if not force and not any(text.rstrip().endswith(e) for e in self._sentence_endings):
            text = text

        self._buffer = ""

        # 单独 task 执行 TTS 合成（不阻塞 token 接收）
        task = asyncio.create_task(self._speak_chunk(text))
        self._tasks = [t for t in self._tasks if not t.done()]
        self._tasks.append(task)

    async def _speak_chunk(self, text: str) -> None:
        try:
            audio = await self.tts.synthesize(text)
            if audio and not self._stop.is_set():
                await play_audio(audio)
        except Exception as e:
            logger.error("TTS chunk failed: %s", e)


# ─────────────────────────────────────────────────────
# 全自动语音对话循环
# ─────────────────────────────────────────────────────

class VoiceLoop:
    """全自动语音对话引擎。"""

    def __init__(self, config: RobotAgentConfig) -> None:
        self.config = config
        self.recorder = AudioRecorder(config.audio)
        self.stt = build_stt(config.stt)
        self.tts = build_tts(config.tts)
        self.speaker = StreamingSpeaker(self.tts)
        self.client = ChatClient(config.server_url, config.robot_id)
        self._running = False

    async def run_continuous(self) -> None:
        """持续监听模式 — 自动检测语音并回复。"""
        logger.info("🚀 Bumi 全自动语音对话模式启动")
        logger.info("  机器人: %s | 服务器: %s", self.config.robot_id, self.config.server_url)
        logger.info("  STT: %s | TTS: %s", self.config.stt.provider, self.config.tts.provider)
        logger.info("  💡 直接对机器人说话，说完自动回复。说话可打断机器人。")
        logger.info("  Ctrl+C 退出
")

        await self.client.connect()
        self._running = True

        # 先播放一个就绪提示音（可选，收到第一条消息就知道已启动）
        try:
            ready_audio = await self.tts.synthesize("Bumi 已就绪")
            if ready_audio:
                await play_audio(ready_audio)
        except Exception:
            pass

        while self._running:
            try:
                await self._one_turn()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("对话轮次异常，继续监听...")
                await asyncio.sleep(0.5)

    async def _one_turn(self) -> None:
        """一轮对话：听 → 想 → 说。"""

        # 1. 听 — VAD 录音
        logger.info("🎤 等待语音...")
        audio_data = self.recorder.record_with_vad(
            silence_timeout=self.config.silence_timeout,
            max_duration=15.0,
        )

        if not audio_data or len(audio_data) < 3200:  # < 0.2s
            return

        # 2. 识别 — STT
        text = self.stt.transcribe(audio_data, self.config.audio.sample_rate)
        if not text:
            logger.info("🔇 识别为空")
            return
        logger.info("🗣  用户: %s", text)

        # 3. 想 — 发送 LLM，流式接收
        logger.info("🤔 思考中...")
        token_stream = self.client.send_text_stream(text)

        # 4. 说 — 流式 TTS 播放（同时检测打断）
        completed = await self.speaker.feed_and_speak(token_stream)

        if not completed:
            logger.info("⏹  被用户打断")
        else:
            logger.info("✅ 回复完成")

    async def run_interactive(self) -> None:
        """交互模式 — 键盘控制（调试用）。"""
        logger.info("Bumi 语音助手（交互模式）")
        await self.client.connect()
        self._running = True

        print("
🎤 命令: Enter=输入文字  /listen=录音  /reset=重置  /quit=退出
")

        loop = asyncio.get_event_loop()
        while self._running:
            try:
                user_input = await loop.run_in_executor(None, input, ">>> ")
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue

            cmd = user_input.strip()

            if cmd == "/quit":
                self._running = False
            elif cmd == "/reset":
                await self.client.reset()
                print("🔄 对话已重置
")
            elif cmd == "/listen":
                audio_data = self.recorder.record_with_vad(
                    silence_timeout=self.config.silence_timeout,
                )
                if audio_data and len(audio_data) >= 3200:
                    text = self.stt.transcribe(audio_data, self.config.audio.sample_rate)
                    if text:
                        print(f"🗣  你说: {text}")
                        await self._speak_response(text)
                    else:
                        print("🔇 识别为空")
                else:
                    print("🔇 未检测到语音")
            elif cmd:
                await self._speak_response(cmd)

        await self.client.close()
        self.recorder.close()

    async def _speak_response(self, text: str) -> None:
        """发送文字，流式获取并播放回复。"""
        print("🤖 Bumi: ", end="", flush=True)
        token_stream = self.client.send_text_stream(text)
        await self.speaker.feed_and_speak(token_stream)
        print()

    async def shutdown(self) -> None:
        self._running = False
        self.speaker.cancel()
        await self.client.close()
        self.recorder.close()


# ─────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Bumi 语音助手")
    parser.add_argument("--mode", choices=["continuous", "interactive"], default="continuous",
                        help="对话模式: continuous=全自动语音 / interactive=键盘控制")
    parser.add_argument("--robot-id", type=str, help="机器人 ID")
    parser.add_argument("--server-url", type=str, help="CommunicationServer URL")
    parser.add_argument("--list-devices", action="store_true", help="列出音频设备并退出")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--device-index", type=int, help="音频输入设备索引")
    parser.add_argument("--stt-provider", type=str, default="openai")
    parser.add_argument("--tts-provider", type=str, default="edge")
    args = parser.parse_args()

    config = RobotAgentConfig.from_env()
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

    voice_loop = VoiceLoop(config)

    async def run():
        try:
            if args.mode == "interactive":
                await voice_loop.run_interactive()
            else:
                await voice_loop.run_continuous()
        except KeyboardInterrupt:
            print("
👋 再见!")
        finally:
            await voice_loop.shutdown()

    asyncio.run(run())


if __name__ == "__main__":
    main()
