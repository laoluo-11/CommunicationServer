"""
DDS Bridge — 连接运控板 DDS 和 CommunicationServer 的桥接器。

数据流:
  运控板 DDS internal_capture (audio/video) → DDSBridge → WebSocket → CommServer
  CommServer → LLM response → TTS → DDS external_playback → 运控板 speaker

用法:
    python -m robot_agent.dds_bridge
    python -m robot_agent.dds_bridge --server-url ws://localhost:8001/ws/chat --robot-id bumi-01
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import struct
import sys
import threading
import time
from collections import deque
from io import BytesIO
from typing import Optional

import websockets

from robot_agent.config import AudioConfig, TTSConfig
from robot_agent.tts import build_tts, play_audio
from robot_agent.voice_ctrl import DDSVoiceController, HAS_PYVOICECTRL, _PYVOICECTRL_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("robot.bridge")


# ─────────────────────────────────────────────────────
# CommServer WebSocket 客户端（支持 image）
# ─────────────────────────────────────────────────────

class CommServerClient:
    """连接 CommunicationServer，发送文本/图像，接收流式 LLM 回复。"""

    def __init__(self, server_url: str, robot_id: str):
        self.url = server_url.rstrip("/") + f"/{robot_id}"
        self.robot_id = robot_id
        self._ws = None

    async def connect(self):
        self._ws = await websockets.connect(self.url)
        logger.info("已连接 CommunicationServer: %s", self.url)

    async def close(self):
        if self._ws:
            await self._ws.close()

    async def send_text(self, text: str) -> str:
        """发送文字，返回完整回复。"""
        await self._ensure_connected()
        await self._ws.send(json.dumps({"type": "chat", "text": text}, ensure_ascii=False))
        full = ""
        async for raw in self._ws:
            msg = json.loads(raw)
            if msg["type"] == "token":
                full += msg["text"]
            elif msg["type"] == "done":
                return full or msg.get("text", "")
            elif msg["type"] == "error":
                logger.error("CommServer error: %s", msg.get("message", ""))
                return ""
        return full

    async def send_text_stream(self, text: str):
        """发送文字，流式 yield token。"""
        await self._ensure_connected()
        await self._ws.send(json.dumps({"type": "chat", "text": text}, ensure_ascii=False))
        async for raw in self._ws:
            msg = json.loads(raw)
            if msg["type"] == "token":
                yield msg["text"]
            elif msg["type"] == "done":
                return
            elif msg["type"] == "error":
                logger.error("CommServer error: %s", msg.get("message", ""))
                return

    async def send_with_image(self, text: str, image_base64: str) -> str:
        """发送文字 + 图像，返回完整回复。"""
        await self._ensure_connected()
        await self._ws.send(json.dumps({
            "type": "chat_with_image",
            "text": text,
            "image_base64": image_base64,
        }, ensure_ascii=False))
        full = ""
        async for raw in self._ws:
            msg = json.loads(raw)
            if msg["type"] == "token":
                full += msg["text"]
            elif msg["type"] == "done":
                return full or msg.get("text", "")
            elif msg["type"] == "error":
                logger.error("CommServer error: %s", msg.get("message", ""))
                return ""
        return full

    async def reset(self):
        await self._ensure_connected()
        await self._ws.send(json.dumps({"type": "reset"}, ensure_ascii=False))
        # consume reset_ok
        raw = await self._ws.recv()
        logger.info("对话已重置")

    async def _ensure_connected(self):
        if self._ws is None or self._ws.closed:
            await self.connect()


# ─────────────────────────────────────────────────────
# DDS Bridge — 主桥接逻辑
# ─────────────────────────────────────────────────────

class DDSBridge:
    """DDS ↔ CommunicationServer 双向桥接。"""

    def __init__(
        self,
        dds_xml_path: str = "",
        server_url: str = "ws://localhost:8001/ws/chat",
        robot_id: str = "bumi-01",
        tts_provider: str = "edge",
        enable_video: bool = True,
        max_video_fps: float = 1.0,  # 每秒最多发几帧给 LLM
    ):
        self.dds_xml_path = dds_xml_path
        self.robot_id = robot_id
        self.enable_video = enable_video
        self.max_video_fps = max_video_fps

        # DDS 控制器
        self.voice = DDSVoiceController(dds_xml_path=dds_xml_path, auto_wakeup=True)

        # CommServer 客户端
        self.comm = CommServerClient(server_url, robot_id)

        # TTS
        self.tts = build_tts(TTSConfig(provider=tts_provider))

        # 状态
        self._running = False
        self._latest_subtitle = ""
        self._subtitle_lock = threading.Lock()
        self._subtitle_event = threading.Event()

        # 视频缓冲
        self._last_frame_time = 0
        self._last_frame_data: Optional[bytes] = None
        self._last_frame_lock = threading.Lock()
        self._frame_width = 640
        self._frame_height = 480

        # 音频播放队列（避免阻塞 DDS 回调）
        self._playback_queue: deque = deque()
        self._playback_lock = threading.Lock()

    # ── 初始化 ───────────────────────────────────

    async def start(self):
        if not HAS_PYVOICECTRL:
            logger.error("pyVoiceCtrl 不可用")
            return

        # 1. 初始化 DDS
        if not self.voice.init(load_configs=True):
            logger.error("DDS 初始化失败")
            return

        # 2. 注册回调
        self.voice._set_audio_callback()  # 音频流回调
        self.voice._ctrl.set_audio_stream_callback(self._on_audio_stream)
        self.voice._ctrl.set_video_stream_callback(self._on_video_stream)
        self.voice.on_subtitle(self._on_subtitle)
        self.voice.on_command(self._on_command)

        # 3. 连接 CommServer
        await self.comm.connect()

        # 4. 唤醒运控板（启动内部采集）
        logger.info("唤醒运控板...")
        self.voice.wakeup(timeout_s=15)

        self._running = True
        logger.info("DDS Bridge 启动完成")
        logger.info("  运控板 → DDS → CommServer → TTS → DDS → 运控板")

    # ── DDS 回调 ───────────────────────────────────

    def _on_subtitle(self, text: str, definite: bool):
        """ASR 识别结果回调"""
        marker = "✓" if definite else "…"
        logger.info("ASR [%s]: %s", marker, text)
        with self._subtitle_lock:
            self._latest_subtitle = text
        self._subtitle_event.set()

    def _on_command(self, cmd_json: str):
        logger.info("指令: %s", cmd_json)

    def _on_audio_stream(self, data, channels, sample_rate):
        """运控板内部采集的音频流回调（原始 PCM int16 list）。

        data: list[int16] — need struct packing
        channels: int
        sample_rate: int
        """
        if not data:
            return
        # 转成 bytes 备用（暂不处理，ASR 由运控板做）
        pass

    def _on_video_stream(self, data, width, height, fmt):
        """运控板内部采集的视频流回调。

        data: list[uint8] — raw YUYV frame
        width, height: int
        fmt: int (pixel format)
        """
        if not self.enable_video:
            return

        now = time.time()
        if now - self._last_frame_time < 1.0 / self.max_video_fps:
            return

        self._last_frame_time = now
        with self._last_frame_lock:
            self._last_frame_data = bytes(data)
            self._frame_width = width
            self._frame_height = height

    def _on_playback_audio(self, pcm_bytes: bytes, channels: int, sample_rate: int):
        """TTS 音频回传播放（本地播放或者发给运控板）"""
        with self._playback_lock:
            self._playback_queue.append((pcm_bytes, channels, sample_rate))

    # ── 处理逻辑 ───────────────────────────────────

    async def _process_subtitle(self):
        """处理 ASR 字幕：发送到 CommServer → LLM → TTS → DDS 播放。"""
        with self._subtitle_lock:
            text = self._latest_subtitle
            self._latest_subtitle = ""
        self._subtitle_event.clear()

        if not text or len(text.strip()) < 2:
            return

        logger.info("📤 发送到 CommServer: %s", text)

        try:
            # 如果有最近的图像帧，一起发给 LLM
            with self._last_frame_lock:
                frame = self._last_frame_data
                w, h = self._frame_width, self._frame_height
                self._last_frame_data = None  # 用完清掉

            if frame and self.enable_video:
                # 转 JPEG base64
                image_b64 = self._yuyv_to_jpeg_base64(frame, w, h)
                if image_b64:
                    logger.info("附带图像 %dx%d", w, h)
                    reply = await self.comm.send_with_image(text, image_b64)
                else:
                    reply = await self.comm.send_text(text)
            else:
                reply = await self.comm.send_text(text)

            if not reply:
                logger.warning("LLM 回复为空")
                return

            logger.info("🤖 LLM: %s", reply)

            # TTS 合成
            try:
                tts_audio = await self.tts.synthesize(reply)
                if tts_audio:
                    # 发送到 DDS external_playback → 运控板播放
                    self._send_tts_to_dds(tts_audio)
            except Exception as e:
                logger.error("TTS 失败: %s", e)

        except Exception as e:
            logger.exception("处理字幕异常: %s", e)

    def _send_tts_to_dds(self, audio_data: bytes):
        """将 TTS 音频写入临时 PCM 文件，通过 DDS external_playback 发送。"""
        import tempfile

        # TTS 输出通常是 MP3（edge-tts）或 WAV，需要转 PCM
        # 此处简化：写入临时文件，用 external_audio_playback_from_file
        with tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as f:
            # TODO: 实际需要将 MP3/WAV 解码为 PCM int16
            # 暂时直接写入原始数据
            f.write(audio_data)
            tmp_path = f.name

        def _send():
            try:
                self.voice._ctrl.external_audio_playback_from_file(
                    tmp_path,
                    channels=2,
                    sample_rate=16000,
                    format=2,
                    duration_ms=10,
                )
            except Exception as e:
                logger.error("DDS 音频播放失败: %s", e)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        t = threading.Thread(target=_send, daemon=True)
        t.start()

    @staticmethod
    def _yuyv_to_jpeg_base64(yuyv_data: bytes, width: int, height: int) -> Optional[str]:
        """YUYV 原始帧 → JPEG base64。"""
        try:
            import numpy as np
            from PIL import Image

            # YUYV → RGB
            yuyv = np.frombuffer(yuyv_data, dtype=np.uint8).reshape(height, width, 2)
            y = yuyv[:, :, 0].astype(np.float32)
            u = yuyv[0::2, :, 1].astype(np.float32)
            v = yuyv[0::2, :, 1].astype(np.float32)
            # 上采样 UV
            u = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1)
            v = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)

            r = y + 1.402 * (v - 128)
            g = y - 0.344136 * (u - 128) - 0.714136 * (v - 128)
            b = y + 1.772 * (u - 128)

            rgb = np.stack([r, g, b], axis=2).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(rgb)

            buf = BytesIO()
            img.save(buf, format="JPEG", quality=80)
            return base64.b64encode(buf.getvalue()).decode()
        except ImportError:
            logger.warning("numpy/PIL 未安装，无法转换视频帧")
            return None
        except Exception as e:
            logger.error("YUYV→JPEG 转换失败: %s", e)
            return None

    # ── 主循环 ───────────────────────────────────

    async def run(self):
        await self.start()

        loop = asyncio.get_event_loop()
        while self._running:
            # 等待 ASR 字幕
            try:
                # 异步等待字幕事件
                await asyncio.wait_for(
                    loop.run_in_executor(None, self._subtitle_event.wait, 0.5),
                    timeout=1.0,
                )
            except asyncio.TimeoutError:
                continue

            if self._subtitle_event.is_set():
                await self._process_subtitle()

    async def shutdown(self):
        self._running = False
        try:
            self.voice.sleep()
        except Exception:
            pass
        self.voice.stop()
        await self.comm.close()


# ─────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DDS Bridge — 运控板 ↔ CommServer 桥接")
    parser.add_argument("--dds-xml", type=str, default="",
                        help="DDS 配置文件路径")
    parser.add_argument("--server-url", type=str,
                        default="ws://localhost:8001/ws/chat",
                        help="CommunicationServer WebSocket URL")
    parser.add_argument("--robot-id", type=str, default="bumi-01",
                        help="机器人 ID")
    parser.add_argument("--tts-provider", type=str, default="edge",
                        help="TTS 后端: edge / openai")
    parser.add_argument("--no-video", action="store_true",
                        help="禁用视频流")
    args = parser.parse_args()

    bridge = DDSBridge(
        dds_xml_path=args.dds_xml,
        server_url=args.server_url,
        robot_id=args.robot_id,
        tts_provider=args.tts_provider,
        enable_video=not args.no_video,
    )

    async def run():
        try:
            await bridge.run()
        except KeyboardInterrupt:
            print("\n👋 再见!")
        finally:
            await bridge.shutdown()

    asyncio.run(run())


if __name__ == "__main__":
    main()
