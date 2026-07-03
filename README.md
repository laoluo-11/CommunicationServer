# CommunicationServer

Bumi 机器人语音对话系统 — 机器人与云端大模型之间的实时智能对话中间件。

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    Jetson 算力板                          │
│                   192.168.55.101                          │
│                   用户: noetix / bumi                      │
│                                                          │
│  ┌──────────────────┐   localhost   ┌────────────────┐  │
│  │   robot-agent    │──WebSocket──→│ Communication  │  │
│  │                  │   :8001      │    Server      │  │   internet   ┌──────────┐
│  │  1. 麦克风采集    │              │                │  │──────────────→│  OpenAI  │
│  │  2. STT 语音→文字 │←──流式回复──│  1. 对话管理    │  │               │  API     │
│  │  3. 发送文字      │              │  2. LLM 调用   │  │               │          │
│  │  4. 接收回复      │              │  3. 会话历史    │  │               │          │
│  │  5. TTS 文字→语音 │              │  :8001         │  │               └──────────┘
│  │  6. 扬声器播放    │              └────────────────┘  │
│  └──────────────────┘                                   │
│         ↑ 麦克风 / ↓ 扬声器 (OS 音频设备)                 │
└─────────────────────────────────────────────────────────┘
          ↑ DDS (500Hz, 电机控制 — 不经过音频)
┌──────────────────┐
│  运控板            │
│  192.168.55.102   │
│  (不对用户开放)    │
└──────────────────┘
```

**两个组件都运行在 Jetson 上**（本服务器与 Jetson 网络不通，无法分开部署）。

### robot-agent — 机器人端

负责音频采集、语音识别、语音合成：

| 组件 | 默认实现 | 说明 |
|------|---------|------|
| 音频采集 | PyAudio + VAD | 麦克风录音，语音活动检测，自动结束 |
| STT | OpenAI Whisper API | 语音→文字 (`whisper-1`) |
| TTS | Microsoft Edge TTS | 文字→语音（免费，高质量中文） |
| 通信 | WebSocket | 连接 CommunicationServer |

启动命令：
```bash
cd ~/CommunicationServer
python -m robot_agent.main

# 指定设备
python -m robot_agent.main --list-devices     # 列出音频设备
python -m robot_agent.main --device-index 1   # 指定麦克风
```

交互方式：
- `Enter` — 直接输入文字对话
- `/listen` — 开始录音，自动识别并回复
- `/reset` — 重置对话历史
- `/devices` — 列出音频设备
- `/quit` — 退出

### CommunicationServer — 服务端

FastAPI 服务，负责对话管理和 LLM 调用：

```bash
cd ~/CommunicationServer
pip install -r requirements.txt
LLM_API_KEY=*** uvicorn app.main:app --host 0.0.0.0 --port 8001
```

## API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/system/health` | GET | 健康检查 |
| `/chat` | POST | 发送文本获取回复 |
| `/chat/reset` | POST | 重置对话 |
| `/ws/chat/{robot_id}` | WebSocket | 流式对话（推荐） |

## 部署（Jetson aarch64）

```bash
# 1. 安装依赖
pip install -r requirements.txt
pip install -r requirements_robot.txt

# 2. 配置环境变量
export LLM_API_KEY=sk-***
export ROBOT_ID=bumi-01

# 3. 启动 CommunicationServer
nohup uvicorn app.main:app --host 0.0.0.0 --port 8001 &

# 4. 启动 robot-agent
python -m robot_agent.main
```

## 项目结构

```
CommunicationServer/
├── app/                         # CommunicationServer (FastAPI)
│   ├── main.py
│   ├── core/config.py
│   ├── api/routes/
│   │   ├── chat.py              # REST 对话
│   │   ├── ws.py                # WebSocket 流式对话
│   │   └── system.py            # 健康检查
│   └── services/
│       ├── llm.py               # LLM 调用
│       └── chat.py              # 会话管理
├── robot_agent/                 # 机器人端 (音频 + STT + TTS)
│   ├── main.py                  # 主循环入口
│   ├── config.py                # 配置
│   ├── audio.py                 # 麦克风录音 / VAD
│   ├── stt.py                   # 语音识别 (Whisper API)
│   ├── tts.py                   # 语音合成 (Edge TTS)
│   └── client.py                # WebSocket 客户端
├── requirements.txt             # 服务端依赖
├── requirements_robot.txt       # 机器人端依赖
└── .env.example
```
