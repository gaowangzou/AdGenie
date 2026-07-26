# 模型路由与 OPD 接入

AdGenie 的 OPD 训练框架负责蒸馏小模型；后端运行时不会直接加载 checkpoint，而是通过 OpenAI-compatible endpoint 调用已经部署好的模型。

## 五个 role

- `agent_orchestration`：主 LangGraph ReAct Agent 的编排模型，接管对话、工具选择和多步执行。
- `image_understanding`：图片理解模型，只接管图片输入；音频、视频和语音输出仍走 Qwen3-Omni 专用 API。
- `video_script`：视频脚本、分镜、镜头运动和生成提示词规划工具。
- `tts_voice`：播客、旁白、广播剧等 TTS 音色、说话人和合成策略规划工具。
- `personal_agent`：结合结构化记忆和会话摘要生成个人化 brief 与风格约束，不写入长期记忆。

## 路由配置

运行侧配置文件是 `agent/backend/storage/model_router.json`。必须同时打开顶层 `enabled` 和具体 role 的 `enabled`，并提供可调用的 `provider`、`model`、`base_url`。

```json
{
  "enabled": true,
  "models": {
    "agent_orchestration": {
      "enabled": true,
      "provider": "openai_compatible",
      "model": "agent_orchestration-test",
      "base_url": "http://127.0.0.1:30000/v1",
      "api_key_env": ""
    }
  }
}
```

`checkpoint` 只记录训练产物位置，不能被后端直接调用。蒸馏模型需要先用 vLLM、SGLang 或其它 OpenAI-compatible 服务部署，再把 endpoint 写入路由表。

## 配置脚本

```bash
cd agent/backend
py scripts/configure_model_router.py enable --role agent_orchestration --provider openai_compatible --model agent_orchestration-test --base-url http://127.0.0.1:30000/v1 --check
py scripts/configure_model_router.py show
py scripts/configure_model_router.py disable --role agent_orchestration
```

## 验证

有真实或临时小模型 endpoint 时，先启动后端，再运行：

```bash
cd agent/backend
py scripts/verify_model_router_e2e.py --backend-url http://127.0.0.1:8000 --role agent_orchestration --model agent_orchestration-test --base-url http://127.0.0.1:30000/v1 --expect-text "router e2e ok"
```

脚本会临时写入 `model_router.json`，请求 `/api/chat`，默认在结束时恢复原配置。

## 回退行为

- 未开启路由、role 未启用或 endpoint 字段不完整时，`create_llm()` 会继续使用默认大模型。
- `agent_orchestration` 创建失败或首个 SSE 错误事件会回退默认大模型。
- `video_script`、`tts_voice`、`personal_agent` 规划工具失败时返回 JSON `fallback_instruction`，由主 Agent 自行规划并继续遵守确认流程。
- `image_understanding` 图片路由失败会回退 Qwen3-Omni；如果图片路由成功但请求语音输出，会降级为文字结果并在 `message` 中说明。
## 媒体后处理依赖

模型路由只决定哪个 LLM 或小模型 endpoint 负责规划/理解。视频拼接和音频后处理不是模型调用问题，而是本地媒体处理问题：

- `concatenate_videos` 通过 moviepy 调用 ffmpeg 写出视频。
- `concatenate_audio`、`select_bgm`、`mix_audio_with_bgm` 通过 pydub 调用 ffmpeg/ffprobe 读取、裁剪和导出音频。

因此即使模型路由关闭、默认大模型正常，缺少系统级 `ffmpeg` / `ffprobe` 时这些工具仍会失败。Windows 可用 `winget install --id Gyan.FFmpeg -e` 安装，安装后必须重开启动后端的终端，让 PATH 生效。
