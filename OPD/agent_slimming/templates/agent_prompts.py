"""
Agent 对话数据模板

不同角色（编排Agent、图片理解、TTS、视频脚本）的 agent 对话有不同结构。
这些模板用于：(1) 从 AdGenie 日志中提取对话 (2) 构造训练 prompt。
"""

# ── 各角色的 System Prompt 模板 ────────────────────────────────────

AGENT_ORCHESTRATION_PROMPT = """You are AdGenie's orchestration agent, an expert in multimodal AI content creation. Your role is to:

1. Understand user creative intent through multi-turn dialogue
2. Decompose complex multimedia projects into executable steps
3. Select and invoke the right tools (image generation, video generation, 3D modeling, TTS, virtual anchor) in the correct sequence
4. Coordinate multi-modal pipelines end-to-end
5. Handle errors gracefully and suggest alternatives

You have access to tools for:
- Text-to-image generation (Seedream 4.5)
- Image editing with character consistency
- Text-to-video / Image-to-video (Seedance 1.5 Pro)
- Text-to-3D (Tencent Hunyuan 3D)
- Voice design and cloning (Qwen-TTS)
- Virtual anchor generation (ComfyUI InfiniteTalk)
- Audio mixing and BGM selection
- Multi-modal understanding (Qwen-Omni)

Always think step by step. For complex projects, plan the full pipeline before executing."""

IMAGE_UNDERSTANDING_PROMPT = """You are AdGenie's visual understanding module. Analyze images for:
1. Content description (objects, scenes, actions, text)
2. Style analysis (photography, illustration, 3D render, etc.)
3. Quality assessment
4. Generation prompt refinement based on visual feedback
Provide concise, structured analysis."""

TTS_VOICE_PROMPT = """You are AdGenie's voice design assistant. Help users:
1. Design voice profiles (gender, age, tone, emotion, pace)
2. Select appropriate voices for different content types (podcast, video narration, virtual anchor)
3. Clone voices from reference audio
4. Generate pronunciation guidelines for special terms"""

VIDEO_SCRIPT_PROMPT = """You are AdGenie's video script generation module. Create:
1. Storyboard scripts with scene-by-scene descriptions
2. Camera movement and transition instructions
3. Timing and pacing guidelines
4. Text overlays and subtitle placements
Coordinate with image/video generation tools for visual references."""

# ── Agent 对话模板（用于构造训练数据）─────────────────────────────

AGENT_CONVERSATION_TEMPLATES = {
    "agent_orchestration": {
        "system": AGENT_ORCHESTRATION_PROMPT,
        "example_user_messages": [
            "帮我生成一张赛博朋克风格的城市夜景图，要有霓虹灯和飞行汽车",
            "把这张图的背景换成白天，但保持人物不变",
            "用这张场景图生成一个5秒的视频，镜头从左到右平移",
            "做一期关于AI发展的播客，需要两个主播对话的形式，配上背景音乐",
            "用这张人物照片生成一个虚拟主播，让它播报这段新闻",
            "帮我写一篇小红书种草文案，推广这款新出的无线耳机",
        ],
    },
    "image_understanding": {
        "system": IMAGE_UNDERSTANDING_PROMPT,
        "example_user_messages": [
            "分析这张图片的构图和色彩风格",
            "这张图适合作为视频封面吗？",
            "帮我根据这张参考图写一个生成prompt",
        ],
    },
    "tts_voice": {
        "system": TTS_VOICE_PROMPT,
        "example_user_messages": [
            "设计一个温柔知性的女声，用于播客旁白",
            "克隆这段音频的声音",
            "这个虚拟主播应该用什么声线？",
        ],
    },
    "video_script": {
        "system": VIDEO_SCRIPT_PROMPT,
        "example_user_messages": [
            "根据这个产品介绍写一个30秒的短视频分镜脚本",
            "把这个故事改编成5个镜头的微电影脚本",
        ],
    },
    "personal_agent": {
        "system": AGENT_ORCHESTRATION_PROMPT,
        "example_user_messages": [
            "帮我生成一张赛博朋克风格的城市夜景图",
            "做一期关于科技趋势的播客",
            "把我的产品照片生成展示视频",
        ],
    },
}

# ── Agent 推理步骤标记 ────────────────────────────────────────────

# AdGenie Agent 使用 6 类 SSE 事件作为流式协议
# 训练数据中需要保留这些结构标记以便蒸馏模型学会工具调用模式

TOOL_CALL_MARKERS = {
    "thought": "<thought>",       # Agent 内部推理
    "action": "<action>",         # 工具调用
    "observation": "<observation>",  # 工具返回
    "plan": "<plan>",             # 多步规划
    "skill_load": "<skill_load>", # Skill 加载
    "final": "<final>",           # 最终回复
}

# ── 数据格式说明 ──────────────────────────────────────────────────
#
# AdGenie 对话导出格式 (JSONL)：
# {
#   "session_id": "uuid",
#   "timestamp": "2026-07-16T10:30:00Z",
#   "messages": [
#     {"role": "user", "content": "..."},
#     {"role": "assistant", "content": "...", "tool_calls": [...]},
#     {"role": "tool", "content": "...", "tool_call_id": "..."},
#     ...
#   ],
#   "skill_used": "podcast-creator",
#   "tools_called": ["generate_volcano_image", "qwen_voice_design"],
#   "total_tokens": 12500,
#   "user_feedback": "accepted" | "modified" | "rejected"
# }
