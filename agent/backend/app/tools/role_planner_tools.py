"""
Role planner tools backed by OPD model router roles.

These tools create real runtime call sites for the remaining distilled roles:
video_script, tts_voice, and personal_agent. They do planning/adaptation work and
then return structured JSON for the main Agent to use with the existing media
creation tools.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.llm.factory import create_llm
from app.services import workspace_service

logger = logging.getLogger(__name__)


def _message_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def _extract_json_object(text: str) -> dict[str, Any]:
    """Best-effort JSON extraction for model responses that may include prose."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else {"items": parsed}
    except Exception:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(stripped[start:end + 1])
            return parsed if isinstance(parsed, dict) else {"items": parsed}
        except Exception:
            pass
    return {"raw_text": text}


def _invoke_role_json(role: str, system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any]:
    model = create_llm(role=role)
    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=json.dumps(user_payload, ensure_ascii=False, indent=2)),
    ])
    text = _message_text(response)
    result = _extract_json_object(text)
    result.setdefault("role", role)
    return result


class VideoScriptPlanInput(BaseModel):
    request: str = Field(description="用户的视频创作需求或商品/剧情描述")
    duration_seconds: Optional[int] = Field(default=None, description="目标视频总时长，单位秒；未知可留空")
    ratio: str = Field(default="16:9", description="视频宽高比，如 16:9、9:16、1:1")
    style: Optional[str] = Field(default=None, description="期望视觉风格，如写实商业、赛博朋克、极简等")
    reference_assets: Optional[str] = Field(default=None, description="可用参考素材路径或简述，如已有图片/角色/商品图")


@tool("plan_video_script", args_schema=VideoScriptPlanInput)
def plan_video_script_tool(
    request: str,
    duration_seconds: Optional[int] = None,
    ratio: str = "16:9",
    style: Optional[str] = None,
    reference_assets: Optional[str] = None,
) -> str:
    """
    使用 video_script 角色模型生成视频脚本、分镜、镜头运动和视频生成提示词。

    适用于短视频、广告片、商品展示、图生视频前的分镜规划。该工具只做规划，
    不生成视频；拿到结果后再调用 generate_volcano_image / generate_volcano_video。
    """
    try:
        system_prompt = """You are AdGenie's video_script role model.
Return only valid JSON. Create practical video planning output for downstream image/video generation tools.
Schema:
{
  "summary": "one sentence plan",
  "duration_seconds": number,
  "ratio": "string",
  "style": "string",
  "shots": [
    {
      "shot": 1,
      "duration_seconds": number,
      "scene": "visual scene description",
      "camera_motion": "camera movement",
      "action": "subject/action",
      "visual_prompt": "prompt for image/video generation",
      "transition": "transition to next shot",
      "text_overlay": "optional overlay text"
    }
  ],
  "negative_prompt": "things to avoid",
  "execution_notes": ["notes for the main agent"]
}"""
        result = _invoke_role_json("video_script", system_prompt, {
            "request": request,
            "duration_seconds": duration_seconds,
            "ratio": ratio,
            "style": style,
            "reference_assets": reference_assets,
        })
        result.setdefault("message", "视频脚本规划完成")
        result.setdefault("next_step", "请将分镜展示给用户，并等待用户确认后再调用图片/视频生成工具。")
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error("video_script planner failed: %s", e, exc_info=True)
        return json.dumps({
            "error": f"视频脚本规划失败: {e}",
            "role": "video_script",
            "fallback_instruction": "规划模型不可用。请主 Agent 根据用户需求和已启用的视频 Skill 自行生成分镜，并展示给用户等待确认；不要重试 plan_video_script。",
        }, ensure_ascii=False)


class TTSVoicePlanInput(BaseModel):
    request: str = Field(description="用户的播客、旁白、有声书、广播剧或配音需求")
    script: Optional[str] = Field(default=None, description="已有脚本或台词；没有可留空，由模型给出建议结构")
    language: str = Field(default="zh", description="主要语言代码，如 zh、en、ja")
    speaker_count: Optional[int] = Field(default=None, description="期望说话人数量；未知可留空")
    reference_audio: Optional[str] = Field(default=None, description="用户上传或已有参考音频路径；没有可留空")


@tool("plan_tts_voice", args_schema=TTSVoicePlanInput)
def plan_tts_voice_tool(
    request: str,
    script: Optional[str] = None,
    language: str = "zh",
    speaker_count: Optional[int] = None,
    reference_audio: Optional[str] = None,
) -> str:
    """
    使用 tts_voice 角色模型规划音色、角色、台词切分、情绪和合成策略。

    该工具不直接合成音频；它输出给 qwen_voice_design、qwen_voice_cloning、
    concatenate_audio 和 mix_audio_with_bgm 使用的结构化计划。
    """
    try:
        system_prompt = """You are AdGenie's tts_voice role model.
Return only valid JSON. Design voices and synthesis plan for podcast, narration, audiobook, radio drama, or virtual-anchor audio.
Schema:
{
  "content_type": "podcast|narration|audiobook|radio_drama|voiceover|other",
  "language": "string",
  "speakers": [
    {
      "name": "speaker name",
      "voice_description": "detailed voice prompt",
      "emotion": "default emotion",
      "pace": "slow|medium|fast",
      "sample_text": "short text for voice preview",
      "use_reference_audio": true
    }
  ],
  "segments": [
    {"speaker": "name", "text": "line to synthesize", "emotion": "emotion", "pause_after_ms": 1200}
  ],
  "bgm_style": "optional background music style",
  "synthesis_notes": ["notes for voice_design / voice_cloning"]
}"""
        result = _invoke_role_json("tts_voice", system_prompt, {
            "request": request,
            "script": script,
            "language": language,
            "speaker_count": speaker_count,
            "reference_audio": reference_audio,
        })
        result.setdefault("message", "TTS 音色与合成规划完成")
        result.setdefault("next_step", "请展示脚本/音色计划和音色样本方案，等待用户确认后再调用 Qwen-TTS 与音频处理工具。")
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error("tts_voice planner failed: %s", e, exc_info=True)
        return json.dumps({
            "error": f"TTS 音色规划失败: {e}",
            "role": "tts_voice",
            "fallback_instruction": "规划模型不可用。请主 Agent 根据音频 Skill 流程自行规划脚本、说话人、音色和合成步骤，展示给用户确认；不要重试 plan_tts_voice。",
        }, ensure_ascii=False)


class PersonalStyleInput(BaseModel):
    request: str = Field(description="用户原始创作需求")
    canvas_id: Optional[str] = Field(default=None, description="当前 canvas_id；未知可留空")
    target_medium: Optional[str] = Field(default=None, description="目标媒介，如 image、video、audio、copy、multimodal")
    current_plan: Optional[str] = Field(default=None, description="主 Agent 已有计划或草稿；没有可留空")


@tool("adapt_personal_style", args_schema=PersonalStyleInput)
def adapt_personal_style_tool(
    request: str,
    canvas_id: Optional[str] = None,
    target_medium: Optional[str] = None,
    current_plan: Optional[str] = None,
) -> str:
    """
    使用 personal_agent 角色模型结合当前长期记忆/会话摘要，生成个人化 brief 和风格约束。

    适用于用户要求符合长期偏好、角色资产、固定品牌风格、常用口吻或历史项目延续的任务。
    该工具只做偏好适配，不写入记忆；需要持久记忆时仍由 write_memory 处理。
    """
    try:
        memory_context = workspace_service.get_workspace_context(query=request, canvas_id=canvas_id)
        system_prompt = """You are AdGenie's personal_agent role model.
Return only valid JSON. Adapt the user's creative request using the provided memory/context without inventing unsupported preferences.
Schema:
{
  "rewritten_brief": "personalized brief",
  "style_constraints": ["specific style requirements"],
  "preferred_tone": "tone for final output",
  "asset_references": ["relevant asset paths or names"],
  "avoid": ["things to avoid"],
  "confidence": "confirmed|inferred|low",
  "memory_overlap_notes": "what came from memory vs current request",
  "next_agent_instructions": ["instructions for the main agent"]
}"""
        result = _invoke_role_json("personal_agent", system_prompt, {
            "request": request,
            "target_medium": target_medium,
            "current_plan": current_plan,
            "workspace_context": memory_context,
        })
        result.setdefault("message", "个人化风格适配完成")
        result.setdefault("next_step", "请主 Agent 将这些个人化约束用于后续工具调用；不要因此写入长期记忆。")
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error("personal_agent adapter failed: %s", e, exc_info=True)
        return json.dumps({
            "error": f"个人化风格适配失败: {e}",
            "role": "personal_agent",
            "fallback_instruction": "个人化适配模型不可用。请主 Agent 仅使用当前请求、对话历史和已注入记忆自行提炼风格约束；不要重试 adapt_personal_style，也不要凭空写入长期记忆。",
        }, ensure_ascii=False)
