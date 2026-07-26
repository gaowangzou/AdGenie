"""
Agent服务 - 处理LangGraph Agent的流式输出
"""
import json
import os
import logging
from typing import List, Dict, Any, AsyncGenerator, Optional
from langgraph.prebuilt import create_react_agent
from app.services.stream_processor import StreamProcessor
from app.services.prompt import get_full_prompt
from app.services import skill_service
from app.tools.volcano_image_generation import generate_volcano_image_tool, edit_volcano_image_tool
from app.tools.model_3d_generation import generate_3d_model_tool
from app.tools.volcano_video_generation import generate_volcano_video_tool
from app.tools.video_concatenation import concatenate_videos_tool
from app.tools.virtual_anchor_generation import detect_face_tool, generate_virtual_anchor_tool
from app.tools.qwen_tts import qwen_voice_design_tool, qwen_voice_cloning_tool
from app.tools.audio_mixing import concatenate_audio_tool, select_bgm_tool, mix_audio_with_bgm_tool
from app.tools.qwen_omni_understanding import qwen_omni_understand_tool
from app.tools.role_planner_tools import plan_video_script_tool, plan_tts_voice_tool, adapt_personal_style_tool
from app.tools.skill_tools import read_skill_file_tool, list_skill_dir_tool, init_skill_tool, write_skill_file_tool, delete_skill_file_tool
from app.tools.workspace_tools import write_memory
from app.llm.factory import create_llm
from app.services import workspace_service, model_router_service

# 使用统一的日志配置
logger = logging.getLogger(__name__)


def create_agent(current_query: Optional[str] = None, canvas_id: Optional[str] = None, model_role: Optional[str] = "agent_orchestration"):
    """创建LangGraph Agent"""
    # 使用 LLM 工厂创建模型实例（默认使用火山引擎）
    model = create_llm(role=model_role)

    # 创建工具列表
    tools = [
        # generate_image_tool,
        # edit_image_tool,
        generate_volcano_image_tool,
        edit_volcano_image_tool,
        generate_3d_model_tool,
        generate_volcano_video_tool,
        concatenate_videos_tool,
        detect_face_tool,
        generate_virtual_anchor_tool,
        # Qwen-TTS工具
        qwen_voice_design_tool,
        qwen_voice_cloning_tool,
        # 音频混音工具
        concatenate_audio_tool,
        select_bgm_tool,
        mix_audio_with_bgm_tool,
        # Qwen3-Omni 多模态理解工具
        qwen_omni_understand_tool,
        # OPD role planner tools
        plan_video_script_tool,
        plan_tts_voice_tool,
        adapt_personal_style_tool,
        # Skill 文件读取工具（Progressive Loading）
        read_skill_file_tool,
        list_skill_dir_tool,
        # Skill 创建工具（skill-creator 工作流支持）
        init_skill_tool,
        write_skill_file_tool,
        delete_skill_file_tool,
        # Workspace 记忆写入工具
        write_memory,
    ]
    logger.info(f"🛠️  注册工具: {[tool.name for tool in tools]}")

    # 动态生成工具列表描述
    tool_descriptions = []
    for tool in tools:
        tool_descriptions.append(f"- {tool.name}: {tool.description}")
    tools_list_text = "\n".join(tool_descriptions)

    # 使用prompt模块生成完整提示词
    skills_context = skill_service.get_skills_context()
    workspace_context = workspace_service.get_workspace_context(query=current_query, canvas_id=canvas_id)
    full_prompt = get_full_prompt(
        tools_list_text=tools_list_text,
        skills_context=skills_context,
        workspace_context=workspace_context,
    )

    # 创建Agent
    agent = create_react_agent(
        name="adgenie_multimodal_agent",
        model=model,
        tools=tools,
        prompt=full_prompt
    )
    
    logger.info("✅ Agent创建成功")
    return agent



def _is_sse_error_event(event: str) -> bool:
    """Return True when a StreamProcessor SSE event is an error payload."""
    if not event.startswith("data: "):
        return False
    data_str = event[len("data: "):].strip()
    if not data_str or data_str == "[DONE]":
        return False
    try:
        payload = json.loads(data_str)
    except Exception:
        return False
    return payload.get("type") == "error"


async def _stream_agent_events(
    agent: Any,
    messages: List[Dict[str, Any]],
    session_id: Optional[str],
) -> AsyncGenerator[str, None]:
    """Stream agent events and preserve client-disconnect handling in one place."""
    processor = StreamProcessor(session_id)
    async for event in processor.process_stream(agent, messages):
        try:
            yield event
        except (GeneratorExit, StopAsyncIteration, ConnectionError, BrokenPipeError, OSError) as e:
            logger.info(f"⚠️ 客户端断开连接: {type(e).__name__}: {str(e)}")
            raise
        except Exception as e:
            logger.warning(f"⚠️ 发送事件时出错: {type(e).__name__}: {str(e)}")
            raise


async def process_chat_stream(
    messages: List[Dict[str, Any]],
    session_id: Optional[str] = None,
    canvas_id: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """
    处理聊天流式响应
    
    Args:
        messages: 消息历史
        session_id: 会话ID
    
    Yields:
        SSE格式的事件流
    """
    try:
        logger.info(f"💬 收到聊天请求: session_id={session_id}, messages_count={len(messages)}")
        
        # 创建Agent（在 langgraph 1.0.0 中，create_react_agent 返回的对象已经是编译后的）
        current_query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                current_query = msg.get("content", "")
                break
        routed_model = model_router_service.resolve_model_for_role("agent_orchestration")
        try:
            agent = create_agent(current_query=current_query, canvas_id=canvas_id, model_role="agent_orchestration")
        except Exception as e:
            if not routed_model:
                raise
            logger.warning(
                "⚠️ 路由模型创建失败，回退默认大模型: role=agent_orchestration, model=%s, base_url=%s, error=%s",
                routed_model.model,
                routed_model.base_url,
                e,
            )
            agent = create_agent(current_query=current_query, canvas_id=canvas_id, model_role=None)
            routed_model = None

        first_event = True

        async for event in _stream_agent_events(agent, messages, session_id):
            if first_event and routed_model and _is_sse_error_event(event):
                logger.warning(
                    "⚠️ 路由模型首包失败，回退默认大模型: role=agent_orchestration, model=%s, base_url=%s",
                    routed_model.model,
                    routed_model.base_url,
                )
                fallback_agent = create_agent(current_query=current_query, canvas_id=canvas_id, model_role=None)
                async for fallback_event in _stream_agent_events(fallback_agent, messages, session_id):
                    yield fallback_event
                return

            first_event = False
            yield event


    except (GeneratorExit, StopAsyncIteration, ConnectionError, BrokenPipeError, OSError) as e:
        # 客户端断开连接，这是正常情况，不需要记录为错误
        logger.info(f"ℹ️ 客户端断开连接，停止流式响应: {type(e).__name__}")
        # 不发送错误事件，因为客户端已经断开
        return
    except Exception as e:
        import traceback
        logger.error(f"❌ 处理聊天流时出错: {str(e)}")
        logger.error(traceback.format_exc())
        try:
            # 尝试发送错误事件（如果客户端还在）
            error_event = {
                "type": "error",
                "error": str(e)
            }
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except:
            # 如果发送失败（客户端已断开），忽略
            pass
