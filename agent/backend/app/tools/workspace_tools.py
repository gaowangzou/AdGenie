"""
Workspace Tools - Agent 主动写入结构化长期记忆的工具。

MEMORY.md 现在是可读镜像，真实长期记忆写入 storage/memory.sqlite3。
"""
import logging
from langchain_core.tools import tool
from app.services import memory_service

logger = logging.getLogger(__name__)

# MEMORY.md 镜像中的合法章节名
_VALID_SECTIONS = ["角色资产", "成功 Prompt 模板", "用户偏好记录"]
_SECTION_TO_TYPE = {
    "角色资产": "角色资产",
    "成功 Prompt 模板": "prompt模板",
    "用户偏好记录": "用户偏好",
}


@tool
def write_memory(section: str, content: str) -> str:
    """
    将重要创作信息写入结构化长期记忆，并刷新 MEMORY.md 镜像。
    用于记录用户满意的角色资产、效果好的 Prompt 模板、用户明确的风格偏好等长期记忆。

    Args:
        section: 写入哪个章节，必须是以下之一：
                 "角色资产" / "成功 Prompt 模板" / "用户偏好记录"
        content: 要写入的 Markdown 内容（一条或多条条目）

    Returns:
        操作结果说明
    """
    if section not in _VALID_SECTIONS:
        return f"❌ 无效的章节名称：'{section}'。有效章节：{', '.join(_VALID_SECTIONS)}"

    try:
        memory_id = memory_service.upsert_memory(
            type=_SECTION_TO_TYPE[section],
            content=content,
            summary=content.strip()[:160],
            scope_type="user",
            scope_id="default_user",
            confidence="confirmed",
            source_session_id="write_memory_tool",
            metadata={"section": section, "source": "agent_confirmed_memory"},
        )
        logger.info(f"✅ 成功写入结构化记忆 {memory_id} [{section}]")
        return f"✅ 已成功写入长期记忆（{memory_id}），并刷新 MEMORY.md 镜像的「{section}」章节。"
    except Exception as e:
        logger.error(f"写入结构化记忆失败: {e}")
        return f"❌ 写入长期记忆失败：{e}"
