"""
显式 Agent State / Observation 建模

复用 langgraph.prebuilt.create_react_agent 现有的 agent<->tools 循环，
不引入新的图结构（DAG / Planner-Executor）。通过两个官方扩展点接入：

- state_schema=AdGenieState：在内置 messages/remaining_steps 之外
  显式声明业务级状态字段。
- pre_model_hook=sync_observations：该 hook 在 "tools" 节点之后、
  "agent"（LLM）节点之前执行，把新产生的 ToolMessage 标准化为
  Observation 并写回 AdGenieState，供下一轮模型决策读取。
"""
import json
import logging
import operator
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages
from langgraph.managed import RemainingSteps
from typing_extensions import Annotated, NotRequired, TypedDict

logger = logging.getLogger(__name__)


class Observation(TypedDict):
    """一次工具调用的标准化观测结果"""

    tool_name: str
    tool_call_id: str
    success: bool
    structured_result: Any
    artifact_uri: Optional[str]
    error_type: Optional[str]
    retryable: bool
    step: int


class ToolHistoryEntry(TypedDict):
    """一次工具调用的动作记录"""

    tool_call_id: str
    tool_name: str
    arguments: Dict[str, Any]
    step: int


class ErrorState(TypedDict):
    """最近一次失败的完整上下文，供调试/前端展示使用"""

    tool_name: str
    tool_call_id: str
    error_type: str
    message: str
    retryable: bool


class AdGenieState(TypedDict):
    """显式建模的 Agent 业务状态，兼容 create_react_agent 对 state_schema 的要求"""

    # create_react_agent 强制要求的 key
    messages: Annotated[Sequence[BaseMessage], add_messages]
    remaining_steps: NotRequired[RemainingSteps]

    # 业务级扩展
    user_goal: NotRequired[Optional[str]]
    current_step: NotRequired[int]
    tool_history: NotRequired[Annotated[List[ToolHistoryEntry], operator.add]]
    observations: NotRequired[Annotated[List[Observation], operator.add]]
    artifacts: NotRequired[Annotated[List[Dict[str, Any]], operator.add]]
    error_state: NotRequired[Optional[ErrorState]]
    retry_count: NotRequired[int]
    execution_status: NotRequired[str]


_ARTIFACT_KEYS = (
    "artifact_uri",
    "image_url",
    "video_url",
    "model_url",
    "audio_url",
    "local_path",
    "image_urls",
    "video_urls",
    "local_paths",
)

# 项目内工具的失败文本前缀约定：
# - "Error"/"错误"/"Exception"/"Traceback"：volcano_image_generation.py、skill_tools.py 等
# - "❌"：workspace_tools.py（write_memory）
_ERROR_PREFIXES = ("error", "错误", "exception", "traceback", "❌")

# 项目内工具的失败 JSON 约定："error" 字段（model_3d_generation.py、
# virtual_anchor_generation.py、video_concatenation.py、volcano_video_generation.py、
# role_planner_tools.py 等大量工具用 json.dumps({"error": ...}) 表示失败，
# 不带 "Error" 前缀，必须单独识别）
_ERROR_JSON_KEYS = ("error",)

_ERROR_TYPE_PATTERNS: Tuple[Tuple["re.Pattern[str]", str, bool], ...] = (
    (re.compile(r"不存在|not found|filenotfound", re.IGNORECASE), "not_found", False),
    (re.compile(r"参数|invalid|valueerror|不能同时|至少一个|不支持的格式", re.IGNORECASE), "invalid_input", False),
    (re.compile(r"超时|timeout", re.IGNORECASE), "timeout", True),
    (re.compile(r"429|rate.?limit", re.IGNORECASE), "rate_limited", True),
    (re.compile(r"未配置|api_key|api key", re.IGNORECASE), "config_error", False),
    (re.compile(r"5\d{2}|internal server|network|connection", re.IGNORECASE), "upstream_error", True),
)


def _classify_error_type(message: str) -> Tuple[str, bool]:
    for pattern, error_type, retryable in _ERROR_TYPE_PATTERNS:
        if pattern.search(message):
            return error_type, retryable
    return "tool_error", True


def _extract_error_message(structured_result: Any, content_str: str) -> str:
    if isinstance(structured_result, dict):
        # "raw" 对应非 JSON 内容的兜底包装（见 _normalize_observation），
        # 必须排在检查列表里，否则非 JSON 错误文本会丢失，退化成 dict 的 str() 表示
        for key in ("error", "message", "raw"):
            value = structured_result.get(key)
            if isinstance(value, str) and value:
                return value
    return content_str


def _looks_like_error_json(structured_result: Any) -> bool:
    if not isinstance(structured_result, dict):
        return False
    for key in _ERROR_JSON_KEYS:
        if structured_result.get(key):
            return True
    if structured_result.get("success") is False:
        return True
    if str(structured_result.get("status", "")).lower() == "error":
        return True
    return False


def _extract_artifact_uri(structured_result: Any) -> Optional[str]:
    if not isinstance(structured_result, dict):
        return None
    for key in _ARTIFACT_KEYS:
        value = structured_result.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, list) and value and isinstance(value[0], str):
            return value[0]
    return None


def _find_tool_call(messages: Sequence[BaseMessage], tool_call_id: str) -> Optional[Dict[str, Any]]:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                if call.get("id") == tool_call_id:
                    return call
    return None


def _normalize_observation(
    tool_message: ToolMessage,
    tool_call: Optional[Dict[str, Any]],
    step: int,
) -> Observation:
    tool_name = tool_message.name or (tool_call.get("name") if tool_call else None) or "unknown_tool"
    raw_content = tool_message.content
    content_str = raw_content if isinstance(raw_content, str) else json.dumps(raw_content, ensure_ascii=False)
    content_lower = content_str.strip().lower()

    structured_result: Any
    try:
        structured_result = json.loads(content_str)
    except (TypeError, ValueError):
        structured_result = {"raw": content_str}

    status_is_error = getattr(tool_message, "status", "success") == "error"
    prefix_is_error = any(content_lower.startswith(p.lower()) for p in _ERROR_PREFIXES)
    json_is_error = _looks_like_error_json(structured_result)

    success = not (status_is_error or prefix_is_error or json_is_error)

    error_type: Optional[str] = None
    retryable = False
    artifact_uri: Optional[str] = None
    if success:
        artifact_uri = _extract_artifact_uri(structured_result)
    else:
        error_message = _extract_error_message(structured_result, content_str)
        error_type, retryable = _classify_error_type(error_message)

    return Observation(
        tool_name=tool_name,
        tool_call_id=tool_message.tool_call_id,
        success=success,
        structured_result=structured_result,
        artifact_uri=artifact_uri,
        error_type=error_type,
        retryable=retryable,
        step=step,
    )


def sync_observations(state: AdGenieState) -> Dict[str, Any]:
    """pre_model_hook：将新增 ToolMessage 标准化为 Observation 并写回 AdGenieState"""
    messages = state.get("messages") or []
    existing_observations = state.get("observations") or []
    seen_ids = {obs["tool_call_id"] for obs in existing_observations}
    step = state.get("current_step") or 0

    new_observations: List[Observation] = []
    new_tool_history: List[ToolHistoryEntry] = []
    new_artifacts: List[Dict[str, Any]] = []

    for msg in messages:
        if not isinstance(msg, ToolMessage) or msg.tool_call_id in seen_ids:
            continue
        step += 1
        tool_call = _find_tool_call(messages, msg.tool_call_id)
        observation = _normalize_observation(msg, tool_call, step)
        new_observations.append(observation)
        new_tool_history.append(
            ToolHistoryEntry(
                tool_call_id=msg.tool_call_id,
                tool_name=observation["tool_name"],
                arguments=(tool_call.get("args") if tool_call else None) or {},
                step=step,
            )
        )
        if observation["success"] and observation["artifact_uri"]:
            new_artifacts.append(
                {
                    "tool_call_id": observation["tool_call_id"],
                    "tool_name": observation["tool_name"],
                    "uri": observation["artifact_uri"],
                    "step": step,
                }
            )

    update: Dict[str, Any] = {}

    if not state.get("user_goal"):
        # messages 是完整历史（含之前轮次），必须倒序取最后一条 HumanMessage
        # 才是本轮用户请求，否则多轮对话里 user_goal 会永远固定成第一轮的输入
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) and msg.content:
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                update["user_goal"] = content
                break

    if new_observations:
        update["observations"] = new_observations
        update["tool_history"] = new_tool_history
        update["current_step"] = step
        if new_artifacts:
            update["artifacts"] = new_artifacts

        last = new_observations[-1]
        if last["success"]:
            update["execution_status"] = "running"
            update["error_state"] = None
        else:
            update["execution_status"] = "tool_error"
            update["retry_count"] = (state.get("retry_count") or 0) + 1
            update["error_state"] = ErrorState(
                tool_name=last["tool_name"],
                tool_call_id=last["tool_call_id"],
                error_type=last["error_type"] or "tool_error",
                message=_extract_error_message(last["structured_result"], str(last["structured_result"])),
                retryable=last["retryable"],
            )

        logger.info(
            "🧭 Observation 同步: step=%s success=%s tool=%s status=%s",
            last["step"],
            last["success"],
            last["tool_name"],
            update["execution_status"],
        )
    elif not state.get("execution_status"):
        update["execution_status"] = "initialized"

    return update


def render_state_context(state: AdGenieState, max_observations: int = 5) -> str:
    """把当前 State 中最近的 Observation 渲染为供模型阅读的文本片段"""
    observations = state.get("observations") or []
    if not observations:
        return ""

    error_state = state.get("error_state")

    lines = [
        "[Agent State]",
        f"current_step: {state.get('current_step', 0)}",
        f"execution_status: {state.get('execution_status', 'initialized')}",
        f"retry_count: {state.get('retry_count', 0)}",
        f"error_state: {error_state if error_state else 'none'}",
        "Recent Observations (most recent last):",
    ]
    for obs in observations[-max_observations:]:
        line = f"- step={obs['step']} tool={obs['tool_name']} success={obs['success']}"
        if obs["success"]:
            if obs.get("artifact_uri"):
                line += f" artifact={obs['artifact_uri']}"
        else:
            line += f" error_type={obs.get('error_type')} retryable={obs.get('retryable')}"
        lines.append(line)

    return "\n".join(lines)
