"""
LLM 工厂 - 根据配置创建对应的 LLM 实例
"""
import os
import logging
from typing import Optional
from langchain_core.language_models import BaseChatModel
from app.llm.base import BaseLLMProvider
from app.llm.volcano import VolcanoLLMProvider
from app.llm.siliconflow import SiliconFlowLLMProvider
from app.services import model_router_service

logger = logging.getLogger(__name__)


def create_llm(provider: Optional[str] = None, role: Optional[str] = None) -> BaseChatModel:
    """
    创建 LLM 实例。

    Args:
        provider: LLM 提供商名称（"volcano" / "siliconflow" / "openai_compatible"）。
        role: 可选的蒸馏模型角色，如 agent_orchestration / video_script。
              当 storage/model_router.json 启用且该 role 配置了可调用 endpoint 时，优先使用路由模型。
    """
    routed = model_router_service.resolve_model_for_role(role)
    model_name_override = None
    base_url_override = None
    api_key_override = None

    if routed:
        provider = routed.provider
        model_name_override = routed.model
        base_url_override = routed.base_url or None
        api_key_override = routed.api_key or None
        logger.info(
            "🧭 使用角色模型路由: role=%s provider=%s model=%s base_url=%s",
            role,
            provider,
            model_name_override,
            base_url_override or "<provider-default>",
        )

    # 如果没有指定 provider，从环境变量读取（默认 volcano）
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "volcano").lower().strip()
    provider = provider.lower().strip()
    
    logger.info(f"🏭 创建 LLM: provider={provider}, role={role or 'default'}")
    
    # 根据 provider 创建对应的实例
    provider_instance: BaseLLMProvider
    
    if provider == "volcano":
        provider_instance = VolcanoLLMProvider(
            model_name=model_name_override,
            base_url=base_url_override,
            api_key=api_key_override,
        )
    elif provider in ("siliconflow", "openai_compatible"):
        provider_instance = SiliconFlowLLMProvider(
            model_name=model_name_override,
            base_url=base_url_override,
            api_key=api_key_override,
        )
    else:
        raise ValueError(
            f"不支持的 LLM 提供商: {provider}。"
            f"支持的提供商: volcano, siliconflow, openai_compatible。"
            f"请在 .env 中设置 LLM_PROVIDER=volcano 或 LLM_PROVIDER=siliconflow"
        )
    
    # 创建并返回模型实例
    model = provider_instance.create_model()
    logger.info(f"✅ LLM 创建成功: provider={provider_instance.get_provider_name()}, role={role or 'default'}")
    
    return model
