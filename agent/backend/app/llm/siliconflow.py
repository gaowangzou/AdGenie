"""
SiliconFlow LLM 实现
"""
import os
import logging
from typing import Optional
from langchain_openai import ChatOpenAI
from app.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class SiliconFlowLLMProvider(BaseLLMProvider):
    """SiliconFlow LLM 提供商"""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        # 从环境变量获取配置
        self.api_key = (api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "")).strip()
        self.base_url = (base_url if base_url is not None else os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")).strip()
        self.model_name = (model_name if model_name is not None else os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-V3.1-Terminus")).strip()
        
        if not self.api_key:
            raise RuntimeError(
                "未配置 OPENAI_API_KEY。请在 backend/.env 中设置，"
                "可参考 env.example（cp env.example .env）。"
            )
    
    def create_model(self) -> ChatOpenAI:
        """创建 SiliconFlow ChatModel 实例"""
        logger.info(f"🔷 创建 SiliconFlow LLM: model={self.model_name}, base_url={self.base_url}")
        
        model = ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=0.7,
            streaming=True,  # 启用流式输出
            max_tokens=2048,
            # 关键：禁止并行工具调用，强制"一次调用一个工具 -> 等结果 -> 再下一次"
            model_kwargs={"parallel_tool_calls": False},
        )
        
        return model
    
    def get_provider_name(self) -> str:
        return "siliconflow"
