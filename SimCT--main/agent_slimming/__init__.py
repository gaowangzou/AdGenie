"""
Agent Self-Slimming Module for AdGenie.

使用 SimCT 跨 tokenizer 在线策略蒸馏，将编排 Agent 从大模型蒸馏为用户专属轻量模型。

针对 AdGenie 多模型协同场景：
  - 编排 Agent（200步递归推理）→ 蒸馏到 2B 小模型，降低延迟和成本
  - 图片理解 → 多模态小模型
  - TTS / 语音 → 语音专项小模型
  - 视频脚本 → 文本生成小模型

核心流程:
  1. Data Collection   — 从用户历史对话中提取 Agent 推理轨迹
  2. Data Preparation  — 转换为 SimCT 训练格式
  3. Teacher Response  — 大模型生成黄金标注
  4. SFT Warmup        — 小模型基础指令微调
  5. OPD Distillation  — 跨 tokenizer 在线策略蒸馏
  6. Model Deployment  — 注册到模型路由表
"""

from .config import SlimmingConfig
from .data_collector import DataCollector
from .data_preparer import DataPreparer
from .slimming_pipeline import SlimmingPipeline
from .router import ModelRouter

__all__ = [
    "SlimmingConfig",
    "DataCollector",
    "DataPreparer",
    "SlimmingPipeline",
    "ModelRouter",
]
