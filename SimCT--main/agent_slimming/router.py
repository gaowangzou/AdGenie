"""
模型路由注册器

蒸馏完成后，将新的轻量模型注册到路由表中。
adgenie 的 Agent 调度层通过此路由表决定每个请求使用哪个模型。

路由规则：
  1. 用户请求 → 意图识别 → 匹配角色
  2. 若存在该角色的蒸馏模型 → 使用蒸馏模型（低延迟、低成本）
  3. 若蒸馏模型 unavailable → 回退到通用大模型
  4. 通用请求（未匹配任何角色）→ 使用默认大模型

路由表格式 (JSON):
{
  "models": {
    "agent_orchestration": {
      "primary": "agent_orchestration-gemma-2-2b-it",
      "fallback": "Qwen2.5-7B-Instruct",
      "checkpoint": "./output/ckpts/agent_orchestration-.../step-100",
      "metrics": {"latency_ms": 45, "cost_per_1k_tokens": 0.0003},
      "distilled_at": "2026-07-16T10:30:00Z"
    },
    ...
  },
  "default": "Qwen2.5-7B-Instruct"
}
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .config import SlimmingConfig, ROLE_PRESETS

logger = logging.getLogger(__name__)

# 路由表默认存放位置（与 adgenie backend 共享）
DEFAULT_ROUTER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "agent", "backend", "storage", "model_router.json"
)


class ModelRouter:
    """
    管理蒸馏后模型的注册和路由信息。

    路由表持久化到 JSON 文件，供 adgenie 的 LLM Factory 读取。
    """

    def __init__(self, config: SlimmingConfig, router_path: Optional[str] = None):
        self.config = config
        self.router_path = router_path or DEFAULT_ROUTER_PATH
        self._ensure_router_file()

    def register(
        self,
        model_name: str,
        model_path: str,
        role: str,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        注册一个蒸馏模型到路由表。

        Args:
            model_name: 模型在路由表中的名称
            model_path: 模型 checkpoint 路径
            role: 角色标签
            metrics: 性能指标 {"latency_ms": ..., "cost_per_1k_tokens": ...}
        """
        router_data = self._load()

        # 获取该角色的预设信息
        preset = ROLE_PRESETS.get(role, ROLE_PRESETS["personal_agent"])

        entry = {
            "primary": model_name,
            "fallback": self.config.teacher_model,
            "checkpoint": model_path,
            "student_architecture": self.config.student_model,
            "teacher_architecture": self.config.teacher_model,
            "kd_algorithm": self.config.kd_algorithm,
            "metrics": metrics or self._estimate_metrics(),
            "distilled_at": datetime.now().isoformat(),
            "training_config": {
                "role": role,
                "learning_rate": self.config.learning_rate,
                "num_epochs": self.config.num_epochs,
            },
        }

        router_data["models"][role] = entry
        self._save(router_data)
        logger.info(f"Registered {model_name} for role '{role}' in router table")

    def unregister(self, role: str) -> None:
        """从路由表移除某个角色的蒸馏模型"""
        router_data = self._load()
        if role in router_data["models"]:
            del router_data["models"][role]
            self._save(router_data)
            logger.info(f"Unregistered model for role '{role}'")

    def get_model_for_role(self, role: str) -> Dict[str, Any]:
        """查询某个角色当前使用的模型信息"""
        router_data = self._load()
        if role in router_data["models"]:
            return router_data["models"][role]
        return {
            "primary": router_data["default"],
            "fallback": router_data["default"],
            "checkpoint": "",
            "note": "No distilled model available, using default",
        }

    def list_models(self) -> Dict[str, Any]:
        """列出所有已注册的蒸馏模型"""
        return self._load()

    def _estimate_metrics(self) -> Dict[str, Any]:
        """估算蒸馏模型的性能指标"""
        # 基于模型参数量粗略估算
        # 7B → 2B: 约 70% 延迟降低, 80% 成本降低
        student_name = self.config.student_model.lower()
        teacher_name = self.config.teacher_model.lower()

        if "2b" in student_name or "3b" in student_name:
            latency_reduction = 0.65
            cost_reduction = 0.75
        elif "1.5b" in student_name or "1b" in student_name:
            latency_reduction = 0.80
            cost_reduction = 0.85
        elif "4b" in student_name:
            latency_reduction = 0.50
            cost_reduction = 0.55
        else:
            latency_reduction = 0.60
            cost_reduction = 0.70

        return {
            "estimated_latency_reduction": f"{latency_reduction:.0%}",
            "estimated_cost_reduction": f"{cost_reduction:.0%}",
            "note": "Estimated metrics; run benchmarks for accurate numbers",
        }

    def _ensure_router_file(self) -> None:
        """确保路由表文件存在"""
        router_path = Path(self.router_path)
        if not router_path.exists():
            router_path.parent.mkdir(parents=True, exist_ok=True)
            default_data = {
                "models": {},
                "default": self.config.teacher_model,
                "updated_at": datetime.now().isoformat(),
            }
            with open(router_path, "w", encoding="utf-8") as f:
                json.dump(default_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Created router table at {router_path}")

    def _load(self) -> Dict[str, Any]:
        with open(self.router_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, data: Dict[str, Any]) -> None:
        data["updated_at"] = datetime.now().isoformat()
        with open(self.router_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
