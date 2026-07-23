"""
Agent 对话数据采集器

从 AdGenie 的用户对话历史中提取 Agent 推理轨迹，
按角色（编排/图片理解/TTS/视频脚本）分类存储。

输入格式: JSONL 文件，每行一条对话记录
输出格式: SimCT 训练所需的 prompt-completion 对
"""
from __future__ import annotations

import json
import logging
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterator

from .config import SlimmingConfig
from .templates.agent_prompts import (
    AGENT_CONVERSATION_TEMPLATES,
    TOOL_CALL_MARKERS,
)

logger = logging.getLogger(__name__)


@dataclass
class ConversationRecord:
    """单条 Agent 对话记录"""
    session_id: str
    timestamp: str
    messages: List[Dict[str, Any]]
    skill_used: Optional[str] = None
    tools_called: List[str] = field(default_factory=list)
    total_tokens: int = 0
    user_feedback: str = "accepted"

    @classmethod
    def from_jsonl_line(cls, line: str) -> "ConversationRecord":
        data = json.loads(line.strip())
        return cls(
            session_id=data.get("session_id", ""),
            timestamp=data.get("timestamp", ""),
            messages=data.get("messages", []),
            skill_used=data.get("skill_used"),
            tools_called=data.get("tools_called", []),
            total_tokens=data.get("total_tokens", 0),
            user_feedback=data.get("user_feedback", "accepted"),
        )


@dataclass
class TrainingExample:
    """单条训练样本"""
    prompt: str
    """用户输入 + system prompt"""
    completion: str
    """Agent 完整回复（含 reasoning + tool_calls + final_response）"""
    role: str
    source_session_id: str


class DataCollector:
    """
    从 AdGenie 对话历史中采集训练数据。

    支持:
    - 按角色过滤（agent_orchestration / image_understanding / tts_voice / video_script）
    - 按用户反馈过滤（只保留 accepted / 包含 modified 作为难例）
    - 最小对话数检查
    - 数据统计和报告
    """

    def __init__(self, config: SlimmingConfig):
        self.config = config
        self.conversation_dir = Path(config.conversation_dir)
        self.role = config.role

    def collect(self) -> List[TrainingExample]:
        """主入口：采集并返回训练样本列表"""
        records = self._load_conversations()
        logger.info(f"Loaded {len(records)} conversation records for role={self.role}")

        if len(records) < self.config.min_conversations:
            logger.warning(
                f"Only {len(records)} conversations found, "
                f"minimum required is {self.config.min_conversations}. "
                f"Consider generating synthetic data or collecting more usage."
            )

        examples = self._convert_to_training_examples(records)
        self._print_stats(records, examples)
        return examples

    def _load_conversations(self) -> List[ConversationRecord]:
        """加载对话 JSONL 文件"""
        records: List[ConversationRecord] = []
        if not self.conversation_dir.exists():
            logger.warning(f"Conversation directory not found: {self.conversation_dir}")
            return records

        jsonl_files = list(self.conversation_dir.glob("*.jsonl"))
        if not jsonl_files:
            # 检查子目录
            jsonl_files = list(self.conversation_dir.rglob("*.jsonl"))

        for jsonl_file in jsonl_files:
            try:
                with open(jsonl_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = ConversationRecord.from_jsonl_line(line)
                            records.append(record)
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.debug(f"Skipping malformed line in {jsonl_file}: {e}")
            except Exception as e:
                logger.warning(f"Error reading {jsonl_file}: {e}")

        # 限制最大数量
        if len(records) > self.config.max_conversations:
            records = random.sample(records, self.config.max_conversations)

        return records

    def _convert_to_training_examples(
        self, records: List[ConversationRecord]
    ) -> List[TrainingExample]:
        """将对话记录转换为训练样本"""
        template = AGENT_CONVERSATION_TEMPLATES.get(
            self.role, AGENT_CONVERSATION_TEMPLATES["personal_agent"]
        )
        system_prompt = template["system"]
        examples: List[TrainingExample] = []

        for record in records:
            # 过滤：只保留用户接受或修改后接受的对话
            if record.user_feedback not in ("accepted", "modified"):
                continue

            # 提取 user ↔ assistant 交互对
            user_messages = []
            assistant_responses = []

            for msg in record.messages:
                if msg.get("role") == "user":
                    user_messages.append(msg.get("content", ""))
                elif msg.get("role") == "assistant":
                    assistant_responses.append(msg.get("content", ""))

            # 配对：每个 user message + 对应的 assistant response
            for i, (user_msg, asst_msg) in enumerate(
                zip(user_messages, assistant_responses)
            ):
                if not user_msg.strip() or not asst_msg.strip():
                    continue

                # 构造 prompt：system + user message
                prompt = self._build_prompt(system_prompt, user_msg)
                # completion = agent 完整回复
                completion = asst_msg

                examples.append(TrainingExample(
                    prompt=prompt,
                    completion=completion,
                    role=self.role,
                    source_session_id=record.session_id,
                ))

        return examples

    def _build_prompt(self, system_prompt: str, user_message: str) -> str:
        """构造训练 prompt（ChatML 格式）"""
        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_message}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _print_stats(
        self, records: List[ConversationRecord], examples: List[TrainingExample]
    ) -> None:
        """打印数据统计报告"""
        total_tokens = sum(r.total_tokens for r in records)
        feedback_dist = defaultdict(int)
        skill_dist = defaultdict(int)
        tool_dist = defaultdict(int)

        for r in records:
            feedback_dist[r.user_feedback] += 1
            if r.skill_used:
                skill_dist[r.skill_used] += 1
            for tool in r.tools_called:
                tool_dist[tool] += 1

        logger.info("=" * 60)
        logger.info(f"  Data Collection Report — {self.role}")
        logger.info("=" * 60)
        logger.info(f"  Conversations:  {len(records)}")
        logger.info(f"  Training examples: {len(examples)}")
        logger.info(f"  Total tokens:    {total_tokens:,}")
        logger.info(f"  Avg tokens/conv: {total_tokens // max(len(records), 1):,}")
        logger.info(f"  Feedback dist:   {dict(feedback_dist)}")
        logger.info(f"  Skills used:     {dict(skill_dist)}")
        logger.info(f"  Top tools:       {dict(sorted(tool_dist.items(), key=lambda x: -x[1])[:5])}")
        logger.info("=" * 60)

    def generate_synthetic_data(self, num_examples: int = 100) -> List[TrainingExample]:
        """
        当真实数据不足时，基于模板生成合成数据用于初始蒸馏。
        这样即使用户没有足够历史对话，也能完成第一次蒸馏。
        """
        template = AGENT_CONVERSATION_TEMPLATES.get(
            self.role, AGENT_CONVERSATION_TEMPLATES["personal_agent"]
        )
        system_prompt = template["system"]
        user_templates = template["example_user_messages"]

        examples = []
        for i in range(num_examples):
            user_msg = random.choice(user_templates)
            prompt = self._build_prompt(system_prompt, user_msg)
            examples.append(TrainingExample(
                prompt=prompt,
                completion="",  # 合成数据没有 completion，需要 teacher 生成
                role=self.role,
                source_session_id=f"synthetic_{i}",
            ))
        return examples

    def export_for_simct(self, examples: List[TrainingExample], output_dir: str) -> str:
        """
        导出为 SimCT/KDFlow 训练格式。

        输出目录结构：
          output_dir/
            train.jsonl    — 训练集 prompt
            eval.jsonl     — 验证集 prompt
            metadata.json  — 数据集元信息
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        random.shuffle(examples)
        split_idx = int(len(examples) * self.config.train_split)
        train_examples = examples[:split_idx]
        eval_examples = examples[split_idx:]

        # 写入 train.jsonl（SimCT 需要的格式：{"messages": [...]}）
        self._write_jsonl(output_path / "train.jsonl", train_examples)
        self._write_jsonl(output_path / "eval.jsonl", eval_examples)

        # 写入元信息
        metadata = {
            "role": self.role,
            "num_train": len(train_examples),
            "num_eval": len(eval_examples),
            "teacher_model": self.config.teacher_model,
            "student_model": self.config.student_model,
            "collected_at": "",
        }
        with open(output_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(
            f"Exported {len(train_examples)} train + {len(eval_examples)} eval "
            f"examples to {output_dir}"
        )
        return str(output_path)

    @staticmethod
    def _write_jsonl(path: Path, examples: List[TrainingExample]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                record = {
                    "messages": [
                        {"role": "system", "content": ""},
                        {"role": "user", "content": ex.prompt},
                    ],
                    "label": ex.completion,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
