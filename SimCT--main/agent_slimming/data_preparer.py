"""
训练数据准备器

将 DataCollector 采集的原始训练样本转换为 SimCT 蒸馏流水线所需格式。

SimCT 支持的训练数据格式：
  - 每条记录: {"messages": [...], "label": "ground truth completion"}
  - messages 中每条的格式: {"role": "system"|"user"|"assistant", "content": "..."}

还负责:
  - 数据清洗（去重、截断、质量过滤）
  - 难例挖掘（用户修改后重新生成的样本权重大）
  - 格式标准化（ChatML / ShareGPT 格式转换）
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

from .config import SlimmingConfig
from .data_collector import TrainingExample

logger = logging.getLogger(__name__)


class DataPreparer:
    """
    将原始对话数据转换为 SimCT 训练格式。

    管道: raw JSONL → TrainingExample → 清洗 → 去重 → 格式标准化 → SimCT 格式
    """

    def __init__(self, config: SlimmingConfig):
        self.config = config

    def prepare(self, examples: List[TrainingExample]) -> List[Dict[str, Any]]:
        """
        完整的准备流水线。

        Returns:
            List of dicts with "messages" key, ready for SimCT training.
        """
        logger.info(f"Preparing {len(examples)} raw examples...")

        # Step 1: 过滤空 completion（仅保留有 ground truth 的样本）
        examples = self._filter_empty_completions(examples)
        logger.info(f"  After filtering empty: {len(examples)}")

        # Step 2: 去重
        examples = self._deduplicate(examples)
        logger.info(f"  After dedup: {len(examples)}")

        # Step 3: 质量过滤
        examples = self._quality_filter(examples)
        logger.info(f"  After quality filter: {len(examples)}")

        # Step 4: 截断超长样本
        examples = self._truncate_long(examples)
        logger.info(f"  After truncation: {len(examples)}")

        # Step 5: 转换为 SimCT 格式
        simct_data = self._to_simct_format(examples)
        logger.info(f"  Final SimCT records: {len(simct_data)}")

        return simct_data

    def _filter_empty_completions(
        self, examples: List[TrainingExample]
    ) -> List[TrainingExample]:
        """过滤没有 completion 的样本（这些需要 teacher 先标注）"""
        return [e for e in examples if e.completion.strip()]

    def _deduplicate(
        self, examples: List[TrainingExample]
    ) -> List[TrainingExample]:
        """基于 prompt+completion 的 MD5 去重"""
        seen: Set[str] = set()
        unique: List[TrainingExample] = []
        for ex in examples:
            key = hashlib.md5(
                (ex.prompt + ex.completion).encode("utf-8")
            ).hexdigest()
            if key not in seen:
                seen.add(key)
                unique.append(ex)
        return unique

    def _quality_filter(
        self, examples: List[TrainingExample]
    ) -> List[TrainingExample]:
        """
        质量过滤规则:
        - completion 长度至少 20 字符
        - prompt 长度至少 10 字符
        - 不包含明显的截断标记
        """
        filtered = []
        for ex in examples:
            if len(ex.completion) < 20:
                continue
            if len(ex.prompt) < 10:
                continue
            # 过滤明显的截断（以 "..." 或 "[truncated]" 结尾）
            if ex.completion.rstrip().endswith("[truncated]"):
                continue
            filtered.append(ex)
        return filtered

    def _truncate_long(
        self, examples: List[TrainingExample]
    ) -> List[TrainingExample]:
        """
        截断超过 max_seq_length 的样本。
        保留前面部分（prompt 头部 + completion 核心内容）。
        """
        max_len = self.config.max_seq_length
        truncated = []
        for ex in examples:
            total_chars = len(ex.prompt) + len(ex.completion)
            if total_chars <= max_len * 4:  # 粗略估计: 4 chars/token
                truncated.append(ex)
            else:
                # 保留 prompt 完整 + completion 前部
                available = max_len * 4 - len(ex.prompt)
                if available > 200:
                    ex.completion = ex.completion[:available] + "\n[truncated]"
                    truncated.append(ex)
        return truncated

    def _to_simct_format(
        self, examples: List[TrainingExample]
    ) -> List[Dict[str, Any]]:
        """
        转换为 SimCT 训练格式:

        {
            "messages": [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}  # 可选，SFT 时使用
            ],
            "label": "ground truth completion",  # 用于 teacher response 的对比
            "metadata": {
                "source_session_id": "...",
                "role": "..."
            }
        }
        """
        records = []
        for ex in examples:
            record = {
                "messages": self._parse_chatml_messages(ex.prompt, ex.completion),
                "label": ex.completion,
                "metadata": {
                    "source_session_id": ex.source_session_id,
                    "role": ex.role,
                },
            }
            records.append(record)
        return records

    @staticmethod
    def _parse_chatml_messages(
        prompt: str, completion: str
    ) -> List[Dict[str, str]]:
        """
        从 ChatML 格式的 prompt 中提取 messages。

        prompt 格式:
          <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n

        返回:
          [{"role": "system", "content": "..."},
           {"role": "user", "content": "..."},
           {"role": "assistant", "content": "..."}]
        """
        messages = []

        # 解析 system
        if "<|im_start|>system\n" in prompt:
            sys_start = prompt.find("<|im_start|>system\n") + len("<|im_start|>system\n")
            sys_end = prompt.find("<|im_end|>", sys_start)
            if sys_end > sys_start:
                messages.append({
                    "role": "system",
                    "content": prompt[sys_start:sys_end].strip(),
                })

        # 解析 user
        if "<|im_start|>user\n" in prompt:
            usr_start = prompt.find("<|im_start|>user\n") + len("<|im_start|>user\n")
            usr_end = prompt.find("<|im_end|>", usr_start)
            if usr_end > usr_start:
                messages.append({
                    "role": "user",
                    "content": prompt[usr_start:usr_end].strip(),
                })

        # 解析 assistant (completion)
        if completion.strip():
            messages.append({
                "role": "assistant",
                "content": completion.strip(),
            })

        return messages

    def prepare_for_sft(
        self, examples: List[TrainingExample], teacher_responses_dir: str
    ) -> List[Dict[str, Any]]:
        """
        准备 SFT 数据：将 teacher 生成的回复与用户 prompt 配对。

        teacher_responses_dir: teacher 生成的回复目录
          (由 SimCT Step 2 生成，格式: {prompt_hash}.json)

        Returns:
            SFT 格式的训练数据
        """
        response_dir = Path(teacher_responses_dir)
        if not response_dir.exists():
            logger.warning(f"Teacher responses dir not found: {teacher_responses_dir}")
            return self.prepare(examples)

        sft_data = []
        for ex in examples:
            prompt_hash = hashlib.md5(ex.prompt.encode("utf-8")).hexdigest()
            response_file = response_dir / f"{prompt_hash}.json"

            if response_file.exists():
                with open(response_file, "r", encoding="utf-8") as f:
                    teacher_data = json.load(f)
                teacher_completion = teacher_data.get("response", ex.completion)
            else:
                teacher_completion = ex.completion

            messages = self._parse_chatml_messages(ex.prompt, teacher_completion)
            sft_data.append({"messages": messages})

        logger.info(f"Prepared {len(sft_data)} SFT examples with teacher responses")
        return sft_data
