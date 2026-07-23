"""
Agent Self-Slimming 配置

定义蒸馏流水线的全部可配置参数，覆盖 AdGenie 多模型协同场景：
  - agent_orchestration: 编排 Agent 模型（200步递归推理）
  - image_understanding: 多模态图片理解模型
  - tts_voice: TTS 语音合成模型
  - video_script: 视频脚本生成模型
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


# ── 预设的蒸馏目标配置 ──────────────────────────────────────────────
# 每个角色有不同的精度/延迟/成本要求，蒸馏到不同架构的小模型上
ROLE_PRESETS = {
    "agent_orchestration": {
        "description": "编排Agent（200步递归推理），核心瓶颈是推理延迟和token成本",
        "teacher_model": "Qwen2.5-7B-Instruct",
        "student_model": "gemma-2-2b-it",
        "priority": "latency",  # 优先降低延迟
        "target_tokens_per_second": 50,
        "target_cost_reduction": 0.70,
    },
    "image_understanding": {
        "description": "多模态图片理解，核心瓶颈是视觉编码器计算量",
        "teacher_model": "Qwen2.5-VL-7B-Instruct",
        "student_model": "Qwen2.5-VL-2B-Instruct",
        "priority": "throughput",
        "target_tokens_per_second": 30,
        "target_cost_reduction": 0.60,
    },
    "tts_voice": {
        "description": "TTS语音合成辅助模型，轻量即可满足需求",
        "teacher_model": "Qwen2.5-7B-Instruct",
        "student_model": "Qwen2.5-1.5B-Instruct",
        "priority": "cost",
        "target_tokens_per_second": 100,
        "target_cost_reduction": 0.85,
    },
    "video_script": {
        "description": "视频分镜脚本生成，需要创意但不需要复杂推理",
        "teacher_model": "Qwen2.5-7B-Instruct",
        "student_model": "Phi-4-mini-instruct",
        "priority": "balanced",
        "target_tokens_per_second": 40,
        "target_cost_reduction": 0.65,
    },
    "personal_agent": {
        "description": "用户专属Agent，学习个人使用习惯和偏好",
        "teacher_model": "Qwen2.5-7B-Instruct",
        "student_model": "gemma-2-2b-it",
        "priority": "personalization",
        "target_tokens_per_second": 60,
        "target_cost_reduction": 0.75,
    },
}


@dataclass
class SlimmingConfig:
    """Agent Self-Slimming 完整配置"""

    # ── 角色选择 ──────────────────────────────────────────────────
    role: str = "personal_agent"
    """蒸馏角色: agent_orchestration | image_understanding | tts_voice | video_script | personal_agent"""

    # ── 模型配置 ──────────────────────────────────────────────────
    teacher_model: str = "Qwen2.5-7B-Instruct"
    student_model: str = "gemma-2-2b-it"
    model_path: str = field(default_factory=lambda: os.getenv("MODEL_PATH", "./models"))
    output_path: str = field(default_factory=lambda: os.getenv("OUTPUT_PATH", "./output/ckpts"))
    data_path: str = field(default_factory=lambda: os.getenv("DATA_PATH", "./data"))

    # ── 数据配置 ──────────────────────────────────────────────────
    conversation_dir: str = "./data/agent_conversations"
    """AdGenie 用户对话历史目录（JSONL 格式）"""
    min_conversations: int = 50
    """最少需要的对话条数才触发蒸馏"""
    max_conversations: int = 5000
    train_split: float = 0.9
    max_seq_length: int = 8192

    # ── 蒸馏算法配置 ──────────────────────────────────────────────
    kd_algorithm: Literal["simple_ctkd", "span_ctkd", "span_ctkd_1to1"] = "span_ctkd"
    kd_loss_fn: Literal["kl", "rkl", "js", "tvd", "skewed_kl", "skewed_rkl"] = "rkl"
    kd_ratio: float = 0.9
    """KD loss 占比，1-kd_ratio 为 CE loss"""

    # ── 训练配置 ──────────────────────────────────────────────────
    num_epochs: int = 1
    learning_rate: float = 5e-7
    lr_warmup_ratio: float = 0.05
    train_batch_size: int = 64
    micro_train_batch_size: int = 1
    num_gpus_per_node: int = 8
    num_nodes: int = 1

    # ── Rollout 配置（On-Policy 蒸馏）────────────────────────────
    rollout_batch_size: int = 64
    rollout_num_engines: int = 8
    n_samples_per_prompt: int = 1
    temperature: float = 0.6
    top_p: float = 0.95
    generate_max_len: int = 4096

    # ── SFT Warmup 配置 ──────────────────────────────────────────
    sft_learning_rate: float = 2e-6
    sft_num_epochs: int = 3
    sft_batch_size: int = 32

    # ── Teacher 配置 ─────────────────────────────────────────────
    teacher_dp_size: int = 8
    teacher_tp_size: int = 1
    teacher_mem_fraction_static: float = 0.5
    teacher_context_length: int = 32768

    # ── 日志与保存 ───────────────────────────────────────────────
    logging_steps: int = 5
    save_steps: int = 50
    use_wandb: bool = False
    wandb_project: str = "agent-self-slimming"

    # ── 增量蒸馏配置 ─────────────────────────────────────────────
    incremental: bool = False
    """是否增量蒸馏（基于已有 checkpoint 继续训练）"""
    base_checkpoint: Optional[str] = None
    """增量蒸馏的起始 checkpoint 路径"""
    feedback_threshold: int = 500
    """累积多少条新反馈后触发增量蒸馏"""

    def __post_init__(self):
        """应用角色预设"""
        if self.role in ROLE_PRESETS:
            preset = ROLE_PRESETS[self.role]
            if self.teacher_model == "Qwen2.5-7B-Instruct":
                self.teacher_model = preset["teacher_model"]
            if self.student_model == "gemma-2-2b-it":
                self.student_model = preset["student_model"]

    @property
    def save_path(self) -> str:
        return os.path.join(
            self.output_path,
            f"{self.role}-{self._model_tag(self.teacher_model)}-to-{self._model_tag(self.student_model)}-lr{self.learning_rate}",
        )

    @property
    def sft_save_path(self) -> str:
        return os.path.join(
            self.output_path,
            f"{self.role}-sft-warmup-{self._model_tag(self.teacher_model)}-lr{self.sft_learning_rate}",
        )

    @staticmethod
    def _model_tag(name: str) -> str:
        return name.lower().replace("/", "-").replace("_", "-")[:30]

    def to_cli_args(self) -> str:
        """转换为 kdflow CLI 参数字符串"""
        args = []
        args.append(f"--num_nodes {self.num_nodes}")
        args.append(f"--num_gpus_per_node {self.num_gpus_per_node}")
        args.append("--backend fsdp2")
        args.append(f"--train_batch_size {self.train_batch_size}")
        args.append(f"--micro_train_batch_size {self.micro_train_batch_size}")
        args.append(f"--learning_rate {self.learning_rate}")
        args.append(f"--lr_warmup_ratio {self.lr_warmup_ratio}")
        args.append(f"--num_epochs {self.num_epochs}")
        args.append(f"--save_path {self.save_path}")
        args.append("--bf16 True")
        args.append("--gradient_checkpointing True")
        args.append("--enable_sleep True")
        args.append(f"--student_name_or_path ${{MODEL_PATH}}/{self.student_model}")
        args.append(f"--teacher_name_or_path ${{MODEL_PATH}}/{self.teacher_model}")
        args.append("--enable_thinking False")
        args.append(f"--rollout_batch_size {self.rollout_batch_size}")
        args.append(f"--rollout_num_engines {self.rollout_num_engines}")
        args.append("--rollout_tp_size 1")
        args.append("--rollout_mem_fraction_static 0.6")
        args.append(f"--n_samples_per_prompt {self.n_samples_per_prompt}")
        args.append(f"--generate_max_len {self.generate_max_len}")
        args.append(f"--temperature {self.temperature}")
        args.append(f"--train_dataset_path ${{DATA_PATH}}/agent_slimming_{self.role}")
        args.append(f"--max_len {self.max_seq_length}")
        args.append("--input_key messages")
        args.append("--apply_chat_template True")
        args.append("--preprocess_num_workers 8")
        args.append("--packing_samples True")
        args.append(f"--kd_ratio {self.kd_ratio}")
        args.append(f"--kd_loss_fn {self.kd_loss_fn}")
        args.append(f"--kd_algorithm {self.kd_algorithm}")
        args.append(f"--teacher_dp_size {self.teacher_dp_size}")
        args.append(f"--teacher_tp_size {self.teacher_tp_size}")
        args.append(f"--teacher_mem_fraction_static {self.teacher_mem_fraction_static}")
        args.append(f"--teacher_context_length {self.teacher_context_length}")
        args.append(f"--logging_steps {self.logging_steps}")
        args.append(f"--save_steps {self.save_steps}")
        if not self.use_wandb:
            args.append("--use_wandb False")
        return " ".join(args)
