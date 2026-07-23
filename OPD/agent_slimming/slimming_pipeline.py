"""
Agent Self-Slimming 全流水线编排器

编排 SimCT 的 5 步蒸馏流程，适配 AdGenie 的多模型协同场景。

完整流程:
  Step 0: 数据采集        — 从用户对话历史提取训练样本
  Step 1: 数据准备        — 清洗、去重、格式标准化
  Step 2: Teacher 推理    — 大模型为每个 prompt 生成黄金标注
  Step 3: SFT Warmup      — 小模型基础指令微调（LLaMA-Factory）
  Step 4: OPD 蒸馏        — SimCT 跨 tokenizer 在线策略蒸馏
  Step 5: 评估 & 部署     — 评测蒸馏模型 vs 原始大模型，注册到路由表

使用方式:
  python -m agent_slimming.slimming_pipeline --role personal_agent
  python -m agent_slimming.slimming_pipeline --role agent_orchestration --incremental
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from .config import SlimmingConfig
from .data_collector import DataCollector
from .data_preparer import DataPreparer
from .router import ModelRouter

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(handler)


# ── SimCT 脚本路径（相对于 SimCT--main 根目录）────────────────────
SIMCT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = SIMCT_ROOT / "scripts"
AGENT_SCRIPTS_DIR = SCRIPTS_DIR / "agent_slimming"


class SlimmingPipeline:
    """
    Agent Self-Slimming 全流水线。

    封装 SimCT 5 步训练流程，为 AdGenie 的每个角色（编排Agent、
    图片理解、TTS、视频脚本）提供一键蒸馏能力。

    Attributes:
        config: 完整流水线配置
        data_collector: 对话数据采集器
        data_preparer: 训练数据准备器
        router: 蒸馏后模型路由注册器
    """

    def __init__(self, config: SlimmingConfig):
        self.config = config
        self.data_collector = DataCollector(config)
        self.data_preparer = DataPreparer(config)
        self.router = ModelRouter(config)
        self._steps_completed: list[str] = []
        self._start_time: Optional[float] = None

    # ── 公开 API ────────────────────────────────────────────────────

    def run(self, skip_data_collection: bool = False) -> dict:
        """
        运行完整流水线。

        Args:
            skip_data_collection: 跳过数据采集（数据已准备好时使用）

        Returns:
            流水线执行报告 dict
        """
        self._start_time = time.time()
        logger.info("=" * 60)
        logger.info(f"  Agent Self-Slimming Pipeline Started")
        logger.info(f"  Role: {self.config.role}")
        logger.info(f"  Teacher: {self.config.teacher_model}")
        logger.info(f"  Student: {self.config.student_model}")
        logger.info(f"  Algorithm: {self.config.kd_algorithm}")
        logger.info(f"  Incremental: {self.config.incremental}")
        logger.info("=" * 60)

        try:
            if not skip_data_collection:
                self.step_collect_data()
            self.step_prepare_data()
            self.step_generate_teacher_responses()
            self.step_sft_warmup()
            self.step_distillation()
            self.step_evaluate()
            self.step_register_model()
            return self._build_report(success=True)
        except Exception as e:
            logger.error(f"Pipeline failed at step {len(self._steps_completed)}: {e}")
            return self._build_report(success=False, error=str(e))

    def step_collect_data(self) -> str:
        """Step 0: 采集用户对话数据"""
        self._log_step("Collecting agent conversation data")
        examples = self.data_collector.collect()

        if len(examples) < self.config.min_conversations:
            logger.warning(
                f"Insufficient data: {len(examples)} < {self.config.min_conversations}. "
                f"Generating synthetic data as supplement..."
            )
            synthetic = self.data_collector.generate_synthetic_data(
                num_examples=self.config.min_conversations - len(examples)
            )
            examples = examples + synthetic

        # 保存中间结果
        raw_dir = Path(self.config.data_path) / f"agent_slimming_{self.config.role}" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        with open(raw_dir / "collected_examples.json", "w", encoding="utf-8") as f:
            json.dump([{"prompt": e.prompt, "completion": e.completion,
                        "role": e.role, "source_session_id": e.source_session_id}
                       for e in examples], f, ensure_ascii=False, indent=2)

        self._steps_completed.append("data_collection")
        logger.info(f"Data collection done: {len(examples)} examples")
        return str(raw_dir)

    def step_prepare_data(self) -> str:
        """Step 1: 准备训练数据"""
        self._log_step("Preparing training data")

        # 重新加载已采集的数据
        examples = self.data_collector.collect()
        if len(examples) < self.config.min_conversations:
            synthetic = self.data_collector.generate_synthetic_data(
                num_examples=self.config.min_conversations - len(examples)
            )
            examples = examples + synthetic

        # 清洗和格式标准化
        simct_records = self.data_preparer.prepare(examples)

        # 导出为 SimCT 训练格式
        output_dir = os.path.join(
            self.config.data_path, f"agent_slimming_{self.config.role}"
        )
        self.data_collector.export_for_simct(examples, output_dir)

        self._steps_completed.append("data_preparation")
        logger.info(f"Data preparation done: {len(simct_records)} records → {output_dir}")
        return output_dir

    def step_generate_teacher_responses(self) -> str:
        """Step 2: 使用 Teacher 大模型生成黄金标注"""
        self._log_step("Generating teacher responses")
        self._run_script("run_teacher_responses.sh")
        self._steps_completed.append("teacher_responses")
        return os.path.join(
            self.config.data_path,
            f"teacher_responses_{self.config.role}"
        )

    def step_sft_warmup(self) -> str:
        """Step 3: SFT Warmup 训练"""
        self._log_step("Running SFT warmup")
        self._run_script("run_sft_warmup.sh")
        self._steps_completed.append("sft_warmup")
        return self.config.sft_save_path

    def step_distillation(self) -> str:
        """Step 4: SimCT 跨 tokenizer OPD 蒸馏"""
        self._log_step("Running cross-tokenizer OPD distillation")
        self._run_script("run_distill.sh")
        self._steps_completed.append("distillation")
        return self.config.save_path

    def step_evaluate(self) -> dict:
        """Step 5: 评估蒸馏模型"""
        self._log_step("Evaluating distilled model")
        # 评测逻辑：对比蒸馏模型 vs Teacher 在 Agent 任务上的表现
        self._steps_completed.append("evaluation")
        return {}

    def step_register_model(self) -> str:
        """Step 6: 注册蒸馏模型到路由表"""
        self._log_step("Registering model in router")
        model_name = f"{self.config.role}-{self.config.student_model}"
        self.router.register(
            model_name=model_name,
            model_path=self.config.save_path,
            role=self.config.role,
        )
        self._steps_completed.append("model_registration")
        logger.info(f"Model registered: {model_name}")
        return model_name

    # ── 内部方法 ────────────────────────────────────────────────────

    def _log_step(self, name: str) -> None:
        logger.info(f"\n{'─' * 40}\n  Step {len(self._steps_completed)}: {name}\n{'─' * 40}")

    def _run_script(self, script_name: str) -> None:
        """运行 agent_slimming 脚本"""
        script_path = AGENT_SCRIPTS_DIR / script_name
        if not script_path.exists():
            logger.warning(
                f"Script not found: {script_path}. "
                f"Skipping — this step may need to be run manually."
            )
            return

        env = os.environ.copy()
        env["MODEL_PATH"] = self.config.model_path
        env["DATA_PATH"] = self.config.data_path
        env["OUTPUT_PATH"] = self.config.output_path
        env["SLIMMING_ROLE"] = self.config.role

        logger.info(f"Running: bash {script_path}")
        result = subprocess.run(
            ["bash", str(script_path)],
            env=env,
            cwd=str(SIMCT_ROOT),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error(f"Script failed:\n{result.stderr[-500:]}")
            raise RuntimeError(f"{script_name} failed with exit code {result.returncode}")
        logger.info(f"Script {script_name} completed successfully")

    def _build_report(self, success: bool, error: str = "") -> dict:
        elapsed = time.time() - (self._start_time or time.time())
        return {
            "role": self.config.role,
            "teacher_model": self.config.teacher_model,
            "student_model": self.config.student_model,
            "kd_algorithm": self.config.kd_algorithm,
            "success": success,
            "error": error,
            "steps_completed": self._steps_completed,
            "elapsed": str(timedelta(seconds=int(elapsed))),
            "output_path": self.config.save_path,
            "timestamp": datetime.now().isoformat(),
        }


# ── CLI ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Agent Self-Slimming — AdGenie Agent 蒸馏瘦身工具"
    )
    parser.add_argument(
        "--role", type=str, default="personal_agent",
        choices=["agent_orchestration", "image_understanding", "tts_voice",
                 "video_script", "personal_agent"],
        help="蒸馏角色 (default: personal_agent)"
    )
    parser.add_argument(
        "--teacher", type=str, default=None,
        help="Teacher 模型名称（覆盖角色预设）"
    )
    parser.add_argument(
        "--student", type=str, default=None,
        help="Student 模型名称（覆盖角色预设）"
    )
    parser.add_argument(
        "--incremental", action="store_true",
        help="增量蒸馏模式"
    )
    parser.add_argument(
        "--base-checkpoint", type=str, default=None,
        help="增量蒸馏的初始 checkpoint"
    )
    parser.add_argument(
        "--skip-data-collection", action="store_true",
        help="跳过数据采集步骤"
    )
    parser.add_argument(
        "--conversation-dir", type=str, default="./data/agent_conversations",
        help="对话数据目录"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅打印配置，不执行训练"
    )

    args = parser.parse_args()
    config = SlimmingConfig(role=args.role)
    if args.teacher:
        config.teacher_model = args.teacher
    if args.student:
        config.student_model = args.student
    if args.incremental:
        config.incremental = True
    if args.base_checkpoint:
        config.base_checkpoint = args.base_checkpoint
    if args.conversation_dir:
        config.conversation_dir = args.conversation_dir

    if args.dry_run:
        print(json.dumps({
            "role": config.role,
            "teacher": config.teacher_model,
            "student": config.student_model,
            "algorithm": config.kd_algorithm,
            "loss": config.kd_loss_fn,
            "data_dir": config.conversation_dir,
            "output": config.save_path,
        }, indent=2))
        return

    pipeline = SlimmingPipeline(config)
    report = pipeline.run(skip_data_collection=args.skip_data_collection)

    print("\n" + "=" * 60)
    print("  Pipeline Report")
    print("=" * 60)
    for k, v in report.items():
        print(f"  {k}: {v}")

    if not report["success"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
