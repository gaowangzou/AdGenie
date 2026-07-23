#!/bin/bash
# ============================================================================
# Agent Self-Slimming — 完整蒸馏流水线
# ============================================================================
# 一键运行从数据采集到模型部署的全流程。
#
# 使用方式:
#   bash run_full_pipeline.sh                                    # 默认：用户专属Agent
#   bash run_full_pipeline.sh --role agent_orchestration         # 编排Agent瘦身
#   bash run_full_pipeline.sh --role image_understanding         # 图片理解模型瘦身
#   bash run_full_pipeline.sh --role tts_voice                   # TTS语音模型瘦身
#   bash run_full_pipeline.sh --role video_script                # 视频脚本模型瘦身
#   bash run_full_pipeline.sh --role personal_agent --incremental # 增量蒸馏
# ============================================================================

set -e

# ── 默认配置 ────────────────────────────────────────────────────────
ROLE="${SLIMMING_ROLE:-personal_agent}"
MODEL_PATH="${MODEL_PATH:-./models}"
DATA_PATH="${DATA_PATH:-./data}"
OUTPUT_PATH="${OUTPUT_PATH:-./output/ckpts}"
CONVERSATION_DIR="${CONVERSATION_DIR:-./data/agent_conversations}"
INCREMENTAL=""
BASE_CHECKPOINT=""
DRY_RUN=""

# ── 解析参数 ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --role) ROLE="$2"; shift 2 ;;
        --teacher) TEACHER_OVERRIDE="$2"; shift 2 ;;
        --student) STUDENT_OVERRIDE="$2"; shift 2 ;;
        --incremental) INCREMENTAL="--incremental"; shift ;;
        --base-checkpoint) BASE_CHECKPOINT="--base-checkpoint $2"; shift 2 ;;
        --conversation-dir) CONVERSATION_DIR="$2"; shift 2 ;;
        --dry-run) DRY_RUN="--dry-run"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── 环境变量 ────────────────────────────────────────────────────────
export MODEL_PATH DATA_PATH OUTPUT_PATH

echo "=========================================="
echo "  Agent Self-Slimming Pipeline"
echo "=========================================="
echo "  Role:       ${ROLE}"
echo "  Model Path: ${MODEL_PATH}"
echo "  Data Path:  ${DATA_PATH}"
echo "  Output:     ${OUTPUT_PATH}"
echo "  Incremental: ${INCREMENTAL:-false}"
echo "=========================================="
echo ""

# ── 检查前置条件 ────────────────────────────────────────────────────
if [ ! -d "${MODEL_PATH}" ]; then
    echo "⚠ Model directory not found: ${MODEL_PATH}"
    echo "  Please download teacher and student model weights first."
    echo "  Teacher: Qwen2.5-7B-Instruct (or equivalent)"
    echo "  Student: gemma-2-2b-it (or equivalent)"
    exit 1
fi

# ── 确保对话数据目录存在 ────────────────────────────────────────────
if [ ! -d "${CONVERSATION_DIR}" ]; then
    echo "⚠ Conversation directory not found: ${CONVERSATION_DIR}"
    echo "  Creating empty directory. Add your AdGenie conversation"
    echo "  JSONL files here for personalized distillation."
    mkdir -p "${CONVERSATION_DIR}"
fi

# ── Step 0 + 1: 数据采集和准备 ──────────────────────────────────────
echo ""
echo "━━━ Step 0+1: Data Collection & Preparation ━━━"
python -m agent_slimming.data_collector 2>/dev/null || \
python -c "
from agent_slimming.data_collector import DataCollector
from agent_slimming.config import SlimmingConfig

config = SlimmingConfig(role='${ROLE}')
config.conversation_dir = '${CONVERSATION_DIR}'
collector = DataCollector(config)
examples = collector.collect()
print(f'Collected {len(examples)} examples')

# 导出为 SimCT 格式
output_dir = '${DATA_PATH}/agent_slimming_${ROLE}'
collector.export_for_simct(examples, output_dir)
print(f'Exported to {output_dir}')
"

echo "✓ Data preparation complete"

# ── Step 2: Teacher 响应生成 ────────────────────────────────────────
echo ""
echo "━━━ Step 2: Teacher Response Generation ━━━"
echo "  Generating gold responses using ${TEACHER_OVERRIDE:-Qwen2.5-7B-Instruct}..."
echo "  (This step requires SGLang and GPU resources)"
echo "  → Run manually: bash scripts/agent_slimming/run_teacher_responses.sh"

# ── Step 3: SFT Warmup ──────────────────────────────────────────────
echo ""
echo "━━━ Step 3: SFT Warmup ━━━"
echo "  Training student model on teacher responses..."
echo "  (This step requires LLaMA-Factory and GPU resources)"
echo "  → Run manually: bash scripts/agent_slimming/run_sft_warmup.sh"

# ── Step 4: OPD 蒸馏 ────────────────────────────────────────────────
echo ""
echo "━━━ Step 4: Cross-Tokenizer OPD Distillation ━━━"
echo "  Distilling teacher → student using SimCT span_ctkd..."
echo "  → Run manually: bash scripts/agent_slimming/run_distill.sh"

# ── 流水线完成 ──────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Pipeline preparation complete!"
echo "=========================================="
echo ""
echo "  Next steps (run on GPU cluster):"
echo "  1. bash scripts/agent_slimming/run_teacher_responses.sh"
echo "  2. bash scripts/agent_slimming/run_sft_warmup.sh"
echo "  3. bash scripts/agent_slimming/run_distill.sh"
echo ""
echo "  Or use the Python pipeline directly:"
echo "  python -m agent_slimming.slimming_pipeline --role ${ROLE} ${INCREMENTAL}"
echo ""
