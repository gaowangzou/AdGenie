#!/bin/bash
# ============================================================================
# Step 0+1: Agent 对话数据采集与训练数据准备
# 从 AdGenie 用户对话历史中提取训练样本，清洗并导出为 SimCT 格式。
# ============================================================================
set -e

ROLE="${SLIMMING_ROLE:-personal_agent}"
MODEL_PATH="${MODEL_PATH:-./models}"
DATA_PATH="${DATA_PATH:-./data}"
CONVERSATION_DIR="${CONVERSATION_DIR:-./data/agent_conversations}"

echo "=========================================="
echo "  Agent Data Collection & Preparation"
echo "=========================================="
echo "  Role:             ${ROLE}"
echo "  Conversation Dir: ${CONVERSATION_DIR}"
echo "  Output Dir:       ${DATA_PATH}/agent_slimming_${ROLE}"
echo "=========================================="

python -c "
from agent_slimming.data_collector import DataCollector
from agent_slimming.data_preparer import DataPreparer
from agent_slimming.config import SlimmingConfig

config = SlimmingConfig(role='${ROLE}')
config.conversation_dir = '${CONVERSATION_DIR}'

collector = DataCollector(config)
examples = collector.collect()

if len(examples) < config.min_conversations:
    print(f'Only {len(examples)} real conversations. Generating synthetic supplement...')
    synthetic = collector.generate_synthetic_data(config.min_conversations - len(examples))
    examples = examples + synthetic
    print(f'Total after synthetic: {len(examples)}')

preparer = DataPreparer(config)
cleaned = preparer.prepare(examples)

for ex in cleaned[:3]:
    msgs = ex.get('messages', [])
    preview = ' → '.join([f\"{m['role']}: {m['content'][:50]}...\" for m in msgs[:3]])
    print(f'  Sample: {preview}')

collector.export_for_simct(examples, '${DATA_PATH}/agent_slimming_${ROLE}')
"

echo ""
echo "Done! Data ready at ${DATA_PATH}/agent_slimming_${ROLE}/"
