#!/bin/bash
# ============================================================================
# Step 2: 生成 Teacher 响应
# 使用 SGLang 启动 Teacher 大模型，为每个训练 prompt 生成 8 条备选回复。
# ============================================================================
set -e

ROLE="${SLIMMING_ROLE:-personal_agent}"
MODEL_PATH="${MODEL_PATH:-./models}"
DATA_PATH="${DATA_PATH:-./data}"
OUTPUT_PATH="${OUTPUT_PATH:-./output/ckpts}"

# 根据角色确定 teacher 模型
case "${ROLE}" in
    agent_orchestration) TEACHER="Qwen2.5-7B-Instruct" ;;
    image_understanding) TEACHER="Qwen2.5-VL-7B-Instruct" ;;
    tts_voice)           TEACHER="Qwen2.5-7B-Instruct" ;;
    video_script)        TEACHER="Qwen2.5-7B-Instruct" ;;
    personal_agent)      TEACHER="Qwen2.5-7B-Instruct" ;;
    *)                   TEACHER="Qwen2.5-7B-Instruct" ;;
esac

TRAIN_DATA="${DATA_PATH}/agent_slimming_${ROLE}/train.jsonl"
EVAL_DATA="${DATA_PATH}/agent_slimming_${ROLE}/eval.jsonl"
OUTPUT_DIR="${DATA_PATH}/teacher_responses_${ROLE}"

echo "Role: ${ROLE}"
echo "Teacher: ${TEACHER}"
echo "Train data: ${TRAIN_DATA}"
echo "Output: ${OUTPUT_DIR}"

if [ ! -f "${TRAIN_DATA}" ]; then
    echo "Error: Training data not found at ${TRAIN_DATA}"
    echo "Run data preparation first: python -m agent_slimming.slimming_pipeline --role ${ROLE}"
    exit 1
fi

# SGLang 参数
TEMPERATURE=0.6
TOP_P=0.95
MAX_TOKENS=4096
NUM_SAMPLES=8
DP_SIZE=8

echo "Starting SGLang server (DP=${DP_SIZE})..."
echo "This requires ${DP_SIZE} GPUs for teacher inference."

# 实际使用时取消注释:
# python -m sglang.launch_server \
#     --model-path "${MODEL_PATH}/${TEACHER}" \
#     --dp "${DP_SIZE}" \
#     --mem-fraction-static 0.5 \
#     --context-length 32768 &

# sleep 30  # 等待服务器启动

# python scripts/sft/generate_responses.py \
#     --input "${TRAIN_DATA}" \
#     --output "${OUTPUT_DIR}" \
#     --temperature "${TEMPERATURE}" \
#     --top-p "${TOP_P}" \
#     --max-tokens "${MAX_TOKENS}" \
#     --n "${NUM_SAMPLES}" \
#     --base-url "http://127.0.0.1:30000"

echo "Teacher response generation complete: ${OUTPUT_DIR}"
