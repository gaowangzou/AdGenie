#!/bin/bash
# ============================================================================
# Step 3: SFT Warmup 训练
# 使用 LLaMA-Factory 对 Student 模型进行基础指令微调。
# Teacher 生成的回复作为训练目标，让 Student 学会基础的 Agent 指令遵循。
# ============================================================================
set -e

ROLE="${SLIMMING_ROLE:-personal_agent}"
MODEL_PATH="${MODEL_PATH:-./models}"
DATA_PATH="${DATA_PATH:-./data}"
OUTPUT_PATH="${OUTPUT_PATH:-./output/ckpts}"

case "${ROLE}" in
    agent_orchestration) STUDENT="gemma-2-2b-it"   TEACHER="Qwen2.5-7B-Instruct" ;;
    image_understanding) STUDENT="Qwen2.5-VL-2B-Instruct" TEACHER="Qwen2.5-VL-7B-Instruct" ;;
    tts_voice)           STUDENT="Qwen2.5-1.5B-Instruct"  TEACHER="Qwen2.5-7B-Instruct" ;;
    video_script)        STUDENT="Phi-4-mini-instruct"    TEACHER="Qwen2.5-7B-Instruct" ;;
    personal_agent)      STUDENT="gemma-2-2b-it"          TEACHER="Qwen2.5-7B-Instruct" ;;
    *)                   STUDENT="gemma-2-2b-it"          TEACHER="Qwen2.5-7B-Instruct" ;;
esac

SFT_OUTPUT="${OUTPUT_PATH}/${ROLE}-sft-warmup-lr2e-6"

echo "Role: ${ROLE}"
echo "Student: ${STUDENT}"
echo "Teacher: ${TEACHER}"
echo "Output: ${SFT_OUTPUT}"
echo ""
echo "SFT Warmup training requires LLaMA-Factory."
echo "Please run the following manually:"
echo ""
echo "  llamafactory-cli train \\"
echo "    --model_name_or_path ${MODEL_PATH}/${STUDENT} \\"
echo "    --dataset agent_slimming_${ROLE} \\"
echo "    --output_dir ${SFT_OUTPUT} \\"
echo "    --num_train_epochs 3 \\"
echo "    --per_device_train_batch_size 4 \\"
echo "    --gradient_accumulation_steps 8 \\"
echo "    --learning_rate 2e-6 \\"
echo "    --lr_scheduler_type cosine \\"
echo "    --bf16 \\"
echo "    --save_steps 20 \\"
echo "    --logging_steps 5"
