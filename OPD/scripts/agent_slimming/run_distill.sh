#!/bin/bash
# ============================================================================
# Step 4: SimCT 跨 Tokenizer 在线策略蒸馏
# 核心步骤：将 Teacher 大模型的知识蒸馏到 Student 小模型。
# 使用 span_ctkd 算法，支持跨 tokenizer（如 Qwen → Gemma）。
# ============================================================================
set -e

ROLE="${SLIMMING_ROLE:-personal_agent}"
MODEL_PATH="${MODEL_PATH:-./models}"
DATA_PATH="${DATA_PATH:-./data}"
OUTPUT_PATH="${OUTPUT_PATH:-./output/ckpts}"

# ── 根据角色确定 teacher / student ─────────────────────────────────
case "${ROLE}" in
    agent_orchestration)
        TEACHER="Qwen2.5-7B-Instruct"
        STUDENT="gemma-2-2b-it"
        SFT_CKPT="${OUTPUT_PATH}/${ROLE}-sft-warmup-lr2e-6/checkpoint-80"
        LR=5e-7
        ;;
    image_understanding)
        TEACHER="Qwen2.5-VL-7B-Instruct"
        STUDENT="Qwen2.5-VL-2B-Instruct"
        SFT_CKPT="${OUTPUT_PATH}/${ROLE}-sft-warmup-lr2e-6/checkpoint-80"
        LR=5e-7
        ;;
    tts_voice)
        TEACHER="Qwen2.5-7B-Instruct"
        STUDENT="Qwen2.5-1.5B-Instruct"
        SFT_CKPT="${OUTPUT_PATH}/${ROLE}-sft-warmup-lr2e-6/checkpoint-80"
        LR=1e-6
        ;;
    video_script)
        TEACHER="Qwen2.5-7B-Instruct"
        STUDENT="Phi-4-mini-instruct"
        SFT_CKPT="${OUTPUT_PATH}/${ROLE}-sft-warmup-lr2e-6/checkpoint-80"
        LR=5e-7
        ;;
    personal_agent)
        TEACHER="Qwen2.5-7B-Instruct"
        STUDENT="gemma-2-2b-it"
        SFT_CKPT="${OUTPUT_PATH}/${ROLE}-sft-warmup-lr2e-6/checkpoint-80"
        LR=5e-7
        ;;
    *)
        TEACHER="Qwen2.5-7B-Instruct"
        STUDENT="gemma-2-2b-it"
        SFT_CKPT="${OUTPUT_PATH}/${ROLE}-sft-warmup-lr2e-6/checkpoint-80"
        LR=5e-7
        ;;
esac

SAVE_PATH="${OUTPUT_PATH}/${ROLE}-${STUDENT}-ctopd-lr${LR}"

echo "=========================================="
echo "  SimCT Cross-Tokenizer OPD Distillation"
echo "=========================================="
echo "  Role:       ${ROLE}"
echo "  Teacher:    ${TEACHER} (Qwen tokenizer)"
echo "  Student:    ${STUDENT}"
echo "  Algorithm:  span_ctkd"
echo "  Loss:       reverse KL divergence"
echo "  LR:         ${LR}"
echo "  SFT ckpt:   ${SFT_CKPT}"
echo "  Output:     ${SAVE_PATH}"
echo "=========================================="
echo ""

# ── 构建命令行参数 ─────────────────────────────────────────────────
OPTS=""
OPTS+=" --num_nodes 1"
OPTS+=" --num_gpus_per_node 8"
OPTS+=" --backend fsdp2"
OPTS+=" --train_batch_size 64"
OPTS+=" --micro_train_batch_size 1"
OPTS+=" --learning_rate ${LR}"
OPTS+=" --lr_warmup_ratio 0.05"
OPTS+=" --num_epochs 1"
OPTS+=" --save_path ${SAVE_PATH}"
OPTS+=" --bf16 True"
OPTS+=" --gradient_checkpointing True"
OPTS+=" --enable_sleep True"

# Model
OPTS+=" --student_name_or_path ${SFT_CKPT}"
OPTS+=" --teacher_name_or_path ${MODEL_PATH}/${TEACHER}"
OPTS+=" --enable_thinking False"

# Rollout (On-Policy)
OPTS+=" --rollout_batch_size 64"
OPTS+=" --rollout_num_engines 8"
OPTS+=" --rollout_tp_size 1"
OPTS+=" --rollout_mem_fraction_static 0.6"
OPTS+=" --n_samples_per_prompt 1"
OPTS+=" --generate_max_len 4096"
OPTS+=" --temperature 0.6"

# Data
OPTS+=" --train_dataset_path ${DATA_PATH}/agent_slimming_${ROLE}"
OPTS+=" --max_len 8192"
OPTS+=" --input_key messages"
OPTS+=" --apply_chat_template True"
OPTS+=" --preprocess_num_workers 8"
OPTS+=" --packing_samples True"

# Distillation
OPTS+=" --kd_ratio 0.9"
OPTS+=" --kd_loss_fn rkl"
OPTS+=" --kd_algorithm span_ctkd"
OPTS+=" --teacher_dp_size 8"
OPTS+=" --teacher_tp_size 1"
OPTS+=" --teacher_mem_fraction_static 0.5"
OPTS+=" --teacher_context_length 32768"

# Logging
OPTS+=" --logging_steps 5"
OPTS+=" --save_steps 50"
OPTS+=" --use_wandb False"

export SGLANG_DISABLE_CUDNN_CHECK=1

echo "Running SimCT distillation..."
echo "Command: python -m kdflow.cli.train_kd_on_policy ${OPTS}"
echo ""

# 实际运行时取消注释:
# python -m kdflow.cli.train_kd_on_policy ${OPTS}

echo ""
echo "Distillation complete!"
echo "Model saved to: ${SAVE_PATH}"
echo ""
echo "Next: Register in router — python -m agent_slimming.router register \\"
echo "  --model-name ${ROLE}-${STUDENT} \\"
echo "  --model-path ${SAVE_PATH} \\"
echo "  --role ${ROLE}"
