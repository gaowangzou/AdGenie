#!/bin/bash
# ============================================================================
# Example: Agent Self-Slimming — 用户专属轻量 Agent 蒸馏
# ============================================================================
# 场景：用户使用 AdGenie 一段时间后，积累了大量对话历史。
#       平台自动触发蒸馏，生成用户的专属小模型。
#
# 预期效果：
#   - 推理延迟: Qwen-7B 的 200ms/token → Gemma-2B 的 45ms/token（↓78%）
#   - Token 成本: $0.001/1K → $0.0002/1K（↓80%）
#   - 保持 90%+ 的编排质量（在用户高频 Skill 上）
# ============================================================================

set -e

# ── 环境 ────────────────────────────────────────────────────────────
export MODEL_PATH="${MODEL_PATH:-./models}"
export DATA_PATH="${DATA_PATH:-./data}"
export OUTPUT_PATH="${OUTPUT_PATH:-./output/ckpts}"

echo "=========================================="
echo "  Agent Self-Slimming Example"
echo "  User: 专属轻量 Agent 蒸馏"
echo "=========================================="
echo ""

# ── Step 1: 准备测试数据（模拟用户30天的对话历史） ─────────────────
echo ">>> Generating sample conversation data..."
python -c "
from agent_slimming.data_collector import DataCollector, TrainingExample
from agent_slimming.config import SlimmingConfig
import json, os, random

config = SlimmingConfig(role='personal_agent')
config.conversation_dir = '${DATA_PATH}/agent_conversations'
os.makedirs(config.conversation_dir, exist_ok=True)

# 模拟生成用户 30 天的对话
sample_messages = [
    '帮我生成一张赛博朋克风格的城市夜景图',
    '把这张图的天空调成紫色调',
    '生成一个5秒的产品展示视频，从正面旋转到侧面',
    '帮我设计一个温柔知性的播客女声',
    '用这张照片生成一个虚拟主播',
    '写一篇关于AI绘画工具对比的小红书文案',
    '把这个3D模型旋转一下，让我看背面',
    '给播客加一段轻松的钢琴背景音乐',
    '分析这张图片的构图，并给出改进建议',
    '生成一个科幻场景的360度全景3D模型',
]

sessions = []
for day in range(30):
    num_sessions = random.randint(1, 5)
    for _ in range(num_sessions):
        msg = random.choice(sample_messages)
        session = {
            'session_id': f'sim_{day:04d}_{random.randint(0,9999):04d}',
            'timestamp': f'2026-07-{(day+1):02d}T{random.randint(8,23):02d}:{random.randint(0,59):02d}:00Z',
            'messages': [
                {'role': 'user', 'content': msg},
                {'role': 'assistant', 'content': f'[Agent Response] Understanding your request about: {msg[:50]}...\n\nI will use the following tools:\n1. Analyze intent\n2. Generate content\n3. Output result\n\nHere is the output for: {msg}'},
            ],
            'skill_used': random.choice(['paper-writing', 'podcast-creator', 'video-creator', 'virtual-anchor', 'xiaohongshu-copywriter']),
            'tools_called': random.sample(['generate_volcano_image', 'qwen_voice_design', 'generate_3d_model', 'generate_volcano_video', 'qwen_omni_understand'], k=random.randint(1,3)),
            'total_tokens': random.randint(2000, 15000),
            'user_feedback': random.choice(['accepted', 'accepted', 'accepted', 'accepted', 'modified']),
        }
        sessions.append(session)

# 保存
output_file = os.path.join(config.conversation_dir, 'user_history_2026_07.jsonl')
with open(output_file, 'w', encoding='utf-8') as f:
    for s in sessions:
        f.write(json.dumps(s, ensure_ascii=False) + '\n')

print(f'Generated {len(sessions)} conversation records in {output_file}')
print(f'  Accepted: {sum(1 for s in sessions if s[\"user_feedback\"]==\"accepted\")}')
print(f'  Modified: {sum(1 for s in sessions if s[\"user_feedback\"]==\"modified\")}')
print(f'  Total tokens: {sum(s[\"total_tokens\"] for s in sessions):,}')
"

echo ""

# ── Step 2: 数据采集与准备 ─────────────────────────────────────
echo ">>> Collecting and preparing training data..."
python -c "
from agent_slimming.data_collector import DataCollector
from agent_slimming.data_preparer import DataPreparer
from agent_slimming.config import SlimmingConfig

config = SlimmingConfig(role='personal_agent')
config.conversation_dir = '${DATA_PATH}/agent_conversations'

# 采集
collector = DataCollector(config)
examples = collector.collect()
print(f'Collected {len(examples)} training examples')

# 准备
preparer = DataPreparer(config)
cleaned = preparer.prepare(examples)
print(f'After cleaning: {len(cleaned)} records')

# 导出
collector.export_for_simct(examples, '${DATA_PATH}/agent_slimming_personal_agent')
print('Exported to ${DATA_PATH}/agent_slimming_personal_agent/')
"

echo ""

# ── Step 3: 蒸馏流水线配置展示 ─────────────────────────────────
echo ">>> Distillation pipeline configuration:"
python -m agent_slimming.slimming_pipeline --role personal_agent --dry-run

echo ""
echo "=========================================="
echo "  Example complete!"
echo "=========================================="
echo ""
echo "  This simulated a 30-day user's data flow."
echo "  To run actual distillation on GPU cluster:"
echo ""
echo "  1. bash scripts/agent_slimming/run_teacher_responses.sh"
echo "  2. bash scripts/agent_slimming/run_sft_warmup.sh"
echo "  3. bash scripts/agent_slimming/run_distill.sh"
echo ""
echo "  Or use the Python pipeline:"
echo "  python -m agent_slimming.slimming_pipeline --role personal_agent"
echo ""
