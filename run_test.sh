#!/usr/bin/env bash
# 全库检索评估脚本
# 用法:
#   ./run_test.sh 3 ./checkpoints/exp_name/best.pt
#   ./run_test.sh 3 ./checkpoints/bs32_lr0.0001_ep50_hn3_0319_1430/best.pt

set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

GPU="${1:-0}"
shift 2>/dev/null || true
CKPT="${1:-}"
shift 2>/dev/null || true

if [[ -z "$CKPT" ]]; then
    echo "用法: ./run_test.sh <gpu_id> <checkpoint_path>"
    echo "例如: ./run_test.sh 0 /home/wenan/RRetrieval/checkpoints/iv_baseline_s2026_unitime_enviroment/best.pt"
    exit 1
fi

# 结果文件保存在 checkpoint 同目录
RESULT_DIR=$(dirname "$CKPT")
RESULT_FILE="$RESULT_DIR/eval_results.json"

export CUDA_VISIBLE_DEVICES="$GPU"

echo "============================================"
echo "  Composed Video Retrieval - Evaluation"
echo "  GPU: $GPU"
echo "  Checkpoint: $CKPT"
echo "  Results: $RESULT_FILE"
echo "============================================"

python -u test.py \
    --checkpoint "$CKPT" \
    --test_json /home/wenan/UniTime-main/data/COIN_test_finalllllly.json \
    --video_pool ./COIN_testing_videos_filtered.txt \
    --batch_size 64 \
    --num_workers 4 \
    --save_results "$RESULT_FILE" \
    "$@"
