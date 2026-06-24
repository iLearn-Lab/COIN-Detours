#!/usr/bin/env bash
# 同 task 检索评估（oracle 上限），不影响 test.py / eval_results.json
# 用法:
#   ./run_test1.sh 3 ./checkpoints/exp_name/best.pt

set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

GPU="${1:-0}"
shift 2>/dev/null || true
CKPT="${1:-}"
shift 2>/dev/null || true

if [[ -z "$CKPT" ]]; then
    echo "用法: ./run_test1.sh <gpu_id> <checkpoint_path>"
    exit 1
fi

RESULT_DIR=$(dirname "$CKPT")
RESULT_FILE="$RESULT_DIR/eval_results_same_task.json"

export CUDA_VISIBLE_DEVICES="$GPU"

echo "============================================"
echo "  Same-Task Retrieval Eval (test1.py)"
echo "  GPU: $GPU"
echo "  Checkpoint: $CKPT"
echo "  Results: $RESULT_FILE"
echo "============================================"

python -u test1.py \
    --checkpoint "$CKPT" \
    --test_json /home/wenan/UniTime-main/data/COIN_test_finalllllly.json \
    --video_pool ./COIN_testing_videos_filtered.txt \
    --batch_size 64 \
    --num_workers 4 \
    --save_results "$RESULT_FILE" \
    "$@"
