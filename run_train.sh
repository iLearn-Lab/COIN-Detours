#!/usr/bin/env bash
# 训练启动脚本
#
# === SigLIP 特征 (默认) ===
#   ./run_train.sh 3                              # GPU 3, 默认参数
#   ./run_train.sh 3 --exp_name baseline_v2       # 指定实验名
#
# === InternVideo 特征 (768-d) ===
#   ./run_train.sh 3 --use_internvideo 1 --feat_dim 768 --exp_name iv_baseline
#   ./run_train.sh 0,1,2,3 --use_internvideo 1 --feat_dim 768 --exp_name iv_4gpu
#   独立 val 集 + 新随机种子 (与 iv_baseline 区分):
#   ./run_train.sh 3 --use_internvideo 1 --feat_dim 768 --seed 123 \
#       --train_json /data/wenan-data/UniTime-main/datasets/COIN_train_unitime_mr_seg.json \
#       --val_json /data/wenan-data/UniTime-main/datasets/COIN_val_unitime_mr_seg.json \
#       --exp_name iv_coin_val_s123
#
# === BLIP-2 特征 (256-d, CoVR) ===
#   先提取文本特征: python extract_blip2_text_coin.py --gpu 0
#   再训练:
#   ./run_train.sh 3 --use_blip2 1 --feat_dim 256 \
#       --max_target_frames 15 --max_history_frames 15 --exp_name blip2_baseline
#
# === 多卡训练 ===
#   ./run_train.sh 0,1,2,3                        # 4张GPU并行
#   ./run_train.sh 0,1 --exp_name ddp_2gpu        # 2张GPU
#
# === 两阶段训练 ===
#   阶段1 (无 hard neg):
#     ./run_train.sh 3 --max_hard_neg 0 --exp_name stage1_baseline
#   阶段2 (从阶段1恢复 + hard neg + 小学习率):
#     ./run_train.sh 0,1,2,3 --resume ./checkpoints/stage1_baseline/best.pt \
#                             --lr 2e-5 --max_epochs 30 --exp_name stage2_hardneg

set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

CONDA_PYTHON="/home/wenan/.conda/envs/tfcovr/bin/python"
TORCHRUN="/home/wenan/.conda/envs/tfcovr/bin/torchrun"

GPUS="${1:-0}"
shift 2>/dev/null || true
export CUDA_VISIBLE_DEVICES="$GPUS"

# count GPUs by commas
NGPU=$(echo "$GPUS" | awk -F',' '{print NF}')

echo "============================================"
echo "  Composed Video Retrieval - Training"
echo "  GPUs: $GPUS  (×${NGPU})"
echo "============================================"

COMMON_ARGS="--batch_size 32 --max_epochs 50 --lr 1e-4 --num_workers 4"

if [ "$NGPU" -gt 1 ]; then
    "$TORCHRUN" --nproc_per_node="$NGPU" \
        train.py $COMMON_ARGS "$@"
else
    "$CONDA_PYTHON" -u train.py $COMMON_ARGS "$@"
fi
