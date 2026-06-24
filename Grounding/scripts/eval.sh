# export CUDA_VISIBLE_DEVICES=0
# export DECORD_EOF_RETRY_MAX=20480

# export MASTER_PORT=29501
# #11405
# START=0
# END=24
# RUN_NAME=COIN_qwen2_train
# python inference_v1.py --model_local_path /mnt/nodestor/wenan/Qwen2-VL-2B-Instruct \
#     --model_finetune_path /mnt/nodestor/wenan/UniTime-new-main/checkpoints/COIN \
#     --video_root /mnt/nodestor/wenan/HTvideo_test \
#     --feat_folder ./feature_COIN \
#     --data_path //mnt/nodestor/wenan/UniTime-new-main/data/COIN_test_human_remain.json \
#     --output_dir ./results/$RUN_NAME/remain\
#     --nf_short 128 \
#     --start $START \
#     --end $END


#!/bin/bash
# ====================================================
# 多卡并行推理脚本 WenAn - YouCook2 Qwen2 推理任务
# ====================================================

# 基础配置
# TOTAL=882
# TOTAL=1132
# TOTAL=71223
# TOTAL=8374
# TOTAL=11405
# NUM_GPUS=6
# CHUNK=$((TOTAL / NUM_GPUS))
# MODEL_LOCAL_PATH=/home/wenan/hf/Qwen2-VL-2B-Instruct
# MODEL_FINETUNE_PATH=/home/wenan/UniTime-main/checkpoints/detours_train_epoch_15s
# # VIDEO_ROOT=/mnt/nodestor/wenan/raw_videos/validation
# # VIDEO_ROOT=/mnt/nodestor/wenan/COINvideos
# VIDEO_ROOT=/home/wenan/detours-video/HTvideo_test
# FEAT_FOLDER=/home/wenan/UniTime-main/feature_detours_test
# # FEAT_FOLDER=/mnt/nodestor/wenan/UniTime-new-main/feature_COIN
# # DATA_PATH=/mnt/nodestor/wenan/UniTime-new-main/data/youcook2_test_human_unitime_intention.json
# # DATA_PATH=/mnt/nodestor/wenan/UniTime-new-main/data/COIN_test_1-17_wenan_converted.json
# DATA_PATH=/home/wenan/UniTime-main/data/detours_format_filtered.json
# RUN_NAME=detours_train_epoch_15s/detours
# MASTER_PORT=29501
# NF_SHORT=128
# GPU_OFFSET=1  # 👈 从第0号GPU开始

# # 创建日志与结果目录
# mkdir -p logs
# mkdir -p results/$RUN_NAME

# # 环境变量
# export DECORD_EOF_RETRY_MAX=20480

# # 启动多卡推理
# for ((i=0; i<$NUM_GPUS; i++)); do
#     START=$((i * CHUNK))
#     if [ $i -eq $((NUM_GPUS - 1)) ]; then
#         END=$TOTAL
#     else
#         END=$((START + CHUNK))
#     fi

#     GPU_ID=$((i + GPU_OFFSET))   # 👈 偏移GPU编号

#     echo ">>> GPU $GPU_ID 处理样本 [$START, $END)"
#     CUDA_VISIBLE_DEVICES=$GPU_ID nohup python /home/wenan/UniTime-main/inference.py \
#         --model_local_path $MODEL_LOCAL_PATH \
#         --model_finetune_path $MODEL_FINETUNE_PATH \
#         --video_root $VIDEO_ROOT \
#         --feat_folder $FEAT_FOLDER \
#         --data_path $DATA_PATH \
#         --output_dir "./results/$RUN_NAME" \
#         --nf_short $NF_SHORT \
#         --start $START \
#         --end $END \
#         > logs/gpu${GPU_ID}.log 2>&1 &
# done

# echo "✅ 所有进程已启动(GPU0-7),查看 logs/gpu*.log 获取输出。"


# TOTAL=8374
# SKIP=1700            # 👈 已完成的数量
# REMAIN=$((TOTAL - SKIP))

# NUM_GPUS=8
# CHUNK=$((REMAIN / NUM_GPUS))

# MODEL_LOCAL_PATH=/mnt/nodestor/wenan/Qwen2-VL-2B-Instruct
# MODEL_FINETUNE_PATH=/mnt/nodestor/wenan/UniTime-new-main/checkpoints/youcook2_qwen2_train_plus_direct_analysis
# VIDEO_ROOT=/mnt/nodestor/wenan/HTvideo_test
# FEAT_FOLDER=/mnt/nodestor/wenan/UniTime-new-main/feature_youcook2
# DATA_PATH=/mnt/nodestor/wenan/UniTime-new-main/data/youcook2_test_human_unitime_intention.json
# RUN_NAME=youcook2_qwen2_train_plus_direct_analysis/test
# MASTER_PORT=29501
# NF_SHORT=128
# GPU_OFFSET=0

# mkdir -p logs
# mkdir -p results/$RUN_NAME

# export DECORD_EOF_RETRY_MAX=20480

# for ((i=0; i<$NUM_GPUS; i++)); do
#     START=$((SKIP + i * CHUNK))
#     if [ $i -eq $((NUM_GPUS - 1)) ]; then
#         END=$TOTAL
#     else
#         END=$((START + CHUNK))
#     fi

#     GPU_ID=$((i + GPU_OFFSET))

#     echo ">>> GPU $GPU_ID 处理样本 [$START, $END)"
#     CUDA_VISIBLE_DEVICES=$GPU_ID nohup python inference_v3.py \
#         --model_local_path $MODEL_LOCAL_PATH \
#         --model_finetune_path $MODEL_FINETUNE_PATH \
#         --video_root $VIDEO_ROOT \
#         --feat_folder $FEAT_FOLDER \
#         --data_path $DATA_PATH \
#         --output_dir "./results/$RUN_NAME" \
#         --nf_short $NF_SHORT \
#         --start $START \
#         --end $END \
#         > logs/gpu${GPU_ID}.log 2>&1 &
# done

# echo "✅ 已启动，从样本1701开始到8374结束。查看 logs/gpu*.log。"


# 基础配置
# TOTAL=11405
TOTAL=1119
GPUS=(4)        # 👈 固定 GPU 列表
NUM_GPUS=${#GPUS[@]}
CHUNK=$((TOTAL / NUM_GPUS))

MODEL_LOCAL_PATH=/home/wenan/hf/Qwen2-VL-2B-Instruct
MODEL_FINETUNE_PATH=/home/wenan/UniTime-main/checkpoints/COIN_2b_2epoch_2e4_15s

VIDEO_ROOT=/home/wenan/COINvideos
# VIDEO_ROOT=/home/wenan/detours-video/HTvideo_test
# FEAT_FOLDER=/home/wenan/UniTime-main/feature_detours_test
FEAT_FOLDER=/home/wenan/UniTime-main/feature_coin_test_2b
# DATA_PATH=/mnt/nodestor/wenan/UniTime-new-main/data/youcook2_test_human_unitime_intention.json
DATA_PATH=/home/wenan/UniTime-main/data/COIN_test_finalllllly.json
# DATA_PATH=/home/wenan/UniTime-main/data/detours_format_filtered.json
RUN_NAME=COIN_2b_2epoch_2e4_15s/COIN-wo-two-stage-inference
MASTER_PORT=29501
NF_SHORT=128
VIDEO1TIME=-1  # background video 时长(秒); -1=从0开始, 正数=取 video1_end 前 N 秒

# 推理脚本所在项目根（在任意 cwd 下执行本脚本均可）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 创建目录
mkdir -p logs
mkdir -p results/$RUN_NAME

# 环境变量
export DECORD_EOF_RETRY_MAX=20480

# 启动推理
for ((i=0; i<$NUM_GPUS; i++)); do
    START=$((i * CHUNK))
    if [ $i -eq $((NUM_GPUS - 1)) ]; then
        END=$TOTAL
    else
        END=$((START + CHUNK))
    fi

    GPU_ID=${GPUS[$i]}   # 👈 直接使用数组里的 GPU

    echo ">>> GPU $GPU_ID 处理样本 [$START, $END)"

    CUDA_VISIBLE_DEVICES=$GPU_ID nohup python inference.py \
        --model_local_path $MODEL_LOCAL_PATH \
        --model_finetune_path $MODEL_FINETUNE_PATH \
        --video_root $VIDEO_ROOT \
        --feat_folder $FEAT_FOLDER \
        --data_path $DATA_PATH \
        --output_dir "./results/$RUN_NAME" \
        --nf_short $NF_SHORT \
        --video1time $VIDEO1TIME \
        --start $START \
        --end $END \
        > logs/gpu${GPU_ID}.log 2>&1 &
done

# echo "✅ 所有进程已启动 (使用 GPU: 0,5,6,7)，查看 logs/gpu*.log 获取输出。"