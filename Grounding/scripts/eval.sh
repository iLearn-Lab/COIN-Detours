
#!/bin/bash

TOTAL=1119
GPUS=(4)
NUM_GPUS=${#GPUS[@]}
CHUNK=$((TOTAL / NUM_GPUS))

MODEL_LOCAL_PATH=/home/wenan/hf/Qwen2-VL-2B-Instruct
MODEL_FINETUNE_PATH=/home/wenan/UniTime-main/checkpoints/COIN_2b_2epoch_2e4_15s

VIDEO_ROOT=/home/wenan/COINvideos
FEAT_FOLDER=/home/wenan/UniTime-main/feature_coin_test_2b
DATA_PATH=/home/wenan/UniTime-main/data/COIN_test_finalllllly.json
RUN_NAME=COIN_2b_2epoch_2e4_15s/COIN-wo-two-stage-inference
MASTER_PORT=29501
NF_SHORT=128
VIDEO1TIME=-1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

mkdir -p logs
mkdir -p results/$RUN_NAME

export DECORD_EOF_RETRY_MAX=20480

for ((i=0; i<$NUM_GPUS; i++)); do
    START=$((i * CHUNK))
    if [ $i -eq $((NUM_GPUS - 1)) ]; then
        END=$TOTAL
    else
        END=$((START + CHUNK))
    fi

    GPU_ID=${GPUS[$i]}

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

