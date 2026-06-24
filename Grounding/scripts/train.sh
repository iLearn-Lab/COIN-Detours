NUM_GPUS=3
DISTRIBUTED_ARGS="
    --nnodes=1 \
    --nproc_per_node ${NUM_GPUS} \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:0
"
export CUDA_VISIBLE_DEVICES=5,6,7
export DECORD_EOF_RETRY_MAX=20480

MODEL_ID=qwen2-vl-2b-instruct
model_local_path="/home/wenan/hf/Qwen2-VL-2B-Instruct"
TRAIN_DATA_PATH="/home/wenan/UniTime-main/datasets/COIN_train_unitime_mr+seg_mr.json"
EVAL_DATA_PATH=None
IMAGE_FOLDER=None
VIDEO_FOLDER="/home/wenan/COINvideos"  #If you specified video_path in the data file, this can be set to none
FEAT_FOLDER="/home/wenan/UniTime-main/feature_coin_train"  #If you specified feature_path in the data file, this can be set to none

FPS=1
CLIP_LENGTH=32

TRAIN_VISION_ENCODER=False                              # whether train the vision encoder
USE_VISION_LORA=False                                   # whether use lora for vision encoder (only effective when `TRAIN_VISION_ENCODER` is True)
TRAIN_VISION_PROJECTOR=False                            # whether train the vision projector (only full finetuning is supported)

USE_LORA=True                                           # whether use lora for llm
Q_LORA=False                                            # whether use q-lora for llm; only effective when `USE_LORA` is True
LORA_R=8                                                # the lora rank (both llm and vision encoder)
LORA_ALPHA=8                                            # the lora alpha (both llm and vision encoder)

RUN_ID=COIN_2b_2epoch_2e4_15s   # 与checkpoint路径保持一致

DS_STAGE=zero2                                         # deepspeed stage; < zero2 | zero3 >
PER_DEVICE_BATCH_SIZE=1                                # batch size per GPU
GRAD_ACCUM=1                                            # gradient accumulation steps
NUM_EPOCHS=2                                          # 设置为2
LR=2e-4                                                 # learning rate
MODEL_MAX_LEN=32768                                      # maximum input length of the model


torchrun $DISTRIBUTED_ARGS /home/wenan/UniTime-main/train.py \
    --model_id $MODEL_ID \
    --model_local_path $model_local_path \
    --data_path $TRAIN_DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --video_folder $VIDEO_FOLDER \
    --fps $FPS \
    --output_dir ./checkpoints/$RUN_ID \
    --report_to tensorboard \
    --run_name $RUN_ID \
    --deepspeed /home/wenan/UniTime-main/ds_configs/${DS_STAGE}.json \
    --bf16 True \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate ${LR} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length $MODEL_MAX_LEN \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --train_vision_encoder $TRAIN_VISION_ENCODER \
    --use_vision_lora $USE_VISION_LORA \
    --train_vision_projector $TRAIN_VISION_PROJECTOR \
    --use_lora $USE_LORA \
    --q_lora $Q_LORA \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --save_strategy "epoch" \
    --clip_length $CLIP_LENGTH \
    --feat_folder $FEAT_FOLDER \
    --video1time 15 \