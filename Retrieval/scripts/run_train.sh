#!/usr/bin/env bash

set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONDA_PYTHON="/home/wenan/.conda/envs/tfcovr/bin/python"
TORCHRUN="/home/wenan/.conda/envs/tfcovr/bin/torchrun"

GPUS="${1:-0}"
shift 2>/dev/null || true
export CUDA_VISIBLE_DEVICES="$GPUS"

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
