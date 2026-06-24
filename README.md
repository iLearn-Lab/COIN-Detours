# COIN-Detours

COIN-Detours: Context-Aware Retrieval and Generative Temporal Grounding for Instructional Video Detours.

This repository contains the minimal code release for composed video retrieval training, evaluation, and hard-negative mining.

## Files

- `Retrieval/train.py`: training entry point
- `Retrieval/test.py`: full-library retrieval evaluation
- `Retrieval/mine_train_hard_negatives.py`: mine model-confusion hard negatives
- `Retrieval/model.py`: dual-tower retrieval model
- `Retrieval/dataset.py`: dataset and feature loading utilities
- `Retrieval/config.py`: default paths and hyperparameters
- `Retrieval/scripts/run_train.sh`, `Retrieval/scripts/run_test.sh`: launch scripts
- `Retrieval/COIN_testing_videos_filtered.txt`: default evaluation video pool list

## Setup

Install dependencies in your Python environment:

```bash
pip install -r requirements.txt
```

Edit `Retrieval/config.py` or pass command-line arguments to point to your local dataset, feature directories, and pretrained text encoder paths.

## Train

```bash
./Retrieval/scripts/run_train.sh 0 --exp_name baseline
```

For InternVideo features:

```bash
./Retrieval/scripts/run_train.sh 0 --use_internvideo 1 --feat_dim 768 --exp_name iv_baseline
```

## Evaluate

```bash
./Retrieval/scripts/run_test.sh 0 ./Retrieval/checkpoints/baseline/best.pt
```

## Mine Hard Negatives

```bash
python Retrieval/mine_train_hard_negatives.py \
  --checkpoint ./Retrieval/checkpoints/baseline/best.pt \
  --output ./mined_train_hard_negatives.json \
  --top_k 5 \
  --max_rank 20
```
