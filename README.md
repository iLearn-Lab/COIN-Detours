# Retrieval

Minimal code release for composed video retrieval training, evaluation, and hard-negative mining.

## Files

- `train.py`: training entry point
- `test.py`: full-library retrieval evaluation
- `test1.py`: same-task retrieval evaluation
- `mine_train_hard_negatives.py`: mine model-confusion hard negatives
- `model.py`: dual-tower retrieval model
- `dataset.py`: dataset and feature loading utilities
- `config.py`: default paths and hyperparameters
- `run_train.sh`, `run_test.sh`, `run_test1.sh`: launch scripts
- `COIN_testing_videos_filtered.txt`: default evaluation video pool list

## Setup

Install dependencies in your Python environment:

```bash
pip install -r requirements.txt
```

Edit `config.py` or pass command-line arguments to point to your local dataset, feature directories, and pretrained text encoder paths.

## Train

```bash
./run_train.sh 0 --exp_name baseline
```

For InternVideo features:

```bash
./run_train.sh 0 --use_internvideo 1 --feat_dim 768 --exp_name iv_baseline
```

## Evaluate

```bash
./run_test.sh 0 ./checkpoints/baseline/best.pt
```

## Mine Hard Negatives

```bash
python mine_train_hard_negatives.py \
  --checkpoint ./checkpoints/baseline/best.pt \
  --output ./mined_train_hard_negatives.json \
  --top_k 5 \
  --max_rank 20
```
