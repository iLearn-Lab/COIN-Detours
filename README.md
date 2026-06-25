# COIN-Detours

COIN-Detours: Context-Aware Retrieval and Generative Temporal Grounding for Instructional Video Detours.

This repository implements a two-stage pipeline: **Retrieval** (composed video retrieval) and **Grounding** (temporal localization with Qwen2-VL).

## Setup

```bash
pip install -r requirements.txt
```

Before training, update paths in `Retrieval/config.py` and in the scripts under `Grounding/scripts/` to match your local data, features, and model checkpoints.

---

## Retrieval Training

The retrieval module maps *(viewing history + text query)* to a target video using a dual-tower architecture with InfoNCE contrastive learning. By default it uses pre-extracted SigLIP frame features; InternVideo and BLIP-2 features are also supported via `--use_internvideo` and `--use_blip2`.

**Single-GPU training:**

```bash
cd Retrieval
./scripts/run_train.sh 0 --exp_name baseline
```

**Multi-GPU training:**

```bash
./scripts/run_train.sh 0,1,2,3 --exp_name ddp_baseline
```

**InternVideo features (768-dim):**

```bash
./scripts/run_train.sh 0 --use_internvideo 1 --feat_dim 768 --exp_name iv_baseline
```

Checkpoints are saved to `Retrieval/checkpoints/<exp_name>/`. `best.pt` is the weight with the highest validation R@1.

**Full-library retrieval evaluation:**

```bash
./scripts/run_test.sh 0 ./checkpoints/baseline/best.pt
```

**(Optional) Mine hard negatives for a second training stage:**

```bash
python mine_train_hard_negatives.py \
  --checkpoint ./checkpoints/baseline/best.pt \
  --output ./mined_train_hard_negatives.json
```

Pass the output file to training via `--neg_json`.

---

## Grounding Training

The grounding module performs temporal localization on retrieved video segments using Qwen2-VL, fine-tuned with LoRA and DeepSpeed ZeRO-2/3.

**Start training:**

```bash
cd Grounding
bash scripts/train.sh
```

Configure the following in `scripts/train.sh` before running:
- `model_local_path`: Qwen2-VL pretrained weights
- `TRAIN_DATA_PATH`: training annotation JSON
- `VIDEO_FOLDER` / `FEAT_FOLDER`: video and pre-extracted feature directories
- `RUN_ID`, `NUM_GPUS`, `LR`, and other hyperparameters

Checkpoints are saved to `Grounding/checkpoints/<RUN_ID>/` by default.

**Inference / evaluation:**

```bash
bash scripts/eval.sh
```

Configure the fine-tuned checkpoint path, test data, and feature directory in `scripts/eval.sh`. Multi-GPU sharded inference is supported; results are written to `Grounding/results/`.
