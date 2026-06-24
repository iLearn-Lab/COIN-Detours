"""Configuration for Composed Video Retrieval (CVR) model."""

import os
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    train_json: str = "/data/wenan-data/UniTime-main/datasets/COIN_train_unitime_mr_seg.json"
    val_json: str = ""   # if set, use this file as validation set (no random split)
    neg_json: str = "/data/wenan-data/UniTime-main/datasets/COIN_train_unitime_mr_seg_same_history_negatives.json"
    output_dir: str = "./checkpoints"

    feature_dir: str = os.path.expanduser("~/siglip_so400m_feature_fps1/COINvideos")
    text_encoder_path: str = "/home/wenan/hf/siglip-so400m-patch14-384"

    use_internvideo: bool = False
    iv_video_feat_dir: str = "/home/wenan/internvideo-features/COIN-internvideo-features"
    iv_query_feat_dirs: str = (
        "/home/wenan/internvideo-features/COIN_query_feats_train,"
        "/home/wenan/internvideo-features/COIN_query_feats_val,"
        "/home/wenan/internvideo-features/COIN_query_feats_test"
    )

    use_blip2: bool = False
    blip2_video_feat_dir: str = "/data/wenan-data/CoVR-master/blip2-vid-embs-large-all"
    blip2_query_feat_dirs: str = (
        "/home/wenan/blip2-features/COIN_query_feats_train,"
        "/home/wenan/blip2-features/COIN_query_feats_val,"
        "/home/wenan/blip2-features/COIN_query_feats_test"
    )
    blip2_covr_root: str = "/data/wenan-data/CoVR-master"
    blip2_ckpt_path: str = "/data/wenan-data/CoVR-master/blip2_finetune_coco.pth"
    blip2_eva_vit_path: str = "/data/wenan-data/CoVR-master/eva_vit_g.pth"
    blip2_bert_path: str = "/data/wenan-data/CoVR-master/bert-base-uncased"

    feat_dim: int = 1152
    embed_dim: int = 768            # retrieval embedding dimension
    num_temporal_layers: int = 4
    num_heads: int = 8
    ff_dim: int = 2048
    dropout: float = 0.1
    max_target_frames: int = 128    # max frames for target video
    max_history_frames: int = 64    # max frames for reference history
    max_text_len: int = 64          # only used in SigLIP mode
    freeze_text_encoder: bool = True
    init_temperature: float = 0.07
    max_hard_neg: int = 3           # max hard negatives per sample
    num_events: int = 4             # event vectors per target video (VideoTower)
    lambda_orth: float = 0.01       # weight for event orthogonality regularization
    query_ablation: str = "full"
    disable_qg_attn: bool = False
    disable_gated_fusion: bool = False
    disable_inter_event_refine: bool = False
    score_reduce: str = "max"
    use_task_masked_infonce: bool = True
    lambda_span_div: float = 0.1    # weight for span overlap (IoU) diversity loss
    lambda_span_reg: float = 0.05   # weight for span width-minimum + temporal ordering
    lambda_grounding: float = 0.0   # weight for span grounding loss (0 = disabled)

    batch_size: int = 32
    lr: float = 1e-4
    text_lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_epochs: int = 50
    val_ratio: float = 0.1
    num_workers: int = 4
    fp16: bool = True
    grad_clip: float = 1.0
    seed: int = 42
    log_interval: int = 20
    eval_epoch_interval: int = 1
    save_epoch_interval: int = 0

    top_k: Tuple[int, ...] = (1, 5, 10, 50)
