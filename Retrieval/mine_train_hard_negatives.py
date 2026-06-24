"""
Mine hard negatives from training data using the current model.

For each training query, run full-library retrieval and extract
videos that rank above the ground-truth as new hard negatives.
These "model-confusion" negatives are more effective than static
rule-based ones because they directly target the model's current
weak discrimination boundaries.

Output format is compatible with the existing neg_json / dataset.py.

Usage:
    CUDA_VISIBLE_DEVICES=3 python mine_train_hard_negatives.py \
        --checkpoint ./checkpoints/stage2_hn_1gpu/best.pt \
        --output ./mined_train_hard_negatives.json \
        --top_k 5 \
        --max_rank 20
"""

import argparse
import json
import os

import faiss
import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from config import Config
from dataset import (COINRetrievalDataset, InternVideoRetrievalDataset,
                     load_data)
from model import ComposedVideoRetriever
from test import VideoPoolDataset, encode_video_pool, encode_all_queries, build_faiss_index


def mine(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp16 = torch.cuda.is_available()

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_cfg = ckpt.get("config", {})
    cfg = Config()
    for k, v in ckpt_cfg.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.fp16 = fp16

    use_iv = cfg.use_internvideo
    video_feat_dir = cfg.iv_video_feat_dir if use_iv else cfg.feature_dir
    if use_iv:
        print(f"[Mode] InternVideo  video_feat_dir={video_feat_dir}")
    else:
        print(f"[Mode] SigLIP  feature_dir={video_feat_dir}")

    model = ComposedVideoRetriever(
        feat_dim=cfg.feat_dim,
        embed_dim=cfg.embed_dim,
        num_temporal_layers=cfg.num_temporal_layers,
        num_heads=cfg.num_heads,
        ff_dim=cfg.ff_dim,
        dropout=0.0,
        text_encoder_path=cfg.text_encoder_path,
        freeze_text_encoder=True,
        init_temperature=cfg.init_temperature,
        num_events=getattr(cfg, "num_events", 4),
        use_pretrained_text_feat=use_iv,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Model loaded (epoch {ckpt.get('epoch', '?')}, num_events={cfg.num_events})")

    print(f"Loading training data: {cfg.train_json}")
    train_data = load_data(cfg.train_json)
    print(f"Training samples: {len(train_data)}")

    vid2path = {}
    for item in train_data:
        path = item["video2_path"]
        vid = os.path.basename(path).split(".")[0]
        vid2path[vid] = path

    pool_vids = list(vid2path.keys())
    print(f"Unique target videos in training set: {len(pool_vids)}")

    ordered_vids, video_embs = encode_video_pool(
        model, pool_vids, video_feat_dir, cfg, device
    )
    print(f"Video pool encoded: {video_embs.shape}")

    index = build_faiss_index(video_embs, use_gpu=False)
    unique_vids = list(dict.fromkeys(ordered_vids))
    is_multi_event = len(ordered_vids) > len(unique_vids)
    if is_multi_event:
        num_events = len(ordered_vids) // len(unique_vids)
        print(f"[MaxSim] {len(unique_vids)} videos × {num_events} events")
    else:
        num_events = 1
        vid_to_pool_idx = {v: i for i, v in enumerate(ordered_vids)}

    if use_iv:
        iv_query_dirs = [d.strip() for d in cfg.iv_query_feat_dirs.split(",")
                         if d.strip()]
        train_ds = InternVideoRetrievalDataset(
            train_data,
            video_feat_dir=cfg.iv_video_feat_dir,
            query_feat_dirs=iv_query_dirs,
            max_target_frames=cfg.max_target_frames,
            max_history_frames=cfg.max_history_frames,
            feat_dim=cfg.feat_dim,
            cache_features=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.text_encoder_path)
        train_ds = COINRetrievalDataset(
            train_data, cfg.feature_dir, tokenizer,
            cfg.max_target_frames, cfg.max_history_frames,
            cfg.max_text_len, cfg.feat_dim,
            cache_features=True,
        )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )

    query_embs, gt_vids, qids = encode_all_queries(
        model, train_loader, device, cfg.fp16
    )
    print(f"Training queries encoded: {query_embs.shape}")

    search_k = args.max_rank

    mined = []
    n_mined = 0
    n_gt_missing = 0
    n_already_rank1 = 0

    for i in range(len(query_embs)):
        qid = int(qids[i])
        gt_vid = gt_vids[i]

        if is_multi_event:
            if gt_vid not in unique_vids:
                n_gt_missing += 1
                mined.append({"qid": qid, "hard_negatives_same_history_diff_query": []})
                continue
            faiss_k = min(index.ntotal,
                          max(search_k * num_events * 4, search_k + num_events))
            row_scores, row_indices = index.search(
                query_embs[i : i + 1], faiss_k
            )
            vid_scores: dict = {}
            for score, idx in zip(row_scores[0].tolist(), row_indices[0].tolist()):
                v = ordered_vids[idx]
                if v not in vid_scores or score > vid_scores[v]:
                    vid_scores[v] = score
            ranked_vids = sorted(
                unique_vids,
                key=lambda v: vid_scores.get(v, -1e9),
                reverse=True,
            )
            if gt_vid in ranked_vids:
                gt_rank = ranked_vids.index(gt_vid) + 1
            else:
                gt_rank = search_k + 1
            false_positive_vids = [
                v for v in ranked_vids[: gt_rank - 1] if v != gt_vid
            ][:args.top_k]
        else:
            gt_pool_idx = vid_to_pool_idx.get(gt_vid, -1)
            if gt_pool_idx == -1:
                n_gt_missing += 1
                mined.append({"qid": qid, "hard_negatives_same_history_diff_query": []})
                continue

            row_scores, row_indices = index.search(
                query_embs[i : i + 1], min(search_k, index.ntotal)
            )
            retrieved = row_indices[0].tolist()

            if gt_pool_idx in retrieved:
                gt_rank = retrieved.index(gt_pool_idx) + 1
            else:
                gt_rank = search_k + 1

            false_positive_vids = [
                ordered_vids[idx]
                for idx in retrieved[: gt_rank - 1]
                if ordered_vids[idx] != gt_vid
            ][:args.top_k]

        if gt_rank == 1:
            n_already_rank1 += 1
            mined.append({"qid": qid, "hard_negatives_same_history_diff_query": []})
            continue

        hard_negs = [
            {"video2_path": vid2path.get(v, f"dummy/{v}.mp4")}
            for v in false_positive_vids
        ]

        mined.append({
            "qid": qid,
            "hard_negatives_same_history_diff_query": hard_negs,
        })

        if hard_negs:
            n_mined += 1

    print(f"\n=== Mining Results ===")
    print(f"  Total training queries:       {len(query_embs)}")
    print(f"  Already at rank 1:            {n_already_rank1}")
    print(f"  GT not in video pool:         {n_gt_missing}")
    print(f"  Queries with mined hard negs: {n_mined}")
    print(f"  Avg hard negs per query:      "
          f"{sum(len(x['hard_negatives_same_history_diff_query']) for x in mined) / len(mined):.2f}")

    with open(args.output, "w") as f:
        json.dump(mined, f, ensure_ascii=False)
    print(f"\nSaved to: {args.output}")
    print(f"Ready to use as --neg_json {args.output} in train.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--output", type=str,
                        default="./mined_train_hard_negatives.json",
                        help="Output neg_json path")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Max hard negatives per query to keep")
    parser.add_argument("--max_rank", type=int, default=20,
                        help="Only look at top-max_rank results for mining")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    mine(args)
