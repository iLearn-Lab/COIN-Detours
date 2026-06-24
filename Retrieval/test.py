"""
Evaluation script for Composed Video Retrieval — full-library retrieval.

Workflow:
  1. Load ALL videos from the video pool file → encode with Video Tower → FAISS index
  2. Encode each test (history + query) with Query Tower
  3. Search the FAISS index for each query
  4. Compute R@1, R@5, R@10, R@50, MedR, MeanR

Usage:
    CUDA_VISIBLE_DEVICES=3 python test.py \
        --checkpoint ./checkpoints/best.pt \
        --test_json /home/wenan/UniTime-main/data/COIN_test_finalllllly.json \
        --video_pool ./COIN_testing_videos_filtered.txt
"""

import argparse
import json
import os
import time

import faiss
import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from config import Config
from dataset import (COINRetrievalDataset, InternVideoRetrievalDataset,
                     load_data, split_by_reference_video)
from model import ComposedVideoRetriever


class VideoPoolDataset(Dataset):
    """Iterate over the full video library, load pre-extracted features."""

    def __init__(self, vid_list, feature_dir, max_frames=128, feat_dim=1152):
        self.feature_dir = feature_dir
        self.max_frames = max_frames
        self.feat_dim = feat_dim
        self.vids = [v for v in vid_list
                     if os.path.exists(os.path.join(feature_dir, f"{v}.pth.tar"))]
        n_skip = len(vid_list) - len(self.vids)
        if n_skip:
            print(f"[VideoPool] {n_skip}/{len(vid_list)} videos skipped "
                  f"(missing features), using {len(self.vids)}")

    def __len__(self):
        return len(self.vids)

    def _load(self, vid):
        path = os.path.join(self.feature_dir, f"{vid}.pth.tar")
        feat = torch.load(path, map_location="cpu", weights_only=False)
        if feat.dtype == torch.float16:
            feat = feat.float()
        return feat

    def _pad(self, feat):
        T = feat.size(0)
        if T > self.max_frames:
            idx = torch.linspace(0, T - 1, self.max_frames).long()
            feat = feat[idx]
            T = self.max_frames
        if T < self.max_frames:
            pad = torch.zeros(self.max_frames - T, self.feat_dim)
            feat = torch.cat([feat, pad])
            mask = torch.zeros(self.max_frames, dtype=torch.bool)
            mask[:T] = True
        else:
            mask = torch.ones(self.max_frames, dtype=torch.bool)
        return feat, mask

    def __getitem__(self, idx):
        vid = self.vids[idx]
        feat = self._load(vid)
        feat, mask = self._pad(feat)
        return feat, mask, vid


@torch.no_grad()
def encode_video_pool(model, vid_list, feature_dir, cfg, device):
    """Encode the full video library → (ordered_vids, embeddings [V, D])."""
    ds = VideoPoolDataset(vid_list, feature_dir,
                          cfg.max_target_frames, cfg.feat_dim)
    loader = DataLoader(ds, batch_size=cfg.batch_size,
                        num_workers=cfg.num_workers, pin_memory=True)
    embs, ordered_vids = [], []
    model.eval()
    for feat, mask, vids in tqdm(loader, desc="Encoding video library"):
        feat, mask = feat.to(device), mask.to(device)
        with autocast(enabled=cfg.fp16):
            emb = model.encode_video(feat, mask)

        if emb.dim() == 3:
            B, num_ev, D = emb.shape
            embs.append(emb.cpu().reshape(B * num_ev, D))
            for v in vids:
                ordered_vids.extend([v] * num_ev)
        else:
            embs.append(emb.cpu())
            ordered_vids.extend(vids)
    embs = torch.cat(embs, dim=0).numpy().astype("float32")
    return ordered_vids, embs


@torch.no_grad()
def encode_all_queries(model, loader, device, fp16=True, query_ablation="full"):
    """Encode all (history + query) pairs → (embs [N,D], target_vids, qids).

    ``query_ablation`` controls the state-conditioned query ablation:
    "full" (default), "text_only" (zero history), "history_only" (zero text).
    """
    use_pretrained = getattr(model, "use_pretrained_text_feat", False)
    embs, target_vids, qids = [], [], []
    model.eval()
    for batch in tqdm(loader, desc="Encoding queries"):
        h_feat = batch["history_feat"].to(device)
        h_mask = batch["history_mask"].to(device)

        if query_ablation == "text_only":
            h_feat = torch.zeros_like(h_feat)

        with autocast(enabled=fp16):
            if use_pretrained:
                pre_text = batch["pre_text_feat"].to(device)
                if query_ablation == "history_only":
                    pre_text = torch.zeros_like(pre_text)
                q_emb = model.encode_query(h_feat, h_mask,
                                           pre_text_feat=pre_text)
            else:
                if query_ablation == "history_only":
                    raise NotImplementedError(
                        "query_ablation='history_only' currently requires "
                        "use_internvideo=1 (pre_text_feat path)."
                    )
                ids = batch["input_ids"].to(device)
                t_mask = batch["attention_mask"].to(device)
                q_emb = model.encode_query(h_feat, h_mask, ids, t_mask)
        embs.append(q_emb.cpu())
        target_vids.extend(batch["target_vid"])
        qids.extend(
            batch["qid"].tolist() if isinstance(batch["qid"], torch.Tensor)
            else batch["qid"]
        )
    embs = torch.cat(embs, dim=0).numpy().astype("float32")
    return embs, target_vids, qids


def build_faiss_index(embeddings: np.ndarray, use_gpu: bool = False):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    if use_gpu and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(embeddings)
    return index


def compute_metrics(query_embs, gt_vids, vid_list, index,
                    top_k=(1, 5, 10, 50),
                    score_reduce: str = "max"):
    """Full-library retrieval: search & compute recall metrics.

    Supports both single-vector and multi-event (MaxSim/MeanSim) modes.
    Multi-event is detected when vid_list has repeated entries
    (each video appears num_events times consecutively).
    """
    assert score_reduce in ("max", "mean"), \
        f"score_reduce must be 'max' or 'mean', got {score_reduce!r}"

    unique_vids = list(dict.fromkeys(vid_list))
    is_multi_event = len(vid_list) > len(unique_vids)
    n_unique = len(unique_vids)

    ranks = []
    results = []
    n_gt_missing = 0

    if is_multi_event:
        mode_name = "MaxSim" if score_reduce == "max" else "MeanSim"
        print(f"[{mode_name}] {n_unique} unique videos × {len(vid_list)//n_unique} events "
              f"= {len(vid_list)} FAISS entries")
        all_scores, all_indices = index.search(query_embs, index.ntotal)

        pool_vid_set = set(vid_list)
        for i in range(len(query_embs)):
            gt_vid = gt_vids[i]
            if gt_vid not in pool_vid_set:
                n_gt_missing += 1
                continue

            if score_reduce == "max":
                vid_scores: dict = {}
                for score, idx in zip(all_scores[i].tolist(), all_indices[i].tolist()):
                    v = vid_list[idx]
                    if v not in vid_scores or score > vid_scores[v]:
                        vid_scores[v] = score
            else:
                sum_scores: dict = {}
                cnt_scores: dict = {}
                for score, idx in zip(all_scores[i].tolist(), all_indices[i].tolist()):
                    v = vid_list[idx]
                    sum_scores[v] = sum_scores.get(v, 0.0) + float(score)
                    cnt_scores[v] = cnt_scores.get(v, 0) + 1
                vid_scores = {
                    v: sum_scores[v] / max(cnt_scores[v], 1)
                    for v in sum_scores
                }

            ranked_vids = sorted(unique_vids,
                                 key=lambda v: vid_scores.get(v, -1e9),
                                 reverse=True)
            rank = ranked_vids.index(gt_vid) + 1
            ranks.append(rank)
            results.append({
                "qid": int(i),
                "gt_vid": gt_vid,
                "rank": int(rank),
                "top5_vids": ranked_vids[:5],
            })
    else:
        vid_to_idx = {v: i for i, v in enumerate(vid_list)}
        max_k = max(top_k)
        scores, indices = index.search(query_embs, min(max_k, index.ntotal))

        for i in range(len(query_embs)):
            gt_idx = vid_to_idx.get(gt_vids[i], -1)
            if gt_idx == -1:
                n_gt_missing += 1
                continue

            retrieved = indices[i].tolist()
            if gt_idx in retrieved:
                rank = retrieved.index(gt_idx) + 1
            else:
                full_scores, full_idx = index.search(
                    query_embs[i : i + 1], index.ntotal
                )
                full_list = full_idx[0].tolist()
                rank = (full_list.index(gt_idx) + 1
                        if gt_idx in full_list else index.ntotal)
            ranks.append(rank)
            results.append({
                "qid": int(i),
                "gt_vid": gt_vids[i],
                "rank": int(rank),
                "top5_vids": [vid_list[j] for j in indices[i][:5].tolist()],
            })

    if n_gt_missing:
        print(f"WARNING: {n_gt_missing} queries have GT video not in pool "
              f"(excluded from metrics)")

    ranks = np.array(ranks)
    metrics = {}
    for k in top_k:
        metrics[f"R@{k}"] = float((ranks <= k).mean() * 100)
    metrics["MedR"] = float(np.median(ranks))
    metrics["MeanR"] = float(np.mean(ranks))
    metrics["num_queries"] = len(ranks)
    metrics["num_videos_in_pool"] = n_unique
    return metrics, results


def main():
    parser = argparse.ArgumentParser(description="Full-library retrieval eval")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_json", type=str, required=True,
                        help="Test set JSON path")
    parser.add_argument("--video_pool", type=str, required=True,
                        help="Text file listing all video IDs in the library "
                             "(one per line)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_results", type=str, default="./eval_results.json")
    parser.add_argument("--faiss_gpu", type=int, default=0,
                        help="Use GPU for FAISS (0=CPU, 1=GPU)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--query_ablation", type=str, default=None,
                        choices=[None, "full", "text_only", "history_only"],
                        help="Override checkpoint's query_ablation mode for eval.")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        fp16 = True
    else:
        device = torch.device("cpu")
        fp16 = False
        print("WARNING: No GPU, running on CPU (fp16 disabled)")

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
    if args.query_ablation is not None:
        cfg.query_ablation = args.query_ablation
    query_ablation = getattr(cfg, "query_ablation", "full") or "full"
    print(f"[Eval] query_ablation = {query_ablation}")

    use_iv = cfg.use_internvideo
    video_feat_dir = (cfg.iv_video_feat_dir if use_iv else cfg.feature_dir)
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
    print(f"Model loaded (epoch {ckpt.get('epoch', '?')})")

    with open(args.video_pool) as f:
        pool_vids = [line.strip() for line in f if line.strip()]
    print(f"Video pool file: {len(pool_vids)} videos")

    t0 = time.time()
    vid_list, video_embs = encode_video_pool(model, pool_vids,
                                             video_feat_dir, cfg, device)
    print(f"Video library encoded: {video_embs.shape}  ({time.time()-t0:.1f}s)")

    index = build_faiss_index(video_embs, use_gpu=bool(args.faiss_gpu))
    print(f"FAISS index: {index.ntotal} vectors, dim={video_embs.shape[1]}")

    test_data = load_data(args.test_json)
    if use_iv:
        iv_query_dirs = [d.strip() for d in cfg.iv_query_feat_dirs.split(",")
                         if d.strip()]
        test_ds = InternVideoRetrievalDataset(
            test_data,
            video_feat_dir=cfg.iv_video_feat_dir,
            query_feat_dirs=iv_query_dirs,
            max_target_frames=cfg.max_target_frames,
            max_history_frames=cfg.max_history_frames,
            feat_dim=cfg.feat_dim,
            cache_features=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.text_encoder_path)
        test_ds = COINRetrievalDataset(
            test_data, cfg.feature_dir, tokenizer,
            cfg.max_target_frames, cfg.max_history_frames,
            cfg.max_text_len, cfg.feat_dim,
            cache_features=True,
        )
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)

    t0 = time.time()
    query_embs, gt_vids, qids = encode_all_queries(model, test_loader,
                                                    device, cfg.fp16,
                                                    query_ablation=query_ablation)
    print(f"Queries encoded: {query_embs.shape}  ({time.time()-t0:.1f}s)")

    metrics, per_query = compute_metrics(
        query_embs,
        gt_vids,
        vid_list,
        index,
        cfg.top_k,
        score_reduce=getattr(cfg, "score_reduce", "max"),
    )

    print("\n" + "=" * 55)
    print("  Full-Library Retrieval Results")
    print("=" * 55)
    for k, v in metrics.items():
        print(f"  {k:>20s} = {v:.2f}")
    print("=" * 55)

    output = {
        "metrics": metrics,
        "checkpoint": args.checkpoint,
        "video_pool": args.video_pool,
        "test_json": args.test_json,
        "per_query": per_query,
    }
    with open(args.save_results, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.save_results}")

    index_dir = os.path.dirname(args.checkpoint)
    index_path = os.path.join(index_dir, "video.index")
    cpu_index = (faiss.index_gpu_to_cpu(index)
                 if args.faiss_gpu and faiss.get_num_gpus() > 0 else index)
    faiss.write_index(cpu_index, index_path)
    vid_map_path = os.path.join(index_dir, "video_vid_list.json")
    with open(vid_map_path, "w") as f:
        json.dump(vid_list, f)
    print(f"FAISS index → {index_path}")
    print(f"Video ID map → {vid_map_path}")


if __name__ == "__main__":
    main()
