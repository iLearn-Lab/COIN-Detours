"""
Same-task retrieval evaluation (oracle upper bound) — does NOT replace test.py.

Compared to test.py (full-library retrieval), this script additionally
restricts the candidate pool per query to videos sharing the COIN task
(category) with the reference video (video1). Task id is parsed from paths:
  .../COINvideos/{task_id}/{youtube_id}.mp4

Workflow is otherwise identical: encode pool → FAISS → encode queries → metrics.
Reports both full-library and same-task metrics for side-by-side comparison.

Usage:
    CUDA_VISIBLE_DEVICES=3 python test1.py \
        --checkpoint ./checkpoints/best.pt \
        --test_json /home/wenan/UniTime-main/data/COIN_test_finalllllly.json \
        --video_pool ./COIN_testing_videos_filtered.txt \
        --save_results ./checkpoints/eval_results_same_task.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from config import Config
from dataset import (COINRetrievalDataset, InternVideoRetrievalDataset,
                     load_data)
from model import ComposedVideoRetriever

# Reuse encoding / FAISS helpers from test.py (no changes to test.py).
from test import (build_faiss_index, compute_metrics, encode_all_queries,
                  encode_video_pool)


# ---------------------------------------------------------------------------
#  Task id helpers
# ---------------------------------------------------------------------------

def task_from_path(path: str) -> str:
    """Extract COIN task folder id from a video path."""
    parts = path.replace("\\", "/").split("/")
    for i, part in enumerate(parts):
        if part == "COINvideos" and i + 1 < len(parts):
            return parts[i + 1]
    return os.path.basename(os.path.dirname(path))


def vid_from_path(path: str) -> str:
    return os.path.basename(path).split(".")[0]


def build_vid_to_task_map(
    pool_vids: Sequence[str],
    coin_video_roots: Sequence[str],
    extra_json_paths: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, str], List[str]]:
    """Map youtube id → task id by scanning COINvideos dirs + optional JSONs."""
    vid_to_task: Dict[str, str] = {}
    pool_set = set(pool_vids)

    for json_path in extra_json_paths or []:
        if not json_path or not os.path.isfile(json_path):
            continue
        data = load_data(json_path)
        for item in data:
            for key in ("video1_path", "video2_path"):
                p = item.get(key)
                if not p:
                    continue
                vid_to_task[vid_from_path(p)] = task_from_path(p)
        print(f"[TaskMap] +{len(data)} samples from {json_path} "
              f"(total mapped so far: {len(vid_to_task)})")

    for root in coin_video_roots:
        if not root or not os.path.isdir(root):
            print(f"[TaskMap] skip missing root: {root}")
            continue
        n_before = len(vid_to_task)
        for task_name in os.listdir(root):
            task_dir = os.path.join(root, task_name)
            if not os.path.isdir(task_dir):
                continue
            for fname in os.listdir(task_dir):
                if not fname.endswith(".mp4"):
                    continue
                vid = os.path.splitext(fname)[0]
                if vid in pool_set and vid not in vid_to_task:
                    vid_to_task[vid] = task_name
        print(f"[TaskMap] scanned {root}: +{len(vid_to_task) - n_before} pool vids")

    missing = [v for v in pool_vids if v not in vid_to_task]
    return vid_to_task, missing


def query_tasks_from_dataset(test_ds) -> List[str]:
    """One task id per query, aligned with DataLoader order (test_ds.data)."""
    return [task_from_path(item["video1_path"]) for item in test_ds.data]


# ---------------------------------------------------------------------------
#  Same-task metrics
# ---------------------------------------------------------------------------

def compute_metrics_same_task(
    query_embs: np.ndarray,
    gt_vids: List[str],
    query_tasks: List[str],
    vid_list: List[str],
    vid_to_task: Dict[str, str],
    index: faiss.Index,
    top_k: Tuple[int, ...] = (1, 5, 10, 50),
    score_reduce: str = "max",
) -> Tuple[dict, list, dict]:
    """Retrieval metrics with candidates restricted to the query's task."""
    assert score_reduce in ("max", "mean"), \
        f"score_reduce must be 'max' or 'mean', got {score_reduce!r}"
    assert len(query_tasks) == len(gt_vids) == len(query_embs), \
        "query_tasks must align with queries"

    unique_vids = list(dict.fromkeys(vid_list))
    is_multi_event = len(vid_list) > len(unique_vids)
    n_unique = len(unique_vids)

    # task -> unique vids in pool (for stats)
    task_pool_vids: Dict[str, List[str]] = defaultdict(list)
    for v in unique_vids:
        t = vid_to_task.get(v)
        if t is not None:
            task_pool_vids[t].append(v)

    ranks = []
    results = []
    n_gt_missing = 0
    n_gt_not_in_task_pool = 0
    n_query_task_unknown = 0
    pool_sizes = []

    if is_multi_event:
        mode_name = "MaxSim" if score_reduce == "max" else "MeanSim"
        print(f"[Same-task {mode_name}] filtering after full FAISS search")
        all_scores, all_indices = index.search(query_embs, index.ntotal)

        for i in range(len(query_embs)):
            gt_vid = gt_vids[i]
            task = query_tasks[i]
            if task is None:
                n_query_task_unknown += 1
                continue

            candidates = task_pool_vids.get(task, [])
            pool_sizes.append(len(candidates))
            if gt_vid not in set(candidates):
                if gt_vid not in set(unique_vids):
                    n_gt_missing += 1
                else:
                    n_gt_not_in_task_pool += 1
                continue

            if score_reduce == "max":
                vid_scores: dict = {}
                for score, idx in zip(all_scores[i].tolist(), all_indices[i].tolist()):
                    v = vid_list[idx]
                    if vid_to_task.get(v) != task:
                        continue
                    if v not in vid_scores or score > vid_scores[v]:
                        vid_scores[v] = score
            else:
                sum_scores: dict = {}
                cnt_scores: dict = {}
                for score, idx in zip(all_scores[i].tolist(), all_indices[i].tolist()):
                    v = vid_list[idx]
                    if vid_to_task.get(v) != task:
                        continue
                    sum_scores[v] = sum_scores.get(v, 0.0) + float(score)
                    cnt_scores[v] = cnt_scores.get(v, 0) + 1
                vid_scores = {
                    v: sum_scores[v] / max(cnt_scores[v], 1)
                    for v in sum_scores
                }

            ranked_vids = sorted(candidates,
                                 key=lambda v: vid_scores.get(v, -1e9),
                                 reverse=True)
            rank = ranked_vids.index(gt_vid) + 1
            ranks.append(rank)
            results.append({
                "qid": int(i),
                "gt_vid": gt_vid,
                "task_id": task,
                "rank": int(rank),
                "task_pool_size": len(candidates),
                "top5_vids": ranked_vids[:5],
            })
    else:
        vid_to_idx = {v: i for i, v in enumerate(vid_list)}
        all_scores, all_indices = index.search(query_embs, index.ntotal)

        for i in range(len(query_embs)):
            gt_vid = gt_vids[i]
            task = query_tasks[i]
            if task is None:
                n_query_task_unknown += 1
                continue

            candidates = task_pool_vids.get(task, [])
            pool_sizes.append(len(candidates))
            gt_idx = vid_to_idx.get(gt_vid, -1)
            if gt_idx == -1:
                n_gt_missing += 1
                continue
            if gt_vid not in candidates:
                n_gt_not_in_task_pool += 1
                continue

            cand_set = set(candidates)
            scored = []
            for score, idx in zip(all_scores[i].tolist(), all_indices[i].tolist()):
                v = vid_list[idx]
                if v in cand_set:
                    scored.append((v, score))
            ranked_vids = [v for v, _ in sorted(scored, key=lambda x: -x[1])]
            rank = ranked_vids.index(gt_vid) + 1
            ranks.append(rank)
            results.append({
                "qid": int(i),
                "gt_vid": gt_vid,
                "task_id": task,
                "rank": int(rank),
                "task_pool_size": len(candidates),
                "top5_vids": ranked_vids[:5],
            })

    if n_gt_missing:
        print(f"WARNING: {n_gt_missing} GT videos not in full pool (excluded)")
    if n_gt_not_in_task_pool:
        print(f"WARNING: {n_gt_not_in_task_pool} GT videos not in same-task pool")
    if n_query_task_unknown:
        print(f"WARNING: {n_query_task_unknown} queries with unknown task")

    ranks_arr = np.array(ranks)
    metrics = {}
    for k in top_k:
        metrics[f"R@{k}"] = float((ranks_arr <= k).mean() * 100) if len(ranks_arr) else 0.0
    metrics["MedR"] = float(np.median(ranks_arr)) if len(ranks_arr) else 0.0
    metrics["MeanR"] = float(np.mean(ranks_arr)) if len(ranks_arr) else 0.0
    metrics["num_queries"] = len(ranks_arr)
    metrics["num_videos_in_pool"] = n_unique
    metrics["avg_task_pool_size"] = float(np.mean(pool_sizes)) if pool_sizes else 0.0
    metrics["median_task_pool_size"] = float(np.median(pool_sizes)) if pool_sizes else 0.0

    diag = {
        "n_gt_missing": n_gt_missing,
        "n_gt_not_in_task_pool": n_gt_not_in_task_pool,
        "n_query_task_unknown": n_query_task_unknown,
        "num_tasks_in_pool": len(task_pool_vids),
    }
    return metrics, results, diag


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Same-task retrieval eval (oracle upper bound)"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_json", type=str, required=True)
    parser.add_argument("--video_pool", type=str, required=True)
    parser.add_argument(
        "--coin_video_roots",
        type=str,
        default="/home/wenan/COINvideos,/data/wenan-data/COINvideos",
        help="Comma-separated COINvideos roots to build vid→task map",
    )
    parser.add_argument(
        "--extra_task_json",
        type=str,
        default="",
        help="Optional extra JSON(s), comma-separated, to supplement task map",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--save_results",
        type=str,
        default="",
        help="Default: <checkpoint_dir>/eval_results_same_task.json",
    )
    parser.add_argument("--faiss_gpu", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--query_ablation", type=str, default=None,
                        choices=[None, "full", "text_only", "history_only"])
    args = parser.parse_args()

    if not args.save_results:
        args.save_results = os.path.join(
            os.path.dirname(args.checkpoint), "eval_results_same_task.json"
        )

    # ---- device ----
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        fp16 = True
    else:
        device = torch.device("cpu")
        fp16 = False
        print("WARNING: No GPU, running on CPU (fp16 disabled)")

    # ---- checkpoint / config ----
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
    video_feat_dir = cfg.iv_video_feat_dir if use_iv else cfg.feature_dir
    print(f"[Mode] {'InternVideo' if use_iv else 'SigLIP'}  "
          f"video_feat_dir={video_feat_dir}")

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
        disable_qg_attn=getattr(cfg, "disable_qg_attn", False),
        disable_gated_fusion=getattr(cfg, "disable_gated_fusion", False),
        disable_inter_event_refine=getattr(cfg, "disable_inter_event_refine", False),
        score_reduce=getattr(cfg, "score_reduce", "max"),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Model loaded (epoch {ckpt.get('epoch', '?')})")

    # ---- video pool ----
    with open(args.video_pool) as f:
        pool_vids = [line.strip() for line in f if line.strip()]
    print(f"Video pool file: {len(pool_vids)} videos")

    coin_roots = [r.strip() for r in args.coin_video_roots.split(",") if r.strip()]
    extra_jsons = [p.strip() for p in args.extra_task_json.split(",") if p.strip()]
    extra_jsons = [args.test_json] + extra_jsons

    vid_to_task, missing_pool = build_vid_to_task_map(
        pool_vids, coin_roots, extra_json_paths=extra_jsons
    )
    if missing_pool:
        print(f"[TaskMap] WARNING: {len(missing_pool)}/{len(pool_vids)} pool videos "
              f"have no task id (excluded from same-task pool)")
        if len(missing_pool) <= 10:
            print(f"  missing: {missing_pool}")

    # ---- encode library & FAISS ----
    t0 = time.time()
    vid_list, video_embs = encode_video_pool(
        model, pool_vids, video_feat_dir, cfg, device
    )
    print(f"Video library encoded: {video_embs.shape}  ({time.time() - t0:.1f}s)")

    index = build_faiss_index(video_embs, use_gpu=bool(args.faiss_gpu))
    print(f"FAISS index: {index.ntotal} vectors, dim={video_embs.shape[1]}")

    # ---- test queries ----
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
    query_tasks = query_tasks_from_dataset(test_ds)
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    t0 = time.time()
    query_embs, gt_vids, qids = encode_all_queries(
        model, test_loader, device, cfg.fp16,
        query_ablation=query_ablation,
    )
    print(f"Queries encoded: {query_embs.shape}  ({time.time() - t0:.1f}s)")

    score_reduce = getattr(cfg, "score_reduce", "max")
    top_k = cfg.top_k

    # ---- full-library baseline (same as test.py) ----
    metrics_full, per_query_full = compute_metrics(
        query_embs, gt_vids, vid_list, index, top_k, score_reduce=score_reduce
    )

    # ---- same-task oracle retrieval ----
    metrics_task, per_query_task, diag_task = compute_metrics_same_task(
        query_embs, gt_vids, query_tasks, vid_list, vid_to_task, index,
        top_k, score_reduce=score_reduce,
    )

    print("\n" + "=" * 55)
    print("  Full-Library Retrieval (baseline, same as test.py)")
    print("=" * 55)
    for k, v in metrics_full.items():
        print(f"  {k:>24s} = {v:.2f}")

    print("\n" + "=" * 55)
    print("  Same-Task Retrieval (oracle upper bound)")
    print("=" * 55)
    for k, v in metrics_task.items():
        print(f"  {k:>24s} = {v:.2f}")
    print(f"  {'avg_task_pool_size':>24s} = {metrics_task['avg_task_pool_size']:.2f}")
    print("=" * 55)

    output = {
        "eval_type": "same_task_oracle",
        "metrics_full_library": metrics_full,
        "metrics_same_task": metrics_task,
        "same_task_diagnostics": diag_task,
        "checkpoint": args.checkpoint,
        "video_pool": args.video_pool,
        "test_json": args.test_json,
        "coin_video_roots": coin_roots,
        "vid_to_task_coverage": {
            "pool_size": len(pool_vids),
            "mapped": len(pool_vids) - len(missing_pool),
            "missing": len(missing_pool),
        },
        "per_query_full_library": per_query_full,
        "per_query_same_task": per_query_task,
    }
    with open(args.save_results, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.save_results}")
    print("(test1.py does not write video.index / video_vid_list.json)")


if __name__ == "__main__":
    main()
