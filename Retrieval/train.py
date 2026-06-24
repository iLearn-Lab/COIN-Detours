"""
Training script for Composed Video Retrieval.

Usage:
    python train.py --batch_size 32 --max_epochs 50 --lr 1e-4

Supports AMP (fp16), cosine-warmup LR schedule, gradient clipping,
and periodic validation with R@K metrics.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer
from tqdm import tqdm

from config import Config
from dataset import (Blip2RetrievalDataset, COINRetrievalDataset,
                     InternVideoRetrievalDataset, load_data,
                     split_by_reference_video)
from model import ComposedVideoRetriever


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed():
    """Initialize DDP if launched via torchrun. Returns (rank, local_rank, world_size)."""
    if "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) <= 1:
        return 0, 0, 1
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return rank, local_rank, world_size


def gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    """All-gather tensors across GPUs, keeping gradient for local shard."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor.detach())
    gathered[rank] = tensor
    return torch.cat(gathered, dim=0)


def cosine_warmup_schedule(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup then cosine decay."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_optimizer(model: ComposedVideoRetriever, cfg: Config):
    text_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "text_model" in name:
            text_params.append(p)
        else:
            other_params.append(p)

    groups = [{"params": other_params, "lr": cfg.lr}]
    if text_params:
        groups.append({"params": text_params, "lr": cfg.text_lr})

    return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)


def _use_pretrained_text_feat(cfg: Config) -> bool:
    return bool(cfg.use_internvideo or cfg.use_blip2)


def _encode_query_batch(model: ComposedVideoRetriever, batch, device):
    history_feat = batch["history_feat"].to(device)
    history_mask = batch["history_mask"].to(device)
    if getattr(model, "use_pretrained_text_feat", False):
        pre_text = batch["pre_text_feat"].to(device)
        return model.encode_query(history_feat, history_mask,
                                  pre_text_feat=pre_text)
    input_ids = batch["input_ids"].to(device)
    text_mask = batch["attention_mask"].to(device)
    return model.encode_query(history_feat, history_mask,
                              input_ids, text_mask)


def _build_datasets(cfg: Config, train_data, val_data, tokenizer, cache: bool):
    use_hard_neg = cfg.max_hard_neg > 0
    if cfg.use_internvideo:
        iv_query_dirs = [d.strip() for d in cfg.iv_query_feat_dirs.split(",")
                         if d.strip()]
        ds_kwargs = dict(
            video_feat_dir=cfg.iv_video_feat_dir,
            query_feat_dirs=iv_query_dirs,
            max_target_frames=cfg.max_target_frames,
            max_history_frames=cfg.max_history_frames,
            feat_dim=cfg.feat_dim,
            cache_features=cache,
        )
        train_ds = InternVideoRetrievalDataset(
            train_data, **ds_kwargs,
            neg_json=cfg.neg_json if use_hard_neg else None,
            max_hard_neg=cfg.max_hard_neg,
        )
        val_ds = InternVideoRetrievalDataset(val_data, **ds_kwargs)
    elif cfg.use_blip2:
        blip2_query_dirs = [d.strip() for d in cfg.blip2_query_feat_dirs.split(",")
                            if d.strip()]
        ds_kwargs = dict(
            video_feat_dir=cfg.blip2_video_feat_dir,
            query_feat_dirs=blip2_query_dirs,
            max_target_frames=cfg.max_target_frames,
            max_history_frames=cfg.max_history_frames,
            feat_dim=cfg.feat_dim,
            cache_features=cache,
        )
        train_ds = Blip2RetrievalDataset(
            train_data, **ds_kwargs,
            neg_json=cfg.neg_json if use_hard_neg else None,
            max_hard_neg=cfg.max_hard_neg,
        )
        val_ds = Blip2RetrievalDataset(val_data, **ds_kwargs)
    else:
        train_ds = COINRetrievalDataset(
            train_data, cfg.feature_dir, tokenizer,
            cfg.max_target_frames, cfg.max_history_frames,
            cfg.max_text_len, cfg.feat_dim,
            cache_features=cache,
            neg_json=cfg.neg_json if use_hard_neg else None,
            max_hard_neg=cfg.max_hard_neg,
        )
        val_ds = COINRetrievalDataset(
            val_data, cfg.feature_dir, tokenizer,
            cfg.max_target_frames, cfg.max_history_frames,
            cfg.max_text_len, cfg.feat_dim,
            cache_features=cache,
        )
    return train_ds, val_ds


@torch.no_grad()
def evaluate(model: ComposedVideoRetriever, loader: DataLoader,
             top_k=(1, 5, 10, 50), device="cuda"):
    model.eval()

    all_video_embs, all_query_embs = [], []
    all_target_vids, all_qids = [], []

    for batch in tqdm(loader, desc="Eval", leave=False):
        target_feat = batch["target_feat"].to(device)
        target_mask = batch["target_mask"].to(device)
        with autocast(enabled=True):
            v_emb = model.encode_video(target_feat, target_mask)
            q_emb = _encode_query_batch(model, batch, device)

        all_video_embs.append(v_emb.cpu())
        all_query_embs.append(q_emb.cpu())
        all_target_vids.extend(batch["target_vid"])
        all_qids.extend(batch["qid"].tolist() if isinstance(batch["qid"], torch.Tensor)
                        else batch["qid"])

    all_video_embs = torch.cat(all_video_embs, dim=0)
    all_query_embs = torch.cat(all_query_embs, dim=0)

    vid_to_idx = {}
    unique_embs = []
    for i, vid in enumerate(all_target_vids):
        if vid not in vid_to_idx:
            vid_to_idx[vid] = len(unique_embs)
            unique_embs.append(all_video_embs[i])
    unique_embs = torch.stack(unique_embs)

    gt_indices = [vid_to_idx[vid] for vid in all_target_vids]

    N_q = all_query_embs.size(0)
    if unique_embs.dim() == 3:
        V, E, D = unique_embs.shape
        raw = all_query_embs @ unique_embs.reshape(V * E, D).T
        sim = raw.reshape(N_q, V, E).max(dim=-1).values
    else:
        sim = all_query_embs @ unique_embs.T

    ranks = []
    for i in range(sim.size(0)):
        sorted_idx = sim[i].argsort(descending=True).tolist()
        rank = sorted_idx.index(gt_indices[i]) + 1
        ranks.append(rank)

    ranks = np.array(ranks)
    metrics = {}
    for k in top_k:
        metrics[f"R@{k}"] = float((ranks <= k).mean() * 100)
    metrics["MedR"] = float(np.median(ranks))
    metrics["MeanR"] = float(np.mean(ranks))
    metrics["num_queries"] = len(ranks)
    metrics["num_videos"] = len(unique_embs)
    return metrics


def train_one_epoch(model, loader, optimizer, scheduler, scaler, cfg, device,
                    epoch, global_step, rank=0, world_size=1):
    raw_model = model.module if isinstance(model, DDP) else model
    model.train()
    meter = defaultdict(float)
    n_batches = 0
    is_main = (rank == 0)
    is_dist = (world_size > 1)

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False, disable=not is_main)
    for batch in pbar:
        target_feat = batch["target_feat"].to(device)
        target_mask = batch["target_mask"].to(device)
        hn_feat = batch["hard_neg_feat"].to(device)
        hn_mask = batch["hard_neg_mask"].to(device)
        hn_valid = batch["hard_neg_valid"].to(device)

        with autocast(enabled=cfg.fp16):
            query_emb = _encode_query_batch(raw_model, batch, device)

            if hn_valid.any():
                B, K, T, D = hn_feat.shape
                valid_flat = hn_valid.reshape(-1)
                valid_hn_feat = hn_feat.reshape(B*K, T, D)[valid_flat]
                valid_hn_mask = hn_mask.reshape(B*K, T)[valid_flat]

                all_vfeat = torch.cat([target_feat, valid_hn_feat])
                all_vmask = torch.cat([target_mask, valid_hn_mask])
                all_vemb = raw_model.encode_video(all_vfeat, all_vmask)

                video_emb = all_vemb[:B]
                if all_vemb.dim() == 3:
                    _, num_ev, embed_dim = all_vemb.shape
                    hn_emb = all_vemb.new_zeros(B * K, num_ev, embed_dim)
                    hn_emb[valid_flat] = all_vemb[B:]
                    hn_emb = hn_emb.reshape(B, K, num_ev, embed_dim)
                else:
                    embed_dim = all_vemb.size(-1)
                    hn_emb = all_vemb.new_zeros(B * K, embed_dim)
                    hn_emb[valid_flat] = all_vemb[B:]
                    hn_emb = hn_emb.reshape(B, K, embed_dim)
            else:
                video_emb = raw_model.encode_video(target_feat, target_mask)
                hn_emb, hn_valid = None, None

            if is_dist:
                video_emb_all = gather_with_grad(video_emb)
                query_emb_all = gather_with_grad(query_emb)
                labels_offset = rank * video_emb.size(0)
            else:
                video_emb_all, query_emb_all, labels_offset = None, None, 0

            loss, stats = raw_model.compute_loss(
                video_emb, query_emb,
                hard_neg_emb=hn_emb, hard_neg_valid=hn_valid,
                video_emb_all=video_emb_all,
                query_emb_all=query_emb_all,
                labels_offset=labels_offset,
            )

            if cfg.lambda_orth > 0 and video_emb.dim() == 3:
                orth_loss = raw_model.orthogonality_loss(video_emb)
                loss = loss + cfg.lambda_orth * orth_loss
                stats["loss_orth"] = orth_loss.item()
            else:
                stats["loss_orth"] = 0.0

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        global_step += 1

        for k, v in stats.items():
            meter[k] += v
        n_batches += 1

        if is_main and global_step % cfg.log_interval == 0:
            avg = {k: v / n_batches for k, v in meter.items()}
            lr_now = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(
                loss=f"{avg['loss']:.4f}",
                acc=f"{avg['acc_q2v']:.2%}",
                lr=f"{lr_now:.2e}",
            )

    avg = {k: v / max(n_batches, 1) for k, v in meter.items()}
    return avg, global_step


def main():
    cfg = Config()

    parser = argparse.ArgumentParser()
    for field_name, field_obj in cfg.__dataclass_fields__.items():
        ftype = field_obj.type
        if ftype == bool:
            parser.add_argument(f"--{field_name}", type=int,
                                default=int(getattr(cfg, field_name)))
        elif ftype in (int, float, str):
            parser.add_argument(f"--{field_name}", type=ftype,
                                default=getattr(cfg, field_name))
    parser.add_argument("--no_cache", action="store_true",
                        help="Skip pre-loading features into RAM")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment name (auto-generated if not set)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint for loading model weights (fine-tune)")
    parser.add_argument(
        "--save_best_only",
        action="store_true",
        help="If set, disable periodic/final checkpoints and only save best.pt",
    )
    args = parser.parse_args()

    no_cache = args.no_cache
    for k, v in vars(args).items():
        if k in ("no_cache", "exp_name", "resume"):
            continue
        if hasattr(cfg, k):
            expected = type(getattr(cfg, k))
            if expected == bool:
                setattr(cfg, k, bool(v))
            else:
                setattr(cfg, k, v)

    rank, local_rank, world_size = setup_distributed()
    is_main = (rank == 0)
    is_dist = (world_size > 1)

    if args.exp_name:
        exp_name = args.exp_name
    else:
        from datetime import datetime
        hn_tag = f"_hn{cfg.max_hard_neg}" if cfg.max_hard_neg > 0 else "_noHN"
        exp_name = (f"bs{cfg.batch_size}_lr{cfg.lr}"
                    f"_ep{cfg.max_epochs}{hn_tag}"
                    f"_{datetime.now().strftime('%m%d_%H%M')}")
    cfg.output_dir = os.path.join(cfg.output_dir, exp_name)
    if is_main:
        os.makedirs(cfg.output_dir, exist_ok=True)
        print(f"Experiment: {exp_name}")
        print(f"Output dir: {cfg.output_dir}")
        from datetime import datetime
        run_meta = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "command": " ".join(sys.argv),
            "exp_name": exp_name,
            "output_dir": cfg.output_dir,
            "no_cache": bool(no_cache),
            "resume": args.resume,
            "save_best_only": bool(args.save_best_only),
            "config": cfg.__dict__,
        }
        with open(os.path.join(cfg.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump(run_meta, f, ensure_ascii=False, indent=2)
        with open(os.path.join(cfg.output_dir, "cmd.txt"), "w", encoding="utf-8") as f:
            f.write(run_meta["command"] + "\n")
    if is_dist:
        dist.barrier()

    set_seed(cfg.seed + rank)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        if is_main:
            print(f"Using {world_size} GPU(s): {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        cfg.fp16 = False
        if is_main:
            print("WARNING: No GPU detected, running on CPU (fp16 disabled)")

    use_pre_text = _use_pretrained_text_feat(cfg)
    tokenizer = None
    if not use_pre_text:
        tokenizer = AutoTokenizer.from_pretrained(cfg.text_encoder_path)

    if is_main:
        print("Loading data …")
    train_data = load_data(cfg.train_json)
    if cfg.val_json:
        val_data = load_data(cfg.val_json)
        if is_main:
            print(f"[Data] train={len(train_data)} (full)  "
                  f"val={len(val_data)} (from {cfg.val_json})")
    else:
        train_data, val_data = split_by_reference_video(
            train_data, cfg.val_ratio, cfg.seed)
        if is_main:
            print(f"[Data] train={len(train_data)}  val={len(val_data)} "
                  f"(random split, seed={cfg.seed})")

    cache = not no_cache
    train_ds, val_ds = _build_datasets(cfg, train_data, val_data, tokenizer, cache)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_dist else None
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=(train_sampler is None),
                              sampler=train_sampler,
                              num_workers=cfg.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size,
                            shuffle=False, num_workers=cfg.num_workers,
                            pin_memory=True)

    if is_main:
        print("Building model …")
    model = ComposedVideoRetriever(
        feat_dim=cfg.feat_dim,
        embed_dim=cfg.embed_dim,
        num_temporal_layers=cfg.num_temporal_layers,
        num_heads=cfg.num_heads,
        ff_dim=cfg.ff_dim,
        dropout=cfg.dropout,
        text_encoder_path=cfg.text_encoder_path,
        freeze_text_encoder=cfg.freeze_text_encoder,
        init_temperature=cfg.init_temperature,
        num_events=cfg.num_events,
        use_pretrained_text_feat=use_pre_text,
    ).to(device)

    if args.resume:
        assert os.path.isfile(args.resume), f"Checkpoint not found: {args.resume}"
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        prev_epoch = ckpt.get("epoch", "?")
        prev_metrics = ckpt.get("metrics", {})
        prev_r1 = prev_metrics.get("R@1", "N/A")
        if is_main:
            print(f"  ✓ Loaded weights from {args.resume}  (epoch {prev_epoch}, R@1={prev_r1})")
        del ckpt

    if is_dist:
        model = DDP(model, device_ids=[local_rank],
                    find_unused_parameters=False)

    raw_model = model.module if is_dist else model
    n_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    if is_main:
        print(f"Trainable parameters: {n_params:,}")

    optimizer = build_optimizer(model, cfg)
    total_steps = len(train_loader) * cfg.max_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = cosine_warmup_schedule(optimizer, warmup_steps, total_steps)
    scaler = GradScaler(enabled=cfg.fp16)

    best_r1 = 0.0
    best_epoch = None
    global_step = 0

    for epoch in range(1, cfg.max_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        t0 = time.time()
        train_stats, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            cfg, device, epoch, global_step,
            rank=rank, world_size=world_size,
        )
        elapsed = time.time() - t0

        if is_main:
            log_msg = (f"[Epoch {epoch}/{cfg.max_epochs}] "
                       f"loss={train_stats['loss']:.4f}  "
                       f"acc_q2v={train_stats['acc_q2v']:.2%}  "
                       f"acc_v2q={train_stats['acc_v2q']:.2%}  "
                       f"scale={train_stats['logit_scale']:.2f}  "
                       f"time={elapsed:.0f}s")
            print(log_msg)

        if is_main and epoch % cfg.eval_epoch_interval == 0:
            eval_model = raw_model
            metrics = evaluate(eval_model, val_loader, cfg.top_k, device)
            metric_str = "  ".join(f"{k}={v:.2f}" for k, v in metrics.items())
            print(f"  [Val] {metric_str}")

            if metrics["R@1"] > best_r1:
                best_r1 = metrics["R@1"]
                best_epoch = epoch
                save_path = os.path.join(cfg.output_dir, "best.pt")
                torch.save({
                    "epoch": epoch,
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "metrics": metrics,
                    "config": cfg.__dict__,
                }, save_path)
                print(f"  ★ New best R@1={best_r1:.2f}%  → saved {save_path}")
                with open(os.path.join(cfg.output_dir, "best.json"), "w", encoding="utf-8") as f:
                    json.dump(
                        {"best_epoch": best_epoch, "best_R@1": best_r1, "metrics": metrics},
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )

        if (not args.save_best_only) and is_main and cfg.save_epoch_interval > 0 and epoch % cfg.save_epoch_interval == 0:
            ckpt_path = os.path.join(cfg.output_dir, f"epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg.__dict__,
            }, ckpt_path)

        if is_dist:
            dist.barrier()

    if is_main:
        if not args.save_best_only:
            final_path = os.path.join(cfg.output_dir, "final.pt")
            torch.save({"epoch": cfg.max_epochs, "model": raw_model.state_dict(),
                        "config": cfg.__dict__}, final_path)
        print(
            f"Training complete.  Best R@1 = {best_r1:.2f}%"
            + (f"  (epoch {best_epoch})" if best_epoch is not None else "")
        )

    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
