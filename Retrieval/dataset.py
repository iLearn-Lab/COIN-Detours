"""
Dataset for Composed Video Retrieval.

Each sample consists of:
  - target video  (full pre-extracted frame features)
  - history video (features up to the jump timestamp)
  - text query
  - (optional) hard negative target videos

Classes
-------
COINRetrievalDataset        -- original SigLIP features + live tokenisation
InternVideoRetrievalDataset -- InternVideo features (768-d) + pre-extracted
                               query embeddings (no tokeniser required)
Blip2RetrievalDataset       -- BLIP-2 features (256-d) from CoVR-style embs
                               ``{task_id}/{vid}.pth`` → ``[T, 32, 256]``
"""

import os
import json
import random
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def task_id_from_path(path: str) -> int:
    """COIN category id from ``.../COINvideos/{task_id}/file.mp4``."""
    parts = path.replace("\\", "/").split("/")
    for i, part in enumerate(parts):
        if part == "COINvideos" and i + 1 < len(parts):
            return int(parts[i + 1])
    return int(os.path.basename(os.path.dirname(path)))


class COINRetrievalDataset(Dataset):
    """Load pre-extracted SigLIP frame features + tokenised queries."""

    def __init__(
        self,
        data_list: List[dict],
        feature_dir: str,
        tokenizer,
        max_target_frames: int = 128,
        max_history_frames: int = 64,
        max_text_len: int = 64,
        feat_dim: int = 1152,
        cache_features: bool = True,
        neg_json: Optional[str] = None,
        max_hard_neg: int = 3,
    ):
        self.feature_dir = feature_dir
        self.tokenizer = tokenizer
        self.max_target_frames = max_target_frames
        self.max_history_frames = max_history_frames
        self.max_text_len = max_text_len
        self.feat_dim = feat_dim
        self.max_hard_neg = max_hard_neg

        self.data = self._filter_valid(data_list)
        self._build_hard_neg_map(neg_json)

        self._cache: Dict[str, torch.Tensor] = {}
        if cache_features:
            self._preload_features()

    # ------------------------------------------------------------------
    #  Filtering & pre-loading
    # ------------------------------------------------------------------

    def _vid_from_path(self, path: str) -> str:
        return os.path.basename(path).split(".")[0]

    def _feat_path(self, vid: str) -> str:
        return os.path.join(self.feature_dir, f"{vid}.pth.tar")

    def _filter_valid(self, data_list: List[dict]) -> List[dict]:
        valid = []
        for item in data_list:
            v1 = self._vid_from_path(item["video1_path"])
            v2 = self._vid_from_path(item["video2_path"])
            if os.path.exists(self._feat_path(v1)) and os.path.exists(self._feat_path(v2)):
                valid.append(item)
        print(f"[Dataset] {len(data_list)} → {len(valid)} valid samples "
              f"({len(data_list) - len(valid)} skipped due to missing features)")
        return valid

    def _build_hard_neg_map(self, neg_json: Optional[str]):
        """Build mapping: qid → list of hard negative target video ids."""
        self.hard_neg_map: Dict[int, List[str]] = {}
        if neg_json is None or not os.path.exists(neg_json):
            print("[Dataset] No hard negative file provided")
            return

        valid_qids = {item["qid"] for item in self.data}
        neg_data = json.load(open(neg_json))

        n_with_neg = 0
        for entry in neg_data:
            qid = entry["qid"]
            if qid not in valid_qids:
                continue
            negs = entry.get("hard_negatives_same_history_diff_query", [])
            neg_vids = []
            for n in negs:
                vid = self._vid_from_path(n["video2_path"])
                if os.path.exists(self._feat_path(vid)):
                    neg_vids.append(vid)
            if neg_vids:
                self.hard_neg_map[qid] = neg_vids[:self.max_hard_neg]
                n_with_neg += 1

        print(f"[Dataset] Hard negatives: {n_with_neg}/{len(valid_qids)} "
              f"samples have hard negs (max {self.max_hard_neg} per sample)")

    def _preload_features(self):
        vids = set()
        for item in self.data:
            vids.add(self._vid_from_path(item["video1_path"]))
            vids.add(self._vid_from_path(item["video2_path"]))
        for neg_vids in self.hard_neg_map.values():
            vids.update(neg_vids)

        print(f"[Dataset] Pre-loading {len(vids)} unique video features into RAM …")
        for vid in tqdm(vids, desc="Loading features to RAM"):
            path = self._feat_path(vid)
            feat = torch.load(path, map_location="cpu", weights_only=False)
            if feat.dtype == torch.float16:
                feat = feat.float()
            self._cache[vid] = feat
        mem_mb = sum(v.nelement() * v.element_size() for v in self._cache.values()) / 1e6
        print(f"[Dataset] Cached {len(self._cache)} videos, ~{mem_mb:.0f} MB in RAM")

    # ------------------------------------------------------------------
    #  Feature helpers
    # ------------------------------------------------------------------

    def _load_feat(self, video_path: str) -> torch.Tensor:
        vid = self._vid_from_path(video_path)
        return self._load_feat_by_vid(vid)

    def _load_feat_by_vid(self, vid: str) -> torch.Tensor:
        if vid in self._cache:
            return self._cache[vid]
        path = self._feat_path(vid)
        feat = torch.load(path, map_location="cpu", weights_only=False)
        if feat.dtype == torch.float16:
            feat = feat.float()
        return feat

    @staticmethod
    def _uniform_sample(feat: torch.Tensor, max_frames: int) -> torch.Tensor:
        """Uniformly sample `max_frames` from a longer sequence."""
        T = feat.size(0)
        if T <= max_frames:
            return feat
        indices = torch.linspace(0, T - 1, max_frames).long()
        return feat[indices]

    def _pad(self, feat: torch.Tensor, max_frames: int
             ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample / pad to fixed length, return (feat, mask)."""
        feat = self._uniform_sample(feat, max_frames)
        T, D = feat.shape
        if T >= max_frames:
            return feat[:max_frames], torch.ones(max_frames, dtype=torch.bool)
        pad = torch.zeros(max_frames - T, D)
        padded = torch.cat([feat, pad], dim=0)
        mask = torch.zeros(max_frames, dtype=torch.bool)
        mask[:T] = True
        return padded, mask

    # ------------------------------------------------------------------
    #  __getitem__
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]

        # ---- target video (full) ----
        target_feat = self._load_feat(item["video2_path"])
        target_feat, target_mask = self._pad(target_feat, self.max_target_frames)

        # ---- reference video history (up to timestamp) ----
        history_feat = self._load_feat(item["video1_path"])
        startends = item["video1_startends"]
        timestamp = int(startends[0] if isinstance(startends, list) else startends)
        timestamp = max(1, min(timestamp, history_feat.size(0)))
        history_feat = history_feat[:timestamp]
        history_feat, history_mask = self._pad(history_feat, self.max_history_frames)

        # ---- text query ----
        query_text = item["annos"][0]["query"]
        text_enc = self.tokenizer(
            query_text,
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # ---- hard negatives ----
        qid = item["qid"]
        T_tgt = self.max_target_frames

        if self.max_hard_neg > 0:
            neg_vids = self.hard_neg_map.get(qid, [])
            hn_feats, hn_masks = [], []
            for vid in neg_vids:
                f = self._load_feat_by_vid(vid)
                f, m = self._pad(f, T_tgt)
                hn_feats.append(f)
                hn_masks.append(m)

            n_valid = len(hn_feats)
            while len(hn_feats) < self.max_hard_neg:
                hn_feats.append(torch.zeros(T_tgt, self.feat_dim))
                hn_masks.append(torch.zeros(T_tgt, dtype=torch.bool))

            hn_valid = torch.zeros(self.max_hard_neg, dtype=torch.bool)
            hn_valid[:n_valid] = True
            hn_feat_out = torch.stack(hn_feats)     # [K, T, D]
            hn_mask_out = torch.stack(hn_masks)     # [K, T]
        else:
            hn_feat_out = torch.empty(0, T_tgt, self.feat_dim)
            hn_mask_out = torch.empty(0, T_tgt, dtype=torch.bool)
            hn_valid = torch.empty(0, dtype=torch.bool)

        # ---- ground-truth temporal window (normalised to [0, 1]) ----
        duration = item.get("duration", 0)
        annos = item.get("annos", [])
        if annos and annos[0].get("window") and duration > 0:
            w = annos[0]["window"][0]
            gt_span = torch.tensor([w[0] / duration, w[1] / duration],
                                   dtype=torch.float32).clamp(0.0, 1.0)
            gt_span_valid = True
        else:
            gt_span = torch.zeros(2, dtype=torch.float32)
            gt_span_valid = False

        return {
            "target_feat": target_feat,
            "target_mask": target_mask,
            "history_feat": history_feat,
            "history_mask": history_mask,
            "input_ids": text_enc["input_ids"].squeeze(0),
            "attention_mask": text_enc["attention_mask"].squeeze(0),
            "qid": qid,
            "target_vid": self._vid_from_path(item["video2_path"]),
            "hard_neg_feat": hn_feat_out,
            "hard_neg_mask": hn_mask_out,
            "hard_neg_valid": hn_valid,
            "gt_span": gt_span,
            "gt_span_valid": torch.tensor(gt_span_valid, dtype=torch.bool),
            "duration": torch.tensor(duration, dtype=torch.float32),
            "task_id": task_id_from_path(item["video1_path"]),
        }


# ---------------------------------------------------------------------------
#  Helpers for train / val split
# ---------------------------------------------------------------------------

def split_by_reference_video(
    data: List[dict], val_ratio: float = 0.1, seed: int = 42
) -> Tuple[List[dict], List[dict]]:
    """Split data ensuring the same reference video does not leak across sets."""
    groups: Dict[str, List[dict]] = {}
    for item in data:
        key = item["video1_path"]
        groups.setdefault(key, []).append(item)

    keys = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)

    n_val = max(1, int(len(keys) * val_ratio))
    val_keys = set(keys[:n_val])

    train_data, val_data = [], []
    for k in keys:
        (val_data if k in val_keys else train_data).extend(groups[k])

    print(f"[Split] train={len(train_data)}  val={len(val_data)}  "
          f"(ref-video groups: train={len(keys) - n_val}, val={n_val})")
    return train_data, val_data


def load_data(json_path: str) -> List[dict]:
    with open(json_path) as f:
        return json.load(f)


def blip2_video_feat_path(feat_dir: str, vid: str, video_path: str) -> str:
    """Resolve CoVR-style BLIP-2 video embedding path."""
    task_id = task_id_from_path(video_path)
    return os.path.join(feat_dir, str(task_id), f"{vid}.pth")


def pool_blip2_qformer_tokens(feat: torch.Tensor) -> torch.Tensor:
    """``[T, 32, 256]`` → ``[T, 256]`` via mean over Q-Former tokens."""
    if feat.dim() == 3:
        return feat.mean(dim=1)
    return feat


def seconds_to_blip2_frames(seconds: float, duration: float,
                            num_frames: int) -> int:
    """Map a timestamp in seconds to a BLIP-2 frame count (fps-independent)."""
    if duration <= 0 or num_frames <= 0:
        return max(1, num_frames)
    ratio = max(0.0, min(float(seconds), float(duration))) / float(duration)
    return max(1, min(num_frames, int(ratio * num_frames) or 1))


# ---------------------------------------------------------------------------
#  InternVideo dataset (pre-extracted video + query features, 768-dim)
# ---------------------------------------------------------------------------

class InternVideoRetrievalDataset(Dataset):
    """Dataset backed by InternVideo (768-d) pre-extracted features.

    Video features  : ``video_feat_dir/{vid}.pth.tar``  → ``[T, 768]``
    Query features  : ``query_feat_dirs[*]/{qid}.pth.tar`` → ``[768]``
                      The query features are spread across several directories
                      (train / val / test).  All dirs are scanned once at
                      construction time to build a ``qid → path`` index.

    The dataset does **not** require a tokeniser: text query embeddings are
    loaded directly from disk and exposed as ``pre_text_feat`` in each batch.
    """

    def __init__(
        self,
        data_list: List[dict],
        video_feat_dir: str,
        query_feat_dirs: List[str],
        max_target_frames: int = 128,
        max_history_frames: int = 64,
        feat_dim: int = 768,
        cache_features: bool = True,
        neg_json: Optional[str] = None,
        max_hard_neg: int = 3,
    ):
        self.video_feat_dir = video_feat_dir
        self.max_target_frames = max_target_frames
        self.max_history_frames = max_history_frames
        self.feat_dim = feat_dim
        self.max_hard_neg = max_hard_neg

        # Build qid → path index across all query feat dirs
        self._qfeat_index: Dict[int, str] = {}
        for d in query_feat_dirs:
            if not os.path.isdir(d):
                print(f"[InternVideoDataset] WARNING: query feat dir not found: {d}")
                continue
            for fname in os.listdir(d):
                if fname.endswith(".pth.tar"):
                    try:
                        qid = int(fname.split(".")[0])
                        self._qfeat_index[qid] = os.path.join(d, fname)
                    except ValueError:
                        pass
        print(f"[InternVideoDataset] Indexed {len(self._qfeat_index)} query feats "
              f"from {len(query_feat_dirs)} dir(s)")

        self.data = self._filter_valid(data_list)
        self._build_hard_neg_map(neg_json)

        self._vid_cache: Dict[str, torch.Tensor] = {}
        self._qfeat_cache: Dict[int, torch.Tensor] = {}
        if cache_features:
            self._preload_features()

    # ------------------------------------------------------------------
    #  Filtering & pre-loading
    # ------------------------------------------------------------------

    def _vid_from_path(self, path: str) -> str:
        return os.path.basename(path).split(".")[0]

    def _vfeat_path(self, vid: str) -> str:
        return os.path.join(self.video_feat_dir, f"{vid}.pth.tar")

    def _filter_valid(self, data_list: List[dict]) -> List[dict]:
        valid = []
        for item in data_list:
            v1 = self._vid_from_path(item["video1_path"])
            v2 = self._vid_from_path(item["video2_path"])
            qid = item["qid"]
            if (os.path.exists(self._vfeat_path(v1))
                    and os.path.exists(self._vfeat_path(v2))
                    and qid in self._qfeat_index):
                valid.append(item)
        skipped = len(data_list) - len(valid)
        print(f"[InternVideoDataset] {len(data_list)} → {len(valid)} valid samples "
              f"({skipped} skipped due to missing features)")
        return valid

    def _build_hard_neg_map(self, neg_json: Optional[str]):
        self.hard_neg_map: Dict[int, List[str]] = {}
        if neg_json is None or not os.path.exists(neg_json):
            print("[InternVideoDataset] No hard negative file provided")
            return
        valid_qids = {item["qid"] for item in self.data}
        neg_data = json.load(open(neg_json))
        n_with_neg = 0
        for entry in neg_data:
            qid = entry["qid"]
            if qid not in valid_qids:
                continue
            negs = entry.get("hard_negatives_same_history_diff_query", [])
            neg_vids = [
                self._vid_from_path(n["video2_path"])
                for n in negs
                if os.path.exists(self._vfeat_path(self._vid_from_path(n["video2_path"])))
            ]
            if neg_vids:
                self.hard_neg_map[qid] = neg_vids[:self.max_hard_neg]
                n_with_neg += 1
        print(f"[InternVideoDataset] Hard negatives: {n_with_neg}/{len(valid_qids)} "
              f"samples have hard negs (max {self.max_hard_neg} per sample)")

    def _preload_features(self):
        # video features
        vids: set = set()
        for item in self.data:
            vids.add(self._vid_from_path(item["video1_path"]))
            vids.add(self._vid_from_path(item["video2_path"]))
        for neg_vids in self.hard_neg_map.values():
            vids.update(neg_vids)
        print(f"[InternVideoDataset] Pre-loading {len(vids)} video features …")
        for vid in tqdm(vids, desc="Loading video feats"):
            self._vid_cache[vid] = self._load_vid_from_disk(vid)

        # query features
        qids = {item["qid"] for item in self.data}
        print(f"[InternVideoDataset] Pre-loading {len(qids)} query features …")
        for qid in tqdm(qids, desc="Loading query feats"):
            self._qfeat_cache[qid] = self._load_qfeat_from_disk(qid)

        vid_mem = sum(v.nelement() * v.element_size()
                      for v in self._vid_cache.values()) / 1e6
        q_mem = sum(v.nelement() * v.element_size()
                    for v in self._qfeat_cache.values()) / 1e6
        print(f"[InternVideoDataset] RAM usage: video={vid_mem:.0f}MB "
              f"query={q_mem:.1f}MB")

    # ------------------------------------------------------------------
    #  Feature loading helpers
    # ------------------------------------------------------------------

    def _load_vid_from_disk(self, vid: str) -> torch.Tensor:
        feat = torch.load(self._vfeat_path(vid), map_location="cpu",
                          weights_only=False)
        if feat.dtype == torch.float16:
            feat = feat.float()
        return feat

    def _load_qfeat_from_disk(self, qid: int) -> torch.Tensor:
        feat = torch.load(self._qfeat_index[qid], map_location="cpu",
                          weights_only=False)
        if feat.dtype == torch.float16:
            feat = feat.float()
        return feat.squeeze()  # ensure [D]

    def _load_vid(self, vid: str) -> torch.Tensor:
        if vid in self._vid_cache:
            return self._vid_cache[vid]
        return self._load_vid_from_disk(vid)

    def _load_qfeat(self, qid: int) -> torch.Tensor:
        if qid in self._qfeat_cache:
            return self._qfeat_cache[qid]
        return self._load_qfeat_from_disk(qid)

    @staticmethod
    def _uniform_sample(feat: torch.Tensor, max_frames: int) -> torch.Tensor:
        T = feat.size(0)
        if T <= max_frames:
            return feat
        indices = torch.linspace(0, T - 1, max_frames).long()
        return feat[indices]

    def _pad(self, feat: torch.Tensor, max_frames: int
             ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self._uniform_sample(feat, max_frames)
        T, D = feat.shape
        if T >= max_frames:
            return feat[:max_frames], torch.ones(max_frames, dtype=torch.bool)
        pad = torch.zeros(max_frames - T, D)
        padded = torch.cat([feat, pad], dim=0)
        mask = torch.zeros(max_frames, dtype=torch.bool)
        mask[:T] = True
        return padded, mask

    # ------------------------------------------------------------------
    #  __getitem__
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]

        # ---- target video (full) ----
        target_vid = self._vid_from_path(item["video2_path"])
        target_feat = self._load_vid(target_vid)
        target_feat, target_mask = self._pad(target_feat, self.max_target_frames)

        # ---- reference video history (up to timestamp) ----
        history_vid = self._vid_from_path(item["video1_path"])
        history_feat = self._load_vid(history_vid)
        startends = item["video1_startends"]
        timestamp = int(startends[0] if isinstance(startends, list) else startends)
        timestamp = max(1, min(timestamp, history_feat.size(0)))
        history_feat = history_feat[:timestamp]
        history_feat, history_mask = self._pad(history_feat, self.max_history_frames)

        # ---- pre-extracted query feature ----
        qid = item["qid"]
        pre_text_feat = self._load_qfeat(qid)   # [D=768]

        # ---- hard negatives ----
        T_tgt = self.max_target_frames
        if self.max_hard_neg > 0:
            neg_vids = self.hard_neg_map.get(qid, [])
            hn_feats, hn_masks = [], []
            for vid in neg_vids:
                f = self._load_vid(vid)
                f, m = self._pad(f, T_tgt)
                hn_feats.append(f)
                hn_masks.append(m)
            n_valid = len(hn_feats)
            while len(hn_feats) < self.max_hard_neg:
                hn_feats.append(torch.zeros(T_tgt, self.feat_dim))
                hn_masks.append(torch.zeros(T_tgt, dtype=torch.bool))
            hn_valid = torch.zeros(self.max_hard_neg, dtype=torch.bool)
            hn_valid[:n_valid] = True
            hn_feat_out = torch.stack(hn_feats)
            hn_mask_out = torch.stack(hn_masks)
        else:
            hn_feat_out = torch.empty(0, T_tgt, self.feat_dim)
            hn_mask_out = torch.empty(0, T_tgt, dtype=torch.bool)
            hn_valid = torch.empty(0, dtype=torch.bool)

        # ---- ground-truth temporal window ----
        duration = item.get("duration", 0)
        annos = item.get("annos", [])
        if annos and annos[0].get("window") and duration > 0:
            w = annos[0]["window"][0]
            gt_span = torch.tensor([w[0] / duration, w[1] / duration],
                                   dtype=torch.float32).clamp(0.0, 1.0)
            gt_span_valid = True
        else:
            gt_span = torch.zeros(2, dtype=torch.float32)
            gt_span_valid = False

        return {
            "target_feat": target_feat,
            "target_mask": target_mask,
            "history_feat": history_feat,
            "history_mask": history_mask,
            "pre_text_feat": pre_text_feat,
            "qid": qid,
            "target_vid": target_vid,
            "hard_neg_feat": hn_feat_out,
            "hard_neg_mask": hn_mask_out,
            "hard_neg_valid": hn_valid,
            "gt_span": gt_span,
            "gt_span_valid": torch.tensor(gt_span_valid, dtype=torch.bool),
            "duration": torch.tensor(duration, dtype=torch.float32),
            "task_id": task_id_from_path(item["video1_path"]),
        }


# ---------------------------------------------------------------------------
#  BLIP-2 dataset (CoVR-style video embs + pre-extracted query features)
# ---------------------------------------------------------------------------

class Blip2RetrievalDataset(Dataset):
    """Dataset backed by BLIP-2 (256-d) pre-extracted features.

    Video features  : ``video_feat_dir/{task_id}/{vid}.pth`` → ``[T, 32, 256]``
                      Mean-pooled over 32 Q-Former tokens → ``[T, 256]``
    Query features  : ``query_feat_dirs[*]/{qid}.pth.tar`` → ``[256]``
    """

    def __init__(
        self,
        data_list: List[dict],
        video_feat_dir: str,
        query_feat_dirs: List[str],
        max_target_frames: int = 15,
        max_history_frames: int = 15,
        feat_dim: int = 256,
        cache_features: bool = True,
        neg_json: Optional[str] = None,
        max_hard_neg: int = 3,
    ):
        self.video_feat_dir = video_feat_dir
        self.max_target_frames = max_target_frames
        self.max_history_frames = max_history_frames
        self.feat_dim = feat_dim
        self.max_hard_neg = max_hard_neg

        self._qfeat_index: Dict[int, str] = {}
        for d in query_feat_dirs:
            if not os.path.isdir(d):
                print(f"[Blip2Dataset] WARNING: query feat dir not found: {d}")
                continue
            for fname in os.listdir(d):
                if fname.endswith(".pth.tar"):
                    try:
                        qid = int(fname.split(".")[0])
                        self._qfeat_index[qid] = os.path.join(d, fname)
                    except ValueError:
                        pass
        print(f"[Blip2Dataset] Indexed {len(self._qfeat_index)} query feats "
              f"from {len(query_feat_dirs)} dir(s)")

        self.data = self._filter_valid(data_list)
        self._build_hard_neg_map(neg_json)

        self._vid_cache: Dict[str, torch.Tensor] = {}
        self._qfeat_cache: Dict[int, torch.Tensor] = {}
        if cache_features:
            self._preload_features()

    def _vid_from_path(self, path: str) -> str:
        return os.path.basename(path).split(".")[0]

    def _vfeat_path(self, vid: str, video_path: str) -> str:
        return blip2_video_feat_path(self.video_feat_dir, vid, video_path)

    def _filter_valid(self, data_list: List[dict]) -> List[dict]:
        valid = []
        for item in data_list:
            v1 = self._vid_from_path(item["video1_path"])
            v2 = self._vid_from_path(item["video2_path"])
            qid = item["qid"]
            if (os.path.exists(self._vfeat_path(v1, item["video1_path"]))
                    and os.path.exists(self._vfeat_path(v2, item["video2_path"]))
                    and qid in self._qfeat_index):
                valid.append(item)
        skipped = len(data_list) - len(valid)
        print(f"[Blip2Dataset] {len(data_list)} → {len(valid)} valid samples "
              f"({skipped} skipped due to missing features)")
        return valid

    def _build_hard_neg_map(self, neg_json: Optional[str]):
        self.hard_neg_map: Dict[int, List[Tuple[str, str]]] = {}
        if neg_json is None or not os.path.exists(neg_json):
            print("[Blip2Dataset] No hard negative file provided")
            return
        valid_qids = {item["qid"] for item in self.data}
        neg_data = json.load(open(neg_json))
        n_with_neg = 0
        for entry in neg_data:
            qid = entry["qid"]
            if qid not in valid_qids:
                continue
            negs = entry.get("hard_negatives_same_history_diff_query", [])
            neg_items = []
            for n in negs:
                vid = self._vid_from_path(n["video2_path"])
                vpath = n["video2_path"]
                if os.path.exists(self._vfeat_path(vid, vpath)):
                    neg_items.append((vid, vpath))
            if neg_items:
                self.hard_neg_map[qid] = neg_items[:self.max_hard_neg]
                n_with_neg += 1
        print(f"[Blip2Dataset] Hard negatives: {n_with_neg}/{len(valid_qids)} "
              f"samples have hard negs (max {self.max_hard_neg} per sample)")

    def _preload_features(self):
        vids: Dict[str, str] = {}
        for item in self.data:
            v1 = self._vid_from_path(item["video1_path"])
            v2 = self._vid_from_path(item["video2_path"])
            vids[v1] = item["video1_path"]
            vids[v2] = item["video2_path"]
        for neg_items in self.hard_neg_map.values():
            for vid, vpath in neg_items:
                vids[vid] = vpath

        print(f"[Blip2Dataset] Pre-loading {len(vids)} video features …")
        for vid, vpath in tqdm(vids.items(), desc="Loading video feats"):
            self._vid_cache[vid] = self._load_vid_from_disk(vid, vpath)

        qids = {item["qid"] for item in self.data}
        print(f"[Blip2Dataset] Pre-loading {len(qids)} query features …")
        for qid in tqdm(qids, desc="Loading query feats"):
            self._qfeat_cache[qid] = self._load_qfeat_from_disk(qid)

        vid_mem = sum(v.nelement() * v.element_size()
                      for v in self._vid_cache.values()) / 1e6
        q_mem = sum(v.nelement() * v.element_size()
                    for v in self._qfeat_cache.values()) / 1e6
        print(f"[Blip2Dataset] RAM usage: video={vid_mem:.0f}MB "
              f"query={q_mem:.1f}MB")

    def _load_vid_from_disk(self, vid: str, video_path: str) -> torch.Tensor:
        feat = torch.load(self._vfeat_path(vid, video_path),
                          map_location="cpu", weights_only=False)
        if feat.dtype == torch.float16:
            feat = feat.float()
        return pool_blip2_qformer_tokens(feat)

    def _load_qfeat_from_disk(self, qid: int) -> torch.Tensor:
        feat = torch.load(self._qfeat_index[qid], map_location="cpu",
                          weights_only=False)
        if feat.dtype == torch.float16:
            feat = feat.float()
        return feat.squeeze()

    def _load_vid(self, vid: str, video_path: str) -> torch.Tensor:
        if vid in self._vid_cache:
            return self._vid_cache[vid]
        return self._load_vid_from_disk(vid, video_path)

    def _load_qfeat(self, qid: int) -> torch.Tensor:
        if qid in self._qfeat_cache:
            return self._qfeat_cache[qid]
        return self._load_qfeat_from_disk(qid)

    @staticmethod
    def _uniform_sample(feat: torch.Tensor, max_frames: int) -> torch.Tensor:
        T = feat.size(0)
        if T <= max_frames:
            return feat
        indices = torch.linspace(0, T - 1, max_frames).long()
        return feat[indices]

    def _pad(self, feat: torch.Tensor, max_frames: int
             ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self._uniform_sample(feat, max_frames)
        T, D = feat.shape
        if T >= max_frames:
            return feat[:max_frames], torch.ones(max_frames, dtype=torch.bool)
        pad = torch.zeros(max_frames - T, D)
        padded = torch.cat([feat, pad], dim=0)
        mask = torch.zeros(max_frames, dtype=torch.bool)
        mask[:T] = True
        return padded, mask

    def _history_slice(self, feat: torch.Tensor, item: dict) -> torch.Tensor:
        startends = item["video1_startends"]
        seconds = float(startends[0] if isinstance(startends, list) else startends)
        duration = float(item.get("duration", 0) or 0)
        n_frames = max(1, feat.size(0))
        end_idx = seconds_to_blip2_frames(seconds, duration, n_frames)
        return feat[:end_idx]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]

        target_vid = self._vid_from_path(item["video2_path"])
        target_feat = self._load_vid(target_vid, item["video2_path"])
        target_feat, target_mask = self._pad(target_feat, self.max_target_frames)

        history_vid = self._vid_from_path(item["video1_path"])
        history_feat = self._history_slice(
            self._load_vid(history_vid, item["video1_path"]), item)
        history_feat, history_mask = self._pad(history_feat, self.max_history_frames)

        qid = item["qid"]
        pre_text_feat = self._load_qfeat(qid)

        T_tgt = self.max_target_frames
        if self.max_hard_neg > 0:
            neg_items = self.hard_neg_map.get(qid, [])
            hn_feats, hn_masks = [], []
            for vid, vpath in neg_items:
                f = self._load_vid(vid, vpath)
                f, m = self._pad(f, T_tgt)
                hn_feats.append(f)
                hn_masks.append(m)
            n_valid = len(hn_feats)
            while len(hn_feats) < self.max_hard_neg:
                hn_feats.append(torch.zeros(T_tgt, self.feat_dim))
                hn_masks.append(torch.zeros(T_tgt, dtype=torch.bool))
            hn_valid = torch.zeros(self.max_hard_neg, dtype=torch.bool)
            hn_valid[:n_valid] = True
            hn_feat_out = torch.stack(hn_feats)
            hn_mask_out = torch.stack(hn_masks)
        else:
            hn_feat_out = torch.empty(0, T_tgt, self.feat_dim)
            hn_mask_out = torch.empty(0, T_tgt, dtype=torch.bool)
            hn_valid = torch.empty(0, dtype=torch.bool)

        duration = item.get("duration", 0)
        annos = item.get("annos", [])
        if annos and annos[0].get("window") and duration > 0:
            w = annos[0]["window"][0]
            gt_span = torch.tensor([w[0] / duration, w[1] / duration],
                                   dtype=torch.float32).clamp(0.0, 1.0)
            gt_span_valid = True
        else:
            gt_span = torch.zeros(2, dtype=torch.float32)
            gt_span_valid = False

        return {
            "target_feat": target_feat,
            "target_mask": target_mask,
            "history_feat": history_feat,
            "history_mask": history_mask,
            "pre_text_feat": pre_text_feat,
            "qid": qid,
            "target_vid": target_vid,
            "hard_neg_feat": hn_feat_out,
            "hard_neg_mask": hn_mask_out,
            "hard_neg_valid": hn_valid,
            "gt_span": gt_span,
            "gt_span_valid": torch.tensor(gt_span_valid, dtype=torch.bool),
            "duration": torch.tensor(duration, dtype=torch.float32),
            "task_id": task_id_from_path(item["video1_path"]),
        }
