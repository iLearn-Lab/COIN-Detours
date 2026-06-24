"""
Composed Video Retrieval — Dual-Tower Model

Query Tower:  (reference_video_history, text_query)  →  embedding
Video Tower:  target_video                            →  embedding

Key components:
  - TemporalTransformer: self-attention over frame-level features
  - QueryGuidedPooling:  text query attends to history video frames
  - GatedFusion:         adaptive fusion of visual & textual signals
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


# ---------------------------------------------------------------------------
#  Building blocks
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with a learnable scale factor."""

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        x = x + self.scale * self.pe[:, : x.size(1)]


        # 标准 Transformer 是： x = x + pe
        # 这里变成了x = x + scale * pe
        # 模型可以自己决定位置编码的重要性
        # 训练过程中可能学到：
        # scale ≈ 0 → 几乎不用位置
        # scale ≈ 1 → 正常使用
        # scale > 1 → 强调位置
        return self.dropout(x)


class TemporalTransformer(nn.Module):
    """Transformer encoder for temporal modelling of frame features."""

    def __init__(self, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Args:
            x:    [B, T, D]
            mask: [B, T]  True = valid position
        """
        pad_mask = ~mask if mask is not None else None
        return self.norm(self.encoder(x, src_key_padding_mask=pad_mask))


class AttentionPooling(nn.Module):
    """Learnable-query attention pooling: [B, T, D] → [B, D]."""

    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        B = x.size(0)
        q = self.query.expand(B, -1, -1)
        pad_mask = ~mask if mask is not None else None
        out, _ = self.attn(q, x, x, key_padding_mask=pad_mask)
        return self.norm(out.squeeze(1))


class EventPooling(nn.Module):
    """DETR/Q-Former style event pooling: [B, T, D] → [B, num_events, D].

    K learnable event queries each attend over all frame features independently,
    capturing different temporal segments / semantic events in the video.
    After cross-attention, event vectors interact via self-attention.
    """

    def __init__(self, d_model: int, num_events: int, num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.num_events = num_events
        self.event_queries = nn.Parameter(torch.randn(1, num_events, d_model) * 0.02)

        # Cross-attention: each event query attends to frame features
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads,
                                                 batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Self-attention: event vectors interact with each other
        self.self_attn = nn.MultiheadAttention(d_model, num_heads,
                                                batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # Per-event feedforward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Args:
            x:    [B, T, D]  frame features after TemporalTransformer
            mask: [B, T]     True = valid frame
        Returns:
            [B, num_events, D]
        """
        B = x.size(0)
        q = self.event_queries.expand(B, -1, -1)   # [B, num_events, D]
        pad_mask = ~mask if mask is not None else None

        # Cross-attention: events ← frames
        ca_out, _ = self.cross_attn(q, x, x, key_padding_mask=pad_mask)
        q = self.norm1(q + ca_out)

        # Self-attention: events interact
        sa_out, _ = self.self_attn(q, q, q)
        q = self.norm2(q + sa_out)

        # FFN
        q = self.norm3(q + self.ffn(q))

        return q  # [B, num_events, D]


class QueryGuidedPooling(nn.Module):
    """Cross-attention: text query → video temporal features."""

    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, text_feat: torch.Tensor, video_feat: torch.Tensor,
                mask: torch.Tensor | None = None):
        """
        Args:
            text_feat:  [B, D]
            video_feat: [B, T, D]
            mask:       [B, T]  True = valid
        """
        q = self.q_proj(text_feat).unsqueeze(1)          # [B, 1, D]
        pad_mask = ~mask if mask is not None else None
        out, _ = self.attn(q, video_feat, video_feat, key_padding_mask=pad_mask)
        return self.norm(out.squeeze(1))                  # [B, D]


class GatedFusion(nn.Module):
    """Adaptive gated fusion of two modality features."""

    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.proj = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU())
#feat_a 输入的是attended ，，feat_b输入的是 text
    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor):
        combined = torch.cat([feat_a, feat_b], dim=-1)
        g = self.gate(combined)
        h = self.proj(combined)
        return g * feat_a + (1 - g) * h  

class ProjectionHead(nn.Module):
    """Two-layer MLP projection into the retrieval embedding space."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
#  Towers
# ---------------------------------------------------------------------------

class VideoTower(nn.Module):
    """Encode a target video into K event-level retrieval vectors."""

    def __init__(self, feat_dim: int, embed_dim: int, num_layers: int,
                 num_heads: int, ff_dim: int, dropout: float,
                 num_events: int = 4):
        super().__init__()
        self.input_norm = nn.LayerNorm(feat_dim)
        self.pos_enc = PositionalEncoding(feat_dim, dropout=dropout)
        self.temporal = TemporalTransformer(feat_dim, num_heads, num_layers,
                                            ff_dim, dropout)
        # self.pool = AttentionPooling(feat_dim, num_heads)                                    
        self.pool = EventPooling(feat_dim, num_events, num_heads, dropout)
        self.proj = ProjectionHead(feat_dim, embed_dim)

    def forward(self, video_feat: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            video_feat: [B, T, D]   pre-extracted frame features
            mask:       [B, T]      True = valid
        Returns:
            # 之前没有加入 num_events 参数，所以是 [B, D]
            [B, num_events, embed_dim]  L2-normalised per-event embeddings
        """
        x = self.input_norm(video_feat)
        x = self.pos_enc(x)
        x = self.temporal(x, mask)
        x = self.pool(x, mask)      # [B, num_events, D]
        x = self.proj(x)            # [B, num_events, embed_dim]  (Linear acts on last dim)
        return F.normalize(x, dim=-1)


class QueryTower(nn.Module):
    """Encode (reference video history + text query) into a single vector."""

    def __init__(self, feat_dim: int, embed_dim: int, num_layers: int,
                 num_heads: int, ff_dim: int, dropout: float,
                 text_encoder_path: str = "",
                 freeze_text_encoder: bool = True,
                 use_pretrained_text_feat: bool = False):
        super().__init__()
        self.freeze_text = freeze_text_encoder
        self.use_pretrained_text_feat = use_pretrained_text_feat

        if use_pretrained_text_feat:
            self.text_model = None
        else:
            full_model = AutoModel.from_pretrained(text_encoder_path)
            self.text_model = full_model.text_model
            del full_model
            if freeze_text_encoder:
                for p in self.text_model.parameters():
                    p.requires_grad = False
        self.text_adapter = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
        )

        # ---- history video encoder ----
        self.input_norm = nn.LayerNorm(feat_dim)
        self.pos_enc = PositionalEncoding(feat_dim, dropout=dropout)
        self.temporal = TemporalTransformer(feat_dim, num_heads, num_layers,
                                            ff_dim, dropout)

        # ---- query-guided pooling + fusion ----
        self.qg_pool = QueryGuidedPooling(feat_dim, num_heads)
        self.fusion = GatedFusion(feat_dim)
        self.proj = ProjectionHead(feat_dim, embed_dim)

    def _encode_text(self, input_ids: torch.Tensor,
                     attention_mask: torch.Tensor) -> torch.Tensor:
        if self.freeze_text:
            with torch.no_grad():
                out = self.text_model(input_ids=input_ids,
                                      attention_mask=attention_mask)
                raw = out.pooler_output           # [B, D]
        else:
            out = self.text_model(input_ids=input_ids,
                                  attention_mask=attention_mask)
            raw = out.pooler_output
        return self.text_adapter(raw)             # [B, D]

    def forward(self, history_feat: torch.Tensor,
                history_mask: torch.Tensor,
                input_ids: Optional[torch.Tensor] = None,
                text_attention_mask: Optional[torch.Tensor] = None,
                pre_text_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            history_feat:       [B, T, D]
            history_mask:       [B, T]
            input_ids:          [B, L]
            text_attention_mask: [B, L]
            pre_text_feat:      [B, D]  (InternVideo / BLIP-2 mode)
        Returns:
            [B, embed_dim]  L2-normalised embedding
        """
        if self.use_pretrained_text_feat:
            text_feat = self.text_adapter(pre_text_feat)
        else:
            text_feat = self._encode_text(input_ids, text_attention_mask)

        h = self.input_norm(history_feat)
        h = self.pos_enc(h)
        h = self.temporal(h, history_mask)         # [B, T, D]

        attended = self.qg_pool(text_feat, h, history_mask)  # [B, D]
        fused = self.fusion(attended, text_feat)              # [B, D]

        out = self.proj(fused)
        return F.normalize(out, dim=-1)


# ---------------------------------------------------------------------------
#  Full model
# ---------------------------------------------------------------------------

class ComposedVideoRetriever(nn.Module):
    """Dual-tower model for composed video retrieval.

    Video Tower produces K event-level vectors per video [B, num_events, D].
    Query Tower produces a single vector per query [B, D].
    Similarity = MaxSim: max over events of (query · event).
    """

    def __init__(self, feat_dim: int = 1152, embed_dim: int = 768,
                 num_temporal_layers: int = 4, num_heads: int = 8,
                 ff_dim: int = 2048, dropout: float = 0.1,
                 text_encoder_path: str = "",
                 freeze_text_encoder: bool = True,
                 init_temperature: float = 0.07,
                 num_events: int = 4,
                 use_pretrained_text_feat: bool = False):
        super().__init__()
        self.num_events = num_events
        self.use_pretrained_text_feat = use_pretrained_text_feat
        self.video_tower = VideoTower(feat_dim, embed_dim, num_temporal_layers,
                                      num_heads, ff_dim, dropout, num_events)
        self.query_tower = QueryTower(feat_dim, embed_dim, num_temporal_layers,
                                      num_heads, ff_dim, dropout,
                                      text_encoder_path, freeze_text_encoder,
                                      use_pretrained_text_feat)
        self.logit_scale = nn.Parameter(
            torch.log(torch.tensor(1.0 / init_temperature))
        )

    def forward(self, target_feat, target_mask,
                history_feat, history_mask,
                input_ids, text_attention_mask):
        video_emb = self.video_tower(target_feat, target_mask)
        query_emb = self.query_tower(history_feat, history_mask,
                                     input_ids, text_attention_mask)
        return video_emb, query_emb

    def encode_video(self, video_feat, mask=None):
        return self.video_tower(video_feat, mask)

    def encode_query(self, history_feat, history_mask,
                     input_ids=None, text_attention_mask=None,
                     pre_text_feat=None):
        return self.query_tower(history_feat, history_mask,
                                input_ids, text_attention_mask,
                                pre_text_feat)

    @staticmethod
    def _maxsim_q2v(query_emb: torch.Tensor,
                    video_emb_all: torch.Tensor) -> torch.Tensor:
        """MaxSim: query [B_q, D] × video [B_v, E, D] → [B_q, B_v].

        sim[i, j] = max_k( query[i] · video[j, k] )
        """
        B_v, E, D = video_emb_all.shape
        B_q = query_emb.size(0)
        # [B_q, B_v*E] then reshape + max over events
        raw = query_emb @ video_emb_all.reshape(B_v * E, D).T   # [B_q, B_v*E]
        return raw.reshape(B_q, B_v, E).max(dim=-1).values       # [B_q, B_v]

    @staticmethod
    def _maxsim_v2q(video_emb: torch.Tensor,
                    query_emb_all: torch.Tensor) -> torch.Tensor:
        """MaxSim: video [B_v, E, D] × query [B_q, D] → [B_v, B_q].

        sim[i, j] = max_k( video[i, k] · query[j] )
        """
        B_v, E, D = video_emb.shape
        B_q = query_emb_all.size(0)
        raw = video_emb.reshape(B_v * E, D) @ query_emb_all.T   # [B_v*E, B_q]
        return raw.reshape(B_v, E, B_q).max(dim=1).values        # [B_v, B_q]

    @staticmethod
    def orthogonality_loss(video_emb: torch.Tensor) -> torch.Tensor:
        """Encourage event vectors to be mutually orthogonal.

        video_emb: [B, num_events, D]  L2-normalised event embeddings.
        Loss = mean over batch of || G - I ||_F^2  where G = V @ V.T  [E, E].
        Off-diagonal entries of G equal cosine similarity between event pairs;
        minimising them pushes events toward orthogonality.
        """
        if video_emb.dim() != 3 or video_emb.size(1) == 1:
            return video_emb.new_tensor(0.0)
        gram = torch.bmm(video_emb, video_emb.transpose(1, 2))     # [B, E, E]
        E = gram.size(1)
        eye = torch.eye(E, device=gram.device, dtype=gram.dtype).unsqueeze(0)
        off_diag = gram - eye                                        # zero out diagonal
        return (off_diag ** 2).sum(dim=(1, 2)).mean()

    def compute_loss(self, video_emb: torch.Tensor,
                     query_emb: torch.Tensor,
                     hard_neg_emb: torch.Tensor | None = None,
                     hard_neg_valid: torch.Tensor | None = None,
                     video_emb_all: torch.Tensor | None = None,
                     query_emb_all: torch.Tensor | None = None,
                     labels_offset: int = 0):
        """
        Symmetric InfoNCE with MaxSim for multi-event video embeddings.

        video_emb:      [B_local, num_events, D]  local positive video events
        query_emb:      [B_local, D]              local query embeddings
        hard_neg_emb:   [B_local, K, num_events, D]  hard negative events (optional)
        hard_neg_valid: [B_local, K]                 True = valid (optional)
        video_emb_all:  [B_total, num_events, D]  gathered (DDP) or None (single-GPU)
        query_emb_all:  [B_total, D]              gathered (DDP) or None (single-GPU)
        labels_offset:  int  rank * B_local for DDP
        """
        logit_scale = self.logit_scale.exp().clamp(max=100.0)

        v_all = video_emb_all if video_emb_all is not None else video_emb
        q_all = query_emb_all if query_emb_all is not None else query_emb

        B_local = query_emb.size(0)
        labels = torch.arange(B_local, device=query_emb.device) + labels_offset

        # ---- q → v loss: MaxSim(query, all_videos) + hard negatives ----
        sim_q2v = logit_scale * self._maxsim_q2v(query_emb, v_all)   # [B_local, B_total]

        if hard_neg_emb is not None and hard_neg_valid is not None and hard_neg_valid.any():
            # hn_emb: [B_local, K, num_events, D]
            B_l, K, E, D_e = hard_neg_emb.shape
            # MaxSim per hard neg: [B_local, K]
            hn_flat = hard_neg_emb.reshape(B_l, K * E, D_e)           # [B_l, K*E, D]
            hn_sim = logit_scale * torch.bmm(
                hn_flat, query_emb.unsqueeze(-1)
            ).squeeze(-1)                                               # [B_l, K*E]
            hn_sim = hn_sim.reshape(B_l, K, E).max(dim=-1).values      # [B_l, K]
            hn_sim = hn_sim.masked_fill(~hard_neg_valid, float("-inf"))
            sim_q2v = torch.cat([sim_q2v, hn_sim], dim=1)             # [B_local, B_total+K]

        loss_q2v = F.cross_entropy(sim_q2v, labels)

        # ---- v → q loss: MaxSim(all_videos, queries) ----
        sim_v2q = logit_scale * self._maxsim_v2q(video_emb, q_all)    # [B_local, B_total]
        loss_v2q = F.cross_entropy(sim_v2q, labels)

        loss = (loss_q2v + loss_v2q) / 2

        with torch.no_grad():
            acc_q2v = (sim_q2v.argmax(1) == labels).float().mean()
            acc_v2q = (sim_v2q.argmax(1) == labels).float().mean()

        return loss, {
            "loss": loss.item(),
            "loss_q2v": loss_q2v.item(),
            "loss_v2q": loss_v2q.item(),
            "acc_q2v": acc_q2v.item(),
            "acc_v2q": acc_v2q.item(),
            "logit_scale": logit_scale.item(),
        }
