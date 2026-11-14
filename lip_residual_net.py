# -*- coding: utf-8 -*-
"""Phoneme-conditioned residual motion for TalkingGaussian lip Gaussians.

This is an additive branch: it does not replace or modify the pretrained TG
Face/Mouth motion networks. It predicts a bounded per-Face-Gaussian XYZ
residual from spatial state, TG's base XYZ motion, frame-aligned phonemes,
and optional AU expression features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhonemeContextEncoder(nn.Module):
    """Encode the existing 8-frame phone_ids/phone_valid window."""
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=64,
                 window_size=8, padding_idx=0):
        super(PhonemeContextEncoder, self).__init__()
        self.window_size = int(window_size)
        self.embedding = nn.Embedding(
            int(vocab_size), int(embedding_dim), padding_idx=int(padding_idx)
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(embedding_dim + 1, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, phone_ids, phone_valid):
        if phone_ids.ndim == 1:
            phone_ids = phone_ids.unsqueeze(0)
        if phone_valid.ndim == 1:
            phone_valid = phone_valid.unsqueeze(0)
        if phone_ids.ndim != 2 or phone_valid.shape != phone_ids.shape:
            raise ValueError("phone_ids and phone_valid must both be [B, T]")
        if phone_ids.shape[1] != self.window_size:
            raise ValueError(
                "Expected phone window {}, got {}".format(
                    self.window_size, phone_ids.shape[1]
                )
            )
        valid = phone_valid.to(dtype=torch.float32).clamp(0.0, 1.0)
        emb = self.embedding(phone_ids.long())
        x = torch.cat([emb, valid.unsqueeze(-1)], dim=-1).transpose(1, 2)
        feat = self.temporal(x).transpose(1, 2)
        mask = valid.unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        mean_feat = (feat * mask).sum(dim=1) / denom
        max_feat = feat.masked_fill(mask == 0, -1e4).amax(dim=1)
        any_valid = (valid.sum(dim=1, keepdim=True) > 0).to(feat.dtype)
        max_feat = torch.where(any_valid > 0, max_feat, torch.zeros_like(max_feat))
        return self.projection(torch.cat([mean_feat, max_feat], dim=-1))


class LipResidualMotionNetwork(nn.Module):
    """Predict bounded residual XYZ for each Face Gaussian.

    Inputs:
      xyz:       [N, 3] canonical Face Gaussian coordinates
      tg_d_xyz:  [N, 3] frozen original TG Face motion
      phone_ids: [8] or [B, 8] frame-aligned phone IDs
      phone_valid:[8] or [B, 8] validity mask
      lip_weight:[N] or [N, 1] S1 soft lip contribution weights
      au_exp:    optional [6] or [B, 6] expression condition

    Output delta_xyz has shape [N, 3]. Apply S1 weights outside the network
    when adding it to TG positions: xyz + tg_d_xyz + lip_weight * delta_xyz.
    """
    def __init__(self, vocab_size, spatial_dim=64, phone_dim=64,
                 hidden_dim=128, max_delta=0.003, use_au=True):
        super(LipResidualMotionNetwork, self).__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if max_delta <= 0:
            raise ValueError("max_delta must be positive")
        self.max_delta = float(max_delta)
        self.use_au = bool(use_au)
        self.phone_encoder = PhonemeContextEncoder(
            vocab_size=vocab_size, embedding_dim=32, hidden_dim=phone_dim
        )
        spatial_in = 3 + 3 + 1 + (6 if self.use_au else 0)
        self.spatial_encoder = nn.Sequential(
            nn.Linear(spatial_in, spatial_dim),
            nn.GELU(),
            nn.Linear(spatial_dim, spatial_dim),
            nn.GELU(),
        )
        self.phone_projection = nn.Linear(phone_dim, spatial_dim)
        self.gate = nn.Sequential(
            nn.Linear(spatial_dim * 2, spatial_dim),
            nn.GELU(),
            nn.Linear(spatial_dim, spatial_dim),
            nn.Sigmoid(),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(spatial_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        # Start as an identity correction: the new branch initially preserves TG.
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    @staticmethod
    def _frame_vector(value, batch, device, dtype, name):
        if value is None:
            return None
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.shape[0] == 1 and batch > 1:
            value = value.expand(batch, -1)
        if value.shape[0] != batch:
            raise ValueError("{} batch dimension mismatch".format(name))
        return value.to(device=device, dtype=dtype)

    def forward(self, xyz, tg_d_xyz, phone_ids, phone_valid,
                lip_weight, au_exp=None, return_gate=False):
        if xyz.ndim != 2 or xyz.shape[-1] != 3:
            raise ValueError("xyz must have shape [N, 3]")
        if tg_d_xyz.shape != xyz.shape:
            raise ValueError("tg_d_xyz must have the same shape as xyz")
        n = xyz.shape[0]
        if lip_weight.ndim == 1:
            lip_weight = lip_weight.unsqueeze(-1)
        if lip_weight.shape != (n, 1):
            raise ValueError("lip_weight must have shape [N] or [N, 1]")

        # This initial implementation is per-frame (one phone window per view).
        if phone_ids.ndim == 2 and phone_ids.shape[0] != 1:
            raise ValueError("Expected one frame phone window [1, 8] or [8]")
        if phone_valid.ndim == 2 and phone_valid.shape[0] != 1:
            raise ValueError("Expected one frame validity window [1, 8] or [8]")
        phone_ids = phone_ids.to(device=xyz.device, dtype=torch.long)
        phone_valid = phone_valid.to(device=xyz.device, dtype=xyz.dtype)
        phone_feat = self.phone_encoder(phone_ids, phone_valid)
        phone_feat = phone_feat.expand(n, -1)

        if self.use_au:
            if au_exp is None:
                au_exp = xyz.new_zeros((1, 6))
            au_exp = self._frame_vector(
                au_exp, 1, xyz.device, xyz.dtype, "au_exp"
            )
            if au_exp.shape[-1] != 6:
                raise ValueError("au_exp must have 6 expression values")
            au_feat = au_exp.expand(n, -1)
        else:
            au_feat = None

        spatial_parts = [xyz, tg_d_xyz, lip_weight.to(xyz.dtype)]
        if au_feat is not None:
            spatial_parts.append(au_feat)
        spatial_feat = self.spatial_encoder(torch.cat(spatial_parts, dim=-1))
        phone_feat = self.phone_projection(phone_feat)
        gate = self.gate(torch.cat([spatial_feat, phone_feat], dim=-1))
        fused = torch.cat([spatial_feat, gate * phone_feat], dim=-1)
        delta = torch.tanh(self.residual_head(fused)) * self.max_delta
        if return_gate:
            return {"delta_xyz": delta, "gate": gate}
        return {"delta_xyz": delta}
