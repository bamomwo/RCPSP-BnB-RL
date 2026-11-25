"""Resource-set branching Actor and strict checkpoint serialization."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rcpsp_bnb_rl.ml.il.featurize import (
    FEATURE_SCHEMA, set_global_feature_dim, set_candidate_feature_dim,
    set_resource_feature_dim, set_interaction_feature_dim,
)


class _TransformerEncoderLayer(nn.Module):
    """Pre-norm attention and feed-forward residual blocks."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        normed = self.norm1(x)
        attended, _ = self.attn(normed, normed, normed, key_padding_mask=key_padding_mask)
        x = x + attended
        return x + self.ffn(self.norm2(x))


class BranchingTransformer(nn.Module):
    """Score ready activities from the unified ``resource_set`` schema."""

    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 ffn_dim: int = 128, dropout: float = 0.0) -> None:
        super().__init__()
        for name, value in (("d_model", d_model), ("n_heads", n_heads),
                            ("n_layers", n_layers), ("ffn_dim", ffn_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1)")
        self.d_model = d_model
        self.observation_schema = FEATURE_SCHEMA
        self.resource_feature_dim = set_resource_feature_dim()
        self.interaction_feature_dim = set_interaction_feature_dim()
        self.cls_proj = nn.Linear(set_global_feature_dim(), d_model)
        self.cand_proj = nn.Linear(set_candidate_feature_dim(), d_model)
        self.resource_proj = nn.Linear(self.resource_feature_dim, d_model)
        self.interaction_proj = nn.Linear(self.interaction_feature_dim, d_model)
        self.resource_global_proj = nn.Linear(d_model, d_model)
        self.layers = nn.ModuleList([
            _TransformerEncoderLayer(d_model, n_heads, ffn_dim, dropout) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.score_head = nn.Linear(d_model, 1)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        candidate_feats: torch.Tensor,
        global_feats: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        resource_feats: Optional[torch.Tensor] = None,
        candidate_resource_feats: Optional[torch.Tensor] = None,
        resource_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return one logit per candidate, masking infeasible actions."""
        if resource_feats is None or candidate_resource_feats is None:
            raise ValueError("Resource-set policy requires resource and interaction features")
        cls_token = self.cls_proj(global_feats).unsqueeze(0)
        cand_emb = self.cand_proj(candidate_feats)
        resource_emb = self.resource_proj(resource_feats)
        interaction_emb = self.interaction_proj(candidate_resource_feats)
        if resource_mask is None:
            resource_mask = torch.ones(resource_emb.shape[0], dtype=torch.bool, device=resource_emb.device)
        if not bool(resource_mask.any()):
            raise ValueError("Resource-set policy received an empty resource set")
        pair_emb = resource_emb.unsqueeze(0) + interaction_emb
        scores = (cand_emb.unsqueeze(1) * pair_emb).sum(dim=-1) / math.sqrt(self.d_model)
        weights = scores.masked_fill(~resource_mask.unsqueeze(0), -1e9).softmax(dim=1)
        cand_emb = cand_emb + (weights.unsqueeze(-1) * pair_emb).sum(dim=1)
        pooled = resource_emb[resource_mask].mean(dim=0)
        cls_token = cls_token + self.resource_global_proj(pooled).unsqueeze(0)
        x = torch.cat([cls_token, cand_emb], dim=0).unsqueeze(0)
        key_padding_mask: Optional[torch.Tensor] = None
        if action_mask is not None:
            cls_unmask = torch.zeros(1, dtype=torch.bool, device=candidate_feats.device)
            key_padding_mask = torch.cat([cls_unmask, ~action_mask]).unsqueeze(0)
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        x = self.final_norm(x).squeeze(0)
        logits = self.score_head(x[1:]).squeeze(-1)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)
        return logits

    def forward_batch(
        self,
        candidate_feats: torch.Tensor,
        global_feats: torch.Tensor,
        action_mask: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        resource_feats: Optional[torch.Tensor] = None,
        candidate_resource_feats: Optional[torch.Tensor] = None,
        resource_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return padded candidate logits for independent scheduling states."""
        if resource_feats is None or candidate_resource_feats is None:
            raise ValueError("Resource-set policy requires resource and interaction features")
        batch_size = candidate_feats.shape[0]
        cls_tokens = self.cls_proj(global_feats)
        cand_emb = self.cand_proj(candidate_feats)
        resource_emb = self.resource_proj(resource_feats)
        interaction_emb = self.interaction_proj(candidate_resource_feats)
        if resource_mask is None:
            resource_mask = torch.ones(resource_emb.shape[:2], dtype=torch.bool, device=resource_emb.device)
        if not bool(resource_mask.any(dim=1).all()):
            raise ValueError("Resource-set policy received an empty resource set")
        pair_emb = resource_emb.unsqueeze(1) + interaction_emb
        scores = torch.einsum("brd,brmd->brm", cand_emb, pair_emb) / math.sqrt(self.d_model)
        weights = scores.masked_fill(~resource_mask.unsqueeze(1), -1e9).softmax(dim=2)
        cand_emb = cand_emb + (weights.unsqueeze(-1) * pair_emb).sum(dim=2)
        denom = resource_mask.sum(dim=1, keepdim=True).clamp_min(1).to(resource_emb.dtype)
        pooled = (resource_emb * resource_mask.unsqueeze(-1)).sum(dim=1) / denom
        cls_tokens = cls_tokens + self.resource_global_proj(pooled)
        x = torch.cat([cls_tokens.unsqueeze(1), cand_emb], dim=1)
        attendable_mask = action_mask if pad_mask is None else action_mask & pad_mask
        cls_col = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
        key_padding_mask = torch.cat([cls_col, ~attendable_mask], dim=1)
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        logits = self.score_head(self.final_norm(x)[:, 1:]).squeeze(-1)
        return logits.masked_fill(~attendable_mask, -1e9)

    @staticmethod
    def listwise_nll(logits: torch.Tensor, lengths: torch.Tensor,
                     targets: torch.Tensor) -> torch.Tensor:
        """Mean negative log probability of the teacher's selected activity."""
        losses: list[torch.Tensor] = []
        offset = 0
        for length, target in zip(lengths.tolist(), targets):
            if length > 0:
                losses.append(-F.log_softmax(logits[offset:offset + length], dim=0)[target])
            offset += length
        return torch.stack(losses).mean() if losses else logits.sum() * 0.0

    @staticmethod
    def listwise_accuracy(logits: torch.Tensor, lengths: torch.Tensor,
                          targets: torch.Tensor) -> torch.Tensor:
        """Top-1 accuracy across teacher states."""
        correct = total = offset = 0
        for length, target in zip(lengths.tolist(), targets.tolist()):
            if length > 0:
                correct += int(int(logits[offset:offset + length].argmax()) == target)
                total += 1
            offset += length
        return logits.new_tensor(correct / total if total else 0.0)


def _policy_config(model: BranchingTransformer) -> dict:
    return {
        "global_dim": set_global_feature_dim(), "candidate_dim": set_candidate_feature_dim(),
        "resource_feature_dim": set_resource_feature_dim(),
        "interaction_feature_dim": set_interaction_feature_dim(),
        "feature_schema": FEATURE_SCHEMA, "observation_schema": FEATURE_SCHEMA,
        "d_model": model.d_model, "n_heads": model.layers[0].attn.num_heads,
        "n_layers": len(model.layers), "ffn_dim": model.layers[0].ffn[0].out_features,
        "dropout": model.layers[0].attn.dropout,
    }


def save_policy_checkpoint(model: BranchingTransformer, path: str, extra: dict | None = None) -> None:
    """Save Actor parameters, architecture, and feature semantics."""
    torch.save({**(extra or {}), "model_state": model.state_dict(), "config": _policy_config(model)}, path)


def load_policy_checkpoint(path: str, device: torch.device | str = "cpu",
                           dropout: float = 0.0) -> BranchingTransformer:
    """Restore a ``resource_set`` Actor without parameter migration.

    Checkpoints produced for an older feature schema are rejected explicitly:
    feature semantics are part of the model contract and cannot be migrated by
    loading weights into differently sized projections.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(f"Checkpoint at {path} is missing 'model_state'")
    config = checkpoint.get("config", {})
    expected = (set_global_feature_dim(), set_candidate_feature_dim(),
                set_resource_feature_dim(), set_interaction_feature_dim())
    widths = tuple(config.get(key) for key in (
        "global_dim", "candidate_dim", "resource_feature_dim", "interaction_feature_dim"))
    if (config.get("feature_schema") != FEATURE_SCHEMA
            or config.get("observation_schema") != FEATURE_SCHEMA or widths != expected):
        raise ValueError(f"Checkpoint must use {FEATURE_SCHEMA} with input widths {expected}. "
                         "Retrain BC or set bc_checkpoint=null to train from scratch.")
    model = BranchingTransformer(
        d_model=config["d_model"], n_heads=config["n_heads"], n_layers=config["n_layers"],
        ffn_dim=config["ffn_dim"], dropout=dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.eval()
