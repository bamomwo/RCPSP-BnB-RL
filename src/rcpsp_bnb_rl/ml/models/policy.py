from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _TransformerEncoderLayer(nn.Module):
    """
    Single pre-norm transformer encoder layer.

    Pre-norm (LayerNorm before attention/FFN) trains more stably than
    post-norm for small models and short sequences.
    """

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=key_padding_mask)
        x = x + attn_out
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Main policy network
# ---------------------------------------------------------------------------

class BranchingTransformer(nn.Module):
    """
    Attention-based branching policy for RCPSP branch-and-bound.

    CLS token (global features) + per-candidate embeddings → transformer encoder
    → logits [R] over candidates. Forward methods retain a zero-valued second
    output for existing BC and B&B inference callers.
    """

    def __init__(
        self,
        global_dim: int,
        candidate_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ffn_dim: int = 128,
        dropout: float = 0.1,
        critic_feature_dim: int = 0,
        resource_feature_dim: int = 0,
        interaction_feature_dim: int = 0,
        policy_only: bool = True,
        value_only: bool = False,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} must be divisible by n_heads={n_heads}"

        self.d_model = d_model
        # Accept legacy constructor arguments without allocating critic modules.
        if value_only:
            raise ValueError("BranchingTransformer is an actor-only model")
        self.critic_feature_dim = 0
        self.policy_only = True
        self.value_only = False
        self.resource_feature_dim = int(resource_feature_dim)
        self.interaction_feature_dim = int(interaction_feature_dim)
        self.resource_set_enabled = (
            self.resource_feature_dim > 0 and self.interaction_feature_dim > 0
        )

        # Input projections
        self.cls_proj = nn.Linear(global_dim, d_model)
        self.cand_proj = nn.Linear(candidate_dim, d_model)

        # Shared weights make resource encoding independent of count and order.
        self.resource_proj = (
            nn.Linear(self.resource_feature_dim, d_model)
            if self.resource_set_enabled else nn.Identity()
        )
        self.interaction_proj = (
            nn.Linear(self.interaction_feature_dim, d_model)
            if self.resource_set_enabled else nn.Identity()
        )
        self.resource_global_proj = (
            nn.Linear(d_model, d_model)
            if self.resource_set_enabled else nn.Identity()
        )

        # Transformer encoder
        self.layers = nn.ModuleList([
            _TransformerEncoderLayer(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Scoring head: maps each candidate embedding → scalar logit
        self.score_head = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self) -> None:
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
        critic_feats: Optional[torch.Tensor] = None,
        resource_feats: Optional[torch.Tensor] = None,
        candidate_resource_feats: Optional[torch.Tensor] = None,
        resource_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unbatched forward pass.

        Parameters: candidate_feats [R, Fc], global_feats [Fg],
                    action_mask [R] bool. critic_feats is ignored for backwards
                    compatibility.
        Returns: logits [R], zero scalar.
        """
        # Project inputs into d_model space
        cls_token = self.cls_proj(global_feats).unsqueeze(0)   # [1, d_model]
        cand_emb = self.cand_proj(candidate_feats)              # [R, d_model]

        if self.resource_set_enabled:
            if resource_feats is None or candidate_resource_feats is None:
                raise ValueError("Resource-set policy requires resource and interaction features")
            resource_emb = self.resource_proj(resource_feats)  # [M, D]
            interaction_emb = self.interaction_proj(candidate_resource_feats)  # [R, M, D]
            if resource_mask is None:
                resource_mask = torch.ones(
                    resource_emb.shape[0], dtype=torch.bool, device=resource_emb.device
                )
            if not bool(resource_mask.any()):
                raise ValueError("Resource-set policy received an empty resource set")
            pair_emb = resource_emb.unsqueeze(0) + interaction_emb
            scores = (cand_emb.unsqueeze(1) * pair_emb).sum(dim=-1) / math.sqrt(self.d_model)
            scores = scores.masked_fill(~resource_mask.unsqueeze(0), -1e9)
            weights = torch.softmax(scores, dim=1)
            cand_emb = cand_emb + (weights.unsqueeze(-1) * pair_emb).sum(dim=1)
            pooled = resource_emb[resource_mask].mean(dim=0)
            cls_token = cls_token + self.resource_global_proj(pooled).unsqueeze(0)

        # Build sequence: [CLS, cand_1, ..., cand_R]  →  [R+1, d_model]
        seq = torch.cat([cls_token, cand_emb], dim=0).unsqueeze(0)  # [1, R+1, d_model]

        # Build key_padding_mask: mask infeasible candidates so they don't
        # pollute attention of feasible ones.  CLS token is always unmasked.
        # Explicit Optional annotation: TorchScript forbids a local changing
        # type (None -> Tensor), so declare it Optional up front.
        key_padding_mask: Optional[torch.Tensor] = None
        if action_mask is not None:
            # key_padding_mask: True = ignore this position
            # CLS (position 0) is always attended; infeasible candidates are masked.
            cls_unmask = torch.zeros(1, dtype=torch.bool, device=candidate_feats.device)
            cand_mask = ~action_mask  # True where infeasible
            key_padding_mask = torch.cat([cls_unmask, cand_mask], dim=0).unsqueeze(0)  # [1, R+1]

        # Transformer encoding
        x = seq
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        x = self.final_norm(x).squeeze(0)  # [R+1, d_model]

        cand_out = x[1:]     # [R, d_model] — candidate representations

        # Score each candidate
        logits = self.score_head(cand_out).squeeze(-1)  # [R]

        # Mask infeasible candidates
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)

        return logits, logits.new_zeros(())

    def forward_batch(
        self,
        candidate_feats: torch.Tensor,
        global_feats: torch.Tensor,
        action_mask: torch.Tensor,
        critic_feats: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
        resource_feats: Optional[torch.Tensor] = None,
        candidate_resource_feats: Optional[torch.Tensor] = None,
        resource_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched forward pass for padded policy updates.

        Parameters: candidate_feats [B, R_max, Fc], global_feats [B, Fg],
                    action_mask [B, R_max] bool, critic_feats ignored,
                    pad_mask [B, R_max] bool (True=real, False=padding).
        Returns: logits [B, R_max], zeros [B].
        """
        B, R_max, _ = candidate_feats.shape

        # Project inputs into d_model space
        cls_tokens = self.cls_proj(global_feats)              # [B, d_model]
        cand_emb = self.cand_proj(candidate_feats)            # [B, R_max, d_model]

        if self.resource_set_enabled:
            if resource_feats is None or candidate_resource_feats is None:
                raise ValueError("Resource-set policy requires resource and interaction features")
            resource_emb = self.resource_proj(resource_feats)  # [B, M_max, D]
            interaction_emb = self.interaction_proj(candidate_resource_feats)  # [B,R,M,D]
            if resource_mask is None:
                resource_mask = torch.ones(
                    resource_emb.shape[:2], dtype=torch.bool, device=resource_emb.device
                )
            if not bool(resource_mask.any(dim=1).all()):
                raise ValueError("Resource-set policy received an empty resource set")
            pair_emb = resource_emb.unsqueeze(1) + interaction_emb
            scores = torch.einsum("brd,brmd->brm", cand_emb, pair_emb)
            scores = scores / math.sqrt(self.d_model)
            scores = scores.masked_fill(~resource_mask.unsqueeze(1), -1e9)
            weights = torch.softmax(scores, dim=2)
            cand_emb = cand_emb + (weights.unsqueeze(-1) * pair_emb).sum(dim=2)
            denom = resource_mask.sum(dim=1, keepdim=True).clamp_min(1).to(resource_emb.dtype)
            pooled = (resource_emb * resource_mask.unsqueeze(-1)).sum(dim=1) / denom
            cls_tokens = cls_tokens + self.resource_global_proj(pooled)

        # Build sequence: [CLS, cand_1, ..., cand_R_max] per item
        seq = torch.cat([cls_tokens.unsqueeze(1), cand_emb], dim=1)  # [B, R_max+1, d_model]

        # Build key_padding_mask [B, R_max+1]: True = IGNORE this position as a
        # key/value in attention.
        #
        # A candidate may participate only when it is BOTH:
        #   1. a real, non-padding position, and
        #   2. a feasible action.
        # Masking on action_mask (not pad_mask) matches forward()'s semantics
        # exactly: infeasible candidates must not pollute the attention of
        # feasible ones or the CLS token. Both forward paths use this mask so
        # rollout and update distributions agree.
        attendable_mask = action_mask
        if pad_mask is not None:
            attendable_mask = attendable_mask & pad_mask

        # CLS (position 0) is always attended, so no query row is ever fully
        # masked (safe even when every candidate is infeasible).
        cls_col = torch.zeros(B, 1, dtype=torch.bool, device=seq.device)
        key_padding_mask = torch.cat([cls_col, ~attendable_mask], dim=1)  # [B, R_max+1]

        # Transformer encoding
        x = seq
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        x = self.final_norm(x)  # [B, R_max+1, d_model]

        cand_out = x[:, 1:]  # [B, R_max, d_model]

        # Score each candidate
        logits = self.score_head(cand_out).squeeze(-1)  # [B, R_max]

        # Mask infeasible and padded positions. Reuse attendable_mask (the same
        # source of truth as the attention mask) rather than action_mask alone:
        # this guarantees padding stays unselectable even if a future caller
        # accidentally marks a padded slot feasible, and keeps the output mask
        # from drifting out of sync with the attention mask.
        logits = logits.masked_fill(~attendable_mask, -1e9)

        return logits, logits.new_zeros((B,))

    def policy_logits(
        self,
        candidate_feats: torch.Tensor,
        global_feats: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convenience wrapper returning only logits (for IL training)."""
        logits, _ = self.forward(candidate_feats, global_feats, action_mask)
        return logits

    # ------------------------------------------------------------------
    # Loss functions (kept on the model for cohesion)
    # ------------------------------------------------------------------

    @staticmethod
    def listwise_nll(
        logits: torch.Tensor,
        lengths: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Listwise NLL loss: softmax over each state's candidates, penalise expert choice."""
        losses: list[torch.Tensor] = []
        offset = 0
        for length, target in zip(lengths.tolist(), targets):
            if length <= 0:
                offset += length
                continue
            state_logits = logits[offset: offset + length]
            log_probs = F.log_softmax(state_logits, dim=0)
            losses.append(-log_probs[target])
            offset += length
        if not losses:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        return torch.stack(losses).mean()

    @staticmethod
    def listwise_accuracy(
        logits: torch.Tensor,
        lengths: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Top-1 accuracy under the listwise softmax."""
        correct = 0
        total = 0
        offset = 0
        for length, target in zip(lengths.tolist(), targets.tolist()):
            if length <= 0:
                offset += length
                continue
            state_logits = logits[offset: offset + length]
            pred = int(torch.argmax(state_logits).item())
            correct += int(pred == target)
            total += 1
            offset += length
        if total == 0:
            return torch.tensor(0.0, device=logits.device)
        return torch.tensor(correct / total, device=logits.device)


# ---------------------------------------------------------------------------
# Actor checkpoint helpers
# ---------------------------------------------------------------------------

def _policy_config(model: BranchingTransformer) -> dict:
    return {
        "global_dim": model.cls_proj.in_features,
        "candidate_dim": model.cand_proj.in_features,
        "d_model": model.d_model,
        "n_heads": model.layers[0].attn.num_heads,
        "n_layers": len(model.layers),
        "ffn_dim": model.layers[0].ffn[0].out_features,
        "dropout": model.layers[0].attn.dropout,
        "resource_feature_dim": model.resource_feature_dim,
        "interaction_feature_dim": model.interaction_feature_dim,
        "feature_schema": (
            "resource_set_v1" if model.resource_set_enabled else "legacy_flat_v1"
        ),
    }


def save_actor_checkpoint(
    model: BranchingTransformer,
    path: str,
    extra: dict | None = None,
) -> None:
    """Save an actor with the architecture and feature schema needed to reload it."""
    payload = dict(extra or {})
    payload.update({
        "model_state": model.state_dict(),
        "config": _policy_config(model),
        "architecture": "actor_only_v1",
    })
    torch.save(payload, path)


def load_actor_checkpoint(
    path: str,
    device: torch.device | str = "cpu",
    dropout: float = 0.0,
) -> BranchingTransformer:
    """Load an actor, including BC and previous shared/separate PPO checkpoints.

    Legacy value heads and separate critic payloads are ignored. Every active
    actor parameter is required and checked for shape compatibility.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(f"Checkpoint at {path} is missing 'model_state'")
    cfg = checkpoint.get("config", {})
    state = checkpoint["model_state"]
    if cfg.get("value_only", False):
        raise ValueError("A value-only checkpoint does not contain an actor")
    global_dim = int(cfg.get("global_dim") or state["cls_proj.weight"].shape[1])
    candidate_dim = int(cfg.get("candidate_dim") or state["cand_proj.weight"].shape[1])
    d_model = int(cfg.get("d_model") or state["cls_proj.weight"].shape[0])
    n_layers = int(cfg.get("n_layers") or sum(
        1 for key in state if key.startswith("layers.") and key.endswith(".norm1.weight")
    ))
    if "n_heads" not in cfg:
        raise ValueError("Checkpoint config must specify n_heads; weights do not encode it")
    ffn_dim = int(cfg.get("ffn_dim") or state["layers.0.ffn.0.weight"].shape[0])
    resource_feature_dim = int(cfg.get("resource_feature_dim", 0))
    interaction_feature_dim = int(cfg.get("interaction_feature_dim", 0))
    # Set-schema checkpoints can recover widths from their input projections.
    if cfg.get("feature_schema") == "resource_set_v1":
        resource_feature_dim = resource_feature_dim or int(state["resource_proj.weight"].shape[1])
        interaction_feature_dim = interaction_feature_dim or int(state["interaction_proj.weight"].shape[1])
    if bool(resource_feature_dim) != bool(interaction_feature_dim):
        raise ValueError("Resource and interaction feature dimensions must both be set or both be zero")
    model = BranchingTransformer(
        global_dim=global_dim,
        candidate_dim=candidate_dim,
        d_model=d_model,
        n_heads=int(cfg["n_heads"]),
        n_layers=n_layers,
        ffn_dim=ffn_dim,
        dropout=dropout,
        resource_feature_dim=resource_feature_dim,
        interaction_feature_dim=interaction_feature_dim,
    ).to(device)
    ignored_prefixes = ("value_head.",)
    if not model.resource_set_enabled:
        # These modules existed but were never used in some legacy flat models.
        ignored_prefixes += ("resource_proj.", "interaction_proj.", "resource_global_proj.")
    actor_state = {
        key: value for key, value in state.items()
        if not key.startswith(ignored_prefixes)
    }
    model.load_state_dict(actor_state, strict=True)
    model.eval()
    return model


# Public compatibility names used by BC training and the B&B runner.
save_policy_checkpoint = save_actor_checkpoint
load_policy_checkpoint = load_actor_checkpoint
