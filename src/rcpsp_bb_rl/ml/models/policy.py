from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _mlp(sizes: Sequence[int], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for in_dim, out_dim in zip(sizes[:-1], sizes[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


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
    → logits [R] over candidates + value scalar from CLS output.
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
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} must be divisible by n_heads={n_heads}"

        self.d_model = d_model
        self.critic_feature_dim = critic_feature_dim

        # Input projections
        self.cls_proj = nn.Linear(global_dim, d_model)
        self.cand_proj = nn.Linear(candidate_dim, d_model)

        # Transformer encoder
        self.layers = nn.ModuleList([
            _TransformerEncoderLayer(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Scoring head: maps each candidate embedding → scalar logit
        self.score_head = nn.Linear(d_model, 1)

        # Value head: CLS token + optional critic features → scalar
        self.value_head = nn.Sequential(
            nn.Linear(d_model + critic_feature_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unbatched forward pass.

        Parameters: candidate_feats [R, Fc], global_feats [Fg],
                    action_mask [R] bool, critic_feats [Fk] optional.
        Returns: logits [R], value scalar.
        """
        R = candidate_feats.shape[0]

        # Project inputs into d_model space
        cls_token = self.cls_proj(global_feats).unsqueeze(0)   # [1, d_model]
        cand_emb = self.cand_proj(candidate_feats)              # [R, d_model]

        # Build sequence: [CLS, cand_1, ..., cand_R]  →  [R+1, d_model]
        seq = torch.cat([cls_token, cand_emb], dim=0).unsqueeze(0)  # [1, R+1, d_model]

        # Build key_padding_mask: mask infeasible candidates so they don't
        # pollute attention of feasible ones.  CLS token is always unmasked.
        key_padding_mask = None
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

        cls_out = x[0]       # [d_model]  — global context after encoding
        cand_out = x[1:]     # [R, d_model] — candidate representations

        # Score each candidate
        logits = self.score_head(cand_out).squeeze(-1)  # [R]

        # Mask infeasible candidates
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)

        # Value from CLS + optional critic features
        if self.critic_feature_dim > 0:
            if critic_feats is None:
                critic_feats = torch.zeros(
                    self.critic_feature_dim,
                    dtype=cls_out.dtype,
                    device=cls_out.device,
                )
            value_in = torch.cat([cls_out, critic_feats], dim=-1)  # [d_model + Fk]
        else:
            value_in = cls_out
        value = self.value_head(value_in).squeeze(-1)  # scalar

        return logits, value

    def forward_batch(
        self,
        candidate_feats: torch.Tensor,
        global_feats: torch.Tensor,
        action_mask: torch.Tensor,
        critic_feats: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched forward pass for GPU-optimized PPO updates.

        Parameters: candidate_feats [B, R_max, Fc], global_feats [B, Fg],
                    action_mask [B, R_max] bool, critic_feats [B, Fk] optional,
                    pad_mask [B, R_max] bool (True=real, False=padding).
        Returns: logits [B, R_max], values [B].
        """
        B, R_max, _ = candidate_feats.shape

        # Project inputs into d_model space
        cls_tokens = self.cls_proj(global_feats)              # [B, d_model]
        cand_emb = self.cand_proj(candidate_feats)            # [B, R_max, d_model]

        # Build sequence: [CLS, cand_1, ..., cand_R_max] per item
        seq = torch.cat([cls_tokens.unsqueeze(1), cand_emb], dim=1)  # [B, R_max+1, d_model]

        # Build key_padding_mask [B, R_max+1]: True = IGNORE in attention.
        # CLS (position 0) is always attended; padded candidate positions are ignored.
        if pad_mask is not None:
            cls_col = torch.zeros(B, 1, dtype=torch.bool, device=seq.device)
            key_padding_mask = torch.cat([cls_col, ~pad_mask], dim=1)  # [B, R_max+1]
        else:
            key_padding_mask = None

        # Transformer encoding
        x = seq
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        x = self.final_norm(x)  # [B, R_max+1, d_model]

        cls_out = x[:, 0]    # [B, d_model]
        cand_out = x[:, 1:]  # [B, R_max, d_model]

        # Score each candidate
        logits = self.score_head(cand_out).squeeze(-1)  # [B, R_max]

        # Mask infeasible and padded positions
        logits = logits.masked_fill(~action_mask, -1e9)

        # State value from CLS token + critic features
        if self.critic_feature_dim > 0:
            if critic_feats is None:
                critic_feats = torch.zeros(
                    B, self.critic_feature_dim,
                    dtype=cls_out.dtype, device=cls_out.device,
                )
            value_in = torch.cat([cls_out, critic_feats], dim=-1)  # [B, d_model+Fk]
        else:
            value_in = cls_out
        values = self.value_head(value_in).squeeze(-1)  # [B]

        return logits, values

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
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_policy_checkpoint(
    model: BranchingTransformer,
    path: str,
    extra: dict | None = None,
) -> None:
    """Save a BranchingTransformer checkpoint."""
    payload = {
        "model_state": model.state_dict(),
        "config": {
            "global_dim": model.cls_proj.in_features,
            "candidate_dim": model.cand_proj.in_features,
            "d_model": model.d_model,
            "n_heads": model.layers[0].attn.num_heads,
            "n_layers": len(model.layers),
            "ffn_dim": model.layers[0].ffn[0].out_features,
            "dropout": 0.0,  # not stored in module; caller may override
            "critic_feature_dim": model.critic_feature_dim,
        },
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_policy_checkpoint(
    path: str,
    device: torch.device | str = "cpu",
    dropout: float = 0.0,
    critic_feature_dim: int | None = None,
) -> BranchingTransformer:
    """
    Load a BranchingTransformer from a checkpoint. If critic_feature_dim is given,
    override the value-head width (for warm-starting RL from a BC checkpoint).
    """
    checkpoint = torch.load(path, map_location=device)
    if "model_state" not in checkpoint:
        raise ValueError(f"Checkpoint at {path} is missing 'model_state'")

    cfg = checkpoint.get("config", {})
    state = checkpoint["model_state"]

    # Recover architecture dims from saved weights if config is incomplete
    global_dim = cfg.get("global_dim") or state["cls_proj.weight"].shape[1]
    candidate_dim = cfg.get("candidate_dim") or state["cand_proj.weight"].shape[1]
    d_model = cfg.get("d_model") or state["cls_proj.weight"].shape[0]
    n_layers = cfg.get("n_layers") or sum(
        1 for k in state if k.startswith("layers.") and k.endswith(".norm1.weight")
    )
    n_heads = cfg.get("n_heads", 4)
    ffn_dim = cfg.get("ffn_dim") or state["layers.0.ffn.0.weight"].shape[0]

    if critic_feature_dim is None:
        critic_feature_dim = int(cfg.get("critic_feature_dim", 0))

    model = BranchingTransformer(
        global_dim=global_dim,
        candidate_dim=candidate_dim,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        ffn_dim=ffn_dim,
        dropout=dropout,
        critic_feature_dim=critic_feature_dim,
    ).to(device)

    # Drop checkpoint weights whose shape no longer matches (e.g. resized value head).
    model_state = model.state_dict()
    filtered = {
        k: v for k, v in state.items()
        if k in model_state and model_state[k].shape == v.shape
    }
    dropped = [k for k in state if k not in filtered]
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if dropped:
        print(
            f"[load_policy_checkpoint] reinitialised {len(dropped)} param(s) "
            f"with changed shape (value-head resize): {dropped}"
        )
    model.eval()
    return model
