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

    Architecture
    ------------
    1. Project global node features into a CLS token embedding  [d_model]
    2. Project each candidate activity's features into an embedding  [R, d_model]
    3. Prepend the CLS token: sequence = [cls | cand_1 | ... | cand_R]  [R+1, d_model]
    4. Run N transformer encoder layers — candidates attend to each other
       and to the global context through the CLS token
    5. Score each candidate position with a linear head  →  logits [R]
    6. (Optional) Estimate state value from the CLS token output  →  scalar

    The CLS token acts as the global context carrier. After encoding, each
    candidate's representation has attended to all other candidates and to
    the global state, solving the isolation problem of the previous MLP.

    Parameters
    ----------
    global_dim    : dimension of global feature vector (Fg)
    candidate_dim : dimension of per-candidate feature vector (Fc)
    d_model       : internal embedding dimension
    n_heads       : number of attention heads (must divide d_model)
    n_layers      : number of transformer encoder layers
    ffn_dim       : feed-forward network hidden dimension
    dropout       : dropout rate applied in attention and FFN
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
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} must be divisible by n_heads={n_heads}"

        self.d_model = d_model

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

        # Value head: maps CLS token → scalar state value (used in RL)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single node (unbatched over nodes).

        Parameters
        ----------
        candidate_feats : [R, Fc]  — one row per ready activity
        global_feats    : [Fg]     — global node features (1D)
        action_mask     : [R] bool — True where action is feasible (optional)

        Returns
        -------
        logits : [R]   — unnormalised scores; higher = preferred branch
        value  : []    — scalar state value estimate
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

        # Mask infeasible candidates with large negative value
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)

        # State value from CLS token
        value = self.value_head(cls_out).squeeze(-1)  # scalar

        return logits, value

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
        """
        Listwise negative log-likelihood loss.

        For each state, apply softmax over its candidates and penalise the
        log-probability of the expert-chosen candidate.

        Parameters
        ----------
        logits  : [Nc]  — concatenated logits for all candidates in the batch
        lengths : [B]   — number of candidates per state
        targets : [B]   — index of expert choice within each state's ready list
        """
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
        },
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_policy_checkpoint(
    path: str,
    device: torch.device | str = "cpu",
    dropout: float = 0.0,
) -> BranchingTransformer:
    """
    Load a BranchingTransformer from a checkpoint file.

    Parameters
    ----------
    path    : path to .pt file produced by save_policy_checkpoint
    device  : torch device to map weights to
    dropout : dropout rate to use at inference (typically 0.0)
    """
    checkpoint = torch.load(path, map_location=device)
    if "model_state" not in checkpoint:
        raise ValueError(f"Checkpoint at {path} is missing 'model_state'")

    cfg = checkpoint.get("config", {})
    state = checkpoint["model_state"]

    # Recover architecture dims from the saved weights when the config
    # was overwritten by the training config (legacy checkpoint format).
    global_dim = cfg.get("global_dim") or state["cls_proj.weight"].shape[1]
    candidate_dim = cfg.get("candidate_dim") or state["cand_proj.weight"].shape[1]
    d_model = cfg.get("d_model") or state["cls_proj.weight"].shape[0]
    n_layers = cfg.get("n_layers") or sum(
        1 for k in state if k.startswith("layers.") and k.endswith(".norm1.weight")
    )
    n_heads = cfg.get("n_heads", 4)
    ffn_dim = cfg.get("ffn_dim") or state["layers.0.ffn.0.weight"].shape[0]

    model = BranchingTransformer(
        global_dim=global_dim,
        candidate_dim=candidate_dim,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        ffn_dim=ffn_dim,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model
