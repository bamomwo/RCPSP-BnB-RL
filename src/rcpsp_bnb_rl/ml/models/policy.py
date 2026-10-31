from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Log-space critic values
# ---------------------------------------------------------------------------
#
# The critic predicts remaining expansions below a node in log-node space:
#     V_log = -ln(1 + |V_raw|) <= 0,   V_raw = -(expansions remaining) <= 0.
# This puts 1-node and 100k-node subtrees on one stationary scale, so no
# running return normalizer is needed. Tree-GAE backups run in raw units.

LOG_VALUE_MIN = -15.0  # exp(15) ~ 3.3M nodes: clamp before inverting


def log_value_to_raw(v_log: torch.Tensor) -> torch.Tensor:
    """V_raw = -(exp(-clamp(V_log, -15, 0)) - 1)."""
    return -torch.expm1(-v_log.clamp(LOG_VALUE_MIN, 0.0))


def raw_return_to_log(y_raw: torch.Tensor) -> torch.Tensor:
    """Y_log = clamp(-ln(1 + max(0, -Y_raw)), -15, 0): the critic regression target.

    The floor matches log_value_to_raw's clamp, so a saturated prediction and a
    saturated target compare as equal instead of producing a false advantage.
    """
    return (-torch.log1p(torch.clamp(-y_raw, min=0.0))).clamp(min=LOG_VALUE_MIN)


def clamp_log_value(v_log: torch.Tensor) -> torch.Tensor:
    """Critic prediction on the same [LOG_VALUE_MIN, 0] range as its targets."""
    return v_log.clamp(LOG_VALUE_MIN, 0.0)


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
    → logits [R] over candidates + value scalar from CLS output. The value is
    emitted in log-node space through -softplus, so it is always <= 0.

    By default both output heads use one encoder. ``policy_only`` and
    ``value_only`` omit the unused head when building independent PPO networks.
    The omitted output is zero to preserve the inference return signature.
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
        policy_only: bool = False,
        value_only: bool = False,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} must be divisible by n_heads={n_heads}"

        self.d_model = d_model
        self.critic_feature_dim = critic_feature_dim
        self.policy_only = bool(policy_only)
        self.value_only = bool(value_only)
        if self.policy_only and self.value_only:
            raise ValueError("A transformer cannot be both policy-only and value-only")
        if self.policy_only and critic_feature_dim:
            raise ValueError("A policy-only transformer cannot consume critic features")
        self.resource_feature_dim = int(resource_feature_dim)
        self.interaction_feature_dim = int(interaction_feature_dim)
        self.resource_set_enabled = (
            self.resource_feature_dim > 0 and self.interaction_feature_dim > 0
        )

        # Input projections
        self.cls_proj = nn.Linear(global_dim, d_model)
        self.cand_proj = nn.Linear(candidate_dim, d_model)

        # Shared resource encoder: the same weights process every resource, so
        # the representation is independent of resource count and order. The
        # modules are defined even for legacy models so TorchScript sees a
        # stable module schema; they are bypassed when resource_set_enabled is
        # false.
        self.resource_proj = nn.Linear(max(self.resource_feature_dim, 1), d_model)
        self.interaction_proj = nn.Linear(max(self.interaction_feature_dim, 1), d_model)
        self.resource_global_proj = nn.Linear(d_model, d_model)

        # Transformer encoder
        self.layers = nn.ModuleList([
            _TransformerEncoderLayer(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Scoring head: maps each candidate embedding → scalar logit
        self.score_head = nn.Identity() if self.value_only else nn.Linear(d_model, 1)

        # Value head: CLS token + optional critic features → scalar
        self.value_head: nn.Module = (
            nn.Identity()
            if self.policy_only
            else nn.Sequential(
                nn.Linear(d_model + critic_feature_dim, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 1),
            )
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
        resource_feats: Optional[torch.Tensor] = None,
        candidate_resource_feats: Optional[torch.Tensor] = None,
        resource_mask: Optional[torch.Tensor] = None,
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

        cls_out = x[0]       # [d_model]  — global context after encoding
        cand_out = x[1:]     # [R, d_model] — candidate representations

        # Score each candidate
        if self.value_only:
            logits = cand_out.new_zeros((R,))
        else:
            logits = self.score_head(cand_out).squeeze(-1)  # [R]

        # Mask infeasible candidates
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)

        # Value from CLS + optional critic features. A policy-only encoder
        # retains the return signature but does not allocate a value head.
        if self.policy_only:
            value = cls_out.new_zeros(())
        else:
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
            value = -F.softplus(self.value_head(value_in).squeeze(-1))  # log-space, <= 0

        return logits, value

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
        # feasible ones, nor the CLS/value token. Using pad_mask alone let
        # infeasible-but-real candidates attend during the PPO update while
        # forward() (rollout/eval) masked them — a train/serve skew that also
        # made the importance ratio != 1 at update step 0.
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

        cls_out = x[:, 0]    # [B, d_model]
        cand_out = x[:, 1:]  # [B, R_max, d_model]

        # Score each candidate
        if self.value_only:
            logits = cand_out.new_zeros((B, R_max))
        else:
            logits = self.score_head(cand_out).squeeze(-1)  # [B, R_max]

        # Mask infeasible and padded positions. Reuse attendable_mask (the same
        # source of truth as the attention mask) rather than action_mask alone:
        # this guarantees padding stays unselectable even if a future caller
        # accidentally marks a padded slot feasible, and keeps the output mask
        # from drifting out of sync with the attention mask.
        logits = logits.masked_fill(~attendable_mask, -1e9)

        # State value from CLS token + critic features
        if self.policy_only:
            values = cls_out.new_zeros((B,))
        else:
            if self.critic_feature_dim > 0:
                if critic_feats is None:
                    critic_feats = torch.zeros(
                        B, self.critic_feature_dim,
                        dtype=cls_out.dtype, device=cls_out.device,
                    )
                value_in = torch.cat([cls_out, critic_feats], dim=-1)  # [B, d_model+Fk]
            else:
                value_in = cls_out
            values = -F.softplus(self.value_head(value_in).squeeze(-1))  # [B], log-space

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
# Separate actor-critic model
# ---------------------------------------------------------------------------

class SeparateActorCritic(nn.Module):
    """Actor-critic with independent transformer trunks.

    The actor and critic use the same input schema, but have distinct
    parameter objects.  Consequently, value-loss gradients cannot update the
    actor representation.  The actor is a policy-only ``BranchingTransformer``
    (no critic features); the critic is a value-enabled ``BranchingTransformer``.
    """

    def __init__(self, actor: BranchingTransformer, critic: BranchingTransformer) -> None:
        super().__init__()
        if not actor.policy_only:
            raise ValueError("The separate actor must be policy-only")
        if not critic.value_only or critic.critic_feature_dim <= 0:
            raise ValueError("The separate critic must be value-only")
        actor_schema = self._schema(actor)
        critic_schema = self._schema(critic)
        if actor_schema != critic_schema:
            raise ValueError("Actor and critic transformer schemas must match")
        if {p.data_ptr() for p in actor.parameters()} & {p.data_ptr() for p in critic.parameters()}:
            raise ValueError("Actor and critic parameters must have independent storage")
        self.actor = actor
        self.critic = critic

    @classmethod
    def from_shared(
        cls, model: BranchingTransformer, dropout: float | None = None,
    ) -> SeparateActorCritic:
        """Split a shared model without changing its initial logits or values.

        Copy both trunks rather than randomly initializing the critic trunk:
        the shared/separate ablation then starts with identical predictions.
        Neither branch retains the unused output head of the other task.
        """
        if model.policy_only or model.value_only:
            raise ValueError("Splitting requires a model with both actor and critic heads")
        cfg = _policy_config(model)
        cfg.pop("feature_schema")
        if dropout is not None:
            cfg["dropout"] = dropout
        # These constructors allocate on CPU and all weights are overwritten.
        # Preserve the rollout RNG stream for the shared/separate comparison.
        with torch.random.fork_rng(devices=[]):
            cfg.update(policy_only=True, critic_feature_dim=0)
            actor = BranchingTransformer(**cfg)
            cfg.update(
                policy_only=False,
                value_only=True,
                critic_feature_dim=model.critic_feature_dim,
            )
            critic = BranchingTransformer(**cfg)
        parameter = next(model.parameters())
        actor.to(device=parameter.device, dtype=parameter.dtype)
        critic.to(device=parameter.device, dtype=parameter.dtype)
        state = model.state_dict()
        actor.load_state_dict({k: state[k] for k in actor.state_dict()})
        critic.load_state_dict({k: state[k] for k in critic.state_dict()})
        result = cls(actor, critic)
        result.train(model.training)
        return result

    @staticmethod
    def _schema(model: BranchingTransformer) -> tuple:
        return (
            model.cls_proj.in_features,
            model.cand_proj.in_features,
            model.d_model,
            model.layers[0].attn.num_heads,
            len(model.layers),
            model.layers[0].ffn[0].out_features,
            model.resource_feature_dim,
            model.interaction_feature_dim,
        )

    @property
    def resource_set_enabled(self) -> bool:
        return self.actor.resource_set_enabled

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
        logits, _ = self.actor(
            candidate_feats, global_feats, action_mask,
            resource_feats=resource_feats,
            candidate_resource_feats=candidate_resource_feats,
            resource_mask=resource_mask,
        )
        _, value = self.critic(
            candidate_feats, global_feats, action_mask, critic_feats,
            resource_feats, candidate_resource_feats, resource_mask,
        )
        return logits, value

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
        actor_logits, _ = self.actor.forward_batch(
            candidate_feats, global_feats, action_mask,
            critic_feats=None, pad_mask=pad_mask,
            resource_feats=resource_feats,
            candidate_resource_feats=candidate_resource_feats,
            resource_mask=resource_mask,
        )
        _, critic_values = self.critic.forward_batch(
            candidate_feats, global_feats, action_mask,
            critic_feats=critic_feats, pad_mask=pad_mask,
            resource_feats=resource_feats,
            candidate_resource_feats=candidate_resource_feats,
            resource_mask=resource_mask,
        )
        return actor_logits, critic_values


# ---------------------------------------------------------------------------
# Checkpoint helpers
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
        "critic_feature_dim": model.critic_feature_dim,
        "policy_only": model.policy_only,
        "value_only": model.value_only,
        "resource_feature_dim": model.resource_feature_dim,
        "interaction_feature_dim": model.interaction_feature_dim,
        "feature_schema": (
            "resource_set_v1" if model.resource_set_enabled else "legacy_flat_v1"
        ),
    }


def save_policy_checkpoint(
    model: BranchingTransformer,
    path: str,
    extra: dict | None = None,
) -> None:
    """Save a BranchingTransformer checkpoint."""
    payload = {
        "model_state": model.state_dict(),
        "config": _policy_config(model),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def save_separate_actor_critic_checkpoint(
    model: SeparateActorCritic,
    path: str,
    extra: dict | None = None,
) -> None:
    """Save a deployment-compatible actor plus the separately trained critic.

    ``model_state`` deliberately contains only the actor.  Existing B&B
    runners can therefore load the checkpoint with ``load_policy_checkpoint``.
    The critic and its architecture are retained for subsequent PPO loading.
    """
    actor = model.actor
    payload = {
        "model_state": actor.state_dict(),
        "critic_state": model.critic.state_dict(),
        "critic_config": _policy_config(model.critic),
        "actor_critic_architecture": "separate_transformers_v1",
        "config": _policy_config(actor),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def _model_from_state(
    cfg: dict,
    state: dict,
    device: torch.device | str,
    dropout: float,
    critic_feature_dim: int | None = None,
) -> BranchingTransformer:
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

    resource_feature_dim = int(cfg.get("resource_feature_dim", 0))
    interaction_feature_dim = int(cfg.get("interaction_feature_dim", 0))

    model = BranchingTransformer(
        global_dim=global_dim,
        candidate_dim=candidate_dim,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        ffn_dim=ffn_dim,
        dropout=dropout,
        critic_feature_dim=critic_feature_dim,
        resource_feature_dim=resource_feature_dim,
        interaction_feature_dim=interaction_feature_dim,
        policy_only=bool(cfg.get("policy_only", False)),
        value_only=bool(cfg.get("value_only", False)),
    ).to(device)

    # Drop checkpoint weights whose shape no longer matches (e.g. resized value head).
    model_state = model.state_dict()
    filtered = {
        k: v for k, v in state.items()
        if k in model_state and model_state[k].shape == v.shape
    }
    dropped = [k for k in state if k not in filtered]
    model.load_state_dict(filtered, strict=False)
    if dropped:
        print(
            f"[load_policy_checkpoint] reinitialised {len(dropped)} param(s) "
            f"with changed shape (value-head resize): {dropped}"
        )
    model.eval()
    return model


def _checkpoint_value_norm(checkpoint: dict) -> Optional[dict]:
    value_norm = checkpoint.get("value_norm")
    if value_norm is not None and not isinstance(value_norm, dict):
        raise ValueError("Checkpoint has an invalid 'value_norm' payload")
    return value_norm


def load_policy_checkpoint(
    path: str,
    device: torch.device | str = "cpu",
    dropout: float = 0.0,
    critic_feature_dim: int | None = None,
    return_value_norm: bool = False,
) -> BranchingTransformer | Tuple[BranchingTransformer, Optional[dict]]:
    """Load a shared policy or only the actor of a separate PPO checkpoint.

    ``critic_feature_dim`` can resize a shared model's value head for PPO.
    Separate checkpoints expose only the deployment actor here; use
    ``load_separate_actor_critic_checkpoint`` to restore both networks and the
    critic's value-normalization state.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "model_state" not in checkpoint:
        raise ValueError(f"Checkpoint at {path} is missing 'model_state'")
    cfg = checkpoint.get("config", {})
    if cfg.get("policy_only", False) and critic_feature_dim not in (None, 0):
        raise ValueError("A policy-only checkpoint requires the separate actor-critic loader for PPO")
    model = _model_from_state(cfg, checkpoint["model_state"], device, dropout, critic_feature_dim)
    if return_value_norm:
        value_norm = None if model.policy_only else _checkpoint_value_norm(checkpoint)
        return model, value_norm
    return model


def load_separate_actor_critic_checkpoint(
    path: str,
    device: torch.device | str = "cpu",
    dropout: float = 0.0,
    critic_feature_dim: int | None = None,
    return_value_norm: bool = False,
) -> SeparateActorCritic | Tuple[SeparateActorCritic, Optional[dict]]:
    """Restore separate PPO networks, or split a BC/shared PPO initialization.

    When starting from BC, value-head expansion is identical to the shared PPO
    path. Both independent trunks then copy that model's initial weights, so
    sharing is the architecture difference at the start of the experiment.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "model_state" not in checkpoint:
        raise ValueError(f"Checkpoint at {path} is missing 'model_state'")
    if checkpoint.get("actor_critic_architecture") == "separate_transformers_v1":
        actor = _model_from_state(
            checkpoint["config"], checkpoint["model_state"], device, dropout,
        )
        critic = _model_from_state(
            checkpoint["critic_config"], checkpoint["critic_state"], device, dropout,
            critic_feature_dim,
        )
        model = SeparateActorCritic(actor, critic)
        saved_critic_dim = int(checkpoint["critic_config"]["critic_feature_dim"])
    else:
        shared = _model_from_state(
            checkpoint.get("config", {}), checkpoint["model_state"], device, dropout,
            critic_feature_dim,
        )
        model = SeparateActorCritic.from_shared(shared, dropout=dropout)
        saved_critic_dim = int(checkpoint.get("config", {}).get("critic_feature_dim", 0))
    model.eval()
    if return_value_norm:
        value_norm = (
            _checkpoint_value_norm(checkpoint)
            if model.critic.critic_feature_dim == saved_critic_dim else None
        )
        return model, value_norm
    return model
