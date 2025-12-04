from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(sizes: Sequence[int], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for in_dim, out_dim in zip(sizes[:-1], sizes[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers) if layers else nn.Sequential()


class PolicyMLP(nn.Module):
    """
    Simple MLP scoring network for imitation learning over ready tasks.

    Expects flattened candidate/global feature tensors:
        candidate_feats: [N, Fc]
        global_feats:    [N, Fg]
    Returns logits of shape [N]; apply sigmoid for probabilities.
    """

    def __init__(
        self,
        global_dim: int,
        candidate_dim: int,
        hidden_sizes: Iterable[int] = (128, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_sizes = list(hidden_sizes)
        input_dim = global_dim + candidate_dim
        mlp_sizes = [input_dim, *hidden_sizes]
        self.backbone = _mlp(mlp_sizes, dropout)
        last_dim = mlp_sizes[-1] if hidden_sizes else input_dim
        self.head = nn.Linear(last_dim, 1)

    def forward(self, candidate_feats: torch.Tensor, global_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            candidate_feats: Tensor of shape [N, Fc]
            global_feats: Tensor of shape [N, Fg]

        Returns:
            logits: Tensor of shape [N]
        """
        x = torch.cat([candidate_feats, global_feats], dim=-1)
        h = self.backbone(x) if len(self.backbone) > 0 else x
        logits = self.head(h).squeeze(-1)
        return logits

    @staticmethod
    def bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Binary cross-entropy loss for 0/1 labels."""
        return F.binary_cross_entropy_with_logits(logits, labels.float())

    @staticmethod
    def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute sigmoid-threshold accuracy for monitoring."""
        preds = (logits > 0).long()
        return (preds == labels).float().mean()
