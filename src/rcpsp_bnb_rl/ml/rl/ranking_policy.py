from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from torch.distributions import Categorical


@dataclass(frozen=True)
class RankingAction:
    """One compound PPO action and the complete order sent to the solver."""

    feasible_order_indices: List[int]
    solver_order_indices: List[int]
    log_prob: torch.Tensor
    entropy: torch.Tensor
    entropy_per_choice: torch.Tensor


def _pack_rankings(
    rankings: Sequence[Sequence[int]],
    *,
    candidate_count: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack variable-length candidate-index permutations for batched replay."""
    batch_size = len(rankings)
    max_length = max((len(ranking) for ranking in rankings), default=0)
    order_cpu = torch.zeros((batch_size, max_length), dtype=torch.long)
    order_mask_cpu = torch.zeros((batch_size, max_length), dtype=torch.bool)
    lengths_cpu = torch.zeros(batch_size, dtype=torch.long)

    for row, ranking_raw in enumerate(rankings):
        ranking = [int(index) for index in ranking_raw]
        if len(set(ranking)) != len(ranking):
            raise ValueError(f"Ranking contains duplicate candidate indices: {ranking}")
        if any(index < 0 or index >= candidate_count for index in ranking):
            raise ValueError(
                f"Ranking contains an index outside [0, {candidate_count}): {ranking}"
            )
        length = len(ranking)
        lengths_cpu[row] = length
        if length:
            order_cpu[row, :length] = torch.tensor(ranking, dtype=torch.long)
            order_mask_cpu[row, :length] = True

    return (
        order_cpu.to(device=device),
        order_mask_cpu.to(device=device),
        lengths_cpu.to(device=device),
    )


def ranking_log_probs_entropy(
    logits_batch: torch.Tensor,
    feasible_mask_batch: torch.Tensor,
    rankings: Sequence[Sequence[int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replay Plackett-Luce rankings and return log-probability and entropy.

    The ranking probability is the product of categorical selections without
    replacement. Entropy is the sum of the conditional categorical entropies
    encountered along the sampled prefix. Its per-choice form is reported only
    as a scale-independent diagnostic; PPO optimizes the total action entropy.
    """
    if logits_batch.ndim != 2:
        raise ValueError(
            f"Expected logits with shape [B, R], got {tuple(logits_batch.shape)}"
        )
    if feasible_mask_batch.shape != logits_batch.shape:
        raise ValueError(
            "Feasible mask shape must equal logits shape: "
            f"{tuple(feasible_mask_batch.shape)} != {tuple(logits_batch.shape)}"
        )
    if len(rankings) != logits_batch.shape[0]:
        raise ValueError(
            f"Expected {logits_batch.shape[0]} rankings, got {len(rankings)}"
        )

    batch_size, candidate_count = logits_batch.shape
    order, order_mask, lengths = _pack_rankings(
        rankings,
        candidate_count=candidate_count,
        device=logits_batch.device,
    )

    represented = torch.zeros_like(feasible_mask_batch, dtype=torch.long)
    if order.shape[1]:
        represented.scatter_add_(1, order, order_mask.to(dtype=torch.long))
    if not torch.equal(represented.bool(), feasible_mask_batch.bool()):
        raise ValueError("Each ranking must contain every feasible candidate exactly once")

    if order.shape[1] == 0:
        zeros = logits_batch.new_zeros(batch_size)
        return zeros, zeros, zeros

    ordered_logits = logits_batch.gather(1, order)
    masked_logits = torch.where(
        order_mask,
        ordered_logits,
        torch.full_like(ordered_logits, -1e9),
    )

    suffix_log_normalizers = torch.flip(
        torch.logcumsumexp(torch.flip(masked_logits, dims=[1]), dim=1),
        dims=[1],
    )
    selected_log_probs = torch.where(
        order_mask,
        masked_logits - suffix_log_normalizers,
        torch.zeros_like(masked_logits),
    )
    log_probs = selected_log_probs.sum(dim=1)

    max_length = order.shape[1]
    positions = torch.arange(max_length, device=logits_batch.device)
    suffix_mask = positions.unsqueeze(0) >= positions.unsqueeze(1)
    conditional_mask = (
        order_mask.unsqueeze(2)
        & order_mask.unsqueeze(1)
        & suffix_mask.unsqueeze(0)
    )
    conditional_log_probs = torch.where(
        conditional_mask,
        masked_logits.unsqueeze(1) - suffix_log_normalizers.unsqueeze(2),
        torch.zeros_like(conditional_mask, dtype=logits_batch.dtype),
    )
    conditional_terms = torch.exp(conditional_log_probs) * conditional_log_probs
    entropies = -conditional_terms.sum(dim=(1, 2))
    stochastic_choices = (lengths - 1).clamp_min(1).to(dtype=entropies.dtype)
    entropy_per_choice = entropies / stochastic_choices
    return log_probs, entropies, entropy_per_choice


def make_ranking_action(
    logits: torch.Tensor,
    feasible_mask: torch.Tensor,
    *,
    sample: bool = True,
) -> RankingAction:
    """Sample or greedily construct one complete feasible activity ranking."""
    if logits.ndim != 1:
        raise ValueError(f"Expected one-dimensional logits, got {tuple(logits.shape)}")
    if feasible_mask.shape != logits.shape:
        raise ValueError(
            f"Feasible mask shape {tuple(feasible_mask.shape)} does not match logits"
        )

    remaining = torch.nonzero(feasible_mask, as_tuple=False).flatten()
    feasible_order: List[int] = []

    while remaining.numel() > 1:
        remaining_logits = logits.index_select(0, remaining)
        if sample:
            position = Categorical(logits=remaining_logits).sample()
        else:
            position = torch.argmax(remaining_logits)
        position_index = int(position.item())
        selected = remaining[position_index]
        feasible_order.append(int(selected.item()))
        remaining = torch.cat(
            (remaining[:position_index], remaining[position_index + 1 :])
        )

    if remaining.numel() == 1:
        feasible_order.append(int(remaining[0].item()))

    infeasible_order = torch.nonzero(~feasible_mask, as_tuple=False).flatten().tolist()
    solver_order = feasible_order + [int(index) for index in infeasible_order]
    log_probs, entropies, entropy_per_choice = ranking_log_probs_entropy(
        logits.unsqueeze(0),
        feasible_mask.unsqueeze(0),
        [feasible_order],
    )
    return RankingAction(
        feasible_order_indices=feasible_order,
        solver_order_indices=solver_order,
        log_prob=log_probs[0],
        entropy=entropies[0],
        entropy_per_choice=entropy_per_choice[0],
    )
