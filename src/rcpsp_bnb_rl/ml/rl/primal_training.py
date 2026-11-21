"""Actor-only learning from complete, matched sibling continuations."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from rcpsp_bnb_rl.ml.observations import batch_observations


def sibling_ranking_loss(logits: torch.Tensor, makespans: torch.Tensor,
                         mask: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Average strict-pair RankNet within states, then across informative states."""
    better = makespans.unsqueeze(2) < makespans.unsqueeze(1)
    pairs = better & mask.unsqueeze(2) & mask.unsqueeze(1)
    counts = pairs.sum(dim=(1, 2))
    informative = counts > 0
    n_pairs = int(counts.sum().item())
    n_states = int(informative.sum().item())
    if not n_states:
        return logits.sum() * 0.0, 0, 0
    differences = logits.unsqueeze(2) - logits.unsqueeze(1)
    per_state = (F.softplus(-differences) * pairs).sum(dim=(1, 2)) / counts.clamp_min(1)
    return per_state[informative].mean(), n_pairs, n_states


def categorical_kl(old_logits: torch.Tensor, logits: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
    """Exact KL(old || current), one value per decision state."""
    old_logp = old_logits.masked_fill(~mask, -1e9).log_softmax(dim=-1)
    logp = logits.masked_fill(~mask, -1e9).log_softmax(dim=-1)
    return (old_logp.exp() * (old_logp - logp) * mask).sum(dim=-1).clamp_min(0)


def _record_batch(records: Sequence, device: torch.device):
    batch = batch_observations([r.observation for r in records], device)
    mask = batch["action_mask"]
    old_logits = torch.full(mask.shape, -1e9, device=device)
    makespans = torch.zeros(mask.shape, device=device)
    for i, record in enumerate(records):
        size = len(record.old_logits)
        old_logits[i, :size] = record.old_logits.to(device)
        makespans[i, :size] = record.makespans.to(device)
    return batch, old_logits, makespans


def _validate_batch_size(batch_size: int) -> None:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Diagnostic batch size must be a positive integer")


@torch.no_grad()
def policy_kl_metrics(actor, records: Sequence, device: torch.device | str,
                      *, batch_size: int = 128) -> dict:
    """Measure exact all-state KL in bounded batches, without pairwise tensors."""
    _validate_batch_size(batch_size)
    total_kl = 0.0
    max_kl = 0.0
    for start in range(0, len(records), batch_size):
        batch, old, _ = _record_batch(records[start:start + batch_size], device)
        logits, _ = actor.forward_batch(**batch)
        kl = categorical_kl(old, logits, batch["action_mask"])
        total_kl += float(kl.double().sum())
        max_kl = max(max_kl, float(kl.max()))
    return {"kl_old": total_kl / max(1, len(records)), "max_state_kl": max_kl}


@torch.no_grad()
def ranking_metrics(actor, records: Sequence, device: torch.device | str,
                    *, batch_size: int = 128) -> dict:
    """Stream diagnostics, preserving their state and pair denominators."""
    _validate_batch_size(batch_size)
    if not records:
        return {"ranknet_loss": 0.0, "ranking_accuracy": None, "kl_old": 0.0,
                "max_state_kl": 0.0, "entropy": 0.0, "pairs": 0, "states": 0}
    loss_sum = correct_sum = entropy_sum = kl_sum = 0.0
    total_pairs = total_states = 0
    max_kl = 0.0
    for start in range(0, len(records), batch_size):
        batch, old, outcomes = _record_batch(records[start:start + batch_size], device)
        logits, _ = actor.forward_batch(**batch)
        mask = batch["action_mask"]
        loss, n_pairs, n_states = sibling_ranking_loss(logits, outcomes, mask)
        pairs = (outcomes.unsqueeze(2) < outcomes.unsqueeze(1)) & mask.unsqueeze(2) & mask.unsqueeze(1)
        differences = logits.unsqueeze(2) - logits.unsqueeze(1)
        correct = (differences > 0).float() + 0.5 * (differences == 0).float()
        loss_sum += float(loss) * n_states
        correct_sum += float(correct[pairs].double().sum())
        total_pairs += n_pairs
        total_states += n_states
        logp = logits.log_softmax(-1)
        entropy_sum += float((-(logp.exp() * logp * mask).sum(-1)).double().sum())
        kl = categorical_kl(old, logits, mask)
        kl_sum += float(kl.double().sum())
        max_kl = max(max_kl, float(kl.max()))
    return {"ranknet_loss": loss_sum / max(1, total_states),
            "ranking_accuracy": correct_sum / total_pairs if total_pairs else None,
            "kl_old": kl_sum / len(records), "max_state_kl": max_kl,
            "entropy": entropy_sum / len(records), "pairs": total_pairs, "states": total_states}


def train_update(actor, optimizer, records: Sequence, device: torch.device | str,
                 *, epochs: int = 2, minibatch_size: int = 128,
                 entropy_coef: float = 0.001, target_kl: float | None = 0.02,
                 max_grad_norm: float = 0.5, metrics_batch_size: int | None = None) -> dict:
    """Freeze rollout labels; stop updates when measured batch KL exceeds target."""
    if not actor.policy_only:
        raise ValueError("Sibling training requires policy_only=True")
    if epochs < 1 or minibatch_size < 1:
        raise ValueError("epochs and minibatch_size must be positive")
    metrics_batch_size = minibatch_size if metrics_batch_size is None else metrics_batch_size
    _validate_batch_size(metrics_batch_size)
    device = torch.device(device)
    actor.eval()  # Gradients remain enabled; dropout must match greedy collection.
    informative = [r for r in records if len(torch.unique(r.makespans[r.observation["action_mask"]])) > 1]
    before = ranking_metrics(actor, records, device, batch_size=metrics_batch_size)
    steps = 0
    stopped = False
    if informative:
        for _ in range(epochs):
            indices = torch.randperm(len(informative)).tolist()
            for start in range(0, len(indices), minibatch_size):
                selected = [informative[i] for i in indices[start:start + minibatch_size]]
                batch, _, outcomes = _record_batch(selected, device)
                logits, _ = actor.forward_batch(**batch)
                mask = batch["action_mask"]
                rank_loss, _, _ = sibling_ranking_loss(logits, outcomes, mask)
                logp = logits.log_softmax(-1)
                entropy = -(logp.exp() * logp * mask).sum(-1).mean()
                loss = rank_loss - entropy_coef * entropy
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite sibling training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                optimizer.step()
                steps += 1
                if target_kl is not None:
                    measured = policy_kl_metrics(actor, records, device, batch_size=metrics_batch_size)
                    if measured["kl_old"] > target_kl:
                        stopped = True
                        break
            if stopped:
                break
    after = ranking_metrics(actor, records, device, batch_size=metrics_batch_size)
    return {"before": before, "after": after, "optimizer_steps": steps,
            "kl_early_stop": stopped, "informative_states": len(informative),
            "fork_states": len(records), "signal_density": len(informative) / max(1, len(records))}
