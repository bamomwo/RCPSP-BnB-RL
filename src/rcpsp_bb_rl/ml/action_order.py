from __future__ import annotations

from typing import List, Optional, Sequence


def policy_ranked_indices(scores: Sequence[float]) -> List[int]:
    """Return candidate indices in descending-score order with stable ties."""
    return sorted(range(len(scores)), key=scores.__getitem__, reverse=True)


def selected_first_policy_order(
    scores: Sequence[float], selected_index: int
) -> List[int]:
    """Place the sampled action first and rank the tail by policy score."""
    ranked = policy_ranked_indices(scores)
    if selected_index < 0 or selected_index >= len(ranked):
        raise ValueError(
            f"Selected candidate index {selected_index} is outside [0, {len(ranked)})"
        )
    return [selected_index] + [index for index in ranked if index != selected_index]


def resolve_activity_order(
    ready_activities: Sequence[int],
    selected_index: int,
    action_order_indices: Optional[Sequence[int]] = None,
) -> List[int]:
    """Convert a candidate-index permutation into the solver's activity order."""
    chosen = ready_activities[selected_index]
    if action_order_indices is None:
        return [chosen] + [
            activity for activity in ready_activities if activity != chosen
        ]

    order_indices = [int(index) for index in action_order_indices]
    expected_indices = list(range(len(ready_activities)))
    if (
        len(order_indices) != len(ready_activities)
        or sorted(order_indices) != expected_indices
    ):
        raise ValueError(
            "action_order_indices must be a complete permutation of "
            f"candidate indices {expected_indices}, got {order_indices}"
        )
    if order_indices[0] != selected_index:
        raise ValueError(
            "action_order_indices must place selected_index first: "
            f"selected_index={selected_index}, order={order_indices}"
        )
    return [ready_activities[index] for index in order_indices]
