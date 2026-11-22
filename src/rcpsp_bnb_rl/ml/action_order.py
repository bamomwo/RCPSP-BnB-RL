from __future__ import annotations

from typing import List, Sequence


def policy_ranked_indices(scores: Sequence[float]) -> List[int]:
    """Return candidate indices in descending-score order with stable ties."""
    return sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
