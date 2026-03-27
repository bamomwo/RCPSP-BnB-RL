from __future__ import annotations

from typing import List, Optional

from rcpsp_bb_rl.bnb.branching_order import NodeLike, ReadyOrderFn, order_by_activity_id

class SerialBranchingScheme:
    """
    Branch on all ready activities.
    """

    def choose_activities(
        self,
        node: NodeLike,
        incumbent: Optional[int],
        order_ready_fn: Optional[ReadyOrderFn],
    ) -> List[int]:
        if order_ready_fn is None:
            return order_by_activity_id(node, incumbent)
        return list(order_ready_fn(node, incumbent))


class ParallelBranchingScheme:
    """
    Branch on at most max_children ready activities.
    """

    def __init__(self, max_children: Optional[int] = None) -> None:
        self.max_children = max_children

    def choose_activities(
        self,
        node: NodeLike,
        incumbent: Optional[int],
        order_ready_fn: Optional[ReadyOrderFn],
    ) -> List[int]:
        if order_ready_fn is None:
            ready_order = order_by_activity_id(node, incumbent)
        else:
            ready_order = list(order_ready_fn(node, incumbent))

        if self.max_children is None:
            return ready_order
        return ready_order[: self.max_children]
