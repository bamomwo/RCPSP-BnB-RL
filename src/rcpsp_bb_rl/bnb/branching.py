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
