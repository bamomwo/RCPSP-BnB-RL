from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class Activity:
    duration: int
    resources: List[int]
    successors: List[int]


@dataclass
class RCPSPInstance:
    num_activities: int
    num_resources: int
    resource_caps: List[int]
    activities: Dict[int, Activity]


def load_instance(path: Path | str) -> RCPSPInstance:
    """Parse an RCPSP instance file into a structured object."""
    path = Path(path)

    with path.open() as f:
        num_acts, num_res = map(int, f.readline().split())
        resource_caps = list(map(int, f.readline().split()))

        if len(resource_caps) != num_res:
            raise ValueError(
                f"Expected {num_res} resource capacities, found {len(resource_caps)}"
            )

        activities: Dict[int, Activity] = {}

        for act_id in range(1, num_acts + 1):
            tokens = f.readline().split()
            if not tokens:
                raise ValueError(f"Missing data for activity {act_id} in {path}")

            duration = int(tokens[0])
            resources = list(map(int, tokens[1 : 1 + num_res]))
            num_succ = int(tokens[1 + num_res])
            succs = list(map(int, tokens[2 + num_res : 2 + num_res + num_succ]))

            activities[act_id] = Activity(
                duration=duration,
                resources=resources,
                successors=succs,
            )

    return RCPSPInstance(
        num_activities=num_acts,
        num_resources=num_res,
        resource_caps=resource_caps,
        activities=activities,
    )
