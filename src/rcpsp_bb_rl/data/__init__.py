from rcpsp_bb_rl.data.dataset import RCPSPDataset, list_instance_paths
from rcpsp_bb_rl.data.parsing import Activity, RCPSPInstance, load_instance
try:
    from rcpsp_bb_rl.data.teacher import generate_trace, solve_optimal_schedule, write_trace
except ModuleNotFoundError:  # pragma: no cover - optional dependency (ortools)
    generate_trace = None
    solve_optimal_schedule = None
    write_trace = None

__all__ = [
    "Activity",
    "RCPSPInstance",
    "RCPSPDataset",
    "list_instance_paths",
    "load_instance",
]

if generate_trace is not None:
    __all__.extend(
        [
            "generate_trace",
            "solve_optimal_schedule",
            "write_trace",
        ]
    )
