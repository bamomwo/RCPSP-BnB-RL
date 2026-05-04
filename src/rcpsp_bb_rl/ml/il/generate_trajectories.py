from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence


def _load_jsonl(path: Path) -> Iterator[Dict]:
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


@dataclass
class TrajectoryRecord:
    """Single logged state/action from a teacher trajectory."""

    raw: Dict
    source: Path

    @property
    def action_task(self) -> int:
        return int(self.raw["action"]["task"])

    @property
    def action_start(self) -> int:
        return int(self.raw["action"]["start"])

    @property
    def ready(self) -> List[int]:
        return [int(r) for r in self.raw["ready"]]

    @property
    def earliest_start(self) -> Dict[int, Optional[int]]:
        # Keys were stringified when logging; preserve None if present.
        es_map = self.raw.get("earliest_start", {})
        return {int(k): (None if v is None else int(v)) for k, v in es_map.items()}


@dataclass
class CandidateExample:
    """
    Flattened example for supervised learning over the ready set.

    label == 1 for the expert-chosen task, 0 otherwise.
    """

    record: TrajectoryRecord
    candidate: int
    label: int
    candidate_earliest_start: Optional[int]


class TrajectoryDataset:
    """
    Lightweight loader for teacher trajectories.

    If flatten_ready=True, __iter__/__getitem__ yield CandidateExample entries
    (one per ready task) to make supervised imitation easy. Otherwise they
    yield TrajectoryRecord instances.
    """

    def __init__(
        self,
        paths: Sequence[Path | str],
        flatten_ready: bool = True,
    ) -> None:
        self.paths = [Path(p) for p in paths]
        self.flatten_ready = flatten_ready

        self._records: List[TrajectoryRecord] = []
        for path in self.paths:
            for raw in _load_jsonl(path):
                self._records.append(TrajectoryRecord(raw=raw, source=path))

        if flatten_ready:
            self._examples: List[CandidateExample] = []
            for rec in self._records:
                es_map = rec.earliest_start
                for task in rec.ready:
                    self._examples.append(
                        CandidateExample(
                            record=rec,
                            candidate=task,
                            label=1 if task == rec.action_task else 0,
                            candidate_earliest_start=es_map.get(task),
                        )
                    )

    def __len__(self) -> int:
        if self.flatten_ready:
            return len(self._examples)
        return len(self._records)

    def __iter__(self) -> Iterator[CandidateExample | TrajectoryRecord]:
        if self.flatten_ready:
            yield from self._examples
        else:
            yield from self._records

    def __getitem__(self, idx: int) -> CandidateExample | TrajectoryRecord:
        if self.flatten_ready:
            return self._examples[idx]
        return self._records[idx]


def load_trajectories(
    root: Path | str,
    pattern: str = "*.jsonl",
    flatten_ready: bool = True,
) -> TrajectoryDataset:
    """
    Convenience helper to load all trajectory files under a directory.

    Example:
        traj_ds = load_trajectories("data/trajectories")
        first = traj_ds[0]  # CandidateExample by default
    """
    root = Path(root)
    paths = sorted(root.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No trajectory files found under {root} matching {pattern}")
    return TrajectoryDataset(paths=paths, flatten_ready=flatten_ready)
