from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List


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
    def ready(self) -> List[int]:
        return [int(activity) for activity in self.raw["ready"]]
