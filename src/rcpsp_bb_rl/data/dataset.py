import random
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence

from rcpsp_bb_rl.data.parsing import RCPSPInstance, load_instance


def list_instance_paths(root: Path | str, patterns: Sequence[str] = ("*.RCP",)) -> List[Path]:
    """Return all instance files under root matching given glob patterns."""
    root = Path(root)
    paths: List[Path] = []
    for pattern in patterns:
        paths.extend(sorted(root.glob(pattern)))
        paths.extend(sorted(p for p in root.rglob(pattern) if p.parent != root))
    # Remove duplicates while preserving order.
    seen = set()
    unique_paths = []
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        unique_paths.append(p)
    return unique_paths


class RCPSPDataset:
    """Lightweight dataset wrapper to iterate or sample RCPSP instances."""

    def __init__(self, root: Path | str, patterns: Sequence[str] = ("*.RCP",), seed: int | None = None) -> None:
        self.root = Path(root)
        self.patterns = patterns
        self.paths = list_instance_paths(self.root, patterns)
        if not self.paths:
            raise FileNotFoundError(f"No instances found under {self.root} with patterns {patterns}")
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.paths)

    def __iter__(self) -> Iterator[RCPSPInstance]:
        for path in self.paths:
            yield load_instance(path)

    def sample_path(self) -> Path:
        return self._rng.choice(self.paths)

    def sample_instance(self) -> RCPSPInstance:
        return load_instance(self.sample_path())

    def instances(self, shuffle: bool = False) -> Iterable[RCPSPInstance]:
        paths = list(self.paths)
        if shuffle:
            self._rng.shuffle(paths)
        for path in paths:
            yield load_instance(path)
