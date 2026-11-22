"""Compact evaluation records and readable training-console summaries."""
from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile


def _number(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Evaluation metrics must be finite numbers or null")
    return value


def summarize_evaluation(evaluation: dict, update: int) -> dict:
    """Attach the update to the five evaluation metrics, retaining their units."""
    if isinstance(update, bool) or not isinstance(update, int) or update < 0:
        raise ValueError("update must be a non-negative integer")
    summary = evaluation["summary"]
    return {
        "update": update,
        **{name: _number(summary.get(name))
           for name in ("solved", "primal_opt", "primal_gap", "dive1_gap", "nodes")},
    }


def format_evaluation(evaluation: dict, entry: dict) -> str:
    """Show proof and known-optimum matches with their distinct denominators."""
    rows = evaluation["instances"]
    known = [row for row in rows if row.get("primal_opt") is not None]

    def attainment(count, total):
        return f"{count}/{total} ({count / total:.0%})" if total else "n/a"

    def gap(value):
        return f"{value:.2f}%" if value is not None else "n/a"

    solved = attainment(sum(bool(row.get("solved")) for row in rows), len(rows))
    primal = attainment(sum(bool(row.get("primal_opt")) for row in known), len(known))
    nodes = f"{entry['nodes']:.0f}" if entry["nodes"] is not None else "n/a"
    return (
        f"[Eval {entry['update']}] solved={solved}  primal_opt={primal}  "
        f"primal_gap={gap(entry['primal_gap'])}  dive1_gap={gap(entry['dive1_gap'])}  nodes={nodes}"
    )


def save_evaluation(path: Path | str, entry: dict) -> None:
    """Append to a pretty JSON array, replacing the destination atomically."""
    path = Path(path)
    entries = []
    if path.exists():
        try:
            entries = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Evaluation log is not valid JSON: {path}") from exc
        if not isinstance(entries, list) or any(not isinstance(item, dict) for item in entries):
            raise ValueError(f"Evaluation log must contain a JSON array of records: {path}")
    entries.append(entry)
    payload = json.dumps(entries, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
