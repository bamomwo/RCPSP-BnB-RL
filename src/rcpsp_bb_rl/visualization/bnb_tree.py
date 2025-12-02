from pathlib import Path
from typing import Dict, Tuple

from graphviz import Digraph

from rcpsp_bb_rl.bnb.core import BBNode


STATUS_COLORS: Dict[str, str] = {
    "solution": "lightgreen",
    "expanded": "lightblue",
    "pruned": "lightcoral",
    "pending": "white",
}


def render_bnb_tree(
    nodes: Tuple[BBNode, ...] | list,
    edges,
    output_dir: Path | str = "reports/figures",
    name: str = "bnb_tree",
    fmt: str = "pdf",
) -> Path:
    """
    Render a Graphviz visualization of the branch-and-bound search tree.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dot = Digraph(comment="B&B Tree", format=fmt)
    dot.attr(rankdir="TB")

    for node in nodes:
        color = STATUS_COLORS.get(node.status, "white")
        label_lines = [
            f"id={node.node_id}",
            f"depth={node.depth}",
            f"lb={node.lower_bound}",
        ]
        if node.action:
            label_lines.append(node.action)
        if node.status == "solution":
            label_lines.append(f"makespan={max(e.finish for e in node.scheduled.values())}")
        label = "\\n".join(label_lines)
        dot.node(str(node.node_id), label=label, style="filled", fillcolor=color, shape="box")

    for parent_id, child_id in edges:
        dot.edge(str(parent_id), str(child_id))

    outpath = dot.render(str(output_dir / name), cleanup=True)
    return Path(outpath)
