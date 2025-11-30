from pathlib import Path
from typing import Dict

import networkx as nx
from graphviz import Digraph

from rcpsp_bb_rl.data.parsing import Activity


def build_precedence_graph(activities: Dict[int, Activity]) -> nx.DiGraph:
    """Build a DAG where edges encode activity precedence constraints."""
    graph = nx.DiGraph()

    for act_id, info in activities.items():
        graph.add_node(
            act_id,
            duration=info.duration,
            resources=info.resources,
        )

    for act_id, info in activities.items():
        for succ in info.successors:
            graph.add_edge(act_id, succ)

    return graph


def render_graphviz(
    graph: nx.DiGraph,
    output_dir: Path | str = "reports/figures",
    name: str = "rcpsp_graph",
    fmt: str = "pdf",
) -> Path:
    """
    Render the precedence graph with graphviz and write to disk.
    `fmt` can be pdf/png/svg.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dot = Digraph(comment="RCPSP", format=fmt)
    dot.attr(rankdir="LR")

    start_node = min(graph.nodes) if graph.nodes else None
    end_node = max(graph.nodes) if graph.nodes else None

    for node, data in graph.nodes(data=True):
        if node == start_node:
            fill = "lightgreen"
        elif node == end_node:
            fill = "lightcoral"
        else:
            fill = "lightblue"

        res_str = ",".join(map(str, data.get("resources", [])))
        label = f"{node}\\n(d={data.get('duration')})\\n(r={res_str})"
        dot.node(str(node), label=label, shape="box", style="filled", fillcolor=fill)

    for u, v in graph.edges:
        dot.edge(str(u), str(v))

    outpath = dot.render(str(output_dir / name), cleanup=True)
    return Path(outpath)
