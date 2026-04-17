"""Query graph representation."""

from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from ..base import QueryRepresentation


class QueryGraph(QueryRepresentation):
    """Graph representation of a SQL query.

    The graph is stored as a NetworkX :class:`~networkx.DiGraph`. Node and edge
    metadata are kept in parallel dictionaries so callers can access rich
    attributes without having to traverse the NetworkX graph.
    """

    def __init__(self, parsed_query: 'ParsedQuery'):
        super().__init__(parsed_query)
        self.graph: nx.DiGraph = nx.DiGraph()
        self.node_counter: int = 0
        self.node_attributes: Dict[int, Dict[str, Any]] = {}
        self.edge_attributes: Dict[Tuple[int, int], Dict[str, Any]] = {}

    def build(self) -> None:
        self._built = True

    def add_node(self, node_type: str, attributes: Dict[str, Any]) -> int:
        node_id = self.node_counter
        self.node_counter += 1

        payload = {'type': node_type, **attributes}
        self.graph.add_node(node_id, **payload)
        self.node_attributes[node_id] = payload
        return node_id

    def add_edge(
        self,
        source: int,
        target: int,
        edge_type: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {'type': edge_type, **(attributes or {})}
        self.graph.add_edge(source, target, **payload)
        self.edge_attributes[(source, target)] = payload

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'query_graph',
            'nodes': list(self.graph.nodes()),
            'edges': list(self.graph.edges()),
            'node_attributes': self.node_attributes,
            'edge_attributes': self.edge_attributes,
            'is_built': self._built,
        }

    def to_string(self) -> str:
        return (
            f"QueryGraph(nodes={self.graph.number_of_nodes()}, "
            f"edges={self.graph.number_of_edges()})"
        )

    def visualize(self, output_path: Optional[str] = None) -> Any:
        # Import matplotlib lazily so that the core library does not require a
        # display-capable backend when visualization is unused.
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue')

        if output_path:
            plt.savefig(output_path)
            return output_path
        return plt.gcf()

    def to_adjacency_matrix(self) -> np.ndarray:
        """Return the dense adjacency matrix, ordered by node id."""
        nodelist = sorted(self.graph.nodes())
        return nx.to_numpy_array(self.graph, nodelist=nodelist)

    def get_subgraph(self, nodes: List[int]) -> 'QueryGraph':
        """Return a new :class:`QueryGraph` containing only ``nodes``."""
        selected = set(nodes)
        subgraph_view = self.graph.subgraph(selected)

        sub = QueryGraph(self.parsed_query)
        sub.node_counter = self.node_counter
        for node in subgraph_view.nodes():
            attrs = self.node_attributes.get(node, {})
            sub.graph.add_node(node, **attrs)
            sub.node_attributes[node] = attrs
        for src, dst in subgraph_view.edges():
            attrs = self.edge_attributes.get((src, dst), {})
            sub.graph.add_edge(src, dst, **attrs)
            sub.edge_attributes[(src, dst)] = attrs
        sub._built = self._built
        return sub
