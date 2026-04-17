"""Query graph representation."""

from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from ...parser.ast_builder import ASTNode
from ..base import QueryRepresentation


class QueryGraph(QueryRepresentation):
    """Graph representation of a SQL query.

    The graph is stored as a NetworkX :class:`~networkx.DiGraph`. Each AST
    node becomes a graph node carrying its ``type``, ``value``, and
    ``attributes``; edges reflect the AST parent-child relationship and are
    labeled by the parent's type (e.g. ``"select-child"``, ``"from-child"``).

    Node and edge metadata are kept in parallel dictionaries so callers can
    access rich attributes without traversing the NetworkX graph.
    """

    def __init__(self, parsed_query: 'ParsedQuery'):
        super().__init__(parsed_query)
        self.graph: nx.DiGraph = nx.DiGraph()
        self.node_counter: int = 0
        self.node_attributes: Dict[int, Dict[str, Any]] = {}
        self.edge_attributes: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.root_node: Optional[int] = None

    # ---------------------------------------------------------------- build
    def build(self) -> None:
        """Populate the graph from ``self.parsed_query.ast``."""
        ast = getattr(self.parsed_query, 'ast', None)
        if ast is not None:
            self.root_node = self._walk(ast)
        self._built = True

    def _walk(self, ast: ASTNode, parent_id: Optional[int] = None) -> int:
        """Recursively add ``ast`` and its subtree to the graph.

        Returns the id assigned to ``ast``.
        """
        node_id = self.add_node(
            node_type=ast.node_type.value,
            attributes={
                'value': ast.value,
                **ast.attributes,
            },
        )
        if parent_id is not None:
            self.add_edge(
                parent_id,
                node_id,
                edge_type=f"{self.node_attributes[parent_id]['type']}-child",
            )
        for child in ast.children:
            self._walk(child, node_id)
        return node_id

    # ----------------------------------------------------------------- CRUD
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

    # ---------------------------------------------------------- serializers
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'query_graph',
            'root': self.root_node,
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
        # Import matplotlib lazily so core usage does not require a display
        # backend.
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(self.graph)
        labels = {
            n: self.node_attributes.get(n, {}).get('type', str(n))
            for n in self.graph.nodes()
        }
        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            labels=labels,
            node_color='lightblue',
        )

        if output_path:
            plt.savefig(output_path)
            return output_path
        return plt.gcf()

    # ----------------------------------------------------------- conversions
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

    # -------------------------------------------------------- introspection
    def iter_nodes_of_type(self, node_type: str):
        """Yield node ids whose stored ``type`` equals ``node_type``."""
        for nid, attrs in self.node_attributes.items():
            if attrs.get('type') == node_type:
                yield nid

    def node_type_histogram(self) -> Dict[str, int]:
        """Return a histogram of node types across the graph."""
        histogram: Dict[str, int] = {}
        for attrs in self.node_attributes.values():
            t = attrs.get('type', 'unknown')
            histogram[t] = histogram.get(t, 0) + 1
        return histogram
