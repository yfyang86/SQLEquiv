"""Logical Query Tree (LQT) representation."""

from typing import Any, Dict, List, Optional

import networkx as nx

from ..base import QueryRepresentation


class LogicalQueryTree(QueryRepresentation):
    """Logical Query Tree representation.

    This representation is a tree-shaped view of the parsed query. Concrete
    building logic is provided by subclasses or plugins; the default
    implementation exposes a minimal NetworkX ``DiGraph`` scaffold.
    """

    def __init__(self, parsed_query: 'ParsedQuery'):
        super().__init__(parsed_query)
        self.tree: nx.DiGraph = nx.DiGraph()
        self.root_node: Optional[int] = None
        self.node_mapping: Dict[Any, int] = {}

    def build(self) -> None:
        self._built = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'logical_query_tree',
            'root': self.root_node,
            'nodes': list(self.tree.nodes()),
            'edges': list(self.tree.edges()),
            'is_built': self._built,
        }

    def to_string(self) -> str:
        return f"LogicalQueryTree(nodes={self.tree.number_of_nodes()})"

    def visualize(self, output_path: Optional[str] = None) -> Any:
        raise NotImplementedError(
            "LogicalQueryTree.visualize is not yet implemented."
        )

    def get_height(self) -> int:
        """Return the height of the tree, or 0 when the tree is empty."""
        if self.root_node is None or self.tree.number_of_nodes() == 0:
            return 0
        lengths = nx.single_source_shortest_path_length(self.tree, self.root_node)
        return max(lengths.values()) if lengths else 0

    def get_leaves(self) -> List[int]:
        """Return all leaf (out-degree 0) nodes of the tree."""
        return [node for node, degree in self.tree.out_degree() if degree == 0]
