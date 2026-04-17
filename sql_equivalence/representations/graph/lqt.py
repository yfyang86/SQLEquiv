"""Logical Query Tree (LQT) representation."""

from typing import Any, Dict, List, Optional

import networkx as nx

from ...parser.ast_builder import ASTNode
from ..base import QueryRepresentation


class LogicalQueryTree(QueryRepresentation):
    """Logical Query Tree representation.

    A rooted, tree-shaped view of the parsed query. Unlike :class:`QueryGraph`,
    an LQT is guaranteed to be acyclic and rooted, which lets equivalence
    checks fall back to efficient tree-edit distance algorithms.
    """

    def __init__(self, parsed_query: 'ParsedQuery'):
        super().__init__(parsed_query)
        self.tree: nx.DiGraph = nx.DiGraph()
        self.root_node: Optional[int] = None
        self.node_mapping: Dict[int, Dict[str, Any]] = {}
        self._node_counter = 0

    # ---------------------------------------------------------------- build
    def build(self) -> None:
        ast = getattr(self.parsed_query, 'ast', None)
        if ast is not None:
            self.root_node = self._walk(ast)
        self._built = True

    def _walk(self, ast: ASTNode, parent_id: Optional[int] = None) -> int:
        node_id = self._node_counter
        self._node_counter += 1

        payload = {
            'type': ast.node_type.value,
            'value': ast.value,
            **ast.attributes,
        }
        self.tree.add_node(node_id, **payload)
        self.node_mapping[node_id] = payload

        if parent_id is not None:
            self.tree.add_edge(parent_id, node_id)

        for child in ast.children:
            self._walk(child, node_id)
        return node_id

    # ---------------------------------------------------------- serialization
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'logical_query_tree',
            'root': self.root_node,
            'nodes': list(self.tree.nodes()),
            'edges': list(self.tree.edges()),
            'node_attributes': self.node_mapping,
            'is_built': self._built,
        }

    def to_string(self) -> str:
        return f"LogicalQueryTree(nodes={self.tree.number_of_nodes()})"

    def visualize(self, output_path: Optional[str] = None) -> Any:
        import matplotlib.pyplot as plt  # lazy

        plt.figure(figsize=(10, 8))
        pos = (
            nx.drawing.nx_pydot.pydot_layout(self.tree, prog='dot')
            if self.tree.number_of_nodes()
            else {}
        )
        labels = {n: attrs.get('type', str(n)) for n, attrs in self.node_mapping.items()}
        nx.draw(self.tree, pos, with_labels=True, labels=labels, node_color='lightgreen')
        if output_path:
            plt.savefig(output_path)
            return output_path
        return plt.gcf()

    # ------------------------------------------------------------- analytics
    def get_height(self) -> int:
        """Return the height of the tree, or 0 when the tree is empty."""
        if self.root_node is None or self.tree.number_of_nodes() == 0:
            return 0
        lengths = nx.single_source_shortest_path_length(self.tree, self.root_node)
        return max(lengths.values()) if lengths else 0

    def get_leaves(self) -> List[int]:
        """Return all leaf (out-degree 0) nodes of the tree."""
        return [node for node, degree in self.tree.out_degree() if degree == 0]

    def iter_subtree(self, node: Optional[int] = None):
        """Yield node ids in pre-order starting from ``node`` (or the root)."""
        start = self.root_node if node is None else node
        if start is None:
            return
        yield start
        for child in self.tree.successors(start):
            yield from self.iter_subtree(child)
