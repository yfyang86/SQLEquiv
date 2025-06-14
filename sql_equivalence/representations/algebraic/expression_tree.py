# sql_equivalence/representations/algebraic/expression_tree.py
"""Expression tree for algebraic expressions."""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
import graphviz

from .operators import AlgebraicOperator

@dataclass
class ExpressionNode:
    """Node in an expression tree."""
    
    operator: AlgebraicOperator
    parent: Optional['ExpressionNode'] = None
    children: List['ExpressionNode'] = field(default_factory=list)
    node_id: int = field(default_factory=lambda: id(None))
    
    def __post_init__(self):
        self.node_id = id(self)
    
    def add_child(self, child: 'ExpressionNode') -> None:
        """Add a child node."""
        self.children.append(child)
        child.parent = self
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return len(self.children) == 0
    
    def is_root(self) -> bool:
        """Check if this is the root node."""
        return self.parent is None
    
    def get_depth(self) -> int:
        """Get the depth of this node in the tree."""
        depth = 0
        current = self
        while current.parent is not None:
            depth += 1
            current = current.parent
        return depth
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'node_id': self.node_id,
            'operator': self.operator.to_dict(),
            'children': [child.to_dict() for child in self.children]
        }

class ExpressionTree:
    """Tree representation of an algebraic expression."""
    
    def __init__(self, root_operator: Optional[AlgebraicOperator] = None):
        """
        Initialize expression tree.
        
        Args:
            root_operator: Root operator of the tree
        """
        self.root = None
        if root_operator:
            self.root = self._build_tree(root_operator)
    
    def _build_tree(self, operator: AlgebraicOperator, 
                   parent: Optional[ExpressionNode] = None) -> ExpressionNode:
        """Recursively build tree from operators."""
        node = ExpressionNode(operator=operator, parent=parent)
        
        # Recursively build children
        for child_op in operator.children:
            child_node = self._build_tree(child_op, node)
            node.add_child(child_node)
        
        return node
    
    def get_height(self) -> int:
        """Get the height of the tree."""
        if not self.root:
            return 0
        
        def _get_height(node: ExpressionNode) -> int:
            if node.is_leaf():
                return 1
            return 1 + max(_get_height(child) for child in node.children)
        
        return _get_height(self.root)
    
    def get_nodes(self) -> List[ExpressionNode]:
        """Get all nodes in the tree."""
        if not self.root:
            return []
        
        nodes = []
        
        def _collect_nodes(node: ExpressionNode):
            nodes.append(node)
            for child in node.children:
                _collect_nodes(child)
        
        _collect_nodes(self.root)
        return nodes
    
    def get_leaves(self) -> List[ExpressionNode]:
        """Get all leaf nodes."""
        return [node for node in self.get_nodes() if node.is_leaf()]
    
    def find_nodes_by_type(self, operator_type: str) -> List[ExpressionNode]:
        """Find all nodes with a specific operator type."""
        return [
            node for node in self.get_nodes() 
            if node.operator.operator_type.name == operator_type.upper()
        ]
    
    def to_string(self) -> str:
        """Convert tree to string representation."""
        if not self.root:
            return ""
        return self.root.operator.to_string()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tree to dictionary representation."""
        return {
            'root': self.root.to_dict() if self.root else None,
            'height': self.get_height(),
            'node_count': len(self.get_nodes()),
            'leaf_count': len(self.get_leaves())
        }
    
    def visualize(self, output_path: Optional[str] = None, 
                  format: str = 'png') -> Union[str, graphviz.Digraph]:
        """
        Visualize the expression tree.
        
        Args:
            output_path: Path to save the visualization
            format: Output format (png, pdf, svg, etc.)
            
        Returns:
            Path to saved file or graphviz object
        """
        dot = graphviz.Digraph(comment='Expression Tree')
        dot.attr(rankdir='TB')
        
        if not self.root:
            return dot
        
        # Add nodes
        def add_nodes(node: ExpressionNode):
            label = f"{node.operator.operator_type.value}\n{node.operator.to_string()[:50]}"
            shape = 'box' if node.is_leaf() else 'ellipse'
            dot.node(str(node.node_id), label=label, shape=shape)
            
            for child in node.children:
                add_nodes(child)
                dot.edge(str(node.node_id), str(child.node_id))
        
        add_nodes(self.root)
        
        if output_path:
            dot.render(output_path, format=format, cleanup=True)
            return f"{output_path}.{format}"
        
        return dot
    
    def clone(self) -> 'ExpressionTree':
        """Create a deep copy of the tree."""
        if not self.root:
            return ExpressionTree()
        
        cloned_root_op = self.root.operator.clone()
        return ExpressionTree(cloned_root_op)