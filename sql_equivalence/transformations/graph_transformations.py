# sql_equivalence/transformations/graph_transformations.py
"""Graph transformation rules for query graphs."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx
import logging

from ..representations.graph.query_graph import QueryGraph

logger = logging.getLogger(__name__)

class GraphTransformation(ABC):
    """Abstract base class for graph transformations."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def is_applicable(self, graph: QueryGraph) -> bool:
        """Check if transformation can be applied to the graph."""
        pass
    
    @abstractmethod
    def apply(self, graph: QueryGraph) -> QueryGraph:
        """Apply transformation to the graph."""
        pass

class NodeMerging(GraphTransformation):
    """Merge equivalent nodes in the query graph."""
    
    def __init__(self):
        super().__init__(
            name="Node Merging",
            description="Merge nodes representing the same entity"
        )
    
    def is_applicable(self, graph: QueryGraph) -> bool:
        """Check if there are mergeable nodes."""
        # Look for nodes with same type and attributes
        nodes = list(graph.graph.nodes())
        
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if self._are_nodes_equivalent(graph, nodes[i], nodes[j]):
                    return True
        
        return False
    
    def apply(self, graph: QueryGraph) -> QueryGraph:
        """Merge equivalent nodes."""
        new_graph = QueryGraph(graph.parsed_query)
        new_graph.graph = graph.graph.copy()
        new_graph.node_attributes = graph.node_attributes.copy()
        new_graph.edge_attributes = graph.edge_attributes.copy()
        
        # Find and merge equivalent nodes
        merged = True
        while merged:
            merged = False
            nodes = list(new_graph.graph.nodes())
            
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    if self._are_nodes_equivalent(new_graph, nodes[i], nodes[j]):
                        self._merge_nodes(new_graph, nodes[i], nodes[j])
                        merged = True
                        break
                if merged:
                    break
        
        return new_graph
    
    def _are_nodes_equivalent(self, graph: QueryGraph, node1: int, node2: int) -> bool:
        """Check if two nodes are equivalent."""
        attrs1 = graph.node_attributes.get(node1, {})
        attrs2 = graph.node_attributes.get(node2, {})
        
        # Same type and key attributes
        return (attrs1.get('type') == attrs2.get('type') and
                attrs1.get('name') == attrs2.get('name'))
    
    def _merge_nodes(self, graph: QueryGraph, node1: int, node2: int) -> None:
        """Merge node2 into node1."""
        # Redirect all edges from node2 to node1
        for predecessor in graph.graph.predecessors(node2):
            if predecessor != node1:
                graph.graph.add_edge(predecessor, node1)
        
        for successor in graph.graph.successors(node2):
            if successor != node1:
                graph.graph.add_edge(node1, successor)
        
        # Remove node2
        graph.graph.remove_node(node2)
        if node2 in graph.node_attributes:
            del graph.node_attributes[node2]

class EdgeReduction(GraphTransformation):
    """Reduce redundant edges in the query graph."""
    
    def __init__(self):
        super().__init__(
            name="Edge Reduction",
            description="Remove redundant edges from the graph"
        )
    
    def is_applicable(self, graph: QueryGraph) -> bool:
        """Check if there are reducible edges."""
        # Look for transitive edges that can be removed
        for node in graph.graph.nodes():
            successors = list(graph.graph.successors(node))
            for succ in successors:
                # Check if there's an indirect path
                for other_succ in successors:
                    if other_succ != succ and nx.has_path(graph.graph, other_succ, succ):
                        return True
        
        return False
    
    def apply(self, graph: QueryGraph) -> QueryGraph:
        """Remove redundant edges."""
        new_graph = QueryGraph(graph.parsed_query)
        new_graph.graph = graph.graph.copy()
        new_graph.node_attributes = graph.node_attributes.copy()
        new_graph.edge_attributes = graph.edge_attributes.copy()
        
        # Remove transitive edges
        edges_to_remove = []
        
        for node in new_graph.graph.nodes():
            successors = list(new_graph.graph.successors(node))
            for succ in successors:
                # Check if there's an indirect path
                for other_succ in successors:
                    if other_succ != succ:
                        try:
                            path = nx.shortest_path(new_graph.graph, other_succ, succ)
                            if len(path) > 1:  # Indirect path exists
                                edges_to_remove.append((node, succ))
                                break
                        except nx.NetworkXNoPath:
                            continue
        
        for edge in edges_to_remove:
            if new_graph.graph.has_edge(*edge):
                new_graph.graph.remove_edge(*edge)
        
        return new_graph

class SubgraphExtraction(GraphTransformation):
    """Extract relevant subgraphs from the query graph."""
    
    def __init__(self, node_types: Optional[Set[str]] = None):
        super().__init__(
            name="Subgraph Extraction",
            description="Extract subgraph with specific node types"
        )
        self.node_types = node_types or {'table', 'join', 'filter'}
    
    def is_applicable(self, graph: QueryGraph) -> bool:
        """Check if subgraph extraction is applicable."""
        # Check if graph has nodes of specified types
        for node, attrs in graph.node_attributes.items():
            if attrs.get('type') in self.node_types:
                return True
        return False
    
    def apply(self, graph: QueryGraph) -> QueryGraph:
        """Extract subgraph with specified node types."""
        new_graph = QueryGraph(graph.parsed_query)
        
        # Find nodes to include
        nodes_to_include = []
        for node, attrs in graph.node_attributes.items():
            if attrs.get('type') in self.node_types:
                nodes_to_include.append(node)
        
        # Create subgraph
        subgraph = graph.graph.subgraph(nodes_to_include).copy()
        new_graph.graph = subgraph
        
        # Copy relevant attributes
        for node in nodes_to_include:
            if node in graph.node_attributes:
                new_graph.node_attributes[node] = graph.node_attributes[node].copy()
        
        for edge in subgraph.edges():
            if edge in graph.edge_attributes:
                new_graph.edge_attributes[edge] = graph.edge_attributes[edge].copy()
        
        return new_graph

class GraphNormalization(GraphTransformation):
    """Normalize query graph structure."""
    
    def __init__(self):
        super().__init__(
            name="Graph Normalization",
            description="Normalize graph to canonical form"
        )
    
    def is_applicable(self, graph: QueryGraph) -> bool:
        """Always applicable for normalization."""
        return True
    
    def apply(self, graph: QueryGraph) -> QueryGraph:
        """Normalize the graph structure."""
        new_graph = QueryGraph(graph.parsed_query)
        
        # Create canonical node ordering
        nodes = list(graph.graph.nodes())
        
        # Sort nodes by type, then by name/id
        def node_key(node):
            attrs = graph.node_attributes.get(node, {})
            return (attrs.get('type', ''), attrs.get('name', ''), node)
        
        nodes.sort(key=node_key)
        
        # Create mapping to new node IDs
        node_mapping = {old: new for new, old in enumerate(nodes)}
        
        # Create new graph with canonical node IDs
        new_graph.graph = nx.relabel_nodes(graph.graph, node_mapping)
        
        # Update attributes
        for old_node, new_node in node_mapping.items():
            if old_node in graph.node_attributes:
                new_graph.node_attributes[new_node] = graph.node_attributes[old_node].copy()
        
        for old_edge, attrs in graph.edge_attributes.items():
            new_edge = (node_mapping.get(old_edge[0]), node_mapping.get(old_edge[1]))
            if new_edge[0] is not None and new_edge[1] is not None:
                new_graph.edge_attributes[new_edge] = attrs.copy()
        
        return new_graph

class PathSimplification(GraphTransformation):
    """Simplify paths in the query graph."""
    
    def __init__(self):
        super().__init__(
            name="Path Simplification",
            description="Simplify complex paths in the graph"
        )
    
    def is_applicable(self, graph: QueryGraph) -> bool:
        """Check if there are simplifiable paths."""
        # Look for chains of single-degree nodes
        for node in graph.graph.nodes():
            if (graph.graph.in_degree(node) == 1 and 
                graph.graph.out_degree(node) == 1):
                return True
        return False
    
    def apply(self, graph: QueryGraph) -> QueryGraph:
        """Simplify paths by removing intermediate nodes."""
        new_graph = QueryGraph(graph.parsed_query)
        new_graph.graph = graph.graph.copy()
        new_graph.node_attributes = graph.node_attributes.copy()
        new_graph.edge_attributes = graph.edge_attributes.copy()
        
        # Find and remove intermediate nodes
        nodes_to_remove = []
        
        for node in new_graph.graph.nodes():
            attrs = new_graph.node_attributes.get(node, {})
            
            # Only remove certain types of intermediate nodes
            if (attrs.get('type') in ['intermediate', 'temp'] and
                new_graph.graph.in_degree(node) == 1 and 
                new_graph.graph.out_degree(node) == 1):
                
                predecessor = list(new_graph.graph.predecessors(node))[0]
                successor = list(new_graph.graph.successors(node))[0]
                
                # Connect predecessor directly to successor
                new_graph.graph.add_edge(predecessor, successor)
                nodes_to_remove.append(node)
        
        for node in nodes_to_remove:
            new_graph.graph.remove_node(node)
            if node in new_graph.node_attributes:
                del new_graph.node_attributes[node]
        
        return new_graph

class CycleDetection(GraphTransformation):
    """Detect and handle cycles in query graphs."""
    
    def __init__(self):
        super().__init__(
            name="Cycle Detection",
            description="Detect and mark cycles in the graph"
        )
    
    def is_applicable(self, graph: QueryGraph) -> bool:
        """Check if graph has cycles."""
        try:
            cycles = nx.find_cycle(graph.graph, orientation='original')
            return len(cycles) > 0
        except nx.NetworkXNoCycle:
            return False
    
    def apply(self, graph: QueryGraph) -> QueryGraph:
        """Mark cycles in the graph."""
        new_graph = QueryGraph(graph.parsed_query)
        new_graph.graph = graph.graph.copy()
        new_graph.node_attributes = graph.node_attributes.copy()
        new_graph.edge_attributes = graph.edge_attributes.copy()
        
        try:
            cycles = list(nx.simple_cycles(new_graph.graph))
            
            # Mark nodes and edges in cycles
            for cycle in cycles:
                for i, node in enumerate(cycle):
                    # Mark node as part of cycle
                    if node not in new_graph.node_attributes:
                        new_graph.node_attributes[node] = {}
                    new_graph.node_attributes[node]['in_cycle'] = True
                    
                    # Mark edge as part of cycle
                    next_node = cycle[(i + 1) % len(cycle)]
                    edge = (node, next_node)
                    if edge not in new_graph.edge_attributes:
                        new_graph.edge_attributes[edge] = {}
                    new_graph.edge_attributes[edge]['in_cycle'] = True
        
        except Exception as e:
            logger.warning(f"Error detecting cycles: {e}")
        
        return new_graph

def get_all_graph_transformations() -> List[GraphTransformation]:
    """Get all available graph transformations."""
    return [
        NodeMerging(),
        EdgeReduction(),
        SubgraphExtraction(),
        GraphNormalization(),
        PathSimplification(),
        CycleDetection(),
    ]

def apply_graph_transformations(graph: QueryGraph,
                              transformations: Optional[List[GraphTransformation]] = None,
                              max_iterations: int = 10) -> Tuple[QueryGraph, List[str]]:
    """
    Apply graph transformations to a query graph.
    
    Args:
        graph: Query graph to transform
        transformations: List of transformations to apply
        max_iterations: Maximum number of iterations
        
    Returns:
        Tuple of (transformed graph, list of applied transformations)
    """
    if transformations is None:
        transformations = get_all_graph_transformations()
    
    transformed = graph
    applied_transformations = []
    iterations = 0
    
    changed = True
    while changed and iterations < max_iterations:
        changed = False
        
        for transformation in transformations:
            if transformation.is_applicable(transformed):
                transformed = transformation.apply(transformed)
                applied_transformations.append(f"{transformation.name}: {transformation.description}")
                changed = True
                iterations += 1
                break
    
    return transformed, applied_transformations