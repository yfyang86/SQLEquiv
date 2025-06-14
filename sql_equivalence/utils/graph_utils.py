# sql_equivalence/utils/graph_utils.py
"""Graph algorithm utilities."""

import hashlib
import json
from typing import Dict, List, Set, Tuple, Optional, Any, Union
import networkx as nx
import numpy as np
from scipy import sparse
import logging

logger = logging.getLogger(__name__)

def graph_to_adjacency_matrix(graph: nx.Graph, 
                            nodelist: Optional[List] = None) -> np.ndarray:
    """
    Convert NetworkX graph to adjacency matrix.
    
    Args:
        graph: NetworkX graph
        nodelist: Order of nodes (optional)
        
    Returns:
        Adjacency matrix as numpy array
    """
    if nodelist is None:
        nodelist = list(graph.nodes())
    
    n = len(nodelist)
    adj_matrix = np.zeros((n, n))
    
    node_to_idx = {node: i for i, node in enumerate(nodelist)}
    
    for u, v in graph.edges():
        if u in node_to_idx and v in node_to_idx:
            i, j = node_to_idx[u], node_to_idx[v]
            adj_matrix[i, j] = 1
            if not graph.is_directed():
                adj_matrix[j, i] = 1
    
    return adj_matrix

def adjacency_matrix_to_graph(adj_matrix: np.ndarray, 
                            directed: bool = True) -> nx.Graph:
    """
    Convert adjacency matrix to NetworkX graph.
    
    Args:
        adj_matrix: Adjacency matrix
        directed: Whether to create directed graph
        
    Returns:
        NetworkX graph
    """
    graph_class = nx.DiGraph if directed else nx.Graph
    graph = graph_class()
    
    n = adj_matrix.shape[0]
    graph.add_nodes_from(range(n))
    
    # Add edges
    edges = np.where(adj_matrix > 0)
    for i, j in zip(edges[0], edges[1]):
        if directed or i <= j:  # Avoid duplicate edges in undirected graphs
            graph.add_edge(i, j, weight=adj_matrix[i, j])
    
    return graph

def compute_graph_hash(graph: nx.Graph) -> str:
    """
    Compute a hash for a graph based on its structure.
    
    Args:
        graph: NetworkX graph
        
    Returns:
        Hash string
    """
    # Create a canonical representation
    # Sort nodes and edges for consistency
    nodes = sorted(graph.nodes())
    edges = sorted(graph.edges())
    
    # Include node and edge attributes
    node_data = []
    for node in nodes:
        attrs = graph.nodes[node]
        node_data.append((node, sorted(attrs.items())))
    
    edge_data = []
    for u, v in edges:
        attrs = graph.edges[u, v]
        edge_data.append(((u, v), sorted(attrs.items())))
    
    # Create hash
    data = {
        'nodes': node_data,
        'edges': edge_data,
        'directed': graph.is_directed()
    }
    
    data_str = json.dumps(data, sort_keys=True)
    return hashlib.sha256(data_str.encode()).hexdigest()

def find_common_subgraph(graph1: nx.Graph, graph2: nx.Graph) -> nx.Graph:
    """
    Find the maximum common subgraph between two graphs.
    
    Args:
        graph1: First graph
        graph2: Second graph
        
    Returns:
        Maximum common subgraph
    """
    # This is a simplified implementation
    # Full implementation would use algorithms like McGregor's algorithm
    
    common = nx.Graph()
    
    # Find common nodes (based on labels if available)
    common_nodes = set(graph1.nodes()) & set(graph2.nodes())
    common.add_nodes_from(common_nodes)
    
    # Find common edges
    for u, v in graph1.edges():
        if u in common_nodes and v in common_nodes and graph2.has_edge(u, v):
            common.add_edge(u, v)
    
    return common

def compute_graph_statistics(graph: nx.Graph) -> Dict[str, Any]:
    """
    Compute various statistics for a graph.
    
    Args:
        graph: NetworkX graph
        
    Returns:
        Dictionary of statistics
    """
    stats = {}
    
    # Basic statistics
    stats['num_nodes'] = graph.number_of_nodes()
    stats['num_edges'] = graph.number_of_edges()
    stats['density'] = nx.density(graph)
    
    # Degree statistics
    degrees = [d for n, d in graph.degree()]
    stats['avg_degree'] = np.mean(degrees) if degrees else 0
    stats['max_degree'] = max(degrees) if degrees else 0
    stats['min_degree'] = min(degrees) if degrees else 0
    
    # Connectivity
    if graph.is_directed():
        stats['is_weakly_connected'] = nx.is_weakly_connected(graph)
        stats['is_strongly_connected'] = nx.is_strongly_connected(graph)
        stats['num_sccs'] = nx.number_strongly_connected_components(graph)
    else:
        stats['is_connected'] = nx.is_connected(graph)
        stats['num_components'] = nx.number_connected_components(graph)
    
    # Centrality measures (for small graphs)
    if stats['num_nodes'] < 1000:
        try:
            stats['avg_betweenness'] = np.mean(list(nx.betweenness_centrality(graph).values()))
            stats['avg_closeness'] = np.mean(list(nx.closeness_centrality(graph).values()))
        except:
            pass
    
    # Clustering
    if not graph.is_directed():
        stats['avg_clustering'] = nx.average_clustering(graph)
    
    return stats

def is_dag(graph: nx.DiGraph) -> bool:
    """
    Check if a directed graph is a DAG (Directed Acyclic Graph).
    
    Args:
        graph: Directed graph
        
    Returns:
        True if graph is a DAG
    """
    return nx.is_directed_acyclic_graph(graph)

def topological_sort_graph(graph: nx.DiGraph) -> List:
    """
    Perform topological sort on a DAG.
    
    Args:
        graph: Directed acyclic graph
        
    Returns:
        List of nodes in topological order
    """
    if not is_dag(graph):
        raise ValueError("Graph must be a DAG for topological sorting")
    
    return list(nx.topological_sort(graph))

def find_strongly_connected_components(graph: nx.DiGraph) -> List[Set]:
    """
    Find strongly connected components in a directed graph.
    
    Args:
        graph: Directed graph
        
    Returns:
        List of sets, each containing nodes in a strongly connected component
    """
    return [set(scc) for scc in nx.strongly_connected_components(graph)]

def compute_pagerank(graph: nx.Graph, alpha: float = 0.85, 
                    max_iter: int = 100) -> Dict[Any, float]:
    """
    Compute PageRank scores for nodes.
    
    Args:
        graph: NetworkX graph
        alpha: Damping parameter
        max_iter: Maximum iterations
        
    Returns:
        Dictionary mapping nodes to PageRank scores
    """
    return nx.pagerank(graph, alpha=alpha, max_iter=max_iter)

def graph_edit_distance(graph1: nx.Graph, graph2: nx.Graph, 
                       node_match: Optional[callable] = None,
                       edge_match: Optional[callable] = None) -> int:
    """
    Compute graph edit distance between two graphs.
    
    Args:
        graph1: First graph
        graph2: Second graph
        node_match: Function to determine node equivalence
        edge_match: Function to determine edge equivalence
        
    Returns:
        Edit distance
    """
    # This is a simplified implementation
    # Full implementation would use more sophisticated algorithms
    
    # Count operations needed
    distance = 0
    
    # Node additions/deletions
    nodes1 = set(graph1.nodes())
    nodes2 = set(graph2.nodes())
    distance += len(nodes1 - nodes2)  # Deletions
    distance += len(nodes2 - nodes1)  # Additions
    
    # Edge additions/deletions (for common nodes)
    common_nodes = nodes1 & nodes2
    for u in common_nodes:
        for v in common_nodes:
            if graph1.has_edge(u, v) and not graph2.has_edge(u, v):
                distance += 1  # Edge deletion
            elif not graph1.has_edge(u, v) and graph2.has_edge(u, v):
                distance += 1  # Edge addition
    
    return distance

def find_graph_isomorphism(graph1: nx.Graph, graph2: nx.Graph,
                          node_match: Optional[callable] = None) -> Optional[Dict]:
    """
    Find isomorphism mapping between two graphs.
    
    Args:
        graph1: First graph
        graph2: Second graph
        node_match: Function to determine node equivalence
        
    Returns:
        Dictionary mapping nodes from graph1 to graph2, or None if not isomorphic
    """
    GM = nx.isomorphism.GraphMatcher(graph1, graph2, node_match=node_match)
    
    if GM.is_isomorphic():
        return GM.mapping
    
    return None

def compute_graph_similarity_matrix(graphs: List[nx.Graph], 
                                  method: str = 'edit_distance') -> np.ndarray:
    """
    Compute pairwise similarity matrix for a list of graphs.
    
    Args:
        graphs: List of graphs
        method: Similarity method ('edit_distance', 'common_nodes', 'spectral')
        
    Returns:
        Similarity matrix
    """
    n = len(graphs)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                if method == 'edit_distance':
                    distance = graph_edit_distance(graphs[i], graphs[j])
                    max_size = max(graphs[i].number_of_nodes() + graphs[i].number_of_edges(),
                                 graphs[j].number_of_nodes() + graphs[j].number_of_edges())
                    similarity = 1.0 - (distance / max_size) if max_size > 0 else 0.0
                
                elif method == 'common_nodes':
                    nodes1 = set(graphs[i].nodes())
                    nodes2 = set(graphs[j].nodes())
                    if nodes1 or nodes2:
                        similarity = len(nodes1 & nodes2) / len(nodes1 | nodes2)
                    else:
                        similarity = 0.0
                
                elif method == 'spectral':
                    # Spectral similarity based on eigenvalues
                    try:
                        spec1 = nx.laplacian_spectrum(graphs[i])
                        spec2 = nx.laplacian_spectrum(graphs[j])
                        # Pad shorter spectrum with zeros
                        max_len = max(len(spec1), len(spec2))
                        spec1 = np.pad(spec1, (0, max_len - len(spec1)))
                        spec2 = np.pad(spec2, (0, max_len - len(spec2)))
                        similarity = 1.0 - np.linalg.norm(spec1 - spec2) / np.linalg.norm(spec1 + spec2)
                    except:
                        similarity = 0.0
                
                else:
                    similarity = 0.0
                
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
    
    return similarity_matrix