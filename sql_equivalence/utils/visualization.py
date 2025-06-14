# sql_equivalence/utils/visualization.py
"""Visualization utilities for query analysis."""

import os
from typing import Any, Dict, List, Optional, Union, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import graphviz
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)

def visualize_query_graph(query_graph: 'QueryGraph', 
                         output_path: Optional[str] = None,
                         layout: str = 'hierarchical',
                         show_attributes: bool = True,
                         interactive: bool = False) -> Union[str, Any]:
    """
    Visualize a query graph.
    
    Args:
        query_graph: QueryGraph object
        output_path: Path to save visualization
        layout: Layout algorithm ('hierarchical', 'spring', 'circular')
        show_attributes: Whether to show node/edge attributes
        interactive: Whether to create interactive visualization
        
    Returns:
        Path to saved file or visualization object
    """
    if interactive:
        return create_interactive_graph(
            query_graph.graph,
            query_graph.node_attributes,
            query_graph.edge_attributes,
            output_path
        )
    
    # Use matplotlib for static visualization
    plt.figure(figsize=(12, 8))
    
    # Determine layout
    if layout == 'hierarchical':
        pos = nx.nx_agraph.graphviz_layout(query_graph.graph, prog='dot')
    elif layout == 'spring':
        pos = nx.spring_layout(query_graph.graph, k=2, iterations=50)
    elif layout == 'circular':
        pos = nx.circular_layout(query_graph.graph)
    else:
        pos = nx.spring_layout(query_graph.graph)
    
    # Draw nodes
    node_colors = []
    for node in query_graph.graph.nodes():
        node_type = query_graph.node_attributes.get(node, {}).get('type', 'default')
        color_map = {
            'table': 'lightblue',
            'column': 'lightgreen',
            'operator': 'orange',
            'function': 'pink',
            'default': 'gray'
        }
        node_colors.append(color_map.get(node_type, 'gray'))
    
    nx.draw_networkx_nodes(
        query_graph.graph, pos,
        node_color=node_colors,
        node_size=1500,
        alpha=0.9
    )
    
    # Draw edges
    nx.draw_networkx_edges(
        query_graph.graph, pos,
        edge_color='gray',
        arrows=True,
        arrowsize=20,
        alpha=0.6
    )
    
    # Draw labels
    if show_attributes:
        labels = {}
        for node in query_graph.graph.nodes():
            attrs = query_graph.node_attributes.get(node, {})
            label = f"{node}\n{attrs.get('type', '')}"
            if 'value' in attrs:
                label += f"\n{attrs['value']}"
            labels[node] = label
    else:
        labels = {node: str(node) for node in query_graph.graph.nodes()}
    
    nx.draw_networkx_labels(
        query_graph.graph, pos,
        labels,
        font_size=8
    )
    
    plt.title("Query Graph Visualization")
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    
    return plt.gcf()

def visualize_expression_tree(expression_tree: 'ExpressionTree',
                            output_path: Optional[str] = None,
                            format: str = 'png',
                            show_details: bool = True) -> Union[str, Any]:
    """
    Visualize an algebraic expression tree.
    
    Args:
        expression_tree: ExpressionTree object
        output_path: Path to save visualization
        format: Output format
        show_details: Whether to show operator details
        
    Returns:
        Path to saved file or graphviz object
    """
    dot = graphviz.Digraph(comment='Expression Tree')
    dot.attr(rankdir='TB')
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
    
    def add_node(node: 'ExpressionNode', parent_id: Optional[str] = None):
        node_id = str(node.node_id)
        
        # Create label
        op = node.operator
        label = op.operator_type.value
        
        if show_details:
            if hasattr(op, 'columns') and op.columns:
                label += f"\n{', '.join(str(c) for c in op.columns[:3])}"
                if len(op.columns) > 3:
                    label += "..."
            elif hasattr(op, 'condition') and op.condition:
                label += f"\n{str(op.condition)[:30]}..."
            elif hasattr(op, 'table_name'):
                label += f"\n{op.table_name}"
        
        # Determine color based on operator type
        color_map = {
            'PROJECT': 'lightblue',
            'SELECT': 'lightgreen',
            'JOIN': 'orange',
            'UNION': 'pink',
            'AGGREGATE': 'lightyellow',
            'RELATION': 'lightgray'
        }
        color = color_map.get(op.operator_type.name, 'white')
        
        dot.node(node_id, label=label, fillcolor=color)
        
        if parent_id:
            dot.edge(parent_id, node_id)
        
        # Recursively add children
        for child in node.children:
            add_node(child, node_id)
    
    if expression_tree.root:
        add_node(expression_tree.root)
    
    if output_path:
        dot.render(output_path, format=format, cleanup=True)
        return f"{output_path}.{format}"
    
    return dot

def visualize_algebraic_expression(algebraic_expr: 'AlgebraicExpression',
                                 output_path: Optional[str] = None,
                                 format: str = 'png') -> Union[str, Any]:
    """
    Visualize an algebraic expression.
    
    Args:
        algebraic_expr: AlgebraicExpression object
        output_path: Path to save visualization
        format: Output format
        
    Returns:
        Path to saved file or visualization object
    """
    if algebraic_expr.expression_tree:
        return visualize_expression_tree(
            algebraic_expr.expression_tree,
            output_path,
            format
        )
    
    # Fallback: create simple text representation
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, algebraic_expr.to_string(),
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes,
            fontsize=12,
            fontfamily='monospace')
    ax.axis('off')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    
    return fig

def create_query_comparison_plot(results: List[Dict[str, Any]],
                               output_path: Optional[str] = None) -> Union[str, Any]:
    """
    Create a comparison plot for multiple query analysis results.
    
    Args:
        results: List of analysis results
        output_path: Path to save visualization
        
    Returns:
        Path to saved file or figure object
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Equivalence Results', 'Confidence Scores',
                       'Execution Times', 'Method Performance'),
        specs=[[{'type': 'bar'}, {'type': 'scatter'}],
               [{'type': 'bar'}, {'type': 'heatmap'}]]
    )
    
    # Extract data
    methods = []
    is_equivalent = []
    confidences = []
    exec_times = []
    
    for result in results:
        for method, method_result in result.get('method_results', {}).items():
            methods.append(method)
            is_equivalent.append(1 if method_result.get('is_equivalent', False) else 0)
            confidences.append(method_result.get('confidence', 0))
            exec_times.append(result.get('execution_time', 0))
    
    # Equivalence results bar chart
    fig.add_trace(
        go.Bar(x=methods, y=is_equivalent, name='Equivalent'),
        row=1, col=1
    )
    
    # Confidence scores scatter plot
    fig.add_trace(
        go.Scatter(x=methods, y=confidences, mode='markers', 
                  marker=dict(size=10), name='Confidence'),
        row=1, col=2
    )
    
    # Execution times bar chart
    fig.add_trace(
        go.Bar(x=methods, y=exec_times, name='Exec Time (s)'),
        row=2, col=1
    )
    
    # Method performance heatmap
    # Create a simple performance matrix
    perf_matrix = [[is_equivalent[i], confidences[i], 1/exec_times[i] if exec_times[i] > 0 else 0]
                   for i in range(len(methods))]
    
    fig.add_trace(
        go.Heatmap(z=perf_matrix, x=['Equivalent', 'Confidence', 'Speed'],
                  y=methods, colorscale='Viridis'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False,
                     title_text="Query Equivalence Analysis Results")
    
    if output_path:
        fig.write_html(output_path)
        return output_path
    
    return fig

def save_visualization(figure: Any, output_path: str, 
                      format: str = 'png', dpi: int = 300) -> str:
    """
    Save a visualization to file.
    
    Args:
        figure: Figure object
        output_path: Output path
        format: Output format
        dpi: DPI for raster formats
        
    Returns:
        Path to saved file
    """
    if hasattr(figure, 'savefig'):
        # Matplotlib figure
        figure.savefig(output_path, format=format, dpi=dpi, bbox_inches='tight')
    elif hasattr(figure, 'write_image'):
        # Plotly figure
        figure.write_image(output_path, format=format)
    elif hasattr(figure, 'render'):
        # Graphviz figure
        figure.render(output_path, format=format, cleanup=True)
        output_path = f"{output_path}.{format}"
    else:
        raise ValueError(f"Unknown figure type: {type(figure)}")
    
    return output_path

def create_interactive_graph(graph: nx.Graph,
                           node_attributes: Dict[int, Dict[str, Any]],
                           edge_attributes: Dict[Tuple[int, int], Dict[str, Any]],
                           output_path: Optional[str] = None) -> Union[str, Any]:
    """
    Create an interactive graph visualization using Plotly.
    
    Args:
        graph: NetworkX graph
        node_attributes: Node attribute dictionary
        edge_attributes: Edge attribute dictionary
        output_path: Path to save HTML file
        
    Returns:
        Path to saved file or plotly figure
    """
    # Get node positions
    pos = nx.spring_layout(graph, k=2, iterations=50)
    
    # Create edge traces
    edge_traces = []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=2, color='gray'),
            hoverinfo='none',
            showlegend=False
        )
        edge_traces.append(edge_trace)
    
    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Create hover text
        attrs = node_attributes.get(node, {})
        text = f"Node: {node}<br>"
        text += f"Type: {attrs.get('type', 'unknown')}<br>"
        for key, value in attrs.items():
            if key != 'type':
                text += f"{key}: {value}<br>"
        node_text.append(text)
        
        # Determine color
        node_type = attrs.get('type', 'default')
        color_map = {
            'table': 'blue',
            'column': 'green',
            'operator': 'orange',
            'function': 'red',
            'default': 'gray'
        }
        node_colors.append(color_map.get(node_type, 'gray'))
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=[str(node) for node in graph.nodes()],
        textposition="top center",
        hovertext=node_text,
        hoverinfo='text',
        marker=dict(
            size=20,
            color=node_colors,
            line=dict(width=2, color='white')
        )
    )
    
    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])
    
    fig.update_layout(
        title="Interactive Query Graph",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    if output_path:
        fig.write_html(output_path)
        return output_path
    
    return fig

def create_similarity_heatmap(similarity_matrix: Any,
                            labels: List[str],
                            output_path: Optional[str] = None) -> Union[str, Any]:
    """
    Create a heatmap visualization of similarity matrix.
    
    Args:
        similarity_matrix: 2D similarity matrix
        labels: Labels for rows/columns
        output_path: Path to save visualization
        
    Returns:
        Path to saved file or figure object
    """
    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=labels,
        y=labels,
        colorscale='Viridis',
        text=similarity_matrix,
        texttemplate='%{text:.2f}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title="Query Similarity Heatmap",
        xaxis_title="Query",
        yaxis_title="Query",
        width=800,
        height=800
    )
    
    if output_path:
        fig.write_html(output_path)
        return output_path
    
    return fig