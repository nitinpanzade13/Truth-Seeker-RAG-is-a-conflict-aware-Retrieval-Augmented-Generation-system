import networkx as nx
import plotly.graph_objects as go
import numpy as np


def build_conflict_graph_nx(conflict_matrix, doc_labels=None, threshold=0.15):
    """
    Build a NetworkX graph from the conflict matrix.
    Edges are added between documents whose conflict score exceeds the threshold.
    """
    n = len(conflict_matrix)
    if doc_labels is None:
        doc_labels = [f"Doc {i+1}" for i in range(n)]

    G = nx.Graph()
    for i in range(n):
        G.add_node(i, label=doc_labels[i])

    for i in range(n):
        for j in range(i + 1, n):
            weight = conflict_matrix[i][j]
            if weight > threshold:
                G.add_edge(i, j, weight=round(weight, 4))

    return G


def plot_conflict_graph_interactive(conflict_matrix, doc_labels=None, threshold=0.15):
    """
    Create an interactive Plotly figure of the conflict graph.
    Node size reflects total conflict penalty; edge width reflects pairwise contradiction.
    """
    G = build_conflict_graph_nx(conflict_matrix, doc_labels, threshold)
    n = len(conflict_matrix)
    if doc_labels is None:
        doc_labels = [f"Doc {i+1}" for i in range(n)]

    pos = nx.spring_layout(G, seed=42, k=2.0)

    # Edges
    edge_x, edge_y = [], []
    edge_widths = []
    edge_texts = []

    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_widths.append(data["weight"])
        edge_texts.append(f"{doc_labels[u]} ↔ {doc_labels[v]}: {data['weight']:.3f}")

    max_w = max(edge_widths) if edge_widths else 1
    normalized_widths = [1 + 6 * (w / max_w) for w in edge_widths]

    # Create edge traces (one per edge for varying widths)
    edge_traces = []
    for idx, (u, v, data) in enumerate(G.edges(data=True)):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        width = normalized_widths[idx]

        # Color from green (low conflict) to red (high conflict)
        ratio = data["weight"] / max_w if max_w > 0 else 0
        r = int(255 * ratio)
        g = int(255 * (1 - ratio))
        color = f"rgb({r},{g},50)"

        edge_traces.append(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode="lines",
            line=dict(width=width, color=color),
            hoverinfo="text",
            text=edge_texts[idx],
            showlegend=False,
        ))

    # Nodes
    node_x = [pos[i][0] for i in range(n)]
    node_y = [pos[i][1] for i in range(n)]

    # Node penalty = sum of conflicts
    penalties = [sum(conflict_matrix[i]) / (n - 1) if n > 1 else 0 for i in range(n)]
    node_sizes = [20 + 40 * p for p in penalties]

    node_colors = penalties
    node_text = [
        f"{doc_labels[i]}<br>Penalty: {penalties[i]:.3f}"
        for i in range(n)
    ]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            colorscale="RdYlGn_r",
            cmin=0, cmax=max(penalties) if penalties else 1,
            colorbar=dict(title="Conflict<br>Penalty"),
            line=dict(width=2, color="white"),
        ),
        text=doc_labels,
        textposition="top center",
        hoverinfo="text",
        hovertext=node_text,
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title="Conflict Graph — Semantic Contradiction Network",
        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="rgba(0,0,0,0)",
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig


def get_graph_stats(conflict_matrix, threshold=0.15):
    """
    Return summary statistics about the conflict graph.
    """
    G = build_conflict_graph_nx(conflict_matrix, threshold=threshold)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    max_possible = n * (n - 1) // 2 if n > 1 else 1

    flat = []
    for i in range(len(conflict_matrix)):
        for j in range(i + 1, len(conflict_matrix)):
            flat.append(conflict_matrix[i][j])

    return {
        "nodes": n,
        "edges": m,
        "density": m / max_possible if max_possible > 0 else 0,
        "max_conflict": max(flat) if flat else 0,
        "mean_conflict": float(np.mean(flat)) if flat else 0,
        "connected_components": nx.number_connected_components(G),
    }
