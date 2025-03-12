import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as PathEffects
from typing import List


def draw_curved_edges(G, pos, edge_labels, ax):
    # Precompute node positions
    node_coords = {node: pos[node] for node in G.nodes()}

    for u, v in G.edges():
        # Get node coordinates
        x1, y1 = node_coords[u]
        x2, y2 = node_coords[v]

        # Curvature parameter
        rad = 0.0

        # Draw the arrow
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>",
            connectionstyle=f"arc3,rad={rad}",
            lw=1.5,
            color="gray",
            alpha=0.7,
            zorder=1
        )
        ax.add_patch(arrow)

        # Get label
        label = edge_labels.get((u, v), "")

        # Calculate the exact center coordinates of the curved path
        # For an arc, we need to get the bezier control point
        # The control point for the arc is:
        dx = x2 - x1
        dy = y2 - y1
        dist = np.sqrt(dx * dx + dy * dy)

        # Compute the midpoint along the curve using parametric equation
        # For an arc3 connection, we need to consider the curvature
        midpoint_x = (x1 + x2) / 2
        midpoint_y = (y1 + y2) / 2

        # Perpendicular offset based on the rad parameter
        # This is how matplotlib's arc3 connection style works
        perp_x = -rad * dy
        perp_y = rad * dx

        # The midpoint is offset from the straight-line midpoint
        # The exact formula depends on the actual arc3 implementation
        label_x = midpoint_x + perp_x
        label_y = midpoint_y + perp_y

        # Create a white, semi-transparent background for the text
        # that's exactly sized to the text content
        text = ax.text(
            label_x, label_y, label,
            fontsize=20,
            ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3",
                      fc="white",
                      ec="none",
                      alpha=0.8),
            zorder=2  # Make sure text is above the lines
        )

        # Add path effects to make text stand out
        text.set_path_effects([
            PathEffects.withStroke(linewidth=4, foreground='white')
        ])


def draw_kt_graph(kt_data: List[any]):
    # Create a directed graph
    G = nx.DiGraph()

    # Keep track of unique nodes (subjects and objects)
    unique_nodes = set()

    # Process each knowledge triplet
    for kt in kt_data:
        subject = kt['subject']
        object_ = kt['object']
        predicate = kt['predicate']

        # Add to unique nodes
        unique_nodes.add(subject)
        unique_nodes.add(object_)

        # Add edge with predicate as label
        G.add_edge(subject, object_, label=predicate)

    # Print some graph info for verification
    print(f"Graph has {len(G.nodes)} nodes and {len(G.edges)} edges")

    # Set up the plot with a suitable figure size
    plt.figure(figsize=(20, 16))

    # Use a force-directed layout for node positioning with adjusted parameters
    # This helps with spacing between nodes
    pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)

    # Draw nodes with good visibility
    node_size = 3000
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_size,
        node_color="skyblue",
        alpha=0.8,
        linewidths=2,
        edgecolors="black")

    # Draw node labels with proper font size and positioning
    nx.draw_networkx_labels(
        G, pos,
        font_size=20,
        font_family="sans-serif",
        font_weight="bold")

    # Get edge labels
    edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}

    # Get current axis
    ax = plt.gca()

    # Draw edges with curved arrows and labels
    draw_curved_edges(G, pos, edge_labels, ax)

    # Remove axis
    plt.axis('off')
    plt.margins(0.1)

    # Add a title
    plt.title("Knowledge Graph", fontsize=32, pad=20)

    # Adjust layout to prevent clipping
    plt.tight_layout()
    return plt
