import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path
import numpy as np
import networkx as nx

def draw_oval(ax, center, text, width, height, color):
    """
    Draw an oval with text inside.
    
    Parameters:
        ax: matplotlib Axes
            The axes to draw the oval on.
        center: tuple
            (x, y) coordinates of the oval's center.
        text: str
            Text to display inside the oval.
        width, height: float
            Width and height of the oval.
        color: str
            Color of the oval.
    """
    # Create the oval patch
    oval = FancyBboxPatch(
        (center[0] - width / 2, center[1] - height / 2),
        width,
        height,
        boxstyle="round,pad=0.3,rounding_size=0.2",
        edgecolor="black",
        facecolor=color,
        lw=1.5,
    )
    ax.add_patch(oval)
    # Add the text
    ax.text(center[0], center[1], text, ha="center", va="center", fontsize=10, color="black")


def draw_bezier(ax, start, end, color="black"):
    """
    Draw a Bezier curve between two points.
    
    Parameters:
        ax: matplotlib Axes
            The axes to draw the curve on.
        start, end: tuple
            (x, y) coordinates of the start and end points of the curve.
        color: str
            Color of the curve.
    """
    # Control points for Bezier curve
    control1 = (start[0], (start[1] + end[1]) / 2)
    control2 = (end[0], (start[1] + end[1]) / 2)
    vertices = [start, control1, control2, end]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    path = Path(vertices, codes)
    patch = PathPatch(path, edgecolor=color, facecolor="none", lw=1.5)
    ax.add_patch(patch)


def plot_graph_with_beziers():
    # Define nouns and their relationships
    nouns = ["apple", "banana", "cherry", "date", "fig"]
    edges = [("apple", "banana"), ("banana", "cherry"), ("cherry", "date"), ("apple", "date"), ("fig", "apple")]
    
    # Create a graph
    G = nx.Graph()
    G.add_edges_from(edges)
    
    # Node positions (circular layout)
    pos = nx.circular_layout(G)
    
    # Plot setup
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.axis("off")
    
    # Draw nodes as ovals
    for node, (x, y) in pos.items():
        draw_oval(ax, (x, y), node, width=0.15, height=0.08, color=np.random.choice(["#ff9999", "#99ccff", "#c2f0c2", "#ffcc99"]))
    
    # Draw edges as Bezier curves
    for start, end in G.edges():
        draw_bezier(ax, pos[start], pos[end])

       # Adjust view limits to zoom out
    # margin = 0.3  # Adjust this value for more or less zoom
    # x_min, x_max = ax.get_xlim()
    # y_min, y_max = ax.get_ylim()
    # ax.set_xlim(x_min - margin, x_max + margin)
    # ax.set_ylim(y_min - margin, y_max + margin)

    
    plt.show()

# Execute the graph plotting
plot_graph_with_beziers()
