import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

import numpy as np

from plot_glyph import draw_binary_glyph, TRUE_GREEN, FALSE_RED

# Configure matplotlib to use LaTeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')  # Match LaTeX font family, e.g., 'serif'
plt.rcParams['font.size'] = 10

# Helper function to draw an entity node
def draw_entity(ax, position, text, color="black", size=11):
   # Use plt.text to draw a rectangle with rounded corners
   orange = (1., 0.8, 0.5)
   light_orange = (1., 0.9, 0.8)
   bbox_style = dict(boxstyle="round,pad=0.5",
                     ec=orange,
                     fc=light_orange,
                     )
   ax.text(*position, text, size=size, ha="center", va="center", color=color, bbox=bbox_style)

# Helper function to draw an arrow with label
def draw_arrow(ax, start, end, relation="observes", is_testdata=False):
    distance = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
    
    # Calculate the shrink values based on the distance and shrink_factor
    shrinkB = 20

    if relation =="observes":
        color = "mediumorchid"
    elif relation == "orbits":
        color = "darkorange"

    if is_testdata:
        arrowstyle = "->"
        lw = 2
        ls="dashed"
    else:
        arrowstyle = "simple"
        lw = 0.3
        ls="solid"

    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=arrowstyle, color=color, 
        linewidth=lw, 
        linestyle=ls,
        shrinkB=shrinkB, 
        mutation_scale=15,
    )
    ax.add_patch(arrow)


def plot_graph(entities,
               train_relations,
               test_relations):
    # Initialize the figure and axis
    cm = 1/2.54  # centimeters in inches
    fig, ax = plt.subplots(figsize=(16*cm, 8*cm))
    ax.set_xlim(-3, 5)
    ax.set_ylim(-2.5, 2.5)
    ax.axis("off")

    for name, pos in entities.items():
        draw_entity(ax, pos, name)

    # Show true relations
    for start_name, relation_name, end_name in train_relations:
        draw_arrow(ax, entities[start_name], entities[end_name], relation_name, is_testdata=False)

    # Show test relations with binary glyphs
    rng = np.random.default_rng(seed=42)
    for start_name, relation_name, end_name, truth_prob in test_relations:
        draw_arrow(ax, entities[start_name], entities[end_name], relation_name, is_testdata=True)

        # Draw binary glyphs 
        n_clf = 3
        drawpoint = ((3*entities[start_name][0] + 2*entities[end_name][0]) / 5, 
                    (3*entities[start_name][1] + 2*entities[end_name][1]) / 5)
        glyph_values = rng.choice([True, False], (1+n_clf,), p=[truth_prob, 1-truth_prob])
        draw_binary_glyph(ax, *drawpoint, eps_set=glyph_values[1:1+n_clf], h0=glyph_values[0], ground_truth=True, size=3)  

    # Add legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="darkorange", label="Orbits"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="mediumorchid", label="Observes"),
        Line2D([0], [0], ls="-", color="grey", label="$\\mathcal{T}_{train}$"),
        Line2D([0], [0], ls="--", color="grey", label="$\\mathcal{T}_{test}$"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True)

    # Show the plot
    plt.tight_layout()
    fig.savefig("../figures/graph_clf_space.pdf")




if __name__ == "__main__":
    
    # Draw nodes (entities)
    entities = {
        "Sun": (0, 0),
        "Mars": (3, 2),
        "Jupiter": (-3, 1),
        "Titan": (-1.5, 2),
        "James Webb": (0, -2),
        "Earth": (2.5, -1.5),
        "Curiosity Rover": (4, 0),
        "Moon": (0, 1.5),
        "Hubble": (-1.5, -1.5),
        "Sirius": (-3, -0.5)
    }
    
    # Draw edges (relations) # (start, end, relation, truth_prob)
    train_relations = [
        ("Earth", "orbits", "Sun"),
        ("Mars", "orbits", "Sun"),
        ("Moon", "orbits", "Earth"),
        ("Moon", "orbits", "Sun"),
        ("Jupiter", "orbits", "Sun"),
        ("James Webb", "orbits", "Earth"),
        ("James Webb", "observes", "Jupiter"),
    ]
    test_relations = [
        ("Earth", "observes", "Curiosity Rover", 0.9),
        ("Curiosity Rover", "observes", "Mars", 0.9),
        ("Curiosity Rover", "orbits", "Sun", 0.9),
        ("Sun", "orbits", "James Webb", 0.1),
    ]

    plot_graph(entities, train_relations, test_relations)
    
