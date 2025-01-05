import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

from plot_glyph import draw_binary_glyph, TRUE_GREEN, FALSE_RED

# Configure matplotlib to use LaTeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')  # Match LaTeX font family, e.g., 'serif'
plt.rcParams['font.size'] = 10

# Helper function to draw an entity node
def draw_entity(ax, position, text, color="black", size=11):
   # Use plt.text to draw a rectangle with rounded corners
   red = (1., 0.5, 0.5)
   light_red = (1., 0.8, 0.8)
   bbox_style = dict(boxstyle="round,pad=0.5",
                     ec=red,
                     fc=light_red,
                     )
   ax.text(*position, text, size=size, ha="center", va="center", color=color, bbox=bbox_style)

# Helper function to draw an arrow with label
def draw_arrow(ax, start, end, color="red"):
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle="-|>", color=color, linewidth=1,
        shrinkA=10, shrinkB=20, mutation_scale=15
    )
    ax.add_patch(arrow)


if __name__ == "__main__":
        
    # Initialize the figure and axis
    cm = 1/2.54  # centimeters in inches
    fig, ax = plt.subplots(figsize=(12*cm, 6*cm))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-3, 3)
    ax.axis("off")

    # Draw nodes (entities)
    entities = {
        "Sun": (0, 0),
        "Mars": (3, 2),
        "Jupiter": (-3, 2),
        "Earth": (0, -2),
        "James Webb": (-4, 0),
        "Curiosity Rover": (5, 0)
    }
    for name, pos in entities.items():
        draw_entity(ax, pos, name)

    # Draw edges (relations)
    relations = [
        (("Earth", "Sun"), "red"),
        (("Mars", "Sun"), "red"),
        (("Jupiter", "Sun"), "red"),
        (("James Webb", "Earth"), "red"),
        (("James Webb", "Jupiter"), "blue"),
        (("Earth", "Curiosity Rover"), "blue"),
        (("Curiosity Rover", "Mars"), "blue"),
    ]
    for (start_name, end_name), color in relations:
        draw_arrow(ax, entities[start_name], entities[end_name], color)

    # Add legend
    legend_elements = [
        Line2D([0], [0], color="red", lw=2, label="Orbits"),
        Line2D([0], [0], color="blue", lw=2, label="Observes")
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True)


    # Draw binary glyphs 
    for r in relations:
        start_name, end_name = r[0]
        color = r[1]
        midpoint = ((3*entities[start_name][0] + 2*entities[end_name][0]) / 5, 
                    (3*entities[start_name][1] + 2*entities[end_name][1]) / 5)
        draw_binary_glyph(ax, *midpoint, [False, True, False], True, True, size=4)  

    # Show the plot
    plt.tight_layout()
    plt.show()
    fig.savefig("../figures/graph_clf_space.pdf")
