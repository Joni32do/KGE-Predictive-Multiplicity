import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

from ..classification.plot_glyph import draw_binary_glyph, TRUE_GREEN, FALSE_RED

# Configure matplotlib to use LaTeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')  # Match LaTeX font family, e.g., 'serif'
plt.rcParams['font.size'] = 11

# Helper function to draw an entity node
def draw_entity(ax, position, text, color="black", size=11):
   # Use plt.text to draw a rectangle with rounded corners
   red = (1., 0.5, 0.5)
   light_red = (1., 0.8, 0.8)
   bbox_style = dict(boxstyle="round,pad=1",
                     ec=red,
                     fc=light_red,
                     )
   ax.text(*position, text, size=size, ha="center", va="center", color=color, bbox=bbox_style)

# Helper function to draw an arrow with label
def draw_arrow(ax, start, end, color="red"):
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle="->", color=color, linewidth=2,
        shrinkA=10, shrinkB=10
    )
    ax.add_patch(arrow)


if __name__ == "__main__":
        
    # Initialize the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-5, 6)
    ax.set_ylim(-3, 3)
    ax.axis("off")

    # Draw nodes (entities)
    entities = {
        "Sun": (0, 0),
        "Mars": (3, 2),
        "Jupiter": (-3, 2),
        "Earth": (-2, -2),
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
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8, frameon=True)


    # Add glyph for relation 
    test_relation = [
        (("James Webb", "Sun"), "red"),
        (("James Webb", "Mars"), "blue"),
        (("James Webb", "Earth"), "red"),
    ]
    
    draw_binary_glyph(ax, -2, 0, [TRUE_GREEN, FALSE_RED,], TRUE_GREEN, True)

    # Show the plot
    plt.tight_layout()
    plt.show()
