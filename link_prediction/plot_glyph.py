import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle, FancyBboxPatch

# Configure matplotlib to use LaTeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')  # Match LaTeX font family, e.g., 'serif'
plt.rcParams['font.size'] = 11

TRUE_GREEN = (143/255, 209/255, 79/255) #(8F, D1, 4F)
FALSE_RED = (240/255, 1/255, 1/255) #(F00101)


def draw_custom_glyph(ax, x, y, n_slices, slice_colors, baseline_color, ground_truth_color, 
                      radius=0.05, inner_radius=0.025, size=1):
    """
    Draw a custom glyph at given (x, y) coordinates with an inner circle and outer ring.
    
    Parameters:
        ax: Matplotlib Axes
            The axes on which to draw the glyphs.
        x, y: float
            Coordinates of the glyph.
        n_slices: int
            Number of slices in the circle.
        slice_colors: list of str or tuple
            Colors for each slice (length should match n_slices).
        baseline_color: str or tuple
            Color of the inner baseline circle.
        ground_truth_color: str or tuple
            Color of the outer ring.
        radius: float, optional
            Radius of the circle (default is 0.1).
        inner_radius: float, optional
            Radius of the inner baseline circle (default is 0.05)
    """
    # Draw the outer ring (ground truth indicator)
    # outer_ring = Circle((x, y), radius + outer_ring_width, 
    #                     facecolor=ground_truth_color, edgecolor='black', zorder=1)
    # ax.add_patch(outer_ring)

    radius = size * radius
    inner_radius = size * inner_radius

    
    # Draw the main divided circle
    theta_step = 360 / n_slices  # Angle step for slices
    for i in range(n_slices):
        start_angle = i * theta_step
        end_angle = start_angle + theta_step
        wedge = Wedge((x, y), radius, start_angle, end_angle, 
                      facecolor=slice_colors[i], edgecolor='black', lw=0.5, zorder=2)
        ax.add_patch(wedge)
    
    # Draw the inner circle (baseline predictor)
    inner_circle = Circle((x, y), inner_radius, 
                          facecolor=baseline_color, edgecolor='black', lw=0.5, zorder=3)
    ax.add_patch(inner_circle)

def draw_binary_glyph(ax, x, y, eps_set, h0, ground_truth, size=1):
    """
    Draw a binary glyph at given (x, y) coordinates with an inner circle and outer ring.
    
    Parameters:
        ax: Matplotlib Axes
            The axes on which to draw the glyphs.
        x, y: float
            Coordinates of the glyph.
        eps_set: list of bools 
            From list of classifiers in the epsilon set
        h0: bool
            From Baseline classifier.
        ground_truth: bool
            Ground truth label.
    """
    n_slices = len(eps_set)
    ground_truth_color = TRUE_GREEN if ground_truth else FALSE_RED
    baseline_color = TRUE_GREEN if h0 else FALSE_RED
    slice_colors = [TRUE_GREEN if h else FALSE_RED for h in eps_set]
    
    draw_custom_glyph(ax, x, y, n_slices, slice_colors, baseline_color, ground_truth_color, size=size)

def explain_binary_glyph(ax):
    """
    Add a legend to the plot explaining the binary glyph.
    
    Parameters:
        ax: Matplotlib Axes
            The axes on which to draw the legend.
    """
    x_center = 0.5
    y_center = 0.5
    draw_custom_glyph(ax, x_center, y_center, 3, [TRUE_GREEN, FALSE_RED, TRUE_GREEN], TRUE_GREEN, FALSE_RED, radius=0.2, inner_radius=0.1, outer_ring_width=0.04)

    ax.annotate("baseline $h_0$", xy=(x_center, y_center), xytext=(0.2, 0.9), arrowprops=dict(facecolor="black", shrink=0.05, width=0.8, headwidth=4, headlength=6), color="black", ha='center', va='center', transform=ax.transAxes, zorder=4.1)
    ax.text(x=0.1, y=0.18, s=r"$h \in S_{\varepsilon}(h_0)$")
    ax.text(x=0.4, y=1.05, s=r"prediction on $\mathcal{T}_{test}$", ha='center')
    ax.text(x=0.4, y=0.0, s=r"data $\mathcal{T}_{train}$", ha='center')
    ax.add_patch(FancyBboxPatch((0.3, 0.45), 0.2, 0.24, fc="none", ec="lightgrey", lw=0.8, zorder=5, boxstyle="round, rounding_size=0.02"))
    for i, col in enumerate(["tab:blue", "tab:orange", "tab:green"]):
        x_aw = 0.27
        y_aw = 0.27
        shrink = 0.925
        ax.arrow(x_aw, y_aw, shrink*((x_center - 0.15*np.cos((i+2)*2*np.pi/3)) - x_aw), 
                            shrink*((y_center - 0.15*np.sin((i+2)*2*np.pi/3)) - y_aw), 
                            head_width=0.02, head_length=0.02, fc=col, ec=col, zorder=4)
    # Aspect 1:2
    ax.set_xlim(0., 0.8)
    ax.set_ylim(-0.45, 1.15)
    # hide the axes
    ax.axis('off')


# Example usage
if __name__ == "__main__":
    n_points = 10
    n_slices = 3

    # Generate random data
    rng = np.random.default_rng(seed=42)
    x = rng.random((n_points, 2))
    colors = [TRUE_GREEN, FALSE_RED]
    slice_colors = rng.choice(colors, (n_slices, n_points), p=[0.8, 0.2])
    ground_truth_colors = rng.choice(colors, n_points, p=[0.8, 0.2])
    baseline_colors = rng.choice(colors, n_points, p=[0.8, 0.2])

    cm = 1/2.54  # centimeters in inches
    fig, axs = plt.subplots(1, 2, figsize=(14*cm, 7*cm))
    ax = axs[0]
    ax.set_aspect('equal')

    # Plot custom glyphs
    for i in range(n_points):
        slices = [slice_colors[j, i] for j in range(n_slices)]
        ground_truth = ground_truth_colors[i]
        baseline = baseline_colors[i]
        draw_custom_glyph(ax, x[i, 0], x[i, 1], n_slices, slices, baseline, ground_truth)

    # Annotation
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Feature $x_1$")
    ax.set_ylabel("Feature $x_2$")
    ax.grid()

    # Legend for the binary glyph
    axs[1].set_aspect('equal')
    explain_binary_glyph(axs[1])
    plt.show()
