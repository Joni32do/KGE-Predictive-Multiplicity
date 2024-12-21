import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle

def draw_custom_glyph(ax, x, y, n_slices, slice_colors, baseline_color, ground_truth_color, 
                      radius=0.05, inner_radius=0.025, outer_ring_width=0.01):
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
            Radius of the inner baseline circle (default is 0.05).
        outer_ring_width: float, optional
            Width of the outer ring (default is 0.02).
    """
    # Draw the outer ring (ground truth indicator)
    outer_ring = Circle((x, y), radius + outer_ring_width, 
                        facecolor=ground_truth_color, edgecolor='black', zorder=1)
    ax.add_patch(outer_ring)
    
    # Draw the main divided circle
    theta_step = 360 / n_slices  # Angle step for slices
    for i in range(n_slices):
        start_angle = i * theta_step
        end_angle = start_angle + theta_step
        wedge = Wedge((x, y), radius, start_angle, end_angle, 
                      facecolor=slice_colors[i], edgecolor='black', zorder=2)
        ax.add_patch(wedge)
    
    # Draw the inner circle (baseline predictor)
    inner_circle = Circle((x, y), inner_radius, 
                          facecolor=baseline_color, edgecolor='black', zorder=3)
    ax.add_patch(inner_circle)

# Example usage
n_points = 10
n_slices = 5
np.random.seed(0)

# Generate random data
x = np.random.rand(n_points, 2)

# Generate random slice colors for each point
true_green = (143/255, 209/255, 79/255) #(8F, D1, 4F)
false_red = (240/255, 1/255, 1/255) #(F00101)
colors = [true_green, false_red]
rng = np.random.default_rng()


slice_colors = rng.choice(colors, (n_slices, n_points), p=[0.8, 0.2])  # RGB colors for each slice and point
ground_truth_colors = rng.choice(colors, n_points, p=[0.8, 0.2])    # RGB colors for outer ring
baseline_colors = rng.choice(colors, n_points, p=[0.8, 0.2])         # RGB colors for inner circle

fig, ax = plt.subplots()
ax.set_aspect('equal')

def not_colliding_with_previous_drawn(x, i):
    minimal_distance = 0.17
    for j in range(i):
        if np.linalg.norm(x[j]-x[i]) < minimal_distance:
            return False
    return True


# Plot custom glyphs
for i in range(n_points):
    slices = [slice_colors[j, i] for j in range(n_slices)]
    ground_truth = ground_truth_colors[i]
    baseline = baseline_colors[i]
    if not_colliding_with_previous_drawn(x, i):
        draw_custom_glyph(ax, x[i, 0], x[i, 1], n_slices, slices, baseline, ground_truth)

ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)
ax.set_xlabel("Feature $x_1$")
ax.set_ylabel("Feature $x_2$")
ax.grid()

plt.show()

fig.savefig("Classification_Plot.eps")
