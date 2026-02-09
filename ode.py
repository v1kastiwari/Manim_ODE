"""
Visualization of the differential equation dy/dx = -x/y

The equation dy/dx = -x/y represents circles: x² + y² = C
(obtained by separating variables: y dy = -x dx)

Isoclines: curves where the slope is constant
For dy/dx = -x/y = k (constant), we get y = -x/k (straight lines through origin)
"""

import numpy as np
import matplotlib.pyplot as plt

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# ============= LEFT PLOT: Direction Field with Solution Curves =============
x = np.linspace(-5, 5, 20)
y = np.linspace(-5, 5, 20)
X, Y = np.meshgrid(x, y)

# Differential equation: dy/dx = -x/y
# Avoid division by zero
U = np.ones_like(X)
V = -X / (Y + 1e-10)

# Normalize arrows for better visualization
N = np.sqrt(U**2 + V**2)
U_norm = U / (N + 0.1)
V_norm = V / (N + 0.1)

# Plot direction field (slope field)
ax1.quiver(X, Y, U_norm, V_norm, alpha=0.6, color='gray', scale=30)

# Plot solution curves (circles: x² + y² = C)
theta = np.linspace(0, 2*np.pi, 200)
radii = np.array([1, 2, 3, 4, 4.5])
for r in radii:
    x_circle = r * np.cos(theta)
    y_circle = r * np.sin(theta)
    ax1.plot(x_circle, y_circle, 'b-', linewidth=2, label='Solution curves' if r == radii[0] else '')

# Plot isoclines (lines y = -x/k where k = dy/dx)
slopes = [-3, -1, -0.5, 0.5, 1, 3]  # Different constant slope values
x_line = np.linspace(-5, 5, 100)

for slope_value in slopes:
    if slope_value != 0:  # Avoid division by zero
        k = -1 / slope_value  # Since dy/dx = k means y = -x/k
        y_line = -x_line / slope_value
        ax1.plot(x_line, y_line, 'r--', alpha=0.5, linewidth=1)
    else:
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=1)

ax1.set_xlim(-5, 5)
ax1.set_ylim(-5, 5)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('Direction Field with Solution Curves and Isoclines\ndy/dx = -x/y', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)

# Add legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='gray', linewidth=0, marker='>', markersize=8, label='Direction Field'),
    Line2D([0], [0], color='blue', linewidth=2, label='Solution Curves (circles)'),
    Line2D([0], [0], color='red', linewidth=1, linestyle='--', label='Isoclines (dy/dx = constant)')
]
ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)

# ============= RIGHT PLOT: Close-up with Isoclines Labeled =============
x_detailed = np.linspace(-4, 4, 25)
y_detailed = np.linspace(-4, 4, 25)
X_d, Y_d = np.meshgrid(x_detailed, y_detailed)
U_d = np.ones_like(X_d)
V_d = -X_d / (Y_d + 1e-10)
N_d = np.sqrt(U_d**2 + V_d**2)
U_d_norm = U_d / (N_d + 0.1)
V_d_norm = V_d / (N_d + 0.1)

ax2.quiver(X_d, Y_d, U_d_norm, V_d_norm, alpha=0.5, color='gray', scale=25)

# Solution curves
for r in [1.5, 2.5, 3.5]:
    x_circle = r * np.cos(theta)
    y_circle = r * np.sin(theta)
    ax2.plot(x_circle, y_circle, 'b-', linewidth=2.5)

# Isoclines with labels
isocline_slopes = [-2, -1, -0.5, 0.5, 1, 2]
colors = plt.cm.RdYlGn(np.linspace(0, 1, len(isocline_slopes)))

for i, slope_val in enumerate(isocline_slopes):
    k_val = -1 / slope_val
    y_iso = -x_line / slope_val
    ax2.plot(x_line, y_iso, color=colors[i], linestyle='--', linewidth=2, 
             label=f'dy/dx = {slope_val}')

ax2.set_xlim(-4, 4)
ax2.set_ylim(-4, 4)
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('y', fontsize=12)
ax2.set_title('Isoclines and Solution Curves\n(Isoclines show where slope is constant)', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.axvline(x=0, color='k', linewidth=0.5)
ax2.legend(loc='upper right', fontsize=9, ncol=2)

plt.tight_layout()
plt.show()

# ============= ANALYSIS =============
print("=" * 60)
print("DIFFERENTIAL EQUATION ANALYSIS: dy/dx = -x/y")
print("=" * 60)
print("\n1. SOLUTION (via separation of variables):")
print("   y dy = -x dx")
print("   ∫y dy = ∫-x dx")
print("   y²/2 = -x²/2 + C")
print("   → x² + y² = 2C  (FAMILY OF CIRCLES)")
print("\n2. ISOCLINES:")
print("   Definition: curves where dy/dx is constant")
print("   For dy/dx = -x/y = k (constant slope):")
print("   → -x/y = k  ⟹  y = -x/k")
print("   → Isoclines are STRAIGHT LINES through the origin")
print("   → Slope of isocline: -1/k")
print("\n3. GEOMETRIC INTERPRETATION:")
print("   • Solution curves are concentric circles centered at origin")
print("   • Isoclines are straight lines radiating from origin")
print("   • At any point, the tangent slope is perpendicular to radius")
print("   • Isoclines tell us the direction of flow along those lines")
print("=" * 60)
