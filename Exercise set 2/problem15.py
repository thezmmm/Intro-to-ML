import matplotlib.pyplot as plt
import numpy as np

# --- Data points ---
points = {
    "A": (-1.5, 0.0),
    "B": (0.0, 0.9),
    "C": (0.2, -1.8),
    "D": (2.0, 0.0),
    "E": (4.0, 1.0),
    "F": (4.0, -1.0),
    "G": (6.0, -0.5),
    "H": (7.0, -1.0),
}

classes = {
    "A": 1, "B": 1, "C": 1, "D":1,
    "E": -1, "F": -1, "G": -1, "H": -1
}

# --- Grid for the circle ---
x1 = np.linspace(-5, 8, 400)
x2 = np.linspace(-3, 6, 400)
X1, X2 = np.meshgrid(x1, x2)
F = (1 + X1)**2 + (2 - X2)**2

plt.figure(figsize=(7, 7))

# --- Draw the boundary curve ---
plt.contour(X1, X2, F, levels=[4], linewidths=2)

# --- Shade inside area ---
plt.contourf(X1, X2, F, levels=[0, 4], alpha=0.25)

# --- Plot points ---
for label, (x, y) in points.items():
    color = "blue" if classes[label] == 1 else "red"
    plt.scatter(x, y, c=color, s=50)
    plt.text(x + 0.1, y + 0.1, label, fontsize=12)

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Data Points with Curve (1 + x1)^2 + (2 - x2)^2 = 4")
plt.gca().set_aspect('equal', 'box')
plt.grid(True)

plt.show()
