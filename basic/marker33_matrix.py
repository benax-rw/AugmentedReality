import matplotlib.pyplot as plt
import numpy as np

# Aruco Marker Binary Matrix
marker_matrix = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 0, 1],
    [1, 1, 1, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 1, 1, 1],
    [1, 0, 1, 1, 0, 1, 1, 1],
    [1, 1, 0, 0, 1, 0, 1, 1],
    [1, 0, 1, 1, 1, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1]
])

# Plot the binary matrix
plt.imshow(marker_matrix, cmap="binary", interpolation="nearest")
plt.title("Aruco Marker Binary Matrix")
plt.axis("off")  # Hide axes for a clean look
plt.show()