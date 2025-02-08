import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load the predefined Aruco dictionary
aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)

# Define parameters
marker_size = 100  # Size of each Aruco marker
padding = 20  # Space between markers
text_space = 25  # Extra space below each marker for the ID
total_markers = 250  # DICT_6X6_250 contains 250 markers

# Determine grid size (square layout)
grid_size = int(np.ceil(np.sqrt(total_markers)))

# Compute final image size
canvas_size = grid_size * (marker_size + padding + text_space) - padding

# Create a blank white canvas
grid_image = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255

# Define text properties
font = cv.FONT_HERSHEY_SIMPLEX
font_scale = 0.6
font_thickness = 2

# Generate and arrange markers with spacing
for marker_id in range(total_markers):
    # Generate marker image
    marker_image = aruco_dict.generateImageMarker(marker_id, marker_size)

    # Compute row and column positions
    row = marker_id // grid_size
    col = marker_id % grid_size

    # Compute marker position with padding
    y_start = row * (marker_size + padding + text_space)
    x_start = col * (marker_size + padding)

    # Place marker on grid
    grid_image[y_start:y_start + marker_size, x_start:x_start + marker_size] = marker_image

    # Add marker ID below the marker
    text = str(marker_id)
    text_size = cv.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = x_start + (marker_size - text_size[0]) // 2  # Center horizontally
    text_y = y_start + marker_size + text_size[1] + 5  # Position in white space

    # Draw the text
    cv.putText(grid_image, text, (text_x, text_y), font, font_scale, 0, font_thickness)

# Show the grid of markers with IDs
plt.figure(figsize=(12, 12))
plt.imshow(grid_image, cmap='gray')
plt.axis("off")
plt.title("Aruco Markers (DICT_6X6_250) with IDs")
plt.show()

# Wait for 'q' key to exit
while True:
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Close the window
cv.destroyAllWindows()