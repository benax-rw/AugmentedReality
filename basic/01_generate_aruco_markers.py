import cv2 as cv

# Load the predefined dictionary of Aruco markers.
# The dictionary contains various pre-generated markers with different sizes and patterns.
# DICT_6X6_250 means:
# - 6x6: The marker consists of a 6x6 grid of black and white squares.
# - 250: The dictionary contains 250 unique markers.
dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)

# Generate an Aruco marker image.
# Parameters:
# - 33: The ID of the marker within the selected dictionary (must be between 0 and 249 for DICT_6X6_250).
# - 200: The size of the output image in pixels (200x200).
markerImage = dictionary.generateImageMarker(33, 200)

# Save the generated marker as a PNG file.
# This allows the marker to be printed and used for detection in computer vision applications.
cv.imwrite("marker33_200.png", markerImage)

print("Aruco marker (ID 33, size 200x200) saved as 'marker33_200.png'.")