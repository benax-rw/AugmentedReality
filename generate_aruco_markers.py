import cv2 as cv
import numpy as np

# Load the predefined dictionary
dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)

# Generate the marker and return as an image
markerImage = dictionary.generateImageMarker(33, 200)

# Save the marker image as a PNG file
cv.imwrite("marker33.png", markerImage)