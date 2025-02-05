import cv2 as cv
import numpy as np

# Load the predefined dictionary for Aruco markers (DICT_6X6_250)
dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)

# Initialize detector parameters using default values
parameters = cv.aruco.DetectorParameters()

# Create the ArucoDetector object
detector = cv.aruco.ArucoDetector(dictionary, parameters)

# Load the image that will be displayed when the marker is found
im_src = cv.imread("bena.png")
if im_src is None:
    print("Error: Image not found!")
    exit()

# Open the webcam or a video file
cap = cv.VideoCapture(0)  # Change '0' to a file path for a video

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video")
        break

    # Detect markers in the current frame
    markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(frame)

    # Check if any markers are detected
    if markerIds is not None:
        # Iterate through the detected markers
        for i, markerId in enumerate(markerIds):
            if markerId == 33:  # Check if the marker ID is 33
                # Get the dimensions of the source image (the image to display)
                h, w = im_src.shape[:2]

                # Create a window and resize it to the size of the image
                cv.namedWindow('Aruco Marker Detection', cv.WINDOW_NORMAL)
                cv.resizeWindow('Aruco Marker Detection', w, h)

                # Display the image in the window
                cv.imshow('Aruco Marker Detection', im_src)

                # Break the loop after finding the first marker with ID 33
                break

    # Display the original frame if no marker is detected
    if markerIds is None:
        cv.imshow('Aruco Marker Detection', frame)

    # Break the loop if the user presses 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv.destroyAllWindows()