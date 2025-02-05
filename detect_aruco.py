import cv2 as cv
import numpy as np

# Load the predefined dictionary for Aruco markers (DICT_6X6_250)
dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)

# Initialize detector parameters using default values
parameters = cv.aruco.DetectorParameters()

# Create the ArucoDetector object
detector = cv.aruco.ArucoDetector(dictionary, parameters)

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
        # Draw the detected markers and their IDs on the frame
        cv.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
        
        # Iterate through the detected markers
        for markerId in markerIds:
            print(f"Detected Marker ID: {markerId}")
        
    # Display the frame with the marker detection
    cv.imshow('Aruco Marker Detection', frame)

    # Break the loop if the user presses 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv.destroyAllWindows()
