import cv2 as cv
import numpy as np
import time

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

# Attempt to open the camera with a retry mechanism
max_retries = 30
retry_count = 0

while retry_count < max_retries:
    cap = cv.VideoCapture(1) # Camera 0 is being used for showing the face of the presenter (Me)
    if cap.isOpened():
        print("✅ Camera opened successfully!")
        break
    else:
        print(f"⚠️ Failed to open camera. Retrying {retry_count + 1}/{max_retries}...")
        retry_count += 1
        time.sleep(1)  # Wait 1 second before retrying

if not cap.isOpened():
    print("❌ Error: Unable to open the camera after multiple attempts.")
    exit()

# Retry mechanism for frame capturing
frame_retries = 30  # Max retries for capturing a frame

while True:
    retry_count = 0  # Reset retry count for frame capture

    while retry_count < frame_retries:
        ret, frame = cap.read()
        if ret:
            break  # Successfully captured a frame
        else:
            print(f"⚠️ Failed to capture video frame. Retrying {retry_count + 1}/{frame_retries}...")
            retry_count += 1
            time.sleep(0.5)  # Wait 0.5 seconds before retrying

    if not ret:
        print("❌ Error: Unable to capture video frames after multiple attempts.")
        break

    # Detect markers in the current frame
    markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(frame)

    # Check if any markers are detected
    if markerIds is not None:
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