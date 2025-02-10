import cv2 as cv
import numpy as np
import time

# Load the predefined dictionary for Aruco markers (DICT_6X6_250)
dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)

# Initialize detector parameters using default values
parameters = cv.aruco.DetectorParameters()

# Create the ArucoDetector object
detector = cv.aruco.ArucoDetector(dictionary, parameters)

# Load the image to display when the marker is found
im_src = cv.imread("selfie.jpg")
if im_src is None:
    # Fallback if the image is not found
    im_src = np.ones((500, 500, 3), dtype=np.uint8) * 255  # White placeholder image
    cv.putText(im_src, "Image not found!", (50, 250), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Attempt to open the camera with a retry mechanism
max_retries = 30
retry_count = 0

while retry_count < max_retries:
    cap = cv.VideoCapture(0)  # Use camera n
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

# Initialize FPS counter
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to capture frame!")
        break

    frame_count += 1

    # Detect Aruco markers in the frame
    markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(frame)

    # Draw detected markers
    if markerIds is not None:
        cv.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)

        for i, markerId in enumerate(markerIds.flatten()):  # Flatten ensures correct iteration
            if markerId.item() == 33:  # Convert NumPy array to integer
                print("🎯 Marker 33 detected! Displaying image...")

                # Display the image
                h, w = im_src.shape[:2]
                cv.namedWindow('Aruco Marker Detection', cv.WINDOW_NORMAL)
                cv.resizeWindow('Aruco Marker Detection', w, h)
                cv.imshow('Aruco Marker Detection', im_src)
                cv.waitKey(3000)  # Keep image displayed for 3 seconds

    # Show the frame with markers highlighted
    cv.imshow('Aruco Marker Detection', frame)

    # FPS Calculation
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    cv.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Break the loop if the user presses 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv.destroyAllWindows()