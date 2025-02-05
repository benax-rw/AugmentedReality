import cv2 as cv
import numpy as np

# Load the predefined dictionary for Aruco markers (DICT_6X6_250)
dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)

# Initialize detector parameters using default values
parameters = cv.aruco.DetectorParameters()

# Create the ArucoDetector object
detector = cv.aruco.ArucoDetector(dictionary, parameters)

# Load the image that will be displayed on the marker (new_scenery.jpg)
im_src = cv.imread("dy.png")
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
                # Get the corner points of the detected marker
                corners = markerCorners[i].reshape(4, 2)
                (topLeft, topRight, bottomRight, bottomLeft) = corners

                # Set the points for homography: the corners of the detected marker
                pts_dst = np.array([topLeft, topRight, bottomRight, bottomLeft], dtype="float32")

                # Define the size of the source image (the image to overlay)
                h, w = im_src.shape[:2]

                # Define the source points: the corners of the source image
                pts_src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")

                # Compute the homography matrix
                H, _ = cv.findHomography(pts_src, pts_dst)

                # Warp the source image to fit on the marker
                warped_image = cv.warpPerspective(im_src, H, (frame.shape[1], frame.shape[0]))

                # Create a mask from the warped image for blending
                mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                cv.fillConvexPoly(mask, pts_dst.astype(int), 255)

                # Invert the mask to black out the marker area on the frame
                frame_masked = cv.bitwise_and(frame, frame, mask=cv.bitwise_not(mask))

                # Add the warped image to the original frame
                frame = cv.add(frame_masked, warped_image)

                # Break the loop after finding the first marker with ID 33
                break

    # Display the frame with the image overlay
    cv.imshow('Aruco Marker Detection', frame)

    # Break the loop if the user presses 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv.destroyAllWindows()
