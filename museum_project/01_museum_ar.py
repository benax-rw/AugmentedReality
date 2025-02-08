import cv2 as cv
import numpy as np

# Load the predefined dictionary (e.g., DICT_6X6_250)
aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
parameters = cv.aruco.DetectorParameters()

# Mapping of Aruco marker IDs to exhibit information
exhibit_info = {
    33: "The Rosetta Stone\nThe key to deciphering Egyptian hieroglyphs. Discovered in 1799.",
    47: "The Mona Lisa\nPainted by Leonardo da Vinci in the early 1500s. A masterpiece of the Renaissance.",
    128: "Samurai Armor\nWorn by Japanese warriors, symbolizing honor and discipline."
}

# Start video capture
cap = cv.VideoCapture(0)  # Use webcam (0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, _ = cv.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        cv.aruco.drawDetectedMarkers(frame, corners, ids)

        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in exhibit_info:
                # Get marker corner positions
                corner = corners[i][0]
                x, y = int(corner[0][0]), int(corner[0][1])  # Top-left corner

                # Draw a stylish semi-transparent text box
                overlay = frame.copy()
                box_height = 100
                cv.rectangle(overlay, (x, y), (x + 300, y + box_height), (0, 0, 0), -1)  # Black background
                frame = cv.addWeighted(overlay, 0.6, frame, 0.4, 0)  # Transparency effect

                # Display exhibit info in the box
                text = exhibit_info[marker_id]
                cv.putText(frame, text, (x + 10, y + 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)

    # Show the frame
    cv.imshow("Museum AR Guide - Point at an Exhibit", frame)

    # Press 'q' to exit
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()