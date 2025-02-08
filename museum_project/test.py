import cv2

if hasattr(cv2.aruco, "detectMarkers"):
    print("✅ Aruco module is available!")
else:
    print("❌ Aruco module is NOT available!")