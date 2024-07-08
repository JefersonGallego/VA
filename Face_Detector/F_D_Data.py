import cv2
from Face_Detector import FaceDetector
# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 3840x2160, 1920x1080, 1280x720, 640x480  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize the FaceDetector object
detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

# Create a named window for the main image display
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", 800, 600)  # Resize window if needed

# Create a named window for displaying the detected face in original resolution
cv2.namedWindow("Detected Face Original", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detected Face Original", 400, 400)  # Resize window if needed

# Create a named window for displaying the detected face in 50x50 resolution
cv2.namedWindow("Detected Face 50x50", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detected Face 50x50", 50, 50)  # Fixed resolution window

# Run the loop to continually get frames from the webcam
while True:
    # Read the current frame from the webcam
    success, img = cap.read()

    # Detect faces in the image
    img, bboxs = detector.findFaces(img, draw=False)

    # Check if any face is detected
    if bboxs:
        # Loop through each bounding box
        for bbox in bboxs:
            # Extract bounding box coordinates
            x, y, w, h = bbox['bbox']

            # Draw a rectangle around the detected face on the main image
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)

            # Extract the region of the detected face in original resolution
            face_region_original = img[y:y + h, x:x + w]

            # Resize the face region to 50x50 pixels
            face_region_50x50 = cv2.resize(face_region_original, (50, 50))

            # Display the face region in the separate window (original resolution)
            cv2.imshow("Detected Face Original", face_region_original)

            # Display the face region in the separate window (50x50 resolution)
            cv2.imshow("Detected Face 50x50", face_region_50x50)

    # Display the main image with detected faces
    cv2.imshow("Image", img)

    # Wait for 1 millisecond, and keep the window ope
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
