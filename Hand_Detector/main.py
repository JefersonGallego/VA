from Hand_Detector import HandDetector
import cv2

def initialize_camera(index=0):
    """
    Initialize the webcam.
    
    :param index: Index of the camera to use. Default is 0 for the built-in camera.
    :return: The VideoCapture object if successful, None otherwise.
    """
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Error: Cannot open camera with index {index}")
        return None
    return cap

def process_frame(detector, frame):
    """
    Process a single frame to detect hands and draw annotations.
    
    :param detector: The HandDetector object.
    :param frame: The image frame to process.
    :return: The processed frame with annotations.
    """
    hands, img = detector.findHands(frame, draw=True, flipType=True)

    # Display number of hands detected
    num_hands = len(hands)
    cv2.putText(img, f'Hands Detected: {num_hands}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    if hands:
        hand1 = hands[0]
        lmList1 = hand1["lmList"]
        fingers1 = detector.fingersUp(hand1)
        print(f'H1 = {fingers1.count(1)}', end=" ")

        length, info, img = detector.findDistance(lmList1[8][0:2], lmList1[12][0:2], img, color=(255, 0, 255), scale=10)

        if len(hands) == 2:
            hand2 = hands[1]
            lmList2 = hand2["lmList"]
            fingers2 = detector.fingersUp(hand2)
            print(f'H2 = {fingers2.count(1)}', end=" ")

            length, info, img = detector.findDistance(lmList1[8][0:2], lmList2[8][0:2], img, color=(255, 0, 0), scale=10)

        print(f" Distance = {length} ")

    return img

def main():
    cap = initialize_camera(0)
    if cap is None:
        return

    detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

    try:
        while True:
            success, img = cap.read()
            if not success:
                print("Error: Unable to capture frame")
                break

            img = process_frame(detector, img)
            cv2.imshow("Image", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
