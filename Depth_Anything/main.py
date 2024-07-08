import cv2
from inference import DepthAnythingInference
import numpy as np


class IntelligentDodgeRobot:
    def __init__(self):
        # Opcional: establecer la resolución de captura de la cámara (si la cámara lo soporta)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 3840x2160, 1920x1080, 1280x720, 640x480  
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.depth_anything = DepthAnythingInference(model_path='depth_anything_vits14.onnx', color=True)

    def frame_processing(self):
        while True:
            t = cv2.waitKey(5)
            ret, frame = self.cap.read()
            if not ret:
                break
            
            img_depth = self.depth_anything.frame_inference(frame)
            cv2.imshow('depth anything', img_depth)
            cv2.imshow('frame in real time', frame)

            # Calcular el valor máximo de profundidad mostrado en la imagen
            max_depth_value = np.max(img_depth)
            print(f"Valor máximo de profundidad: {max_depth_value}")

            if t == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()

robot_intelligent = IntelligentDodgeRobot()
robot_intelligent.frame_processing()

