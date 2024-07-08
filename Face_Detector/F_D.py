import cv2
from Face_Detector import FaceDetector

# Inicializar la captura de video desde la cámara
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 3840x2160, 1920x1080, 1280x720, 640x480  
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Verificar si la cámara se abrió correctamente
if not video.isOpened():
    print("Error: No se pudo acceder a la cámara.")
    exit()

# Inicializar el detector de rostros
detector = FaceDetector()

# Contador de imágenes guardadas
saved_image_counter = 0

try:
    while True:
        # Leer un cuadro de la cámara
        ret, img = video.read()
        if not ret:
            print("Error: No se pudo leer el cuadro de la cámara.")
            break

        # Detectar rostros en el cuadro
        img, bboxes = detector.findFaces(img, draw=True)

        # Mostrar el número de rostros detectados en tiempo real
        if bboxes:
            cv2.putText(img, f'Rostros detectados: {len(bboxes)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Dibujar las coordenadas (x, y) en la imagen
            for bbox in bboxes:
                x, y, w, h = bbox['bbox']
                cv2.putText(img, f'({x},{y})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Mostrar la imagen con los rostros detectados
        cv2.imshow("Resultado: ", img)

        # Esperar por una tecla presionada
        key = cv2.waitKey(1)
        if key == 27:  # ESC para salir
            break
        elif key == ord('s'):  # 's' para guardar la imagen
            cv2.imwrite(f'rostro_detectado_{saved_image_counter}.jpg', img)
            saved_image_counter += 1
            print(f'Imagen guardada como rostro_detectado_{saved_image_counter}.jpg')

except Exception as e:
    print(f"Error: {e}")

finally:
    # Liberar la cámara y cerrar todas las ventanas
    video.release()
    cv2.destroyAllWindows()