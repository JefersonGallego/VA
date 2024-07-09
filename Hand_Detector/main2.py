import cv2
from Hand_Detector import HandDetector

# Inicializa la captura de video desde la cámara web
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Configura la resolución del video
video.set(3, 1280)
video.set(4, 720)

# Inicializa el detector de manos
detector = HandDetector()

# Lista para almacenar los puntos donde se dibujará
drawing_points = []

while True:
    # Captura un fotograma del video
    check, img = video.read()

    # Detecta manos en el fotograma y dibuja las manos encontradas
    hands, img = detector.findHands(img, draw=True)

    # Si se detectan manos
    if hands:
        # Obtiene la primera mano detectada
        hand = hands[0]

        # Obtiene la lista de puntos de referencia de la mano
        lm_list = hand["lmList"]

        # Verifica cuáles dedos están levantados
        fingers_up = detector.fingersUp(hand)

        # Cuenta el número de dedos levantados
        fingers_up_count = fingers_up.count(1)

        # Lógica de dibujo basada en los dedos levantados
        if fingers_up_count == 1:
            x, y = lm_list[8][0], lm_list[8][1]
            cv2.circle(img, (x, y), 15, (0, 0, 255), cv2.FILLED)
            drawing_points.append((x, y))
        elif fingers_up_count != 3:
            # Si no hay exactamente un dedo índice levantado o tres dedos levantados, añade un punto de pausa
            drawing_points.append((0, 0))
        elif fingers_up_count == 3:
            # Si hay tres dedos levantados, limpia el dibujo
            drawing_points = []

        # Dibuja círculos y líneas basados en los puntos almacenados en drawing_points
        for idx, point in enumerate(drawing_points):
            x, y = point
            if x != 0:
                cv2.circle(img, (x, y), 10, (0, 0, 255), cv2.FILLED)
            if idx >= 1:
                prev_x, prev_y = drawing_points[idx - 1]
                if x != 0 and prev_x != 0:
                    cv2.line(img, (x, y), (prev_x, prev_y), (0, 0, 255), 20)

    # Voltea la imagen horizontalmente para una vista en espejo
    img_flip = cv2.flip(img, 1)
    
    # Muestra la imagen en una ventana
    cv2.imshow('Hand Drawing', img_flip)
    
    # Si se presiona la tecla 'Esc', sale del bucle
    if cv2.waitKey(1) == 27:
        break

# Libera la cámara y cierra las ventanas
video.release()
cv2.destroyAllWindows()

