import cv2
from Face_Detector import FaceMeshDetector,Distancia,Cuadro

def main():
    
    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector(staticMode=False, maxFaces=5, minDetectionCon=0.5, minTrackCon=0.5)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 3840x2160, 1920x1080, 1280x720, 640x480  
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    distancia = Distancia
    cuadro = Cuadro
    face_count = 0

    while True:
        success, img = cap.read()
        img_copy = img.copy()  # Copia de la imagen original para dibujar los rectángulos de los ojos

        img, faces = detector.findFaceMesh(img, draw=True)
        face_count = len(faces)

        if faces:
            for idx, face in enumerate(faces):
                # Selecciona puntos de referencia para calcular métricas
                p1 = face[1]
                p4 = face[10]
                p2 = face[159]
                p3 = face[386]

                # Distancia
                a, _, _ = distancia.findDistance_3(p1, p2, p3, img, metric='side_length')
                b, _, _ = distancia.findDistance(p1, p2, img, metric='euclidean')
                _, _, _ = distancia.findDistance_3(p1, p2, p3, img_copy, metric='side_length')
                _, _, _ = distancia.findDistance(p1, p2, img_copy, metric='euclidean')
                
                # Cuadros
                img_copy = cuadro.drawEyeRectangles(img_copy, face)
                img_copy = cuadro.drawEyebrowRectangles(img_copy, face)
                img_copy = cuadro.drawEarRectangles(img_copy, face)
                img_copy = cuadro.drawNoseRectangle(img_copy, face)
                img_copy = cuadro.drawMouthRectangle(img_copy, face)              
                img_copy = cuadro.drawJawRectangle(img_copy, face)
                img_copy = cuadro.drawForeheadRectangle(img_copy, face)

                # Logitudes
                length_ab, length_bc, length_ca = a

                # Imprime las longitudes de los lados y el número de la cara
                print(f"Cara {idx+1}: Distancia {b:.2f} Lado AB: {length_ab:.2f}, Lado BC: {length_bc:.2f}, Lado CA: {length_ca:.2f}")

        # Muestra el número de caras detectadas en la imagen
        cv2.putText(img, f"Caras: {face_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Image", img)
        cv2.imshow("Eye Rectangles", img_copy)  # Muestra la imagen con los rectángulos de los ojos
        key = cv2.waitKey(1)
        if key == ord('q'):  # Si se presiona 'q', cierra el programa
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
