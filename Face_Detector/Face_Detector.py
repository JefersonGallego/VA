import cv2
import mediapipe as mp
import math

class FaceMeshDetector:
    
    """   Detector de malla facial """

    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        """
        :param staticMode: En modo estático, la detección se realiza en cada imagen: más lento.
        :param maxFaces: Número máximo de caras a detectar.
        :param minDetectionCon: Umbral mínimo de confianza para la detección.
        :param minTrackCon: Umbral mínimo de confianza para el seguimiento.
        """
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        # Inicializa los utilitarios de dibujo y el módulo de malla facial de MediaPipe.
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=self.staticMode,
                                                 max_num_faces=self.maxFaces,
                                                 min_detection_confidence=self.minDetectionCon,
                                                 min_tracking_confidence=self.minTrackCon)
        # Configuración de especificaciones de dibujo.
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self, img, draw=True):
        """
        Encuentra los puntos de referencia faciales en una imagen en formato BGR.
        :param img: Imagen en la que encontrar los puntos de referencia faciales.
        :param draw: Bandera para dibujar la salida en la imagen.
        :return: Imagen con o sin dibujos y los puntos de referencia faciales.
        """
        # Convierte la imagen de BGR a RGB.
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Procesa la imagen para encontrar los puntos de referencia faciales.
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    # Dibuja los puntos de referencia faciales en la imagen.
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                               self.drawSpec, self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])
                faces.append(face)
        return img, faces

class FaceDetector:

    """  Detector de Cara   """

    def __init__(self, minDetectionCon=0.5, modelSelection=0):
        """
        :param minDetectionCon: Minimum confidence value ([0.0, 1.0]) for face
        detection to be considered successful. See details in
        https://solutions.mediapipe.dev/face_detection#min_detection_confidence.

        :param modelSelection: 0 or 1. 0 to select a short-range model that works
        best for faces within 2 meters from the camera, and 1 for a full-range
        model best for faces within 5 meters. See details in
        https://solutions.mediapipe.dev/face_detection#model_selection.
        """
        self.minDetectionCon = minDetectionCon
        self.modelSelection = modelSelection
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(min_detection_confidence=self.minDetectionCon,
                                                                model_selection=self.modelSelection)

    def findFaces(self, img, draw=True):
        """
        Find faces in an image and return the bbox info
        :param img: Image to find the faces in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings.
                 Bounding Box list.
        """

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                if detection.score[0] > self.minDetectionCon:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, ic = img.shape
                    bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                        int(bboxC.width * iw), int(bboxC.height * ih)
                    cx, cy = bbox[0] + (bbox[2] // 2), \
                             bbox[1] + (bbox[3] // 2)
                    bboxInfo = {"id": id, "bbox": bbox, "score": detection.score, "center": (cx, cy)}
                    bboxs.append(bboxInfo)
                    if draw:
                        img = cv2.rectangle(img, bbox, (255, 0, 0), 2)

                        cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                                    (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                                    2, (255, 0, 0), 2)
        return img, bboxs

class Distancia:

    """ Distancia: Logitudes, Angulos, Grados, Area, Perimetro """
    
    def __init__():
        ()

    def findDistance(p1, p2, img=None, metric='chebyshev'):
        """
        Encuentra la distancia entre dos puntos de referencia basados en sus números de índice.
        :param p1: Punto 1.
        :param p2: Punto 2.
        :param img: Imagen para dibujar.
        :param distance_type: Tipo de distancia a calcular ('euclidean', 'manhattan', 'chebyshev').
        :return: Distancia entre los puntos, información de la línea y la imagen con el dibujo.
        """
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if metric == 'euclidean':
            # Calcula la distancia euclidiana entre los dos puntos.
            length = math.hypot(x2 - x1, y2 - y1)
        elif metric == 'manhattan':
            # Calcula la distancia Manhattan entre los dos puntos.
            length = abs(x2 - x1) + abs(y2 - y1)
        elif metric == 'chebyshev':
            # Calcula la distancia de Chebyshev entre los dos puntos.
            length = max(abs(x2 - x1), abs(y2 - y1))
        else:
            raise ValueError("Tipo de distancia no válido. Selecciona entre 'euclidean', 'manhattan' o 'chebyshev'.")

        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            # Dibuja los puntos y la línea entre ellos en la imagen.
            cv2.circle(img, (x1, y1), 3, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 3, (255, 0, 0), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
            cv2.circle(img, (cx, cy), 3, (255, 0, 0), cv2.FILLED)
            return length, info, img
        else:
            return length, info
    
    def findDistance_3(p1, p2, p3, img=None, metric='side_length'):
        """
        Calcula métricas geométricas entre tres puntos de referencia.
        :param p1: Punto 1.
        :param p2: Punto 2.
        :param p3: Punto 3.
        :param img: Imagen para dibujar (opcional).
        :param metric: Métrica a calcular ('side_length', 'angles', 'area', 'additional_properties').
        :return: Valor de la métrica calculada y opcionalmente información adicional y la imagen con el dibujo.
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        if metric == 'side_length':
            # Calcular las longitudes de los lados del triángulo
            length_ab = math.hypot(x2 - x1, y2 - y1)
            length_bc = math.hypot(x3 - x2, y3 - y2)
            length_ca = math.hypot(x1 - x3, y1 - y3)
            metric_value = (length_ab, length_bc, length_ca)

        elif metric == 'angles':
            # Calcular los ángulos entre los lados del triángulo
            ab = (x2 - x1, y2 - y1)
            bc = (x3 - x2, y3 - y2)
            ca = (x1 - x3, y1 - y3)

            def dot_product(v1, v2):
                return v1[0]*v2[0] + v1[1]*v2[1]

            def vector_length(v):
                return math.sqrt(v[0]**2 + v[1]**2)

            angle_ab_bc = math.acos(dot_product(ab, bc) / (vector_length(ab) * vector_length(bc)))
            angle_bc_ca = math.acos(dot_product(bc, ca) / (vector_length(bc) * vector_length(ca)))
            angle_ca_ab = math.acos(dot_product(ca, ab) / (vector_length(ca) * vector_length(ab)))

            # Convertir de radianes a grados
            angle_ab_bc_deg = math.degrees(angle_ab_bc)
            angle_bc_ca_deg = math.degrees(angle_bc_ca)
            angle_ca_ab_deg = math.degrees(angle_ca_ab)

            metric_value = (angle_ab_bc_deg, angle_bc_ca_deg, angle_ca_ab_deg)

        elif metric == 'area':
            # Calcular el área del triángulo usando la fórmula de determinante
            area = 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
            metric_value = area

        elif metric == 'perimeter':

            # Ejemplo: calcular el perímetro del triángulo
            length_ab = math.hypot(x2 - x1, y2 - y1)
            length_bc = math.hypot(x3 - x2, y3 - y2)
            length_ca = math.hypot(x1 - x3, y1 - y3)
            perimeter = length_ab + length_bc + length_ca
            metric_value = perimeter

        else:
            raise ValueError("Métrica no válida. Selecciona entre 'side_length', 'angles', 'area' o 'additional_properties'.")

        info = (x1, y1, x2, y2, x3, y3)
        if img is not None:
            # Dibujar los puntos y las líneas entre ellos en la imagen (si se proporciona)
            cv2.circle(img, (x1, y1), 3, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 3, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x3, y3), 3, (255, 0, 0), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
            cv2.line(img, (x2, y2), (x3, y3), (255, 255, 255), 1)
            cv2.line(img, (x3, y3), (x1, y1), (255, 255, 255), 1)

            return metric_value, info, img 
        else:
            return metric_value

class Cuadro:
    
    """ Cuadros: Partes de Cara """
    
    def __init__():
        ()  

    def drawNoseRectangle(img, face, margin_x=10, margin_y=40):
        """
        Dibuja un rectángulo alrededor de la nariz usando puntos específicos.
        :param img: Imagen en la que dibujar el rectángulo.
        :param face: Puntos de referencia faciales.
        :return: Imagen con el rectángulo dibujado.
        """
        # Puntos específicos de la nariz
        nose_points = [face[1], face[2], face[98], face[327]]
        
        # Coordenadas mínimas y máximas de la nariz
        x_min = min(point[0] for point in nose_points) - margin_x
        y_min = min(point[1] for point in nose_points) - margin_y
        x_max = max(point[0] for point in nose_points) + margin_x
        y_max = max(point[1] for point in nose_points)  
        
        # Dibujar el rectángulo
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 0), 2)  
        
        return img

    def drawEyeRectangles(img, face):
        """
        Dibuja rectángulos alrededor del ojo izquierdo y derecho usando puntos específicos.
        :param img: Imagen en la que dibujar los rectángulos.
        :param face: Puntos de referencia faciales.
        :return: Imagen con los rectángulos dibujados.
        """
        # Puntos específicos del ojo izquierdo y derecho
        left_eye_points = [face[159], face[23], face[133], face[33]]
        right_eye_points = [face[386], face[253], face[362], face[263]]
        
        # Coordenadas mínimas y máximas del ojo izquierdo
        left_x_min = min(point[0] for point in left_eye_points)
        left_y_min = min(point[1] for point in left_eye_points)
        left_x_max = max(point[0] for point in left_eye_points)
        left_y_max = max(point[1] for point in left_eye_points)
        
        # Coordenadas mínimas y máximas del ojo derecho
        right_x_min = min(point[0] for point in right_eye_points)
        right_y_min = min(point[1] for point in right_eye_points)
        right_x_max = max(point[0] for point in right_eye_points)
        right_y_max = max(point[1] for point in right_eye_points)
        
        # Dibujar los rectángulos
        cv2.rectangle(img, (left_x_min, left_y_min), (left_x_max, left_y_max), (0, 255, 0), 2)
        cv2.rectangle(img, (right_x_min, right_y_min), (right_x_max, right_y_max), (0, 255, 0), 2)
        
        return img

    def drawEyebrowRectangles(img, face, margin_x=20, margin_y=10):
        """
        Dibuja rectángulos alrededor de las cejas izquierda y derecha usando puntos específicos.
        :param img: Imagen en la que dibujar los rectángulos.
        :param face: Puntos de referencia faciales.
        :return: Imagen con los rectángulos dibujados.
        """
        # Puntos específicos de las cejas izquierda y derecha
        left_eyebrow_points = [face[55], face[65]]
        right_eyebrow_points = [face[285], face[295]]
        
        # Coordenadas mínimas y máximas de la ceja izquierda
        left_x_min = min(point[0] for point in left_eyebrow_points) - margin_x
        left_y_min = min(point[1] for point in left_eyebrow_points) - margin_y
        left_x_max = max(point[0] for point in left_eyebrow_points) + margin_x
        left_y_max = max(point[1] for point in left_eyebrow_points) + margin_y
        
        # Coordenadas mínimas y máximas de la ceja derecha
        right_x_min = min(point[0] for point in right_eyebrow_points) - margin_x
        right_y_min = min(point[1] for point in right_eyebrow_points) - margin_y
        right_x_max = max(point[0] for point in right_eyebrow_points) + margin_x
        right_y_max = max(point[1] for point in right_eyebrow_points) + margin_y
        
        # Dibujar los rectángulos
        cv2.rectangle(img, (left_x_min, left_y_min), (left_x_max, left_y_max), (255, 0, 0), 2)  # Azul
        cv2.rectangle(img, (right_x_min, right_y_min), (right_x_max, right_y_max), (255, 0, 0), 2)  # Rojo
        
        return img

    def drawEarRectangles(img, face, margin=20):
        """
        Dibuja rectángulos alrededor de las orejas izquierda y derecha usando puntos específicos.
        :param img: Imagen en la que dibujar los rectángulos.
        :param face: Puntos de referencia faciales.
        :return: Imagen con los rectángulos dibujados.
        """
        # Puntos específicos de las orejas izquierda y derecha
        left_ear_points = [face[234], face[93]]
        right_ear_points = [face[454], face[323]]
        
        # Coordenadas mínimas y máximas de la oreja izquierda
        left_x_min = min(point[0] for point in left_ear_points) - margin
        left_y_min = min(point[1] for point in left_ear_points) - margin
        left_x_max = max(point[0] for point in left_ear_points) + margin
        left_y_max = max(point[1] for point in left_ear_points) + margin
        
        # Coordenadas mínimas y máximas de la oreja derecha
        right_x_min = min(point[0] for point in right_ear_points) - margin
        right_y_min = min(point[1] for point in right_ear_points) - margin
        right_x_max = max(point[0] for point in right_ear_points) + margin
        right_y_max = max(point[1] for point in right_ear_points) + margin
        
        # Dibujar los rectángulos
        cv2.rectangle(img, (left_x_min, left_y_min), (left_x_max, left_y_max), (0, 0, 255), 2)  # Magenta
        cv2.rectangle(img, (right_x_min, right_y_min), (right_x_max, right_y_max), (0, 0, 255), 2)  # Celeste
        
        return img

    def drawMouthRectangle(img, face, margin_x=10, margin_y=15):
        """
        Dibuja un rectángulo alrededor de la boca usando puntos específicos.
        :param img: Imagen en la que dibujar el rectángulo.
        :param face: Puntos de referencia faciales.
        :return: Imagen con el rectángulo dibujado.
        """
        # Puntos específicos de la boca
        mouth_points = [face[61], face[291], face[13], face[14]]
        
        # Coordenadas mínimas y máximas de la boca
        mouth_x_min = min(point[0] for point in mouth_points) - margin_x
        mouth_y_min = min(point[1] for point in mouth_points) - margin_y
        mouth_x_max = max(point[0] for point in mouth_points) + margin_x
        mouth_y_max = max(point[1] for point in mouth_points) + margin_y
        
        # Dibujar el rectángulo
        cv2.rectangle(img, (mouth_x_min, mouth_y_min), (mouth_x_max, mouth_y_max), (255, 0, 255), 2)  # Magenta
        
        return img
    
    def drawJawRectangle(img, face, margin_x=40, margin_y=10):
        """
        Dibuja un rectángulo alrededor de la mandíbula usando puntos específicos.
        :param img: Imagen en la que dibujar el rectángulo.
        :param face: Puntos de referencia faciales.
        :return: Imagen con el rectángulo dibujado.
        """
        # Puntos específicos de la mandíbula
        jaw_points = [face[0], face[17], face[199]]
    
        # Coordenadas mínimas y máximas de la mandíbula
        x_min = min(point[0] for point in jaw_points) - margin_x
        y_min = min(point[1] for point in jaw_points) - margin_y
        x_max = max(point[0] for point in jaw_points) + margin_x
        y_max = max(point[1] for point in jaw_points) + margin_y
    
        # Dibujar el rectángulo
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 165, 0), 2)  # Color naranja
    
        return img

    def drawForeheadRectangle(img, face,margin_x=40, margin_y=40):
        """
        Dibuja un rectángulo alrededor de la frente usando puntos específicos.
        :param img: Imagen en la que dibujar el rectángulo.
        :param face: Puntos de referencia faciales.
        :return: Imagen con el rectángulo dibujado.
        """
        # Puntos específicos de la frente
        forehead_points = [face[10], face[67], face[297]]
    
        # Coordenadas mínimas y máximas de la frente
        x_min = min(point[0] for point in forehead_points) - margin_x 
        y_min = min(point[1] for point in forehead_points) - margin_y
        x_max = max(point[0] for point in forehead_points) + margin_x
        y_max = max(point[1] for point in forehead_points) + margin_y
    
        # Dibujar el rectángulo
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (75, 0, 130), 2)  # Color índigo
    
        return img

#if __name__ == "__main__":
#   main()
