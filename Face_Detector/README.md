### FACE DETECTOR SISTEMA

### DESCRIPCION DE CLASES

### FaceMeshDetector
Utiliza la biblioteca MediaPipe para detectar mallas faciales, proporcionando una estructura detallada de puntos de referencia en la cara. Permite la detección y dibujo de estos puntos en una imagen, facilitando el análisis facial detallado.

- [findFaceMesh()]: Encuentra los puntos de referencia faciales en una imagen y opcionalmente los dibuja.

### FaceDetector
Utiliza la biblioteca MediaPipe para detectar caras en una imagen, proporcionando la información de las cajas delimitadoras (bounding boxes) y las coordenadas de los centros de las caras detectadas.

- [findFaces()]: Encuentra caras en una imagen y devuelve la información de las cajas delimitadoras.

### Distancia
Diseñada para calcular diversas métricas geométricas entre puntos, incluyendo distancias (euclidiana, Manhattan, Chebyshev), longitudes de lados de triángulos, ángulos entre lados, área y perímetro de triángulos.

- [findDistance()]: Encuentra la distancia entre dos puntos de referencia basados en sus coordenadas (x, y).

- [findDistance_3()]: Calcula métricas geométricas entre tres puntos de referencia.

### Cuadro
Diseñada para dibujar rectángulos alrededor de diferentes partes de la cara (nariz, ojos, cejas, orejas, boca, mandíbula y frente) utilizando puntos específicos de referencia facial.

- [drawNoseRectangle()]: Dibuja un rectángulo alrededor de la nariz utilizando puntos específicos de referencia facial.

- [drawEyeRectangle()]: Dibuja rectángulos alrededor de los ojos izquierdo y derecho utilizando puntos específicos de referencia facial.

- [drawEyebrowRectangle()]: Dibuja rectángulos alrededor de las cejas izquierda y derecha utilizando puntos específicos de referencia facial.

- [drawEarRectangle()]: Dibuja rectángulos alrededor de las orejas izquierda y derecha utilizando puntos específicos de referencia facial.

- [drawMouthRectangle()]: Dibuja un rectángulo alrededor de la boca utilizando puntos específicos de referencia facial.

- [drawJawRectangle()]: Dibuja un rectángulo alrededor de la mandíbula utilizando puntos específicos de referencia facial.

- [drawForeheadRectangle()]: Dibuja un rectángulo alrededor de la frente utilizando puntos específicos de referencia facial.

### .py

 - [F_D.py]:  Este proyecto utiliza OpenCV para detectar rostros en tiempo real desde la cámara de la computadora. Detecta automáticamente los rostros y muestra el número de rostros detectados junto con sus coordenadas en pantalla. Permite al usuario guardar imágenes con los rostros detectados al presionar la tecla 's'. El programa se detiene con la tecla 'ESC' y libera los recursos de la cámara al finalizar.

 - [F_D_Data.py]: Este proyecto utiliza OpenCV para detectar rostros en tiempo real desde la cámara de la computadora. Muestra la imagen principal con los rostros detectados y dos ventanas adicionales: una con el rostro detectado en su resolución original y otra con el mismo rostro redimensionado a 50x50 píxeles.

 - [main.py]: Este proyecto utiliza OpenCV y un detector de malla facial para analizar imágenes de video en tiempo real desde la cámara de la computadora. Detecta múltiples caras y calcula diversas métricas geométricas como distancias y dibuja rectángulos alrededor de diferentes partes del rostro (ojos, cejas, orejas, nariz, boca, mandíbula y frente). Muestra las métricas calculadas y el número de caras detectadas en tiempo real.

 - [Face_Detector]: Proporciona funcionalidades avanzadas para la detección y análisis de rostros utilizando Python y OpenCV. Incluye métodos para localizar y delimitar rostros en imágenes o videos en tiempo real, así como para identificar puntos clave como ojos, nariz y boca dentro de los rostros detectados. Este módulo está diseñado para integrarse con la biblioteca OpenCV (cv2) y permite ajustar parámetros como la confianza mínima de detección y el número máximo de rostros a identificar, facilitando aplicaciones de visión por computadora que requieren análisis facial preciso y eficiente.


