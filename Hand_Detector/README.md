### HAND DETECTOR SISTEMA

### DESCRIPCION DE CLASE

### HandDetector
Diseñada para detectar y rastrear manos utilizando la biblioteca MediaPipe. Proporciona varias funcionalidades como obtener la lista de puntos de referencia de la mano, dibujar las manos en la imagen, contar cuántos dedos están levantados y calcular la distancia entre dos puntos

- [findHands()]: Detecta y rastrea manos en una imagen proporcionada.

- [fingersUp()]: Determina cuántos dedos están levantados en una mano.

- [ffindDistance()]: Calcula la distancia entre dos puntos específicos en la mano y, opcionalmente, dibuja esta información en una imagen.


### .py

 - [main.py]: Detecta manos en tiempo real a través de la cámara web. La detección de manos incluye el seguimiento de 21 puntos de referencia en cada mano, la identificación de qué cuantos dedos están levantados y el cálculo de distancias entre puntos específicos en las manos.

- [main2.py]: Detecta manos en un video en tiempo real desde la cámara web y permite dibujar utilizando los puntos de referencia de las manos detectadas.

- [Face_Detector]: Detectar y analizar manos en imágenes en tiempo real desde una cámara. Proporciona funciones para identificar puntos de referencia clave en las manos detectadas, contar dedos levantados y calcular distancias entre puntos específicos.


