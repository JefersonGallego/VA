### Depth Inference System
Este proyecto utiliza un modelo ONNX para realizar inferencias de profundidad en tiempo real a partir de cuadros de video capturados por una cámara. El sistema consta de tres módulos principales: inference.py, transform.py y main.py.a.

### DESCRIPCION DE MODULOS

### inference.py
Este módulo se encarga de realizar las inferencias de profundidad utilizando un modelo ONNX.

* Funciones:

- [load_depth_model()]: Carga el modelo ONNX especificado.
- [preprocess_depth_output()]: Preprocesa la salida del modelo para generar un mapa de profundidad escalado a 255.
- [visualize_depth_color()]: Aplica un colormap a la imagen de profundidad para visualización.


* Clase DepthAnythingInference:

- [__init__()]: Constructor que inicializa el modelo y determina si se aplicará coloración.

- [frame_inference()]: Prepara la imagen, ejecuta el modelo para obtener la predicción de profundidad y procesa la salida.

### transform.py
Este módulo contiene varias transformaciones que se aplican a las imágenes para preparar los datos antes de ingresarlos a un modelo de red neuronal.

* Clases de Transformación:

- [Resize]: Redimensiona la imagen al tamaño especificado, opcionalmente manteniendo la relación de aspecto.
- [NormalizeImage]: Normaliza la imagen con media y desviación estándar específicas.
- [PrepareForNet]: Prepara la imagen para su uso como entrada en una red neuronal, cambiando el orden de los ejes y asegurando que los datos sean contiguos.

* Funciones:

- [load_image()]: Carga y transforma una imagen desde un archivo.
- [load_frameload_frame_resize]: Redimensiona un cuadro de video al tamaño objetivo y lo prepara para la inferencia.

### main.py
Este módulo controla el flujo principal del programa, capturando cuadros de video en tiempo real, aplicando el modelo de inferencia de profundidad, y mostrando tanto el cuadro original como el mapa de profundidad generado.

### depth_anything_vits14.onx
Modelo de estimacion

