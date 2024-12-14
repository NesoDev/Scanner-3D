# Scanner 3D

Este proyecto implementa un escáner 3D utilizando un ESP32-CAM y una base giratoria controlada por un motor paso a paso. El sistema captura un video del objeto mientras gira y procesa las imágenes para crear un modelo 3D.

## Estructura del Proyecto

- `/backend`: Código Python para procesamiento de imágenes y generación del modelo 3D
  - `video_processor.py`: Procesamiento de video y detección de contornos
  - `point_cloud.py`: Generación y limpieza de nube de puntos
  - `mesh_generator.py`: Generación de malla 3D
  - `format_converter.py`: Conversión de OBJ a GLB
  - `server.py`: Servidor web para comunicación con el frontend

- `/frontend`: Interfaz web para visualización del modelo 3D
  - `index.html`: Página principal
  - `styles.css`: Estilos CSS
  - `main.js`: Lógica de visualización usando Three.js

- `/esp32`: Código Arduino para el ESP32-CAM y control del motor

## Requisitos

- Python 3.8+
- OpenCV
- NumPy
- Trimesh
- Flask
- Three.js

## Instalación

1. Instalar dependencias de Python:
```bash
pip install -r requirements.txt
```

2. Cargar el código en el ESP32-CAM usando Arduino IDE

3. Abrir el frontend en un navegador web
