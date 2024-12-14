from rembg import remove, new_session
from pathlib import Path
from PIL import Image
import multiprocessing
import os
import numpy as np
import logging
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from functools import partial
import threading
from scipy.spatial import Delaunay
import math
from typing import List, Tuple, Optional, Dict, Any
import trimesh
import json

# Configuración segura de multiprocessing
multiprocessing.set_start_method('fork', force=True)

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Semáforo global para controlar concurrencia
_thread_lock = threading.Lock()

def clear_dir(path: Path) -> None:
    """
    Limpia todos los archivos de un directorio
    
    Args:
        path (Path): Ruta del directorio a limpiar
    """
    try:
        for archivo in path.iterdir():
            if archivo.is_file():
                archivo.unlink()
        logger.info(f"Directorio {path} limpiado exitosamente")
    except Exception as e:
        logger.error(f"Error al limpiar el directorio {path}: {str(e)}")
        raise

def process_single_image(input_file: Path, output_path: Path, session):
    """
    Procesa una sola imagen de manera segura con control de concurrencia.
    """
    try:
        # Usar un lock para operaciones críticas
        with threading.Lock():
            input_frame = Image.open(input_file)
            output_frame = remove(input_frame, session=session, alpha_matting=False, post_process_mask=False)
            
            # Convertir la imagen de PIL a numpy array
            output_frame_np = np.array(output_frame)

            # Verificar si la imagen tiene un canal alfa
            if output_frame_np.shape[2] == 4:
                # Separar los canales
                b, g, r, a = cv2.split(output_frame_np)

                # Aplicar un umbral más agresivo al canal alfa
                _, a_thresh = cv2.threshold(a, 127, 255, cv2.THRESH_BINARY)

                # Crear kernel para operaciones morfológicas
                kernel = np.ones((3,3), np.uint8)
                
                # Aplicar operaciones morfológicas para limpiar bordes
                a_clean = cv2.morphologyEx(a_thresh, cv2.MORPH_CLOSE, kernel)
                a_clean = cv2.morphologyEx(a_clean, cv2.MORPH_OPEN, kernel)

                # Crear una imagen de fondo blanco
                output_image = np.ones_like(output_frame_np, dtype=np.uint8) * 255

                # Aplicar la máscara limpia
                visible_mask = a_clean > 0
                
                # Crear la silueta: negro sobre blanco
                output_image[visible_mask] = [0, 0, 0, 255]  # Silueta negra
                output_image[~visible_mask] = [255, 255, 255, 255]  # Fondo blanco

                # Convertir a imagen PIL y guardar
                output_file = output_path / f"{input_file.stem}.png"
                Image.fromarray(output_image).save(output_file, format='PNG')
                logger.info(f"Procesada imagen: {input_file.name}")
                return True
            else:
                logger.error(f"La imagen procesada no tiene canal alfa: {input_file}")
                return False
    except Exception as e:
        logger.error(f"Error procesando {input_file}: {str(e)}")
        return False

def remove_bg(input_path: Path, output_path: Path) -> bool:
    """
    Remueve el fondo de todas las imágenes en el directorio de entrada usando threads
    """
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Obtener lista de archivos a procesar
        image_files = [
            f for f in input_path.iterdir() 
            if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ]
        
        if not image_files:
            logger.warning("No se encontraron imágenes para procesar")
            return False

        # Usar threads en lugar de procesos
        num_threads = min(32, len(image_files))  # Limitamos a 32 threads máximo
        logger.info(f"Procesando {len(image_files)} imágenes usando {num_threads} threads")
        
        # Crear una única sesión de rembg
        session = new_session("u2net_human_seg")
        
        # Procesar imágenes con manejo de errores mejorado
        results = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Usar as_completed para manejar resultados a medida que se completan
            futures = {
                executor.submit(process_single_image, f, output_path, session): f 
                for f in image_files
            }
            
            for future in as_completed(futures):
                original_file = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error procesando {original_file}: {str(e)}")
                    results.append(False)
        
        # Verificar resultados
        success_count = sum(1 for r in results if r)
        logger.info(f"Procesadas {success_count} de {len(image_files)} imágenes")
        
        return success_count > 0
        
    except Exception as e:
        logger.error(f"Error al remover fondo: {str(e)}")
        return False
    finally:
        # Limpiar recursos de manera segura
        try:
            multiprocessing.util.Finalize(None, multiprocessing.util._run_finalizers, exitpriority=1)
        except Exception:
            pass

def convert_to_silhouette(input_path: Path):
    """
    Convierte una imagen con fondo transparente a una silueta negra.

    Args:
        input_image_path (Path): Ruta de la imagen de entrada (con fondo transparente).
        output_image_path (Path): Ruta para guardar la imagen resultante con la silueta negra.
    """

    for path in input_path.iterdir():

        # Leer la imagen incluyendo el canal alfa
        frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        if frame is None:
            raise FileNotFoundError(f"No se pudo encontrar la imagen: {path}")

        # Verificar si la imagen tiene un canal alfa
        if frame.shape[2] != 4:
            raise ValueError("La imagen no tiene un canal alfa.")

        # Separar los canales
        b, g, r, a = cv2.split(frame)

        # Crear una máscara de las áreas visibles (donde alfa no es 0)
        visible_mask = a > 0

        # Crear una imagen de fondo blanco (del mismo tamaño que la imagen original)
        output_image = np.ones_like(frame, dtype=np.uint8) * 255 

        # Convertir las áreas visibles a negro (puedes elegir cualquier color)
        output_image[:, :, 0] = visible_mask * 0  # Canal B
        output_image[:, :, 1] = visible_mask * 0  # Canal G
        output_image[:, :, 2] = visible_mask * 0  # Canal R
        output_image[:, :, 3] = a  # Mantener el canal alfa original

        # Guardar la imagen resultante
        cv2.imwrite(str(path), output_image)


def extract_silhouette_contours(input_path: Path) -> List[np.ndarray]:
    """
    Extrae los puntos x,y del contorno de siluetas negras sobre fondo blanco
    
    Args:
        input_path (Path): Directorio con imágenes de siluetas
    
    Returns:
        List[np.ndarray]: Lista de contornos, donde cada contorno es un array de puntos (x,y)
    """
    all_contours = []
    
    # Obtener archivos de imagen ordenados por número de frame
    image_files = sorted(
        [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']],
        key=lambda x: int(x.stem.split('-')[-1])  # Extrae el número después de 'frame_'
    )
    
    for i, image_file in enumerate(image_files):
        # Leer imagen en modo binario (negro y blanco)
        image = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
        
        # Umbralizar para asegurar imagen binaria (negro y blanco)
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Encontrar contornos externos
        contours, _ = cv2.findContours(
            binary, 
            cv2.RETR_EXTERNAL,  # Solo contornos externos
            cv2.CHAIN_APPROX_SIMPLE  # Comprimir puntos de contorno
        )
        
        # Seleccionar el contorno más grande (presumiblemente el objeto)
        if contours:
            # Ordenar contornos por área en orden descendente
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Tomar el contorno más grande
            main_contour = sorted_contours[0]
            
            # Extraer puntos x,y del contorno
            contour_points = main_contour.reshape(-1, 2)
            print(f"Procesando contorno {i} de frame_{i:03d}: {len(contour_points)} puntos")
            
            all_contours.append(contour_points)

            # Guardar contorno visualizado
            black_canvas = np.zeros_like(image)
            cv2.drawContours(black_canvas, [main_contour], -1, (255, 255, 255), 2)
            output_file = Path("scan") / f"contour_{i:03d}.png"
            cv2.imwrite(str(output_file), black_canvas)
            
            logger.info(f"Contorno extraído de {image_file.name}")
        else:
            logger.warning(f"No se encontró contorno en {image_file.name}")
            # Añadir un contorno vacío para mantener la secuencia
            all_contours.append(np.array([]))
    
    return all_contours

def process_contour_points(contours: List[np.ndarray]) -> Dict[str, Any]:
    """
    Procesa los puntos de contorno para preparar datos 3D
    
    Args:
        contours (List[np.ndarray]): Lista de contornos con puntos x,y
    
    Returns:
        Dict[str, Any]: Diccionario con información procesada de los contornos
    """
    # Verificar que haya contornos
    if not contours:
        logger.error("No se encontraron contornos")
        return {}
    
    # Calcular estadísticas de los contornos
    contour_stats = {
        'total_contours': len(contours),
        'points_per_contour': [len(contour) for contour in contours],
        'contour_points': [contour.tolist() for contour in contours]  # Convertir a lista
    }
    
    # Calcular centroide de cada contorno
    centroids = []
    for contour in contours:
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centroids.append((cx, cy))
    
    contour_stats['centroids'] = centroids
    
    return contour_stats

def convert_contours_to_3d(contours: List[List[List[int]]], rotation_step: float = 3.0, radius_scale: float = 1.0) -> List[List[Tuple[float, float, float]]]:
    """
    Convierte contornos 2D a puntos 3D considerando rotación y espaciado
    
    Args:
        contours (List[List[List[int]]]): Lista de contornos con puntos [x, y]
        rotation_step (float): Ángulo de rotación entre contornos (grados)
        radius_scale (float): Factor de escala para el radio
    
    Returns:
        List[List[Tuple[float, float, float]]]: Puntos 3D de los contornos
    """
    contour_points_3d = []
    
    # Calcular centro promedio de todos los contornos
    all_x = []
    all_y = []
    for contour in contours:
        for point in contour:
            try:
                x, y = float(point[0]), float(point[1])
                all_x.append(x)
                all_y.append(y)
            except (IndexError, ValueError, TypeError):
                continue
    
    if not all_x or not all_y:
        logger.error("No se pudieron extraer puntos válidos")
        return []
    
    center_x = np.mean(all_x)
    center_y = np.mean(all_y)
    
    logger.info(f"Centro del objeto: ({center_x}, {center_y})")
    
    for idx, contour in enumerate(contours):
        # Filtrar y convertir puntos a float
        filtered_contour = []
        for point in contour:
            try:
                if len(point) >= 2:
                    x, y = float(point[0]), float(point[1])
                    # Centrar los puntos
                    x -= center_x
                    y -= center_y
                    filtered_contour.append([x, y])
            except (IndexError, ValueError, TypeError) as e:
                logger.warning(f"Punto de contorno inválido: {point}. Error: {e}")
        
        # Saltar contornos vacíos o con pocos puntos
        if len(filtered_contour) < 3:
            logger.warning(f"Contorno {idx} tiene insuficientes puntos: {len(filtered_contour)}")
            continue
        
        # Calcular ángulo de rotación
        angle = math.radians(idx * rotation_step)
        
        # Convertir puntos 2D a 3D
        points_3d = []
        for point in filtered_contour:
            x, y = point
            
            # Rotar puntos alrededor del eje Y
            x_3d = x * math.cos(angle)  # X se convierte en X*cos(θ)
            y_3d = y  # Y permanece igual
            z_3d = x * math.sin(angle)  # X contribuye a Z con sin(θ)
            
            # Aplicar escala
            x_3d *= radius_scale
            z_3d *= radius_scale
            
            points_3d.append((x_3d, y_3d, z_3d))
        
        contour_points_3d.append(points_3d)
    
    return contour_points_3d

def connect_vertical_points(contour_points_3d: List[List[Tuple[float, float, float]]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Conecta puntos verticalmente entre contornos adyacentes para crear volumen
    
    Args:
        contour_points_3d (List[List[Tuple[float, float, float]]]): Puntos 3D de contornos
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Vértices y caras de la malla 3D
    """
    # Verificar que haya al menos dos contornos
    if len(contour_points_3d) < 2:
        logger.warning("Se requieren al menos dos contornos para conexión vertical")
        return np.array([], dtype=np.float32), np.array([], dtype=np.int32)
    
    # Preparar listas de vértices y caras
    vertices = []
    faces = []
    
    # Interpolar contornos para asegurar mismo número de puntos
    def interpolate_contour(contour, target_points):
        try:
            from scipy.interpolate import interp1d
            
            # Extraer coordenadas
            x, y, z = zip(*contour)
            
            # Crear funciones de interpolación
            t_orig = np.linspace(0, 1, len(x))
            t_new = np.linspace(0, 1, target_points)
            
            f_x = interp1d(t_orig, x, kind='linear')
            f_y = interp1d(t_orig, y, kind='linear')
            f_z = interp1d(t_orig, z, kind='linear')
            
            # Interpolar puntos
            new_x = f_x(t_new)
            new_y = f_y(t_new)
            new_z = f_z(t_new)
            
            return [(float(x), float(y), float(z)) for x, y, z in zip(new_x, new_y, new_z)]
        except Exception as e:
            logger.error(f"Error en interpolación: {e}")
            return None
    
    # Determinar número de puntos para interpolación (usar un valor razonable)
    target_points = 100  # Número fijo de puntos por contorno
    
    # Interpolar todos los contornos al mismo número de puntos
    interpolated_contours = []
    for contour in contour_points_3d:
        interpolated = interpolate_contour(contour, target_points)
        if interpolated is not None:
            interpolated_contours.append(interpolated)
    
    # Verificar que tengamos contornos interpolados
    if len(interpolated_contours) < 2:
        logger.error("No hay suficientes contornos interpolados")
        return np.array([], dtype=np.float32), np.array([], dtype=np.int32)
    
    # Generar vértices y caras triangulares
    for i in range(len(interpolated_contours) - 1):
        contour1 = interpolated_contours[i]
        contour2 = interpolated_contours[i + 1]
        
        # Añadir vértices de ambos contornos
        start_idx = len(vertices)
        vertices.extend(contour1)
        vertices.extend(contour2)
        
        # Generar caras triangulares entre contornos
        for j in range(target_points):
            next_j = (j + 1) % target_points
            
            # Índices de los vértices
            v1 = start_idx + j
            v2 = start_idx + target_points + j
            v3 = start_idx + target_points + next_j
            v4 = start_idx + next_j
            
            # Crear dos triángulos para formar un quad
            faces.append([v1, v2, v3])  # Primer triángulo
            faces.append([v1, v3, v4])  # Segundo triángulo
    
    # Generar tapas superior e inferior con triángulos
    def create_cap(contour_points, is_top=False):
        # Calcular centroide
        centroid_x = np.mean([p[0] for p in contour_points])
        centroid_y = np.mean([p[1] for p in contour_points])
        centroid_z = contour_points[0][2]
        
        # Añadir centroide a vértices
        centroid_idx = len(vertices)
        vertices.append((float(centroid_x), float(centroid_y), float(centroid_z)))
        
        # Crear triángulos para la tapa
        start_idx = len(vertices) - len(contour_points) - 1
        for j in range(len(contour_points)):
            next_j = (j + 1) % len(contour_points)
            if is_top:
                faces.append([centroid_idx, start_idx + j, start_idx + next_j])
            else:
                faces.append([centroid_idx, start_idx + next_j, start_idx + j])
    
    # Crear tapas
    create_cap(interpolated_contours[0], is_top=False)
    create_cap(interpolated_contours[-1], is_top=True)
    
    try:
        # Convertir a numpy arrays asegurando tipos de datos consistentes
        vertices_array = np.array(vertices, dtype=np.float32)
        faces_array = np.array(faces, dtype=np.int32)
        
        # Verificar formas
        logger.info(f"Forma de vértices: {vertices_array.shape}")
        logger.info(f"Forma de caras: {faces_array.shape}")
        
        return vertices_array, faces_array
    except Exception as e:
        logger.error(f"Error al convertir arrays: {e}")
        return np.array([], dtype=np.float32), np.array([], dtype=np.int32)

def generate_3d_mesh(input_path: Path, output_path: Path, rotation_step: float = 3.0, radius_scale: float = 1.0) -> Optional[Path]:
    """
    Genera malla 3D a partir de contornos de siluetas
    
    Args:
        input_path (Path): Directorio con imágenes de entrada
        output_path (Path): Directorio de salida
        rotation_step (float): Ángulo de rotación entre contornos (grados)
        radius_scale (float): Factor de escala para el radio
    
    Returns:
        Optional[Path]: Ruta del archivo de malla generado, o None si falla
    """
    try:
        # 1. Extraer contornos de siluetas
        silhouette_contours = extract_silhouette_contours(input_path)
        
        # Registrar número de contornos extraídos
        logger.info(f"Número de contornos extraídos: {len(silhouette_contours)}")
        
        # 2. Convertir contornos a puntos 3D
        contour_points_3d = convert_contours_to_3d(
            silhouette_contours, 
            rotation_step=rotation_step,
            radius_scale=radius_scale
        )
        
        # Registrar número de contornos 3D
        logger.info(f"Número de contornos 3D: {len(contour_points_3d)}")
        
        # 3. Conectar puntos verticalmente
        vertices, faces = connect_vertical_points(contour_points_3d)
        
        # Registrar información de vértices y caras
        logger.info(f"Número de vértices: {len(vertices)}")
        logger.info(f"Número de caras: {len(faces)}")
        
        # 4. Crear malla con trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # 5. Asignar el color #252525 (gris oscuro) a las caras de la malla
        mesh.visual.face_colors = np.array([37, 37, 37, 255])  # Gris oscuro en formato RGBA

        
        # 6. Preparar ruta de salida
        output_path.mkdir(parents=True, exist_ok=True)
        glb_path = output_path / "scanned_object.glb"
        obj_path = output_path / "scanned_object.obj"
        
        # 7. Exportar modelo
        mesh.export(str(glb_path), file_type='glb')
        mesh.export(str(obj_path), file_type='obj')
        
        logger.info(f"Modelo 3D generado en {glb_path}")
        return glb_path
    
    except Exception as e:
        logger.error(f"Error al generar modelo 3D: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def get_points_cloud(input_path: Path, output_path: Path) -> bool:
    """
    Obtiene la nube de puntos y genera modelo 3D
    
    Args:
        input_path (Path): Directorio con imágenes de entrada
        output_path (Path): Directorio de salida
    
    Returns:
        bool: True si el procesamiento fue exitoso, False en caso contrario
    """
    try:
        # 1. Extraer contornos de siluetas
        silhouette_contours = extract_silhouette_contours(input_path)
        
        # 2. Procesar puntos de contorno
        contour_data = process_contour_points(silhouette_contours)
        
        # 3. Guardar datos de contorno
        output_path.mkdir(parents=True, exist_ok=True)
        contour_file = output_path / "contour_points.json"
        
        with open(contour_file, 'w') as f:
            json.dump(contour_data, f, indent=2)
        
        # 4. Generar modelo 3D
        glb_model = generate_3d_mesh(input_path, output_path / "3d_models")
        
        return glb_model is not None
    
    except Exception as e:
        logger.error(f"Error en get_points_cloud: {str(e)}")
        return False