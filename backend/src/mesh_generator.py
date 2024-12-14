import numpy as np
import open3d as o3d
from typing import List, Tuple

class MeshGenerator:
    def __init__(self):
        """
        Inicializa el generador de mallas 3D
        """
        self.vertices = []
        self.frames_points = []
        
    def add_contour_points(self, points: List[Tuple[float, float]], angle: float):
        """
        Añade puntos de contorno de un frame específico
        
        Args:
            points: Lista de puntos (x,y) del contorno
            angle: Ángulo de rotación del frame en radianes
        """
        # Convertir puntos 2D a 3D usando el ángulo de rotación
        frame_points = []
        for x, y in points:
            # Calcular coordenadas 3D
            x3d = x * np.cos(angle)
            y3d = x * np.sin(angle)
            z3d = y
            frame_points.append([x3d, y3d, z3d])
        self.frames_points.append(frame_points)
            
    def generate_mesh(self) -> o3d.geometry.TriangleMesh:
        """
        Genera la malla 3D conectando los puntos entre frames adyacentes
        
        Returns:
            Malla triangular de Open3D
        """
        vertices = []
        triangles = []
        vertex_index = 0
        
        # Procesar cada par de frames consecutivos
        for i in range(len(self.frames_points)):
            current_frame = self.frames_points[i]
            next_frame = self.frames_points[(i + 1) % len(self.frames_points)]
            
            # Asegurar que ambos frames tengan el mismo número de puntos
            min_points = min(len(current_frame), len(next_frame))
            current_frame = current_frame[:min_points]
            next_frame = next_frame[:min_points]
            
            # Añadir vértices de ambos frames
            vertices.extend(current_frame)
            
            # Crear triángulos entre frames
            for j in range(min_points - 1):
                # Índices para el cuadrilátero entre frames
                v1 = vertex_index + j
                v2 = vertex_index + j + 1
                v3 = vertex_index + min_points + j
                v4 = vertex_index + min_points + j + 1
                
                # Crear dos triángulos
                triangles.append([v1, v2, v3])
                triangles.append([v2, v4, v3])
                
            vertex_index += min_points
            
        # Añadir últimos vértices
        vertices.extend(self.frames_points[0])
        
        # Crear malla con Open3D
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices))
        mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
        
        # Optimizar la malla
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.compute_vertex_normals()
        
        return mesh
        
    def clean(self):
        """
        Limpia los datos de la malla actual
        """
        self.vertices = []
        self.frames_points = []
