import numpy as np
import open3d as o3d
from typing import List, Tuple

class PointCloudProcessor:
    def __init__(self):
        """
        Inicializa el procesador de nube de puntos
        """
        pass

    def create_point_cloud(self, points: List[Tuple[float, float, float]]) -> o3d.geometry.PointCloud:
        """
        Crea una nube de puntos a partir de una lista de puntos
        
        Args:
            points: Lista de puntos 3D
            
        Returns:
            Objeto PointCloud de Open3D
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        return pcd

    def clean_point_cloud(self, pcd: o3d.geometry.PointCloud, 
                         nb_neighbors: int = 30,  
                         std_ratio: float = 1.5) -> o3d.geometry.PointCloud:  
        """
        Limpia la nube de puntos eliminando outliers y mejorando la calidad
        
        Args:
            pcd: Nube de puntos
            nb_neighbors: Número de vecinos a considerar
            std_ratio: Ratio de desviación estándar
            
        Returns:
            Nube de puntos limpia
        """
        # Eliminar outliers estadísticos
        cleaned_pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                                      std_ratio=std_ratio)
        
        # Aplicar voxel downsampling para uniformizar la densidad de puntos
        voxel_size = 0.005  
        cleaned_pcd = cleaned_pcd.voxel_down_sample(voxel_size)
        
        # Estimar normales con parámetros optimizados
        cleaned_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.05, max_nn=50))  
        
        # Orientar normales hacia el exterior con más iteraciones
        cleaned_pcd.orient_normals_consistent_tangent_plane(200)
        
        return cleaned_pcd

    def create_mesh(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
        """
        Crea una malla de alta calidad a partir de la nube de puntos
        
        Args:
            pcd: Nube de puntos procesada
            
        Returns:
            Malla triangular optimizada
        """
        # Reconstrucción de superficie usando Poisson con mayor profundidad
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=10, width=0, scale=1.1, linear_fit=True)
        
        # Recortar triángulos con baja densidad usando un umbral adaptativo
        density_threshold = np.quantile(densities, 0.05)  
        vertices_to_remove = densities < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        # Optimizar malla con múltiples iteraciones de suavizado
        mesh.filter_smooth_taubin(number_of_iterations=50)
        
        # Simplificar la malla manteniendo la calidad
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=100000)
        
        # Reparar la malla
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        return mesh

    def process_points(self, points: List[Tuple[float, float, float]]) -> o3d.geometry.TriangleMesh:
        """
        Procesa la lista de puntos completa para generar una malla
        
        Args:
            points: Lista de puntos 3D
            
        Returns:
            Malla triangular final
        """
        # Crear nube de puntos
        pcd = self.create_point_cloud(points)
        
        # Limpiar nube de puntos
        cleaned_pcd = self.clean_point_cloud(pcd)
        
        # Crear malla
        mesh = self.create_mesh(cleaned_pcd)
        
        return mesh

    def save_mesh(self, mesh: o3d.geometry.TriangleMesh, output_path: str):
        """
        Guarda la malla en formato OBJ
        
        Args:
            mesh: Malla triangular
            output_path: Ruta donde guardar el archivo
        """
        o3d.io.write_triangle_mesh(output_path, mesh)
