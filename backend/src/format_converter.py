import trimesh
import numpy as np
from pathlib import Path

class ModelConverter:
    @staticmethod
    def obj_to_glb(obj_path: str, glb_path: str):
        """
        Convierte un archivo OBJ a GLB
        
        Args:
            obj_path: Ruta al archivo OBJ
            glb_path: Ruta donde guardar el archivo GLB
        """
        # Cargar la malla desde OBJ
        mesh = trimesh.load(obj_path)
        
        # Asegurarse de que la malla esté centrada y escalada apropiadamente
        mesh.vertices -= mesh.center_mass
        scale = 1.0 / np.max(np.abs(mesh.vertices))
        mesh.vertices *= scale
        
        # Exportar como GLB
        mesh.export(glb_path)
