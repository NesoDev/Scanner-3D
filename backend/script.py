import trimesh

# Carga el archivo .obj
mesh = trimesh.load('scan/3d_models/scanned_object.obj')

# Renderiza en una ventana interactiva
mesh.show()