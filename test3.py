from app.mesh.PoissonMeshOpen3D import PoissonMeshOpen3DABC
from app.scan.Scan import Scan

scan = Scan(scan_name="TestScan")
scan.import_points_from_file(file_path="src/Камеры/data_3.dxf")

mesh = PoissonMeshOpen3DABC(depth=14,
                            width=0,
                            scale=1.1).init_mesh_from_scan(scan)
print(mesh)
# mesh.trim_by_bpa(scale=1)
# mesh.plot()

mesh = mesh.mesh
import open3d as o3d
import numpy as np

# Создание или загрузка меша


# Создание воксельной модели из меша
voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(
    mesh,
    voxel_size=0.5
)

# Визуализация
# o3d.visualization.draw_geometries([voxel_grid])

import pyvista as pv


def o3d_mesh_to_pyvista(o3d_mesh):
    """Конвертация Open3D TriangleMesh в PyVista PolyData"""

    # Получение вершин и треугольников
    vertices = np.asarray(o3d_mesh.vertices)
    triangles = np.asarray(o3d_mesh.triangles)

    # Создание PyVista mesh
    faces = np.insert(triangles, 0, 3, axis=1).flatten()
    pyvista_mesh = pv.PolyData(vertices, faces)

    # Добавление цветов если есть
    if o3d_mesh.has_vertex_colors():
        colors = np.asarray(o3d_mesh.vertex_colors)
        pyvista_mesh['colors'] = colors

    return pyvista_mesh


# pyvista_mesh = o3d_mesh_to_pyvista(mesh)
# plotter = pv.Plotter()
# plotter.add_mesh(pyvista_mesh, color='lightblue', show_edges=True)
# plotter.show()

import open3d as o3d
import pyvista as pv
import numpy as np


# Альтернативный способ - создание вокселей как кубов
def o3d_voxelgrid_to_pyvista_cubes(o3d_voxel_grid):
    """Создание вокселей как отдельных кубов"""

    voxels = list(o3d_voxel_grid.get_voxels())

    if not voxels:
        return None

    # Создаем объединенный mesh из всех вокселей
    combined_mesh = pv.PolyData()
    voxel_size = o3d_voxel_grid.voxel_size

    for i, voxel in enumerate(voxels):
        # Создаем куб для каждого вокселя
        center = o3d_voxel_grid.get_voxel_center_coordinate(voxel.grid_index)

        # Создаем куб centered в данной позиции
        cube = pv.Cube(center=center, x_length=voxel_size,
                       y_length=voxel_size, z_length=voxel_size)

        if i == 0:
            combined_mesh = cube
        else:
            combined_mesh = combined_mesh.merge(cube)

    return combined_mesh

# Пример использования
# o3d_mesh = o3d.geometry.TriangleMesh.create_sphere()
pyvista_mesh = o3d_voxelgrid_to_pyvista_cubes(voxel_grid)

# Визуализация в PyVista
plotter = pv.Plotter()
plotter.add_mesh(pyvista_mesh, color='lightblue', show_edges=True)
plotter.show()