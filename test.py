import open3d as o3d
import numpy as np

from app.scan.Scan import Scan

# Создание случайного облака точек
scan = Scan(scan_name="TestScan")
print(scan)
scan.import_points_from_file(file_path="src/Камеры/data_3.dxf")
print(scan)
points = scan.get_points_array()

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)




# ВЫЧИСЛЯЕМ НОРМАЛИ - это обязательно!
pcd.estimate_normals()

# ОРИЕНТИРУЕМ НОРМАЛИ - это важно для качества
pcd.orient_normals_consistent_tangent_plane(10)

# Теперь Poisson реконструкция должна работать
mesh_poisson, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=9
)

# # Визуализируем
# o3d.visualization.draw_geometries([mesh_poisson])


def visualize_with_wireframe(mesh):
    """Визуализация с wireframe поверхностью"""
    mesh.compute_vertex_normals()

    # Создаем копию меша для wireframe
    wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    wireframe.paint_uniform_color([0, 0, 0])  # Черный цвет для каркаса

    # Визуализация вместе
    o3d.visualization.draw_geometries([pcd, wireframe],
                                      mesh_show_wireframe=False)  # Отключаем встроенный wireframe

visualize_with_wireframe(mesh_poisson)