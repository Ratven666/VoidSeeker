import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

from app.mesh.BallPivotingAlgorithmMeshOpen3D import BallPivotingAlgorithmMeshOpen3D
from app.mesh.MeshOpen3D import MeshOpen3D


class PoissonMeshOpen3D(MeshOpen3D):

    def __init__(self, depth=9, width=0, scale=1.1):
        super().__init__()
        self.depth = depth
        self.width = width
        self.scale = scale

    def _calculate_mesh(self):
        self._estimate_normals()
        # Poisson reconstruction
        self.pcd.orient_normals_consistent_tangent_plane(10)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            self.pcd, depth=self.depth, width=self.width, scale=self.scale
        )
        # Убираем вершины с низкой плотностью (артефакты)
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        self.mesh = mesh
        return mesh

    def trim_by_bpa(self, scale=1.0):
        bpa_mesh = BallPivotingAlgorithmMeshOpen3D().init_mesh_from_scan(self.scan)
        trimmed_mesh = self._trim_poisson_with_bpa(bpa_mesh, scale)
        self.mesh = trimmed_mesh
        return trimmed_mesh

    def _trim_poisson_with_bpa(self, bpa_mesh, scale):
        """
        Обрезает Poisson mesh используя BPA mesh как эталон границ
        """
        bpa_mesh = bpa_mesh.mesh
        if len(bpa_mesh.triangles) == 0:
            return self.mesh
        # Метод 1: Обрезка по расширенному bounding box BPA
        bpa_vertices = np.asarray(bpa_mesh.vertices)
        bpa_bbox_min = np.min(bpa_vertices, axis=0)
        bpa_bbox_max = np.max(bpa_vertices, axis=0)
        # Расширяем bbox для захвата всей геометрии BPA
        bbox_range = bpa_bbox_max - bpa_bbox_min
        bbox_min = bpa_bbox_min - bbox_range * (scale - 1.0)
        bbox_max = bpa_bbox_max + bbox_range * (scale - 1.0)
        # Метод 2: Расстояние до BPA поверхности
        poisson_vertices = np.asarray(self.mesh.vertices)
        # Создаем KDTree для быстрого поиска расстояний до BPA
        bpa_kdtree = KDTree(bpa_vertices)
        # Вычисляем расстояния от каждой вершины Poisson до ближайшей вершины BPA
        distances, _ = bpa_kdtree.query(poisson_vertices)
        # Автоматический подбор порога расстояния
        original_points = np.asarray(self.pcd.points)
        point_spacing = self._estimate_point_spacing(original_points)
        distance_threshold = point_spacing * 3.0  # Порог в 3 средних расстояния

        print(f"Порог расстояния для обрезки: {distance_threshold:.4f}")
        # Комбинированная маска: внутри bbox И близко к BPA поверхности
        inside_bbox = np.all(
            (poisson_vertices >= bbox_min) & (poisson_vertices <= bbox_max),
            axis=1
        )
        close_to_bpa = distances <= distance_threshold
        vertices_to_keep = inside_bbox & close_to_bpa
        # Создаем маску для треугольников (все вершины должны удовлетворять условиям)
        triangles_to_keep = []
        for i, triangle in enumerate(self.mesh.triangles):
            if (vertices_to_keep[triangle[0]] and
                    vertices_to_keep[triangle[1]] and
                    vertices_to_keep[triangle[2]]):
                triangles_to_keep.append(i)
        # Создаем обрезанную сетку
        trimmed_mesh = o3d.geometry.TriangleMesh()
        trimmed_mesh.vertices = self.mesh.vertices
        trimmed_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(self.mesh.triangles)[triangles_to_keep])
        # Копируем нормали
        if len(self.mesh.vertex_normals) == len(self.mesh.vertices):
            trimmed_mesh.vertex_normals = o3d.utility.Vector3dVector(
                np.asarray(self.mesh.vertex_normals)[vertices_to_keep]
            )
        print(f"Обрезка Poisson: {len(triangles_to_keep)}/{len(self.mesh.triangles)} треугольников сохранено")
        return trimmed_mesh

    @staticmethod
    def _estimate_point_spacing(points):
        """
        Оценивает среднее расстояние между точками в облаке
        """
        from scipy.spatial import KDTree
        tree = KDTree(points)
        distances, _ = tree.query(points, k=2)  # k=2 чтобы исключить расстояние до себя
        avg_spacing = np.mean(distances[:, 1])  # Берем расстояния до ближайших соседей
        return avg_spacing

if __name__ == "__main__":
    from app.scan.Scan import Scan

    scan = Scan(scan_name="TestScan")
    scan.import_points_from_file(file_path="../../src/Камеры/data_3.dxf")

    mesh = PoissonMeshOpen3D(depth=12,
                             width=0,
                             scale=1.1).init_mesh_from_scan(scan)
    print(mesh)
    mesh.trim_by_bpa(scale=1)
    mesh.fill_holes_connected_components()
    mesh.plot()
