from copy import deepcopy

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

from app.mesh.mesh_trimmers.MeshTrimmerABC import MeshTrimmerABC


class BorderMeshMeshTrimmer(MeshTrimmerABC):

    def __init__(self, base_mesh, border_mesh, scale=1.0):
        super().__init__(base_mesh)
        self.scale = scale
        self.border_mesh = border_mesh

    def _custom_trim_logic(self):
        """
        Обрезает mesh используя border_mesh mesh как эталон границ
        Симметричная обрезка - удаляются только треугольники, выходящие за пределы border_mesh
        """
        border_mesh = self.border_mesh.mesh
        if len(border_mesh.triangles) == 0:
            return self.base_mesh.mesh

        # Получаем вершины обеих сеток
        border_mesh_vertices = np.asarray(border_mesh.vertices)
        poisson_vertices = np.asarray(self.base_mesh.mesh.vertices)

        # Создаем KDTree для граничной поверхности
        border_mesh_kdtree = KDTree(border_mesh_vertices)

        # Метод 1: Определяем bounding box граничной поверхности
        border_mesh_bbox_min = np.min(border_mesh_vertices, axis=0)
        border_mesh_bbox_max = np.max(border_mesh_vertices, axis=0)

        # Расширяем bbox для захвата всей геометрии border_mesh
        bbox_range = border_mesh_bbox_max - border_mesh_bbox_min
        bbox_min = border_mesh_bbox_min - bbox_range * (self.scale - 1.0)
        bbox_max = border_mesh_bbox_max + bbox_range * (self.scale - 1.0)

        # Метод 2: Определяем, какие точки находятся ВНУТРИ граничной поверхности
        # Используем ray casting для определения внутренних/внешних точек
        inside_border = self._check_points_inside_mesh(poisson_vertices, border_mesh)

        # Комбинированная маска: точки внутри bbox И внутри граничной поверхности
        inside_bbox = np.all(
            (poisson_vertices >= bbox_min) & (poisson_vertices <= bbox_max),
            axis=1
        )

        # Сохраняем только точки, которые находятся внутри граничной поверхности
        # ИЛИ близко к ней (для плавного перехода)
        vertices_to_keep = inside_bbox & inside_border

        # Альтернативный подход: использовать расстояние до поверхности для плавной обрезки
        distances, _ = border_mesh_kdtree.query(poisson_vertices)
        original_points = np.asarray(self.base_mesh.pcd.points)
        point_spacing = self._estimate_point_spacing(original_points)
        distance_threshold = point_spacing * 2.0

        # Расширяем маску: точки внутри ИЛИ близко к поверхности
        close_to_border = distances <= distance_threshold
        vertices_to_keep = vertices_to_keep | (inside_bbox & close_to_border)

        # Создаем маску для треугольников (все вершины должны удовлетворять условиям)
        triangles_to_keep = []
        for i, triangle in enumerate(self.base_mesh.mesh.triangles):
            if (vertices_to_keep[triangle[0]] and
                    vertices_to_keep[triangle[1]] and
                    vertices_to_keep[triangle[2]]):
                triangles_to_keep.append(i)

        # Создаем обрезанную сетку
        trimmed_mesh = o3d.geometry.TriangleMesh()
        trimmed_mesh.vertices = o3d.utility.Vector3dVector(
            np.asarray(self.base_mesh.mesh.vertices)[vertices_to_keep]
        )

        # Переназначаем индексы треугольников
        vertex_mapping = {}
        new_vertex_count = 0
        for old_idx, keep in enumerate(vertices_to_keep):
            if keep:
                vertex_mapping[old_idx] = new_vertex_count
                new_vertex_count += 1

        new_triangles = []
        for triangle_idx in triangles_to_keep:
            triangle = self.base_mesh.mesh.triangles[triangle_idx]
            new_triangle = [vertex_mapping[triangle[0]],
                            vertex_mapping[triangle[1]],
                            vertex_mapping[triangle[2]]]
            new_triangles.append(new_triangle)

        trimmed_mesh.triangles = o3d.utility.Vector3iVector(np.array(new_triangles))

        # Копируем нормали
        if len(self.base_mesh.mesh.vertex_normals) == len(self.base_mesh.mesh.vertices):
            trimmed_mesh.vertex_normals = o3d.utility.Vector3dVector(
                np.asarray(self.base_mesh.mesh.vertex_normals)[vertices_to_keep]
            )

        print(f"Обрезка {self.base_mesh.__class__}: {len(triangles_to_keep)}/"
              f"{len(self.base_mesh.mesh.triangles)} треугольников сохранено")
        print(f"Вершин: {np.sum(vertices_to_keep)}/{len(poisson_vertices)} сохранено")

        return trimmed_mesh

    def _check_points_inside_mesh(self, points, mesh):
        """
        Проверяет, какие точки находятся внутри меша с помощью ray casting
        """
        # Создаем сцену для ray casting
        scene = o3d.t.geometry.RaycastingScene()
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        mesh_id = scene.add_triangles(mesh_t)

        # Преобразуем точки в тензоры
        points_t = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)

        # Вычисляем расстояния до меша (отрицательные значения означают внутри)
        signed_distances = scene.compute_signed_distance(points_t)

        # Точки с отрицательным расстоянием находятся внутри меша
        inside_points = signed_distances.numpy() < 0

        return inside_points

    @staticmethod
    def _estimate_point_spacing(points):
        """
        Оценивает среднее расстояние между точками в облаке
        """
        tree = KDTree(points)
        distances, _ = tree.query(points, k=2)
        avg_spacing = np.mean(distances[:, 1])
        return avg_spacing
