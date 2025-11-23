import time

import open3d as o3d
import numpy as np
from scipy.spatial import ConvexHull

from app.mesh.ConvexHullMeshOpen3D import ConvexHullMeshOpen3D
from app.mesh.mesh_trimmers.BorderMeshMeshTrimmer import BorderMeshMeshTrimmer
from app.mesh.PoissonMeshOpen3D import PoissonMeshOpen3D
from app.mesh.mesh_trimmers.ZLevelsMeshTrimmer import ZLevelsMeshTrimmer
from app.scan.Scan import Scan
from app.scan.ScanPoint import ScanPoint
from app.scan.filters.DistanceBasedAnomalyFilter import DistanceBasedAnomalyFilter
from app.scan.filters.NormalBasedAnomalyFilter import NormalBasedAnomalyFilter

scan = Scan(scan_name="TestScan")
scan.import_points_from_file(file_path="src/Камеры/data_3.dxf")
# scan.plot(point_size=8)

print(scan)
# scan.filter_scan(filter_cls=DistanceBasedAnomalyFilter)
# scan.filter_scan(filter_cls=DistanceBasedAnomalyFilter)
# scan.filter_scan(filter_cls=DistanceBasedAnomalyFilter)
# scan.filter_scan(filter_cls=DistanceBasedAnomalyFilter)


print(scan)
# scan.plot(point_size=10)

#
poisson_mesh = PoissonMeshOpen3D(depth=12,
                                 width=0.5,
                                 scale=2.5).init_mesh_from_scan(scan)
# poisson_mesh.plot()

z_level = scan.borders["z_max"]
print(z_level)
poisson_mesh = ZLevelsMeshTrimmer(base_mesh=poisson_mesh, z_level=z_level).trim_mesh()
# poisson_mesh.plot()


import numpy as np
import open3d as o3d
import pyvista as pv


class SurfaceVoxelizer:
    def __init__(self, voxel_size=0.5):
        """
        Вокселизатор, который гарантирует полное покрытие поверхности

        Args:
            voxel_size (float): Размер вокселя
        """
        self.voxel_size = voxel_size

    def voxelize_surface_complete(self, mesh):
        """
        Вокселизация поверхности с полным покрытием

        Args:
            mesh: Open3D треугольный меш
        """
        # Получаем все вершины меша
        vertices = np.asarray(mesh.vertices)

        if len(vertices) == 0:
            print("Нет вершин в меше")
            return None

        # Находим ограничивающий объем меша
        bbox = mesh.get_axis_aligned_bounding_box()
        min_bound = bbox.get_min_bound()
        max_bound = bbox.get_max_bound()

        # Выравниваем границы чтобы они были кратны размеру вокселя
        aligned_min = self._align_to_grid(min_bound, 'floor')
        aligned_max = self._align_to_grid(max_bound, 'ceil')

        # Добавляем один слой вокселей по границам для гарантии покрытия
        aligned_min -= self.voxel_size
        aligned_max += self.voxel_size

        # Создаем равномерную сетку вокселей
        dimensions = ((aligned_max - aligned_min) / self.voxel_size).astype(int)

        print(f"Создаем сетку размером: {dimensions}")
        print(f"Количество вокселей: {np.prod(dimensions)}")

        # Создаем ImageData сетку
        grid = pv.ImageData()
        grid.dimensions = dimensions + 1  # +1 потому что dimensions указывает на количество ячеек
        grid.origin = aligned_min
        grid.spacing = (self.voxel_size, self.voxel_size, self.voxel_size)

        # Создаем облако точек из всех вершин меша
        point_cloud = pv.PolyData(vertices)

        # Проецируем точки меша на воксельную сетку
        # Воксель считается заполненным если в него попадает хотя бы одна вершина меша
        sampled_grid = grid.sample(point_cloud)

        # Заполняем воксели, которые содержат вершины меша
        occupancy = np.zeros(grid.n_cells, dtype=bool)

        # Для каждой вершины меша находим соответствующий воксель
        for vertex in vertices:
            # Вычисляем индекс вокселя для этой вершины
            idx = ((vertex - aligned_min) / self.voxel_size).astype(int)
            idx = np.clip(idx, 0, np.array(dimensions) - 1)

            # Преобразуем 3D индекс в линейный
            linear_idx = idx[0] + idx[1] * dimensions[0] + idx[2] * dimensions[0] * dimensions[1]
            if linear_idx < len(occupancy):
                occupancy[linear_idx] = True

        # Также учитываем воксели, которые содержат грани меша
        triangles = np.asarray(mesh.triangles)
        for triangle in triangles:
            # Берем вершины треугольника
            tri_vertices = vertices[triangle]

            # Находим ограничивающий объем треугольника
            tri_min = tri_vertices.min(axis=0)
            tri_max = tri_vertices.max(axis=0)

            # Находим диапазон вокселей, покрывающих этот треугольник
            min_idx = ((tri_min - aligned_min) / self.voxel_size).astype(int)
            max_idx = ((tri_max - aligned_min) / self.voxel_size).astype(int) + 1

            # Ограничиваем индексы размерами сетки
            min_idx = np.clip(min_idx, 0, dimensions)
            max_idx = np.clip(max_idx, 0, dimensions)

            # Помечаем все воксели в этом диапазоне
            for x in range(min_idx[0], max_idx[0]):
                for y in range(min_idx[1], max_idx[1]):
                    for z in range(min_idx[2], max_idx[2]):
                        linear_idx = x + y * dimensions[0] + z * dimensions[0] * dimensions[1]
                        if linear_idx < len(occupancy):
                            occupancy[linear_idx] = True

        # Добавляем маску занятости в сетку
        grid['occupancy'] = occupancy.astype(np.uint8)

        # Извлекаем только занятые воксели
        voxel_grid = grid.threshold(0.5, scalars='occupancy')

        return voxel_grid

    def _align_to_grid(self, point, align_type='round'):
        """Выравнивание точки к сетке вокселей"""
        if align_type == 'round':
            return np.round(point / self.voxel_size) * self.voxel_size
        elif align_type == 'floor':
            return np.floor(point / self.voxel_size) * self.voxel_size
        elif align_type == 'ceil':
            return np.ceil(point / self.voxel_size) * self.voxel_size
        else:
            return point

    def check_coverage(self, mesh, voxel_grid):
        """Проверка покрытия поверхности вокселями"""
        vertices = np.asarray(mesh.vertices)

        if len(vertices) == 0 or voxel_grid is None:
            return 0.0

        covered_count = 0
        total_vertices = len(vertices)

        # Для каждой вершины проверяем, попадает ли она в какой-либо воксель
        for vertex in vertices:
            # Находим ближайшую точку в воксельной сетке
            closest_point = voxel_grid.find_closest_point(vertex)
            distance = np.linalg.norm(vertex - closest_point)

            # Считаем вершину покрытой если расстояние меньше половины размера вокселя
            if distance < self.voxel_size / 2:
                covered_count += 1

        coverage = covered_count / total_vertices * 100
        print(f"Покрытие вершин: {coverage:.2f}% ({covered_count}/{total_vertices})")
        return coverage

    def visualize_with_coverage(self, mesh, voxel_grid):
        """Визуализация с проверкой покрытия"""
        plotter = pv.Plotter(shape=(1, 2))

        # Левая панель: воксельная модель
        plotter.subplot(0, 0)
        if voxel_grid is not None:
            plotter.add_mesh(voxel_grid,
                             style='wireframe',
                             color='blue',
                             line_width=2,
                             show_edges=True,
                             label='Voxel Grid')

        # Конвертируем Open3D mesh в PyVista для отображения
        mesh_pv = self._convert_open3d_to_pyvista(mesh)
        if mesh_pv is not None:
            plotter.add_mesh(mesh_pv,
                             color='red',
                             opacity=0.5,
                             label='Original Mesh')

        plotter.add_legend()
        plotter.add_text("Воксельная модель", font_size=12)

        # Правая панель: проверка покрытия
        plotter.subplot(0, 1)

        # Создаем визуализацию покрытия
        vertices = np.asarray(mesh.vertices)
        covered_points = []
        uncovered_points = []

        for vertex in vertices:
            closest_point = voxel_grid.find_closest_point(vertex)
            distance = np.linalg.norm(vertex - closest_point)

            if distance < self.voxel_size / 2:
                covered_points.append(vertex)
            else:
                uncovered_points.append(vertex)

        # Отображаем покрытые точки зеленым
        if covered_points:
            covered_pcd = pv.PolyData(np.array(covered_points))
            plotter.add_mesh(covered_pcd,
                             color='green',
                             point_size=5,
                             render_points_as_spheres=True,
                             label='Covered vertices')

        # Отображаем непокрытые точки красным
        if uncovered_points:
            uncovered_pcd = pv.PolyData(np.array(uncovered_points))
            plotter.add_mesh(uncovered_pcd,
                             color='red',
                             point_size=5,
                             render_points_as_spheres=True,
                             label='Uncovered vertices')

        # Добавляем воксельную сетку для контекста
        if voxel_grid is not None:
            plotter.add_mesh(voxel_grid,
                             style='wireframe',
                             color='blue',
                             line_width=1,
                             opacity=0.3,
                             show_edges=True)

        coverage = len(covered_points) / len(vertices) * 100
        plotter.add_legend()
        plotter.add_text(f"Покрытие: {coverage:.1f}%", font_size=12)

        plotter.show()

        return coverage

    def _convert_open3d_to_pyvista(self, open3d_mesh):
        """Конвертирует Open3D mesh в PyVista mesh"""
        try:
            vertices = np.asarray(open3d_mesh.vertices)
            faces = np.asarray(open3d_mesh.triangles)

            if len(faces) > 0:
                faces_pv = np.insert(faces, 0, 3, axis=1)
                mesh_pv = pv.PolyData(vertices, faces_pv.ravel())
                return mesh_pv
        except Exception as e:
            print(f"Ошибка конвертации меша: {e}")

        return None


# Пример использования
if __name__ == "__main__":


    # Создаем вокселизатор
    voxelizer = SurfaceVoxelizer(voxel_size=0.25)  # Можно настроить размер вокселя

    # Вокселизируем с полным покрытием
    print("Вокселизация поверхности...")
    voxel_grid = voxelizer.voxelize_surface_complete(poisson_mesh.mesh)

    if voxel_grid is not None:
        print(f"Создано вокселей: {voxel_grid.n_cells}")

        # Проверяем покрытие
        print("Проверка покрытия...")
        coverage = voxelizer.check_coverage(poisson_mesh.mesh, voxel_grid)

        # Визуализируем с проверкой покрытия
        print("Визуализация...")
        final_coverage = voxelizer.visualize_with_coverage(poisson_mesh.mesh, voxel_grid)

        print(f"Итоговое покрытие поверхности: {final_coverage:.2f}%")

        # Дополнительная простая визуализация
        if final_coverage < 95:  # Если покрытие недостаточное
            print("Предупреждение: покрытие поверхности менее 95%")
            print("Рекомендуется уменьшить размер вокселя")
    else:
        print("Не удалось создать воксельную модель")