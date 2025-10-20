import pyvista as pv
import numpy as np
import open3d as o3d

from abc import ABC, abstractmethod

from app.mesh.mesh_plotters.MeshPlotterABC import MeshPlotterABC


class MeshPlotterPyVista(MeshPlotterABC):
    """
    Класс для визуализации поверхностей Open3D с использованием PyVista
    """

    def __init__(self, background_color='white', show_axes=True,
                 show_grid=False, window_size=(1200, 800),
                 cmap='viridis', edge_color='black', edge_visibility=True,
                 lighting=True, smooth_shading=False, point_size=1):
        """
        Инициализация визуализатора мешей PyVista

        Args:
            background_color (str): Цвет фона
            show_axes (bool): Показывать оси координат
            show_grid (bool): Показывать сетку
            window_size (tuple): Размер окна (ширина, высота)
            cmap (str): Цветовая карта для меша
            edge_color (str): Цвет ребер
            edge_visibility (bool): Показывать ребра
            lighting (bool): Включить освещение
            smooth_shading (bool): Сглаживание shading
            point_size (int): Размер точек (для облаков точек)
        """
        self.background_color = background_color
        self.show_axes = show_axes
        self.show_grid = show_grid
        self.window_size = window_size
        self.cmap = cmap
        self.edge_color = edge_color
        self.edge_visibility = edge_visibility
        self.lighting = lighting
        self.smooth_shading = smooth_shading
        self.point_size = point_size
        self.plotter = None

    def plot(self, mesh):
        """
        Визуализация меша Open3D

        Args:
            mesh: Объект меша Open3D (TriangleMesh, PointCloud или Geometry)

        Returns:
            pyvista.Plotter: Объект плоттера PyVista
        """
        if mesh is None:
            raise ValueError("Меш не может быть None")
        # Создание плоттера
        self.plotter = pv.Plotter(window_size=self.window_size)
        # Конвертация Open3D меша в PyVista
        pv_mesh = self._convert_open3d_to_pyvista(mesh.mesh)
        if pv_mesh is None:
            raise ValueError("Не удалось конвертировать меш Open3D в формат PyVista")
        # Визуализация в зависимости от типа данных
        if isinstance(pv_mesh, pv.PolyData):
            if pv_mesh.n_cells > 0:
                # Это поверхность с треугольниками
                self._plot_surface_mesh(pv_mesh)
            else:
                # Это облако точек
                self._plot_point_cloud(pv_mesh)
        elif isinstance(pv_mesh, pv.UnstructuredGrid):
            # Это объемный меш
            self._plot_volume_mesh(pv_mesh)
        # Настройка сцены
        self._setup_scene()
        self.plotter.show()
        return self.plotter

    def _convert_open3d_to_pyvista(self, o3d_mesh):
        """
        Конвертация Open3D объекта в PyVista

        Args:
            o3d_mesh: Объект Open3D (TriangleMesh, PointCloud, etc.)

        Returns:
            PyVista mesh object
        """
        if isinstance(o3d_mesh, o3d.geometry.TriangleMesh):
            return self._convert_triangle_mesh(o3d_mesh)
        elif isinstance(o3d_mesh, o3d.geometry.PointCloud):
            return self._convert_point_cloud(o3d_mesh)
        else:
            raise ValueError(f"Неподдерживаемый тип Open3D геометрии: {type(o3d_mesh)}")

    def _convert_triangle_mesh(self, o3d_mesh):
        """
        Конвертация Open3D TriangleMesh в PyVista PolyData
        """
        # Получение вершин и треугольников
        vertices = np.asarray(o3d_mesh.vertices)
        triangles = np.asarray(o3d_mesh.triangles)
        # Создание PyVista меша
        faces = np.insert(triangles, 0, 3, axis=1).flatten()
        pv_mesh = pv.PolyData(vertices, faces)
        return pv_mesh

    def _convert_point_cloud(self, o3d_pointcloud):
        """
        Конвертация Open3D PointCloud в PyVista PolyData
        """
        # Получение точек
        points = np.asarray(o3d_pointcloud.points)
        # Создание облака точек PyVista
        pv_cloud = pv.PolyData(points)
        # Добавление цветов, если есть
        if o3d_pointcloud.has_colors():
            colors = np.asarray(o3d_pointcloud.colors)
            pv_cloud['colors'] = colors
        # Добавление нормалей, если есть
        if o3d_pointcloud.has_normals():
            normals = np.asarray(o3d_pointcloud.normals)
            pv_cloud['Normals'] = normals
        return pv_cloud

    def _plot_surface_mesh(self, pv_mesh):
        """
        Визуализация поверхностного меша
        """
        # Проверка наличия цветов вершин
        has_vertex_colors = 'vertex_colors' in pv_mesh.array_names
        if has_vertex_colors:
            # Использование цветов вершин
            self.plotter.add_mesh(
                pv_mesh,
                scalars='vertex_colors',
                rgb=True,
                show_edges=self.edge_visibility,
                edge_color=self.edge_color,
                lighting=self.lighting,
                smooth_shading=self.smooth_shading
            )
        else:
            # Использование цветовой карты
            self.plotter.add_mesh(
                pv_mesh,
                color='lightblue',
                show_edges=self.edge_visibility,
                edge_color=self.edge_color,
                lighting=self.lighting,
                smooth_shading=self.smooth_shading,
                cmap=self.cmap
            )

    def _plot_point_cloud(self, pv_cloud):
        """
        Визуализация облака точек
        """
        has_colors = 'colors' in pv_cloud.array_names

        if has_colors:
            self.plotter.add_points(
                pv_cloud,
                scalars='colors',
                rgb=True,
                point_size=self.point_size,
                render_points_as_spheres=True
            )
        else:
            self.plotter.add_points(
                pv_cloud,
                color='red',
                point_size=self.point_size,
                render_points_as_spheres=True
            )

    def _plot_volume_mesh(self, pv_volume):
        """
        Визуализация объемного меша
        """
        self.plotter.add_mesh(
            pv_volume,
            show_edges=self.edge_visibility,
            edge_color=self.edge_color,
            lighting=self.lighting,
            cmap=self.cmap
        )

    def _setup_scene(self):
        """Настройка сцены"""
        self.plotter.set_background(self.background_color)

        if self.show_axes:
            self.plotter.add_axes()

        if self.show_grid:
            self.plotter.show_grid()

        # Установка изометрического вида
        self.plotter.view_isometric()
