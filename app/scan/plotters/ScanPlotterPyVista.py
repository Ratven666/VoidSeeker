import pyvista as pv
import numpy as np
from app.scan.ScanPoint import ScanPoint
from app.scan.plotters.ScanPlotterABC import ScanPlotterABC


class ScanPlotterPyVista(ScanPlotterABC):
    """
    Класс для визуализации скана с использованием PyVista
    """

    def __init__(self, point_size=5, background_color='white',
                 show_axes=True, show_grid=True, window_size=(800, 600)):
        """
        Инициализация визуализатора PyVista

        Args:
            point_size (int): Размер точек при отображении
            background_color (str): Цвет фона
            show_axes (bool): Показывать оси координат
            show_grid (bool): Показывать сетку
            window_size (tuple): Размер окна (ширина, высота)
        """
        self.point_size = point_size
        self.background_color = background_color
        self.show_axes = show_axes
        self.show_grid = show_grid
        self.window_size = window_size
        self.plotter = None

    def plot(self, scan, title=None, **kwargs):
        """
        Визуализация скана

        Args:
            scan (Scan): Объект скана для визуализации
            title (str): Заголовок графика
            **kwargs: Дополнительные параметры

        Returns:
            pyvista.Plotter: Объект плоттера PyVista
        """
        if len(scan) == 0:
            print("Предупреждение: Скан не содержит точек для визуализации")
            return None

        # Создание плоттера
        self.plotter = pv.Plotter(window_size=self.window_size)

        # Получение данных точек
        points_array = scan.get_points_array()

        # Создание облака точек PyVista
        point_cloud = pv.PolyData(points_array)

        # Добавление цвета точек, если доступно
        colors = self._get_point_colors(scan)
        if colors is not None:
            point_cloud['colors'] = colors

        # Добавление облака точек на сцену
        if colors is not None:
            self.plotter.add_points(point_cloud,
                                    scalars='colors',
                                    rgb=True,
                                    point_size=self.point_size,
                                    render_points_as_spheres=True)
        else:
            self.plotter.add_points(point_cloud,
                                    color='blue',
                                    point_size=self.point_size,
                                    render_points_as_spheres=True)

        # Настройка сцены
        if title is None:
            title = f"Скан: {scan.name}"

        self.plotter.add_title(title, font_size=18)
        self.plotter.set_background(self.background_color)

        if self.show_axes:
            self.plotter.add_axes()

        if self.show_grid:
            self.plotter.show_grid()

        # Настройка камеры для лучшего обзора
        self.plotter.view_isometric()
        self.plotter.show(**kwargs)
        return self.plotter

    def _get_point_colors(self, scan):
        colors = []
        has_colors = False

        for point in scan:
            if hasattr(point, 'color') and point.color != (0, 0, 0):
                colors.append(point.color)
                has_colors = True
            else:
                colors.append((0, 0, 1))  # Синий по умолчанию

        if has_colors:
            # Нормализация цветов к диапазону [0, 1] для PyVista
            return np.array(colors) / 255.0 if max(max(color) for color in colors) > 1 else np.array(colors)
        else:
            return None
