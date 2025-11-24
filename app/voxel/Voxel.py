import numpy as np

from app.base.Point import Point


class Voxel:

    def __init__(self, point_xyz, size):
        """
        min_corner: np.array([xmin, ymin, zmin])
        size: float (длина ребра вокселя)
        """
        self.min_corner = np.array(point_xyz, dtype=float)
        self.size = float(size)
        self.max_corner = self.min_corner + self.size
        self.status = None

    def get_voxels_points(self):
        """
        Возвращает список 8 координатных вершин вокселя (numpy-массивы)
        Порядок: все комбинации (xmin/xmax, ymin/ymax, zmin/zmax)
        """
        xmin, ymin, zmin = self.min_corner
        xmax, ymax, zmax = self.max_corner
        return [
            np.array([xmin, ymin, zmin]),
            np.array([xmax, ymin, zmin]),
            np.array([xmin, ymax, zmin]),
            np.array([xmax, ymax, zmin]),
            np.array([xmin, ymin, zmax]),
            np.array([xmax, ymin, zmax]),
            np.array([xmin, ymax, zmax]),
            np.array([xmax, ymax, zmax])
        ]

    def contains_point(self, point: Point):
        """True если точка строго внутри вокселя, False иначе"""
        point = np.array([point.x, point.y, point.z], dtype=float)
        return np.all((point > self.min_corner) & (point < self.max_corner))

    def __str__(self):
        return (f"{self.__class__.__name__} (BasePoint: {self.min_corner}, "
                f"Size: {self.size} Status: {self.status}")
