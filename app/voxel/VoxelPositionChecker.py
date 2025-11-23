import numpy as np

class VoxelPositionChecker:
    def __init__(self, voxel, ray_tracer):
        """
        voxel: объект Voxel с методом get_voxels_points()
        ray_tracer: объект HorizontalRayTracer с методом is_point_inside(point)
        """
        self.voxel = voxel
        self.ray_tracer = ray_tracer

    def voxel_status(self):
        """
        Проверяет положение вокселя:
        - 'inside' если все вершины внутри поверхности
        - 'on_surface' если хотя бы одна вершина внутри, но не все
        - 'outside' если все вершины снаружи
        """
        points = self.voxel.get_voxels_points()
        statuses = [self.ray_tracer.is_point_inside(pt) for pt in points]
        count_inside = sum(statuses)
        if count_inside == len(points):
            return "inside"
        elif count_inside > 0:
            return "on_surface"
        else:
            return "outside"
