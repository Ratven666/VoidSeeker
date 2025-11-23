import numpy as np

import numpy as np

class HorizontalRayTracer:
    def __init__(self, mesh):
        self.mesh = mesh  # Open3D TriangleMesh
        self._point_cache = {}  # Кэш для ускорения повторных проверок

    def _make_key(self, point):
        # Ключ строим из округленных координат, используем все 3 компоненты
        p = np.array(point)
        # Можно округлять до 5-6 знаков, либо приводить к int с опорным шагом
        return (round(float(p[0]), 6), round(float(p[1]), 6), round(float(p[2]), 6))

    def is_point_inside(self, point):
        """ Возвращает True если точка внутри поверхности, иначе False.
            Оптимизировано за счет кеширования.
        """
        key = self._make_key(point)
        if key in self._point_cache:
            return self._point_cache[key]
        count = 0
        x0, y0 = point[0], point[1]
        for tri in np.asarray(self.mesh.mesh.triangles):
            verts = np.asarray(self.mesh.mesh.vertices)[tri]
            xy = verts[:, :2]
            if self._ray_intersects_triangle((x0, y0), xy):
                count += 1
        result = count % 2 == 1
        self._point_cache[key] = result
        return result

    @staticmethod
    def _ray_intersects_triangle(pt, tri_xy):
        x0, y0 = pt
        n = 3
        intersections = 0
        for i in range(n):
            x1, y1 = tri_xy[i]
            x2, y2 = tri_xy[(i+1)%n]
            if (y1 > y0) != (y2 > y0):
                x_intersect = (x2 - x1) * (y0 - y1) / (y2 - y1 + 1e-12) + x1
                if x_intersect > x0:
                    intersections += 1
        return intersections % 2 == 1


# class HorizontalRayTracer:
#     def __init__(self, mesh):
#         self.mesh = mesh  # предполагается объект open3d.geometry.TriangleMesh
#
#     def is_point_inside(self, point):
#         """
#         point: np.array([x, y, z])
#         Возвращает True если точка внутри, иначе False.
#         """
#         count = 0
#         x0, y0 = point[0], point[1]
#         for tri in np.asarray(self.mesh.mesh.triangles):
#             verts = np.asarray(self.mesh.mesh.vertices)[tri]
#             # Рассматриваем только проекцию на XY
#             xy = verts[:, :2]
#             # Формируем двумерный треугольник
#             if self._ray_intersects_triangle((x0, y0), xy):
#                 count += 1
#         return count % 2 == 1
#
#     @staticmethod
#     def _ray_intersects_triangle(pt, tri_xy):
#         """
#         pt: (x0, y0) — точка, из которой выпускается положительно-направленный по X (или Y) горизонтальный луч.
#         tri_xy: (3, 2) — координаты вершин треугольника в XY.
#         """
#         x0, y0 = pt
#         n = 3
#         intersections = 0
#         for i in range(n):
#             x1, y1 = tri_xy[i]
#             x2, y2 = tri_xy[(i+1)%n]
#             if (y1 > y0) != (y2 > y0):  # Рёбра пересекают горизонталь Y = y0
#                 x_intersect = (x2 - x1) * (y0 - y1) / (y2 - y1 + 1e-12) + x1
#                 if x_intersect > x0:
#                     intersections += 1
#         return intersections % 2 == 1


if __name__ == "__main__":
    from app.mesh.PoissonMeshOpen3D import PoissonMeshOpen3D
    from app.mesh.mesh_trimmers.ZLevelsMeshTrimmer import ZLevelsMeshTrimmer
    from app.scan.Scan import Scan

    scan = Scan(scan_name="TestScan")
    scan.import_points_from_file(file_path="../../../src/Камеры/data_3.dxf")

    mesh = PoissonMeshOpen3D(depth=12,
                             width=0.5,
                             scale=1.1).init_mesh_from_scan(scan)
    print(mesh)
    # mesh.plot()
    z_level = scan.borders["z_max"]
    print(z_level)
    mesh = ZLevelsMeshTrimmer(base_mesh=mesh, z_level=z_level).trim_mesh()
    # mesh.plot()
    sprt = HorizontalRayTracer(mesh=mesh,)

    for x in range(42020, 42042):
        for y in range(55150, 55151):
            p = [x, y, 106]
            print(p, sprt.is_point_inside(point=p))
