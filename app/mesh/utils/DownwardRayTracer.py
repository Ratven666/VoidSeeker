import numpy as np

class DownwardRayTracer:
    """
    Трассировка: чёткая even-odd + on_surface.
    Дополнительное кэширование пересечений для оптимизации массовых проверок.
    """

    def __init__(self, mesh):
        self.mesh = mesh.mesh  # либо просто mesh, если без оболочки
        self._point_cache = {}       # кэш статусов точек
        self._intersections_cache = {}  # кэш пересечений по XY

    def _make_key(self, point):
        # Ключ по координатам (XY достаточно для пересечений; XYZ — для статуса)
        return tuple([round(float(p), 6) for p in point])

    def _key_xy(self, x0, y0):
        # Ключ для кэша по XY-координате
        return (round(float(x0), 6), round(float(y0), 6))

    def check_point_status(self, point, eps=1e-4):
        key_xyz = self._make_key(point)
        if key_xyz in self._point_cache:
            return self._point_cache[key_xyz]

        x0, y0, z0 = point
        key_xy = self._key_xy(x0, y0)

        # Кэш пересечений по XY — ускоряет массовые вертикальные лучи
        if key_xy in self._intersections_cache:
            z_intersections = self._intersections_cache[key_xy]
        else:
            z_intersections = []
            vertices = np.asarray(self.mesh.vertices)
            triangles = np.asarray(self.mesh.triangles)
            for tri in triangles:
                verts = vertices[tri]
                xy = verts[:, :2]
                if self._point_in_triangle_2d((x0, y0), xy):
                    z_surface = self._point_z_on_triangle(x0, y0, verts)
                    z_intersections.append(z_surface)
            self._intersections_cache[key_xy] = z_intersections

        on_surface = any(abs(z0 - zsurf) < eps for zsurf in z_intersections)
        below = [z for z in z_intersections if z < z0 - eps]

        if on_surface:
            status = "on_surface"
        elif len(below) % 2 == 1:
            status = "inside"
        else:
            status = "outside"

        self._point_cache[key_xyz] = status
        return status

    def is_point_inside(self, point, eps=1e-4):
        return self.check_point_status(point, eps) == "inside"

    @staticmethod
    def _point_in_triangle_2d(pt, tri_xy):
        x, y = pt
        x1, y1 = tri_xy[0]
        x2, y2 = tri_xy[1]
        x3, y3 = tri_xy[2]
        det = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)
        if abs(det) < 1e-12:
            return False
        a = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / det
        b = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / det
        c = 1 - a - b
        return (0 <= a <= 1) and (0 <= b <= 1) and (0 <= c <= 1)

    @staticmethod
    def _point_z_on_triangle(x0, y0, verts):
        p1, p2, p3 = verts
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        if abs(n[2]) < 1e-12:
            return p1[2]
        return (-n[0]*(x0 - p1[0]) - n[1]*(y0 - p1[1])) / n[2] + p1[2]
