import numpy as np
from typing import Union, Tuple, List


class VectorRayTracer:
    """
    РЈРЅРёРІРµСЂСЃР°Р»СЊРЅР°СЏ С‚СЂР°СЃСЃРёСЂРѕРІРєР° Р»СѓС‡РµР№ РІРґРѕР»СЊ РїСЂРѕРёР·РІРѕР»СЊРЅРѕРіРѕ РІРµРєС‚РѕСЂР°.

    РџРѕРґРґРµСЂР¶РёРІР°РµС‚:
    - РўСЂР°СЃСЃРёСЂРѕРІРєСѓ РІРґРѕР»СЊ РїСЂРѕРёР·РІРѕР»СЊРЅРѕРіРѕ РЅР°РїСЂР°РІР»РµРЅРёСЏ (default: РІРЅРёР· РїРѕ Z)
    - Even-odd С‚РµСЃС‚ РґР»СЏ РѕРїСЂРµРґРµР»РµРЅРёСЏ РїСЂРёРЅР°РґР»РµР¶РЅРѕСЃС‚Рё С‚РѕС‡РєРё Рє СЃРµС‚РєРµ
    - Р”РµС‚РµРєС‚РёСЂРѕРІР°РЅРёРµ С‚РѕС‡РµРє РЅР° РїРѕРІРµСЂС…РЅРѕСЃС‚Рё
    - РљСЌС€РёСЂРѕРІР°РЅРёРµ РїРµСЂРµСЃРµС‡РµРЅРёР№ РґР»СЏ РѕРїС‚РёРјРёР·Р°С†РёРё РјР°СЃСЃРѕРІС‹С… РїСЂРѕРІРµСЂРѕРє
    - Р Р°Р±РѕС‚Сѓ СЃ РЅРµРЅРѕСЂРјР°Р»РёР·РѕРІР°РЅРЅС‹РјРё РІРµРєС‚РѕСЂР°РјРё

    Args:
        mesh: Open3D TriangleMesh РёР»Рё РѕР±СЉРµРєС‚ СЃ .mesh
        direction: РќР°РїСЂР°РІР»РµРЅРёРµ С‚СЂР°СЃСЃРёСЂРѕРІРєРё РєР°Рє np.ndarray РёР»Рё tuple [x, y, z]
                  Default: (0, 0, -1) - РІРЅРёР· РїРѕ РѕСЃРё Z
    """

    def __init__(
            self,
            mesh,
            direction: Union[Tuple[float, float, float], np.ndarray] = None
    ):
        """РРЅРёС†РёР°Р»РёР·Р°С†РёСЏ С‚СЂРµР№СЃРµСЂР° СЃ СЃРµС‚РєРѕР№ Рё РЅР°РїСЂР°РІР»РµРЅРёРµРј."""
        # РР·РІР»РµС‡РµРЅРёРµ СЃР°РјРѕР№ СЃРµС‚РєРё (РµСЃР»Рё РѕР±С‘СЂРЅСѓС‚Р° РІ РєР»Р°СЃСЃ)
        self.mesh = mesh.mesh if hasattr(mesh, 'mesh') else mesh

        # РЈСЃС‚Р°РЅРѕРІРєР° РЅР°РїСЂР°РІР»РµРЅРёСЏ (РїРѕ СѓРјРѕР»С‡Р°РЅРёСЋ РІРЅРёР·)
        if direction is None:
            # direction = np.array([0.0, 0.0, -1.0])
            direction = np.array([1.0, 1., 0.])
        else:
            direction = np.asarray(direction, dtype=float)

        # РќРѕСЂРјР°Р»РёР·Р°С†РёСЏ РІРµРєС‚РѕСЂР° РЅР°РїСЂР°РІР»РµРЅРёСЏ
        self.direction = direction / np.linalg.norm(direction)

        # РљСЌС€Рё
        self._point_cache = {}  # РљСЌС€ СЃС‚Р°С‚СѓСЃРѕРІ С‚РѕС‡РµРє
        self._intersections_cache = {}  # РљСЌС€ РїРµСЂРµСЃРµС‡РµРЅРёР№ РїРѕ РїСЂРѕРµРєС†РёРё
        self._projection_axis = None  # РћСЃРё РїСЂРѕРµРєС†РёРё (РІС‹С‡РёСЃР»СЏСЋС‚СЃСЏ Р»РµРЅРёРІРѕ)

    def set_direction(self, direction: Union[Tuple[float, float, float], np.ndarray]):
        """РР·РјРµРЅРёС‚СЊ РЅР°РїСЂР°РІР»РµРЅРёРµ С‚СЂР°СЃСЃРёСЂРѕРІРєРё Рё РѕС‡РёСЃС‚РёС‚СЊ РєСЌС€Рё."""
        direction = np.asarray(direction, dtype=float)
        self.direction = direction / np.linalg.norm(direction)
        self._invalidate_cache()

    def _invalidate_cache(self):
        """РћС‡РёСЃС‚РёС‚СЊ РєСЌС€Рё РїСЂРё РёР·РјРµРЅРµРЅРёРё РЅР°РїСЂР°РІР»РµРЅРёСЏ."""
        self._point_cache.clear()
        self._intersections_cache.clear()
        self._projection_axis = None

    def _get_projection_axes(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Р’С‹С‡РёСЃР»РёС‚СЊ РґРІР° РѕСЂС‚РѕРіРѕРЅР°Р»СЊРЅС‹С… РІРµРєС‚РѕСЂР° РґР»СЏ РїСЂРѕРµРєС†РёРё РЅР° РїР»РѕСЃРєРѕСЃС‚СЊ,
        РїРµСЂРїРµРЅРґРёРєСѓР»СЏСЂРЅСѓСЋ РЅР°РїСЂР°РІР»РµРЅРёСЋ С‚СЂР°СЃСЃРёСЂРѕРІРєРё.
        """
        if self._projection_axis is not None:
            return self._projection_axis

        # РќР°Р№С‚Рё РІРµРєС‚РѕСЂ, РЅРµ РїР°СЂР°Р»Р»РµР»СЊРЅС‹Р№ direction
        if abs(self.direction[0]) < 0.9:
            aux = np.array([1.0, 0.0, 0.0])
        else:
            aux = np.array([0.0, 1.0, 0.0])

        # Р”РІР° РѕСЂС‚РѕРіРѕРЅР°Р»СЊРЅС‹С… РІРµРєС‚РѕСЂР°
        u = np.cross(self.direction, aux)
        u = u / np.linalg.norm(u)
        v = np.cross(self.direction, u)
        v = v / np.linalg.norm(v)

        self._projection_axis = (u, v)
        return u, v

    def _project_point(self, point: np.ndarray) -> Tuple[float, float]:
        """
        РЎРїСЂРѕРµС†РёСЂРѕРІР°С‚СЊ С‚РѕС‡РєСѓ РЅР° РїР»РѕСЃРєРѕСЃС‚СЊ РїРµСЂРїРµРЅРґРёРєСѓР»СЏСЂРЅСѓСЋ РЅР°РїСЂР°РІР»РµРЅРёСЋ.
        Р’РѕР·РІСЂР°С‰Р°РµС‚ РєРѕРѕСЂРґРёРЅР°С‚С‹ (u, v) РІ 2D.
        """
        u_axis, v_axis = self._get_projection_axes()
        proj_u = np.dot(point, u_axis)
        proj_v = np.dot(point, v_axis)
        return proj_u, proj_v

    def _make_key(self, point: np.ndarray) -> Tuple:
        """РљР»СЋС‡ РґР»СЏ РєСЌС€Р° РїРѕ 3D РєРѕРѕСЂРґРёРЅР°С‚Р°Рј."""
        return tuple([round(float(p), 6) for p in point])

    def _key_projection(self, u: float, v: float) -> Tuple:
        """РљР»СЋС‡ РґР»СЏ РєСЌС€Р° РїРѕ 2D РїСЂРѕРµРєС†РёРё."""
        return (round(float(u), 6), round(float(v), 6))

    def _point_in_triangle_2d(
            self,
            pt: Tuple[float, float],
            tri_xy: np.ndarray
    ) -> bool:
        """
        РџСЂРѕРІРµСЂРєР° РїСЂРёРЅР°РґР»РµР¶РЅРѕСЃС‚Рё С‚РѕС‡РєРё С‚СЂРµСѓРіРѕР»СЊРЅРёРєСѓ РІ 2D (Р±Р°СЂРёС†РµРЅС‚СЂРёС‡РµСЃРєРёРµ РєРѕРѕСЂРґРёРЅР°С‚С‹).
        """
        x, y = pt
        x1, y1 = tri_xy[0]
        x2, y2 = tri_xy[1]
        x3, y3 = tri_xy[2]

        det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)

        if abs(det) < 1e-12:
            return False

        a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / det
        b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / det
        c = 1 - a - b

        return (0 <= a <= 1) and (0 <= b <= 1) and (0 <= c <= 1)

    def _point_z_on_triangle(self, pt: np.ndarray, verts: np.ndarray) -> float:
        """
        Р’С‹С‡РёСЃР»РёС‚СЊ РєРѕРѕСЂРґРёРЅР°С‚Сѓ РїРµСЂРµСЃРµС‡РµРЅРёСЏ Р»СѓС‡Р° СЃ С‚СЂРµСѓРіРѕР»СЊРЅРёРєРѕРј.
        pt вЂ” С‚РѕС‡РєР° РЅР° Р»СѓС‡Рµ (РїСЂРѕРµРєС†РёСЏ), verts вЂ” РІРµСЂС€РёРЅС‹ С‚СЂРµСѓРіРѕР»СЊРЅРёРєР°.
        """
        p1, p2, p3 = verts
        v1 = p2 - p1
        v2 = p3 - p1

        # РќРѕСЂРјР°Р»СЊ Рє С‚СЂРµСѓРіРѕР»СЊРЅРёРєСѓ
        n = np.cross(v1, v2)
        n_norm = np.linalg.norm(n)

        if n_norm < 1e-12:
            return p1 @ self.direction  # Р’С‹СЂРѕР¶РґРµРЅРЅС‹Р№ С‚СЂРµСѓРіРѕР»СЊРЅРёРє

        n = n / n_norm

        # РџР°СЂР°РјРµС‚СЂ t РЅР° Р»СѓС‡Рµ: ray(t) = pt + t * direction
        # РЈСЂР°РІРЅРµРЅРёРµ РїР»РѕСЃРєРѕСЃС‚Рё: n В· (ray(t) - p1) = 0
        denom = np.dot(n, self.direction)

        if abs(denom) < 1e-12:
            return p1 @ self.direction  # Р›СѓС‡ РїР°СЂР°Р»Р»РµР»РµРЅ РїР»РѕСЃРєРѕСЃС‚Рё

        t = np.dot(n, p1 - pt) / denom
        intersection_point = pt + t * self.direction

        return intersection_point @ self.direction

    def check_point_status(
            self,
            point: Union[np.ndarray, Tuple],
            eps: float = 1e-4
    ) -> str:
        """
        РћРїСЂРµРґРµР»РёС‚СЊ СЃС‚Р°С‚СѓСЃ С‚РѕС‡РєРё: 'inside', 'outside' РёР»Рё 'on_surface'.

        Args:
            point: РљРѕРѕСЂРґРёРЅР°С‚С‹ С‚РѕС‡РєРё [x, y, z]
            eps: Р”РѕРїСѓСЃРє РґР»СЏ РѕРїСЂРµРґРµР»РµРЅРёСЏ РїСЂРёРЅР°РґР»РµР¶РЅРѕСЃС‚Рё РїРѕРІРµСЂС…РЅРѕСЃС‚Рё

        Returns:
            'inside', 'outside' РёР»Рё 'on_surface'
        """
        point = np.asarray(point, dtype=float)
        key_xyz = self._make_key(point)

        # РџСЂРѕРІРµСЂРєР° РєСЌС€Р° РїРѕ РїРѕР»РЅРѕР№ 3D РїРѕР·РёС†РёРё
        if key_xyz in self._point_cache:
            return self._point_cache[key_xyz]

        # РџСЂРѕРµРєС†РёСЏ С‚РѕС‡РєРё
        u0, v0 = self._project_point(point)
        key_proj = self._key_projection(u0, v0)

        # РљСЌС€ РїРµСЂРµСЃРµС‡РµРЅРёР№ РїРѕ РїСЂРѕРµРєС†РёРё
        if key_proj in self._intersections_cache:
            t_intersections = self._intersections_cache[key_proj]
        else:
            t_intersections = []
            vertices = np.asarray(self.mesh.vertices)
            triangles = np.asarray(self.mesh.triangles)

            # Р›СѓС‡: ray(t) = point + t * direction
            point_proj_uv = np.array([u0, v0])

            for tri in triangles:
                verts = vertices[tri]
                verts_proj = np.array([
                    self._project_point(verts[0])[:2],
                    self._project_point(verts[1])[:2],
                    self._project_point(verts[2])[:2]
                ])

                # РџСЂРѕРІРµСЂРєР°: РЅР°С…РѕРґРёС‚СЃСЏ Р»Рё РїСЂРѕРµРєС†РёСЏ С‚РѕС‡РєРё РІ РїСЂРѕРµРєС†РёРё С‚СЂРµСѓРіРѕР»СЊРЅРёРєР°
                if self._point_in_triangle_2d(point_proj_uv, verts_proj):
                    # Р’С‹С‡РёСЃР»РёС‚СЊ РєРѕРѕСЂРґРёРЅР°С‚Сѓ РїРµСЂРµСЃРµС‡РµРЅРёСЏ Р»СѓС‡Р° СЃ С‚СЂРµСѓРіРѕР»СЊРЅРёРєРѕРј
                    t_intersect = self._point_z_on_triangle(point, verts)
                    t_intersections.append(t_intersect)

            self._intersections_cache[key_proj] = t_intersections

        # РћРїСЂРµРґРµР»РµРЅРёРµ СЃС‚Р°С‚СѓСЃР° С‡РµСЂРµР· even-odd С‚РµСЃС‚
        point_t = point @ self.direction  # РљРѕРѕСЂРґРёРЅР°С‚Р° С‚РѕС‡РєРё РІРґРѕР»СЊ РЅР°РїСЂР°РІР»РµРЅРёСЏ

        on_surface = any(
            abs(point_t - t_surf) < eps for t_surf in t_intersections
        )

        behind = [t for t in t_intersections if t < point_t - eps]

        if on_surface:
            status = "on_surface"
        elif len(behind) % 2 == 1:
            status = "inside"
        else:
            status = "outside"

        self._point_cache[key_xyz] = status
        return status

    def is_point_inside(
            self,
            point: Union[np.ndarray, Tuple],
            eps: float = 1e-4
    ) -> bool:
        """РџСЂРѕРІРµСЂРёС‚СЊ, РЅР°С…РѕРґРёС‚СЃСЏ Р»Рё С‚РѕС‡РєР° РІРЅСѓС‚СЂРё СЃРµС‚РєРё."""
        return self.check_point_status(point, eps) == "inside"

    def is_point_outside(
            self,
            point: Union[np.ndarray, Tuple],
            eps: float = 1e-4
    ) -> bool:
        """РџСЂРѕРІРµСЂРёС‚СЊ, РЅР°С…РѕРґРёС‚СЃСЏ Р»Рё С‚РѕС‡РєР° СЃРЅР°СЂСѓР¶Рё СЃРµС‚РєРё."""
        return self.check_point_status(point, eps) == "outside"

    def is_point_on_surface(
            self,
            point: Union[np.ndarray, Tuple],
            eps: float = 1e-4
    ) -> bool:
        """РџСЂРѕРІРµСЂРёС‚СЊ, РЅР°С…РѕРґРёС‚СЃСЏ Р»Рё С‚РѕС‡РєР° РЅР° РїРѕРІРµСЂС…РЅРѕСЃС‚Рё СЃРµС‚РєРё."""
        return self.check_point_status(point, eps) == "on_surface"

    def batch_check(
            self,
            points: Union[List, np.ndarray],
            eps: float = 1e-4
    ) -> np.ndarray:
        """
        РџСЂРѕРІРµСЂРёС‚СЊ СЃС‚Р°С‚СѓСЃ РјРЅРѕР¶РµСЃС‚РІР° С‚РѕС‡РµРє.

        Args:
            points: РњР°СЃСЃРёРІ С„РѕСЂРјС‹ (N, 3) РёР»Рё СЃРїРёСЃРѕРє С‚РѕС‡РµРє
            eps: Р”РѕРїСѓСЃРє

        Returns:
            РњР°СЃСЃРёРІ СЃС‚Р°С‚СѓСЃРѕРІ ['inside', 'outside', 'on_surface']
        """
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Points РґРѕР»Р¶РµРЅ Р±С‹С‚СЊ РјР°СЃСЃРёРІРѕРј С„РѕСЂРјС‹ (N, 3)")

        statuses = np.array([
            self.check_point_status(p, eps) for p in points
        ])
        return statuses

    def get_direction(self) -> np.ndarray:
        """РџРѕР»СѓС‡РёС‚СЊ РЅРѕСЂРјР°Р»РёР·РѕРІР°РЅРЅРѕРµ РЅР°РїСЂР°РІР»РµРЅРёРµ С‚СЂР°СЃСЃРёСЂРѕРІРєРё."""
        return self.direction.copy()

    def clear_cache(self):
        """РћС‡РёСЃС‚РёС‚СЊ РєСЌС€Рё Р±РµР· РёР·РјРµРЅРµРЅРёСЏ РЅР°РїСЂР°РІР»РµРЅРёСЏ."""
        self._point_cache.clear()
        self._intersections_cache.clear()