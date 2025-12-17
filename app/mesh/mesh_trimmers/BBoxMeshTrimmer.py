import numpy as np
import open3d as o3d
from app.mesh.mesh_trimmers.MeshTrimmerABC import MeshTrimmerABC
from typing import Tuple, List, Dict, Set, Optional
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.qhull import QhullError


class BBoxMeshTrimmer(MeshTrimmerABC):
    """
    Класс для обрезки поверхности по кубоиду (axis-aligned bounding box).

    Особенности:
    - Грани результирующей поверхности совпадают с гранями bbox
    - Поддерживает опцию заполнения граничных полигонов
    - Для корректного заполнения создаёт дополнительные точки на граничных гранях
    - Гарантирует точное совпадение с гранями bbox
    - Использует Delaunay триангуляцию с агрессивной фильтрацией
    - Удаляет все полигоны за пределами контура
    """

    def __init__(self, base_mesh, bbox_bounds: Tuple[Tuple[float, float],
    Tuple[float, float],
    Tuple[float, float]],
                 fill_borders: bool = True,
                 border_face_resolution: int = 1):
        """
        Инициализация BBoxMeshTrimmer
        """
        super().__init__(base_mesh)
        self.bbox_bounds = bbox_bounds
        self.fill_borders = fill_borders
        self.border_face_resolution = max(1, border_face_resolution)
        self.metadata = {}

        # Распаковка границ
        self.x_min, self.x_max = bbox_bounds[0]
        self.y_min, self.y_max = bbox_bounds[1]
        self.z_min, self.z_max = bbox_bounds[2]

        # Валидация границ
        assert self.x_min < self.x_max, "x_min должен быть меньше x_max"
        assert self.y_min < self.y_max, "y_min должен быть меньше y_max"
        assert self.z_min < self.z_max, "z_min должен быть меньше z_max"

    def _is_point_in_bbox(self, point: np.ndarray, eps: float = 1e-10) -> bool:
        """Проверка, находится ли точка внутри bbox"""
        return (self.x_min - eps <= point[0] <= self.x_max + eps and
                self.y_min - eps <= point[1] <= self.y_max + eps and
                self.z_min - eps <= point[2] <= self.z_max + eps)

    def _is_point_on_boundary(self, point: np.ndarray,
                              eps: float = 1e-10) -> Tuple[bool, Set[str]]:
        """Проверка, находится ли точка на границе bbox."""
        boundary_faces = set()

        if abs(point[0] - self.x_min) < eps:
            boundary_faces.add('x_min')
        if abs(point[0] - self.x_max) < eps:
            boundary_faces.add('x_max')
        if abs(point[1] - self.y_min) < eps:
            boundary_faces.add('y_min')
        if abs(point[1] - self.y_max) < eps:
            boundary_faces.add('y_max')
        if abs(point[2] - self.z_min) < eps:
            boundary_faces.add('z_min')
        if abs(point[2] - self.z_max) < eps:
            boundary_faces.add('z_max')

        return len(boundary_faces) > 0, boundary_faces

    def _get_intersection_point(self, p1: np.ndarray,
                                p2: np.ndarray) -> Tuple[Optional[np.ndarray], Set[str]]:
        """Найти точку пересечения отрезка p1-p2 с поверхностью bbox."""
        t_values = {}
        eps = 1e-10

        # X-грани (x_min и x_max)
        if abs(p2[0] - p1[0]) > eps:
            for bound_x in [self.x_min, self.x_max]:
                t_x = (bound_x - p1[0]) / (p2[0] - p1[0])
                if -eps <= t_x <= 1 + eps:
                    point = p1 + t_x * (p2 - p1)
                    if (self.y_min - eps <= point[1] <= self.y_max + eps and
                            self.z_min - eps <= point[2] <= self.z_max + eps):
                        key = 'x_min' if bound_x == self.x_min else 'x_max'
                        t_values[key] = (max(0, min(1, t_x)), point)

        # Y-грани (y_min и y_max)
        if abs(p2[1] - p1[1]) > eps:
            for bound_y in [self.y_min, self.y_max]:
                t_y = (bound_y - p1[1]) / (p2[1] - p1[1])
                if -eps <= t_y <= 1 + eps:
                    point = p1 + t_y * (p2 - p1)
                    if (self.x_min - eps <= point[0] <= self.x_max + eps and
                            self.z_min - eps <= point[2] <= self.z_max + eps):
                        key = 'y_min' if bound_y == self.y_min else 'y_max'
                        t_values[key] = (max(0, min(1, t_y)), point)

        # Z-грани (z_min и z_max)
        if abs(p2[2] - p1[2]) > eps:
            for bound_z in [self.z_min, self.z_max]:
                t_z = (bound_z - p1[2]) / (p2[2] - p1[2])
                if -eps <= t_z <= 1 + eps:
                    point = p1 + t_z * (p2 - p1)
                    if (self.x_min - eps <= point[0] <= self.x_max + eps and
                            self.y_min - eps <= point[1] <= self.y_max + eps):
                        key = 'z_min' if bound_z == self.z_min else 'z_max'
                        t_values[key] = (max(0, min(1, t_z)), point)

        if not t_values:
            return None, set()

        # Выбираем точку с наименьшим t (первое пересечение)
        nearest_key = min(t_values.keys(), key=lambda k: t_values[k][0])
        t_nearest, intersection_point = t_values[nearest_key]

        # Проецируем точку точно на грань bbox
        eps_snap = 1e-9
        if abs(intersection_point[0] - self.x_min) < eps_snap:
            intersection_point[0] = self.x_min
        elif abs(intersection_point[0] - self.x_max) < eps_snap:
            intersection_point[0] = self.x_max

        if abs(intersection_point[1] - self.y_min) < eps_snap:
            intersection_point[1] = self.y_min
        elif abs(intersection_point[1] - self.y_max) < eps_snap:
            intersection_point[1] = self.y_max

        if abs(intersection_point[2] - self.z_min) < eps_snap:
            intersection_point[2] = self.z_min
        elif abs(intersection_point[2] - self.z_max) < eps_snap:
            intersection_point[2] = self.z_max

        # Определяем все грани, которые пересекает эта точка
        faces = set([nearest_key])

        if abs(intersection_point[0] - self.x_min) < eps_snap:
            faces.add('x_min')
        if abs(intersection_point[0] - self.x_max) < eps_snap:
            faces.add('x_max')
        if abs(intersection_point[1] - self.y_min) < eps_snap:
            faces.add('y_min')
        if abs(intersection_point[1] - self.y_max) < eps_snap:
            faces.add('y_max')
        if abs(intersection_point[2] - self.z_min) < eps_snap:
            faces.add('z_min')
        if abs(intersection_point[2] - self.z_max) < eps_snap:
            faces.add('z_max')

        return intersection_point, faces

    def _create_boundary_fill_vertices(self,
                                       boundary_face: str,
                                       edge_vertices: List[np.ndarray],
                                       resolution: int) -> List[np.ndarray]:
        """
        Создать дополнительные вершины на граничной грани для корректного заполнения.
        """
        if len(edge_vertices) < 3:
            return edge_vertices.copy()

        edge_vertices = np.array(edge_vertices)

        # Определяем координаты грани
        if boundary_face in ['x_min', 'x_max']:
            coords_2d = edge_vertices[:, [1, 2]]  # (y, z)
        elif boundary_face in ['y_min', 'y_max']:
            coords_2d = edge_vertices[:, [0, 2]]  # (x, z)
        else:  # z_min, z_max
            coords_2d = edge_vertices[:, [0, 1]]  # (x, y)

        # Сортируем вершины по периметру контура (по углам от центра)
        center = coords_2d.mean(axis=0)
        angles = np.arctan2(coords_2d[:, 1] - center[1], coords_2d[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        sorted_vertices = edge_vertices[sorted_indices]

        # Добавляем промежуточные вершины на каждом ребре контура
        all_vertices = []
        n = len(sorted_vertices)

        for i in range(n):
            current = sorted_vertices[i]
            next_vertex = sorted_vertices[(i + 1) % n]

            # Добавляем текущую вершину
            all_vertices.append(current.copy())

            # Добавляем промежуточные вершины между текущей и следующей
            if resolution > 1:
                for j in range(1, resolution):
                    t = j / resolution
                    intermediate = current + t * (next_vertex - current)
                    all_vertices.append(intermediate)

        return np.array(all_vertices)

    def _custom_trim_logic(self):
        """Основная логика обрезки поверхности по bbox"""

        if len(self.base_mesh.mesh.triangles) == 0:
            self.metadata = {}
            return self.base_mesh.mesh

        vertices = np.asarray(self.base_mesh.mesh.vertices, dtype=np.float64)
        triangles = np.asarray(self.base_mesh.mesh.triangles)

        new_vertices = []
        new_triangles = []
        vertex_mapping = {}
        new_vertex_count = 0

        # Словарь для сбора вершин на граничных гранях
        boundary_vertices_by_face = {
            'x_min': [], 'x_max': [],
            'y_min': [], 'y_max': [],
            'z_min': [], 'z_max': []
        }

        # === Этап 1: Обрезка треугольников ===
        for triangle_idx, triangle in enumerate(triangles):
            triangle_vertices = vertices[triangle]

            # Классификация вершин треугольника
            in_bbox_list = []
            on_boundary_faces_list = []

            for i, v_idx in enumerate(triangle):
                v = triangle_vertices[i]
                in_bbox = self._is_point_in_bbox(v)
                on_boundary, boundary_faces = self._is_point_on_boundary(v)

                in_bbox_list.append(in_bbox)
                on_boundary_faces_list.append(boundary_faces)

                # Сохраняем граничные вершины для возможного заполнения
                if on_boundary:
                    for face in boundary_faces:
                        if not any(np.allclose(v, vv) for vv in boundary_vertices_by_face[face]):
                            boundary_vertices_by_face[face].append(v.copy())

            inside_count = sum(in_bbox_list)

            # Логика обрезки в зависимости от количества внутренних вершин
            if inside_count == 3:
                # === Случай 1: Всё внутри bbox ===
                new_triangle = []
                for v_idx in triangle:
                    if v_idx not in vertex_mapping:
                        new_vertices.append(vertices[v_idx])
                        vertex_mapping[v_idx] = new_vertex_count
                        new_vertex_count += 1
                    new_triangle.append(vertex_mapping[v_idx])
                new_triangles.append(new_triangle)

            elif inside_count == 2:
                # === Случай 2: Две вершины внутри, одна снаружи ===
                valid_indices = [i for i in range(3) if in_bbox_list[i]]
                invalid_idx = [i for i in range(3) if not in_bbox_list[i]][0]

                valid_indices_sorted = sorted(valid_indices)

                # Добавляем две валидные вершины
                new_valid_indices = []
                for i in valid_indices_sorted:
                    v_idx = triangle[i]
                    if v_idx not in vertex_mapping:
                        new_vertices.append(vertices[v_idx].copy())
                        vertex_mapping[v_idx] = new_vertex_count
                        new_vertex_count += 1
                    new_valid_indices.append(vertex_mapping[v_idx])

                # Находим пересечения отрезков со сторонами bbox
                new_cut_indices = []
                for valid_idx in valid_indices_sorted:
                    p_valid = triangle_vertices[valid_idx]
                    p_invalid = triangle_vertices[invalid_idx]

                    intersection, faces = self._get_intersection_point(p_invalid, p_valid)
                    if intersection is not None:
                        new_vertices.append(intersection.copy())
                        new_cut_indices.append(new_vertex_count)

                        # Добавляем в список для граничного заполнения
                        for face in faces:
                            if not any(np.allclose(intersection, v) for v in boundary_vertices_by_face[face]):
                                boundary_vertices_by_face[face].append(intersection.copy())

                        new_vertex_count += 1

                # Создаём два новых треугольника
                if len(new_cut_indices) == 2:
                    new_triangles.append([new_valid_indices[0], new_valid_indices[1], new_cut_indices[0]])
                    new_triangles.append([new_valid_indices[1], new_cut_indices[1], new_cut_indices[0]])

            elif inside_count == 1:
                # === Случай 3: Одна вершина внутри, две снаружи ===
                valid_idx = [i for i in range(3) if in_bbox_list[i]][0]
                invalid_indices = [i for i in range(3) if not in_bbox_list[i]]

                # Добавляем валидную вершину
                v_idx = triangle[valid_idx]
                if v_idx not in vertex_mapping:
                    new_vertices.append(vertices[v_idx].copy())
                    vertex_mapping[v_idx] = new_vertex_count
                    new_vertex_count += 1
                new_valid_idx = vertex_mapping[v_idx]

                # Находим два пересечения
                new_cut_indices = []
                for inv_idx in invalid_indices:
                    p_valid = triangle_vertices[valid_idx]
                    p_invalid = triangle_vertices[inv_idx]

                    intersection, faces = self._get_intersection_point(p_invalid, p_valid)
                    if intersection is not None:
                        new_vertices.append(intersection.copy())
                        new_cut_indices.append(new_vertex_count)

                        # Добавляем в список для граничного заполнения
                        for face in faces:
                            if not any(np.allclose(intersection, v) for v in boundary_vertices_by_face[face]):
                                boundary_vertices_by_face[face].append(intersection.copy())

                        new_vertex_count += 1

                # Создаём один новый треугольник
                if len(new_cut_indices) == 2:
                    new_triangles.append([new_valid_idx, new_cut_indices[0], new_cut_indices[1]])

        # === Этап 2: Добавляем граничное заполнение, если требуется ===
        if self.fill_borders:
            self._add_boundary_fill(new_vertices, new_triangles, new_vertex_count,
                                    boundary_vertices_by_face)

        # === Этап 3: Создаём результирующую поверхность ===
        result_mesh = o3d.geometry.TriangleMesh()

        if len(new_triangles) == 0:
            print("Предупреждение: Все треугольники удалены при обрезке по bbox")
            self.metadata = {}
            return o3d.geometry.TriangleMesh()

        result_mesh.vertices = o3d.utility.Vector3dVector(np.array(new_vertices))
        result_mesh.triangles = o3d.utility.Vector3iVector(np.array(new_triangles))
        result_mesh.compute_vertex_normals()
        result_mesh.compute_triangle_normals()

        # === Этап 4: Сохраняем метаданные ===
        bbox = result_mesh.get_axis_aligned_bounding_box()
        surface_area = result_mesh.get_surface_area()

        try:
            volume = result_mesh.get_volume()
        except Exception:
            volume = None

        self.metadata = {
            "bbox": bbox,
            "surface_area": surface_area,
            "volume": volume,
            "triangles_count": len(new_triangles),
            "vertices_count": len(new_vertices),
            "fill_borders": self.fill_borders,
            "boundary_vertices_by_face": {k: len(v) for k, v in boundary_vertices_by_face.items()}
        }

        print(f"Обрезка по BBox:")
        print(f" Параметры BBox:")
        print(f"  X: [{self.x_min:.3f}, {self.x_max:.3f}]")
        print(f"  Y: [{self.y_min:.3f}, {self.y_max:.3f}]")
        print(f"  Z: [{self.z_min:.3f}, {self.z_max:.3f}]")
        print(f" Результаты:")
        print(f"  Треугольники: {len(new_triangles)} создано")
        print(f"  Вершины: {len(new_vertices)} создано")
        print(f"  Граничное заполнение: {'включено' if self.fill_borders else 'отключено'}")
        print(f"  Площадь поверхности: {surface_area:.4f}")
        if volume is not None:
            print(f"  Объём: {volume:.4f}")

        print(f" Граничные вершины по гранях:")
        for face, verts in boundary_vertices_by_face.items():
            if len(verts) > 0:
                print(f"  {face}: {len(verts)} вершин")

        return result_mesh

    def _add_boundary_fill(self, vertices: List, triangles: List,
                           vertex_count: int, boundary_vertices_by_face: Dict):
        """Добавляет полигоны для заполнения граничных граней bbox"""

        for face, edge_verts in boundary_vertices_by_face.items():
            if len(edge_verts) < 3:
                continue

            # Создаём дополнительные вершины для заполнения
            all_verts = self._create_boundary_fill_vertices(
                face, edge_verts, self.border_face_resolution
            )

            if len(all_verts) < 3:
                continue

            # Проецируем вершины на 2D координаты грани
            face_vertices_2d = self._project_to_face(face, all_verts)

            # Триангулируем грань
            face_indices = self._triangulate_face_delaunay(face_vertices_2d)

            # АГРЕССИВНАЯ ФИЛЬТРАЦИЯ: удаляем полигоны за пределами контура
            face_indices = self._aggressive_filter_triangles(
                face_vertices_2d, face_indices
            )

            # Добавляем треугольники в результирующую поверхность
            for tri_idx in face_indices:
                triangle = [vertex_count + i for i in tri_idx]
                triangles.append(triangle)

            # Добавляем вершины
            for v in all_verts:
                vertices.append(v.copy())
                vertex_count += 1

    def _project_to_face(self, face: str, vertices_3d) -> np.ndarray:
        """Проецирует 3D вершины на 2D координаты граничной грани"""
        verts_2d = []

        for v in vertices_3d:
            if face in ['x_min', 'x_max']:
                verts_2d.append([v[1], v[2]])
            elif face in ['y_min', 'y_max']:
                verts_2d.append([v[0], v[2]])
            else:  # z_min, z_max
                verts_2d.append([v[0], v[1]])

        return np.array(verts_2d)

    def _triangulate_face_delaunay(self, vertices_2d: np.ndarray) -> List[List[int]]:
        """
        Триангуляция граничной грани используя Delaunay триангуляцию.
        """
        triangles = []
        n = len(vertices_2d)

        if n < 3:
            return triangles

        # Для малых контуров используем прямое разбиение
        if n <= 4:
            if n == 3:
                return [[0, 1, 2]]
            elif n == 4:
                return [[0, 1, 2], [0, 2, 3]]

        try:
            points = np.array(vertices_2d, dtype=np.float64)

            # Добавляем небольшой шум для избежания вырожденных случаев
            epsilon = 1e-12
            points = points + np.random.normal(0, epsilon, points.shape)

            tri = Delaunay(points)

            # Преобразуем индексы Delaunay в список треугольников
            for simplex in tri.simplices:
                triangles.append(list(simplex))

            return triangles

        except (QhullError, Exception) as e:
            # Если Delaunay не сработал, используем веер как fallback
            print(f"Предупреждение: Delaunay не сработала для {n} вершин, используем веер")
            return self._triangulate_face_fan(vertices_2d)

    def _triangulate_face_fan(self, vertices_2d: np.ndarray) -> List[List[int]]:
        """Триангуляция веером из первой вершины (fallback метод)"""
        triangles = []
        n = len(vertices_2d)

        if n < 3:
            return triangles

        for i in range(1, n - 1):
            triangles.append([0, i, i + 1])

        return triangles

    def _aggressive_filter_triangles(self,
                                     vertices_2d: np.ndarray,
                                     triangles: List[List[int]]) -> List[List[int]]:
        """
        АГРЕССИВНАЯ фильтрация треугольников, которые находятся за пределами контура.

        Используются несколько методов проверки:
        1. Проверка центра треугольника
        2. Проверка всех вершин треугольника
        3. Проверка выпуклой оболочки
        """
        if len(triangles) == 0:
            return triangles

        # Построим выпуклую оболочку контура
        try:
            hull = ConvexHull(vertices_2d)
        except QhullError:
            # Если не можем построить выпуклую оболочку, возвращаем все
            return triangles

        # Получаем точки на границе выпуклой оболочки
        hull_points = vertices_2d[hull.vertices]

        # Вычисляем минимальное и максимальное расстояние внутри контура
        centroid = hull_points.mean(axis=0)
        distances_from_center = np.linalg.norm(hull_points - centroid, axis=1)
        max_dist_from_center = distances_from_center.max()
        min_dist_from_center = distances_from_center.min()

        filtered = []

        for tri_idx in triangles:
            # Получаем вершины треугольника
            tri_verts = vertices_2d[tri_idx]
            tri_center = tri_verts.mean(axis=0)

            # Проверка 1: Все ли вершины треугольника внутри выпуклой оболочки?
            all_inside = all(
                self._point_in_convex_hull(v, hull_points)
                for v in tri_verts
            )

            # Проверка 2: Центр треугольника близко к центру контура?
            dist_to_center = np.linalg.norm(tri_center - centroid)
            center_reasonable = dist_to_center <= (max_dist_from_center * 1.2)

            # Проверка 3: Все вершины близко друг к другу?
            max_edge_len = np.linalg.norm(
                np.max(tri_verts, axis=0) - np.min(tri_verts, axis=0)
            )
            typical_edge_len = np.linalg.norm(
                np.max(hull_points, axis=0) - np.min(hull_points, axis=0)
            ) / 10  # Примерная длина ребра контура

            reasonable_size = max_edge_len <= (typical_edge_len * 3)

            # Включаем треугольник, если выполнены все условия
            if all_inside and center_reasonable and reasonable_size:
                filtered.append(tri_idx)

        # Если отфильтровали слишком много, возвращаем с менее строгой фильтрацией
        if len(filtered) == 0 and len(triangles) > 0:
            print(f"Предупреждение: Агрессивная фильтрация удалила все полигоны!")
            print(f"  Использую менее строгую фильтрацию...")

            # Менее строгая фильтрация: только проверка центра
            for tri_idx in triangles:
                tri_verts = vertices_2d[tri_idx]
                tri_center = tri_verts.mean(axis=0)

                # Просто проверяем, что центр не слишком далеко от контура
                dist_to_hull = self._point_distance_to_hull(tri_center, hull_points)
                if dist_to_hull < (max_dist_from_center * 0.5):
                    filtered.append(tri_idx)

        return filtered

    def _point_in_convex_hull(self, point: np.ndarray, hull_points: np.ndarray) -> bool:
        """
        Проверяет, находится ли точка внутри выпуклой оболочки.
        """
        try:
            all_points = np.vstack([hull_points, point])
            new_hull = ConvexHull(all_points)

            # Если выпуклая оболочка не изменилась по размеру,
            # значит точка была внутри
            return len(new_hull.vertices) == len(hull_points)
        except QhullError:
            return False

    def _point_distance_to_hull(self, point: np.ndarray, hull_points: np.ndarray) -> float:
        """
        Вычисляет расстояние от точки до ближайшей точки выпуклой оболочки.
        """
        distances = np.linalg.norm(hull_points - point, axis=1)
        return distances.min()

    def get_metadata(self) -> Dict:
        """Возвращает метаданные обрезки"""
        return self.metadata


# === Пример использования ===
if __name__ == "__main__":
    from app.mesh.MeshDxf import MeshDxf
    from app.scan.Scan import Scan

    # Пример использования
    mesh = MeshDxf().init_mesh_from_dxf("../../../src/Рудное тело.dxf")

    scan = Scan(scan_name="TestScan")
    scan.import_points_from_file(file_path="../../../src/Камеры/data_3.dxf")

    # Определяем bbox для обрез
    bbox_bounds = (
        (scan.borders["x_min"] - 20, scan.borders["x_max"] + 20),  # x_min, x_max
        (scan.borders["y_min"] - 20, scan.borders["y_max"] + 20),  # y_min, y_max
        (scan.borders["z_min"] - 20, scan.borders["z_max"] + 20)  # z_min, z_max
    )

    # Обрезаем поверхность с заполнением граничных граней
    trimmer = BBoxMeshTrimmer(
        base_mesh=mesh,
        bbox_bounds=bbox_bounds,
        fill_borders=True,
        border_face_resolution=0.1
    )

    result_mesh = trimmer.trim_mesh()
    metadata = trimmer.get_metadata()

    # Визуализируем
    result_mesh.plot()