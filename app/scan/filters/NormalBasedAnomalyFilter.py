import numpy as np
from sklearn.neighbors import KDTree
from app.scan.filters.ScanFilterABC import ScanFilterABC


class NormalBasedAnomalyFilter(ScanFilterABC):
    """
    Фильтр для удаления аномальных точек на основе анализа нормалей.

    Алгоритм:
    1. Для каждой точки вычисляем нормаль к локальной поверхности через PCA
    2. Сравниваем нормаль точки с нормалями её соседей
    3. Вычисляем степень согласованности нормалей
    4. Удаляем точки с сильно отклоняющимися нормалями
    """

    def __init__(self, k_neighbors=15, angle_threshold_degrees=45.0,
                 consistency_threshold=0.5, min_neighbors_for_normal=3):
        """
        Инициализация фильтра

        Args:
            k_neighbors (int): Количество ближайших соседей для анализа
            angle_threshold_degrees (float): Максимальный допустимый угол между нормалями в градусах
            consistency_threshold (float): Порог согласованности (0-1)
            min_neighbors_for_normal (int): Минимальное количество соседей для вычисления нормали
        """
        self.k_neighbors = k_neighbors
        self.angle_threshold = np.radians(angle_threshold_degrees)
        self.consistency_threshold = consistency_threshold
        self.min_neighbors_for_normal = min_neighbors_for_normal

    def filter(self, scan):
        """
        Фильтрация аномальных точек по нормалям

        Args:
            scan (Scan): Объект скана для фильтрации

        Returns:
            list: Отфильтрованный список точек ScanPoint
        """
        if len(scan) < self.k_neighbors + 1:
            print(f"Warning: Too few points ({len(scan)}) for normal computation")
            return scan._points

        # Получаем массив точек
        points_array = scan.get_points_array()

        # Строим KD-дерево для быстрого поиска соседей
        tree = KDTree(points_array)

        # Находим k ближайших соседей для каждой точки
        distances, indices = tree.query(points_array, k=self.k_neighbors + 1)

        # Убираем саму точку из результатов
        neighbor_indices = indices[:, 1:]

        # Предварительно вычисляем все нормали
        all_normals = self._compute_all_normals(points_array, neighbor_indices)

        # Фильтруем точки на основе согласованности нормалей
        filtered_points = []
        consistency_scores = []

        for i, point in enumerate(scan):
            if all_normals[i] is not None:
                consistency = self._compute_normal_consistency(i, all_normals, neighbor_indices[i])
                consistency_scores.append(consistency)

                if consistency >= self.consistency_threshold:
                    filtered_points.append(point)
            else:
                # Если не удалось вычислить нормаль, сохраняем точку
                filtered_points.append(point)
                consistency_scores.append(1.0)

        removed_count = len(scan) - len(filtered_points)
        avg_consistency = np.mean(consistency_scores) if consistency_scores else 1.0
        print(f"Normal-based filtering removed {removed_count} anomalous points "
              f"({len(filtered_points)} points remaining, avg_consistency={avg_consistency:.3f})")

        return filtered_points

    def _compute_all_normals(self, points_array, neighbor_indices):
        """
        Вычисляет нормали для всех точек
        """
        normals = []
        for i in range(len(points_array)):
            neighbors_idx = neighbor_indices[i]
            if len(neighbors_idx) >= self.min_neighbors_for_normal:
                normal = self._compute_single_normal(points_array[i], points_array[neighbors_idx])
                normals.append(normal)
            else:
                normals.append(None)
        return normals

    def _compute_single_normal(self, point, neighbors):
        """
        Вычисляет нормаль для одной точки через PCA
        """
        try:
            # Центрируем точки относительно текущей точки
            centered = neighbors - point

            # Вычисляем ковариационную матрицу
            cov_matrix = np.cov(centered, rowvar=False)

            # Находим собственные значения и векторы
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

            # Нормаль - это собственный вектор, соответствующий наименьшему собственному значению
            normal = eigenvectors[:, np.argmin(eigenvalues)]

            # Нормализуем нормаль
            normal_norm = np.linalg.norm(normal)
            if normal_norm > 1e-10:
                normal = normal / normal_norm
                return normal

        except Exception:
            pass

        return None

    def _compute_normal_consistency(self, point_idx, all_normals, neighbor_indices):
        """
        Вычисляет степень согласованности нормали точки с нормалями соседей
        """
        point_normal = all_normals[point_idx]
        if point_normal is None:
            return 1.0  # Если нормаль не вычислена, считаем точку нормальной

        consistent_neighbors = 0
        total_comparable = 0

        for neighbor_idx in neighbor_indices:
            if neighbor_idx < len(all_normals) and all_normals[neighbor_idx] is not None:
                neighbor_normal = all_normals[neighbor_idx]

                # Вычисляем угол между нормалями
                angle = self._angle_between_normals(point_normal, neighbor_normal)

                if angle <= self.angle_threshold:
                    consistent_neighbors += 1
                total_comparable += 1

        if total_comparable == 0:
            return 1.0  # Если не с чем сравнивать, считаем точку нормальной

        return consistent_neighbors / total_comparable

    def _angle_between_normals(self, normal1, normal2):
        """
        Вычисляет угол между двумя нормалями в радианах
        """
        dot_product = np.clip(np.dot(normal1, normal2), -1.0, 1.0)
        angle = np.arccos(dot_product)
        # Берем минимальный угол (нормали могут быть направлены в противоположные стороны)
        return min(angle, np.pi - angle)

    def get_normal_statistics(self, scan):
        """
        Возвращает статистику по нормалям для анализа
        """
        if len(scan) < self.k_neighbors + 1:
            return None

        points_array = scan.get_points_array()
        tree = KDTree(points_array)
        distances, indices = tree.query(points_array, k=self.k_neighbors + 1)
        neighbor_indices = indices[:, 1:]

        all_normals = self._compute_all_normals(points_array, neighbor_indices)

        # Вычисляем углы между соседними нормалями
        angles = []
        valid_normals = 0

        for i in range(len(all_normals)):
            if all_normals[i] is not None:
                valid_normals += 1
                for neighbor_idx in neighbor_indices[i][:5]:  # Проверяем первых 5 соседей
                    if neighbor_idx < len(all_normals) and all_normals[neighbor_idx] is not None:
                        angle = self._angle_between_normals(all_normals[i], all_normals[neighbor_idx])
                        angles.append(np.degrees(angle))

        return {
            'valid_normals': valid_normals,
            'total_points': len(scan),
            'coverage': valid_normals / len(scan),
            'mean_angle_deg': np.mean(angles) if angles else 0,
            'std_angle_deg': np.std(angles) if angles else 0,
            'max_angle_deg': np.max(angles) if angles else 0
        }

