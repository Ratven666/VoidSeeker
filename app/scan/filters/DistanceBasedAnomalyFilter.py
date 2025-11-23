import numpy as np
from sklearn.neighbors import KDTree
from app.scan.filters.ScanFilterABC import ScanFilterABC


class DistanceBasedAnomalyFilter(ScanFilterABC):
    """
    Фильтр для удаления аномальных точек на основе анализа расстояний до ближайших соседей.

    Алгоритм:
    1. Для каждой точки находим k ближайших соседей
    2. Вычисляем среднее расстояние до соседей
    3. Анализируем статистическое распределение средних расстояний
    4. Удаляем точки, где среднее расстояние значительно превышает типичное
    """

    def __init__(self, k_neighbors=15, distance_threshold_std=2.0, method='statistical'):
        """
        Инициализация фильтра

        Args:
            k_neighbors (int): Количество ближайших соседей для анализа
            distance_threshold_std (float): Порог в стандартных отклонениях для фильтрации
            method (str): Метод фильтрации ('statistical', 'percentile', 'absolute')
        """
        self.k_neighbors = k_neighbors
        self.distance_threshold_std = distance_threshold_std
        self.method = method

        if method not in ['statistical', 'percentile', 'absolute']:
            raise ValueError("Method must be 'statistical', 'percentile', or 'absolute'")

    def filter(self, scan):
        """
        Фильтрация аномальных точек по расстояниям

        Args:
            scan (Scan): Объект скана для фильтрации

        Returns:
            list: Отфильтрованный список точек ScanPoint
        """
        if len(scan) < self.k_neighbors + 1:
            print(f"Warning: Too few points ({len(scan)}) for k_neighbors={self.k_neighbors}")
            return scan._points

        # Получаем массив точек
        points_array = scan.get_points_array()

        # Строим KD-дерево для быстрого поиска соседей
        tree = KDTree(points_array)

        # Находим k ближайших соседей для каждой точки
        distances, indices = tree.query(points_array, k=self.k_neighbors + 1)

        # Убираем саму точку из результатов (расстояние = 0)
        neighbor_distances = distances[:, 1:]

        # Вычисляем средние расстояния для каждой точки
        mean_distances = np.mean(neighbor_distances, axis=1)

        # Определяем порог фильтрации
        threshold = self._compute_threshold(mean_distances)

        # Фильтруем точки
        filtered_points = []
        for i, point in enumerate(scan):
            if mean_distances[i] <= threshold:
                filtered_points.append(point)

        removed_count = len(scan) - len(filtered_points)
        print(f"Distance-based filtering removed {removed_count} anomalous points "
              f"({len(filtered_points)} points remaining, threshold={threshold:.4f})")

        return filtered_points

    def _compute_threshold(self, mean_distances):
        """
        Вычисляет порог фильтрации в зависимости от выбранного метода
        """
        if self.method == 'statistical':
            # Статистический метод: среднее + N * стандартное отклонение
            overall_mean = np.mean(mean_distances)
            overall_std = np.std(mean_distances)

            if overall_std < 1e-10:  # Если все расстояния одинаковы
                return overall_mean * 1.5

            return overall_mean + self.distance_threshold_std * overall_std

        elif self.method == 'percentile':
            # Метод процентилей: отсекаем верхние X%
            percentile = 100 - self.distance_threshold_std
            return np.percentile(mean_distances, percentile)

        else:  # absolute
            # Абсолютный порог
            return self.distance_threshold_std

    def get_distance_statistics(self, scan):
        """
        Возвращает статистику по расстояниям для анализа
        """
        if len(scan) < self.k_neighbors + 1:
            return None

        points_array = scan.get_points_array()
        tree = KDTree(points_array)
        distances, _ = tree.query(points_array, k=self.k_neighbors + 1)
        mean_distances = np.mean(distances[:, 1:], axis=1)

        return {
            'min': np.min(mean_distances),
            'max': np.max(mean_distances),
            'mean': np.mean(mean_distances),
            'std': np.std(mean_distances),
            'median': np.median(mean_distances),
            'q95': np.percentile(mean_distances, 95)
        }
