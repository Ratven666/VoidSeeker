import open3d as o3d
import numpy as np
from scipy import ndimage

from app.scan.Scan import Scan

import open3d as o3d
import numpy as np

scan = Scan(scan_name="TestScan")
print(scan)
scan.import_points_from_file(file_path="src/Камеры/data_3.dxf")
print(scan)
points = scan.get_points_array()

# Создаем облако точек Open3D
pcd = o3d.geometry.PointCloud()

def visualize_voxel_wireframe(voxel_grid):
    """
    Визуализация вокселей как каркаса
    """
    # Создаем линии для каждого вокселя
    line_sets = []

    for voxel in voxel_grid.get_voxels():
        center = voxel_grid.get_voxel_center_coordinate(voxel.grid_index)
        half_size = voxel_grid.voxel_size / 2

        # Углы вокселя
        corners = []
        for dx in [-1, 1]:
            for dy in [-1, 1]:
                for dz in [-1, 1]:
                    corner = center + np.array([dx, dy, dz]) * half_size
                    corners.append(corner)

        corners = np.array(corners)

        # Линии куба (12 линий)
        lines = [
            [0, 1], [1, 3], [3, 2], [2, 0],  # нижняя грань
            [4, 5], [5, 7], [7, 6], [6, 4],  # верхняя грань
            [0, 4], [1, 5], [2, 6], [3, 7]  # вертикальные линии
        ]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color([0.8, 0.8, 0.8])  # Серый цвет

        line_sets.append(line_set)

    # Визуализация
    o3d.visualization.draw_geometries(line_sets,
                                      window_name="Wireframe Voxels")


def efficient_wireframe_visualization(voxel_grid):
    """
    Более эффективная визуализация каркаса
    """
    all_points = []
    all_lines = []
    line_color = [0.6, 0.6, 0.8]  # Голубовато-серый

    for i, voxel in enumerate(voxel_grid.get_voxels()):
        center = voxel_grid.get_voxel_center_coordinate(voxel.grid_index)
        half_size = voxel_grid.voxel_size / 2

        # Углы вокселя
        corners = []
        for dx in [-1, 1]:
            for dy in [-1, 1]:
                for dz in [-1, 1]:
                    corner = center + np.array([dx, dy, dz]) * half_size
                    corners.append(corner)

        corners = np.array(corners)
        start_idx = len(all_points)
        all_points.extend(corners)

        # Линии куба
        lines = np.array([
            [0, 1], [1, 3], [3, 2], [2, 0],  # нижняя грань
            [4, 5], [5, 7], [7, 6], [6, 4],  # верхняя грань
            [0, 4], [1, 5], [2, 6], [3, 7]  # вертикальные линии
        ]) + start_idx

        all_lines.extend(lines)

    if all_points:
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.array(all_points))
        line_set.lines = o3d.utility.Vector2iVector(np.array(all_lines))
        line_set.paint_uniform_color(line_color)

        o3d.visualization.draw_geometries([line_set],
                                          window_name="Efficient Wireframe")

def voxel_grid_to_array(voxel_grid):
    """
    Конвертирует VoxelGrid в 3D numpy array
    """
    # Получаем все воксели
    voxels = list(voxel_grid.get_voxels())
    if not voxels:
        return np.array([]), np.array([0, 0, 0]), voxel_grid.voxel_size

    # Находим границы индексов
    indices = np.array([voxel.grid_index for voxel in voxels])
    min_bounds = np.min(indices, axis=0)
    max_bounds = np.max(indices, axis=0)

    # Создаем массив
    shape = (max_bounds - min_bounds + 1).astype(int)
    voxel_array = np.zeros(shape, dtype=bool)

    # Заполняем массив
    for voxel in voxels:
        idx = voxel.grid_index - min_bounds
        if all(0 <= idx[i] < shape[i] for i in range(3)):
            voxel_array[tuple(idx)] = True

    return voxel_array, min_bounds, voxel_grid.voxel_size


def array_to_voxel_grid(voxel_array, origin, voxel_size):
    """
    Конвертирует 3D numpy array обратно в VoxelGrid
    """
    # Создаем облако точек из заполненных вокселей
    points = []
    indices = np.argwhere(voxel_array)

    for idx in indices:
        # Конвертируем индекс в мировые координаты
        world_coord = (idx + origin) * voxel_size
        points.append(world_coord)

    if not points:
        # Возвращаем пустую воксельную сетку
        empty_pcd = o3d.geometry.PointCloud()
        empty_pcd.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0]]))
        return o3d.geometry.VoxelGrid.create_from_point_cloud(empty_pcd, voxel_size)

    points = np.array(points)

    # Создаем новую воксельную сетку
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    new_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)

    return new_voxel_grid


def create_spherical_kernel(radius):
    """
    Создает сферическое ядро для морфологических операций
    """
    size = 2 * radius + 1
    kernel = np.zeros((size, size, size), dtype=bool)
    center = radius

    for i in range(size):
        for j in range(size):
            for k in range(size):
                distance = np.sqrt((i - center) ** 2 + (j - center) ** 2 + (k - center) ** 2)
                if distance <= radius:
                    kernel[i, j, k] = True

    return kernel


def fill_voxel_gaps_morphological(voxel_grid, kernel_size=3):
    """
    Заполняет промежутки между вокселями с помощью морфологических операций
    """
    # Конвертируем воксельную сетку в 3D массив
    voxel_array, origin, voxel_size = voxel_grid_to_array(voxel_grid)

    if voxel_array.size == 0:
        return voxel_grid

    # Применяем морфологическое закрытие (dilation + erosion)
    kernel = np.ones((kernel_size, kernel_size, kernel_size), dtype=bool)
    filled_array = ndimage.binary_closing(voxel_array, structure=kernel, iterations=1)

    # Конвертируем обратно в VoxelGrid
    filled_voxel_grid = array_to_voxel_grid(filled_array, origin, voxel_size)

    return filled_voxel_grid


def fill_voxel_holes(voxel_grid):
    """
    Заполняет внутренние полости в воксельной модели
    """
    voxel_array, origin, voxel_size = voxel_grid_to_array(voxel_grid)

    if voxel_array.size == 0:
        return voxel_grid

    # Используем scipy для заполнения отверстий
    from scipy.ndimage import binary_fill_holes

    # Заполняем отверстия в 3D
    filled_array = binary_fill_holes(voxel_array)

    filled_voxel_grid = array_to_voxel_grid(filled_array, origin, voxel_size)
    return filled_voxel_grid


def dilate_voxels(voxel_grid, dilation_radius=1):
    """
    Расширяет воксели для заполнения промежутков
    """
    voxel_array, origin, voxel_size = voxel_grid_to_array(voxel_grid)

    if voxel_array.size == 0:
        return voxel_grid

    # Создаем сферический структурный элемент
    from scipy.ndimage import binary_dilation

    # Создаем сферическое ядро для дилатации
    kernel = create_spherical_kernel(dilation_radius)

    # Применяем дилатацию
    dilated_array = binary_dilation(voxel_array, structure=kernel)

    dilated_voxel_grid = array_to_voxel_grid(dilated_array, origin, voxel_size)
    return dilated_voxel_grid


def comprehensive_voxel_filling(voxel_grid, point_cloud=None, method='combined'):
    """
    Комплексное заполнение вокселей с выбором метода
    """
    original_voxel_count = len(list(voxel_grid.get_voxels()))
    print(f"Исходное количество вокселей: {original_voxel_count}")

    if method == 'dilation':
        # Метод 1: Дилатация
        filled_grid = dilate_voxels(voxel_grid, dilation_radius=1)

    elif method == 'morphological':
        # Метод 2: Морфологическое закрытие
        filled_grid = fill_voxel_gaps_morphological(voxel_grid, kernel_size=3)

    elif method == 'hole_filling':
        # Метод 3: Заполнение отверстий
        filled_grid = fill_voxel_holes(voxel_grid)

    elif method == 'combined':
        # Комбинированный подход
        # 1. Дилатация для заполнения мелких промежутков
        temp_grid = dilate_voxels(voxel_grid, dilation_radius=1)

        # 2. Заполнение отверстий
        temp_grid = fill_voxel_holes(temp_grid)

        # 3. Морфологическое закрытие для сглаживания
        filled_grid = fill_voxel_gaps_morphological(temp_grid, kernel_size=2)
    else:
        print(f"Неизвестный метод: {method}")
        return voxel_grid

    filled_voxel_count = len(list(filled_grid.get_voxels()))
    print(f"Количество вокселей после заполнения: {filled_voxel_count}")
    print(f"Добавлено вокселей: {filled_voxel_count - original_voxel_count}")

    return filled_grid


def simple_filling_example():
    """
    Простой пример заполнения вокселей
    """


    # scan = Scan(scan_name="TestScan")
    # print(scan)
    # scan.import_points_from_file(file_path="src/Камеры/data_3.dxf")
    # print(scan)
    # points = scan.get_points_array()
    #
    # # Создаем облако точек Open3D
    # pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Создаем воксельную сетку
    voxel_size = 0.3
    original_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)

    print("Исходная воксельная сетка:")
    print(f"Количество вокселей: {len(list(original_voxel_grid.get_voxels()))}")

    # Визуализация исходной сетки
    o3d.visualization.draw_geometries([original_voxel_grid],
                                      window_name="Исходная воксельная сетка")

    # Заполняем промежутки
    filled_voxel_grid = comprehensive_voxel_filling(original_voxel_grid, method='combined')

    # Визуализация заполненной сетки
    o3d.visualization.draw_geometries([filled_voxel_grid],
                                      window_name="Заполненная воксельная сетка")

    return original_voxel_grid, filled_voxel_grid

def visualize_voxel_grid_basic(voxel_grid):
    """
    Базовая визуализация воксельной сетки
    """
    o3d.visualization.draw_geometries([voxel_grid],
                                     window_name="Voxel Grid",
                                     width=1024,
                                     height=768)

# Запуск примера
if __name__ == "__main__":
    original, filled = simple_filling_example()
    efficient_wireframe_visualization(filled)