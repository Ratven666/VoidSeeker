
# Создание воксельной сетки
voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(
    poisson_mesh.mesh,
    voxel_size=0.1  # размер вокселя
)


def create_filled_voxels_raycasting(mesh, voxel_size=0.05):
    """
    Создание заполненной воксельной модели с помощью ray casting
    """
    # Получаем bounding box меша
    bbox = mesh.get_axis_aligned_bounding_box()
    min_bound = bbox.min_bound
    max_bound = bbox.max_bound

    # Создаем сцену для ray casting
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    # Создаем сетку вокселей
    x_coords = np.arange(min_bound[0], max_bound[0], voxel_size)
    y_coords = np.arange(min_bound[1], max_bound[1], voxel_size)
    z_coords = np.arange(min_bound[2], max_bound[2], voxel_size)

    # Центры вокселей
    xx, yy, zz = np.meshgrid(x_coords + voxel_size / 2,
                             y_coords + voxel_size / 2,
                             z_coords + voxel_size / 2,
                             indexing='ij')

    voxel_centers = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)

    # Проверяем какие воксели внутри меша
    query_points = o3d.core.Tensor(voxel_centers, dtype=o3d.core.Dtype.Float32)
    signed_distance = scene.compute_signed_distance(query_points)

    # Отбираем воксели внутри (отрицательное расстояние)
    inside_mask = signed_distance.numpy() < 0
    inside_centers = voxel_centers[inside_mask]

    # Создаем точечное облако из внутренних точек
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(inside_centers)

    # Создаем воксельную сетку
    filled_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        pcd, voxel_size=voxel_size
    )

    return filled_voxel_grid

# voxel_grid = create_filled_voxels_raycasting(poisson_mesh.mesh, voxel_size=0.5)

# Визуализация
o3d.visualization.draw_geometries([voxel_grid])

# for voxel in voxel_grid.get_voxels():
#     print(voxel)
import open3d as o3d
import numpy as np
from collections import defaultdict


def split_voxel_grid_into_layers(voxel_grid):
    # Получаем все воксели через get_voxels()
    voxels = voxel_grid.get_voxels()

    if not voxels:
        print("Воксельная сетка пуста")
        return []

    # Группируем воксели по Y-координате (вертикальная ось)
    layers = defaultdict(list)

    for voxel in voxels:
        # grid_index содержит целочисленные координаты вокселя
        y_index = voxel.grid_index[2]  # Y-координата в индексах вокселей
        layers[y_index].append(voxel)

    return layers


def create_voxel_layers(voxel_grid):
    layers_dict = split_voxel_grid_into_layers(voxel_grid)

    # Сортируем слои по Y-координате
    sorted_layers = sorted(layers_dict.items(), key=lambda x: x[0])

    voxel_layers = []

    for layer_index, (y_index, voxels) in enumerate(sorted_layers):
        # Получаем реальные координаты из индексов
        voxel_size = voxel_grid.voxel_size
        origin = voxel_grid.origin

        layer_voxels = []

        for voxel in voxels:
            # Вычисляем реальные координаты центра вокселя
            center = np.array(voxel.grid_index) * voxel_size + origin + voxel_size / 2.0
            layer_voxels.append({
                'grid_index': voxel.grid_index,
                'center': center,
                'color': voxel.color
            })

        voxel_layers.append({
            'layer_index': layer_index,
            'y_index': y_index,
            'y_coordinate': y_index * voxel_size + origin[1] + voxel_size / 2.0,
            'voxels': layer_voxels,
            'voxel_count': len(voxels)
        })

    return voxel_layers


def visualize_layers(voxel_grid):
    layers = create_voxel_layers(voxel_grid)

    print(f"Всего слоев: {len(layers)}")
    print(f"Размер вокселя: {voxel_grid.voxel_size}")
    print(f"Начало координат: {voxel_grid.origin}")

    for i, layer in enumerate(layers):
        print(
            f"Слой {i}: Y-индекс={layer['y_index']}, Y-координата={layer['y_coordinate']:.3f}, вокселей: {layer['voxel_count']}")

        # Создаем точечное облако для визуализации текущего слоя
        points = []
        colors = []

        for voxel_data in layer['voxels']:
            points.append(voxel_data['center'])
            colors.append(voxel_data['color'])

        if points:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            # Визуализация одного слоя
            o3d.visualization.draw_geometries([pcd],
                                              window_name=f"Слой {i}, Y={layer['y_coordinate']:.3f}")

visualize_layers(voxel_grid)

# mesh.trim_by_bpa(scale=1)
# conv_hull_mesh.plot()

# bpa_mesh = BallPivotingAlgorithmMeshOpen3D().init_mesh_from_scan(scan)

# tr_mesh = BorderMeshMeshTrimmer(poisson_mesh, conv_hull_mesh, scale=1).trim_mesh()
# tr_mesh = MeshTrimmers(poisson_mesh).trim_by_border_mesh(bpa_mesh, scale=1.2, outside_only=True)

# z_level = scan.borders["z_max"]
# print(z_level)
# tr_mesh = ZLevelsMeshTrimmer(base_mesh=poisson_mesh, z_level=z_level).trim_mesh()
# tr_mesh.plot()
