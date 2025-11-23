import open3d as o3d
import numpy as np

def make_cube(center, size, color):
    mesh = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size)
    mesh.translate(center - 0.5 * size)
    mesh.paint_uniform_color(color[:3])
    mesh.compute_vertex_normals()
    return mesh

class VoxelModelOpen3DVisualizer:
    def __init__(self, voxelmodel):
        self.voxelmodel = voxelmodel
        try:
            self.surfacemesh = voxelmodel.mesh.mesh
        except AttributeError:
            self.surfacemesh = voxelmodel.mesh

    def to_cubemeshes(self, only_surface=True, status_colors=None):
        if status_colors is None:
            status_colors = {
                "inside": [0.2, 0.8, 0.2, 0.5],      # green (alpha=0.5)
                "on_surface": [1.0, 0.6, 0.1, 0.5],  # orange (alpha=0.5)
                "outside": [0.8, 0.8, 0.8, 0.2]      # gray  (alpha=0.2)
            }
        voxels = self.voxelmodel.get_voxels()
        if only_surface:
            voxels = [v for v in voxels if getattr(v, "status", None) == "on_surface"]
        cubes = []
        for v in voxels:
            center = v.min_corner + 0.5 * v.size
            color = status_colors.get(getattr(v, "status", "outside"), [0.8, 0.8, 0.8, 0.5])
            cubes.append(make_cube(center, v.size, color))
        return cubes

    def plot(self, only_surface=True):
        self.surfacemesh.compute_vertex_normals()
        self.surfacemesh.paint_uniform_color([0.3, 0.5, 1.0])
        cubes = self.to_cubemeshes(only_surface=only_surface)
        # Просто вызов draw_geometries: Open3D отрисует кубики с заданным цветом, прозрачность регулируйте в GUI или экспортом
        o3d.visualization.draw_geometries(
            [self.surfacemesh] + cubes,
            window_name="Voxel Cubes & Surface",
            width=1024, height=900
        )
