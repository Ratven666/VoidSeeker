import pyvista as pv
import numpy as np


class VoxelModelPlotter:
    def __init__(self, voxelmodel):
        self.voxelmodel = voxelmodel
        try:
            self.surfacemesh = voxelmodel.mesh.mesh
        except AttributeError:
            self.surfacemesh = None

    def plot(self, cube_opacity=0.8, cube_color="#ffaa33", cube_edges=False, surface_opacity=0.3,
                            surface_color="blue"):
        voxels = self.voxelmodel.get_voxels()
        plotter = pv.Plotter()

        # Воксели только на поверхности
        surface_voxels = [v for v in voxels if getattr(v, "status", None) == "on_surface"]
        for voxel in surface_voxels:
            center = voxel.min_corner + 0.5 * voxel.size
            cube = pv.Cube(center=center, x_length=voxel.size, y_length=voxel.size, z_length=voxel.size)
            plotter.add_mesh(cube, color=cube_color, opacity=cube_opacity, show_edges=cube_edges)

        # Исходная mesh-поверхность (если есть)
        if self.surfacemesh is not None:
            surf = self.o3d_mesh_to_pyvista(self.surfacemesh)
            plotter.add_mesh(surf, color=surface_color, opacity=surface_opacity, style="surface")

        plotter.show_grid()
        plotter.show()

    @staticmethod
    def o3d_mesh_to_pyvista(mesh):
        import open3d as o3d
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        faces = np.hstack([np.full((triangles.shape[0], 1), 3), triangles]).flatten()
        return pv.PolyData(vertices, faces)
