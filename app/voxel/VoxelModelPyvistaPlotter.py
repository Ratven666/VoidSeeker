import pyvista as pv
import numpy as np

class VoxelModelPyvistaPlotter:
    def __init__(self, voxelmodel, surface_mesh=None):
        self.voxelmodel = voxelmodel
        if surface_mesh is not None:
            # Если это open3d mesh, конвертируем, если уже pyvista -- используем как есть
            if hasattr(surface_mesh, "vertices") and hasattr(surface_mesh, "triangles"):
                self.surface_mesh = self.o3d_mesh_to_pyvista(surface_mesh)
            else:
                self.surface_mesh = surface_mesh
        else:
            self.surface_mesh = self.o3d_mesh_to_pyvista(self.voxelmodel.mesh.mesh)

    def plot(self, only_surface=True, cube_opacity=0.2, surface_opacity=1, cube_edges=True):
        voxels = self.voxelmodel.get_voxels()
        colors = {
            "inside": "#22cc22",
            "on_surface": "#ffaa33",
            "outside": "#cccccc"
        }
        grouped_cubes = {"inside": [], "on_surface": [], "outside": []}

        for v in voxels:
            status = getattr(v, "status", "outside")
            if only_surface and status != "on_surface":
                continue
            center = v.min_corner + 0.5 * v.size
            cube = pv.Cube(center=center, x_length=v.size, y_length=v.size, z_length=v.size)
            grouped_cubes[status].append(cube)

        plotter = pv.Plotter()
        for status, cube_list in grouped_cubes.items():
            if not cube_list:
                continue
            mb = pv.MultiBlock(cube_list)
            plotter.add_mesh(mb,
                             color=colors.get(status, "#cccccc"),
                             opacity=cube_opacity,
                             show_edges=cube_edges,
                             smooth_shading=False)

        if self.surface_mesh is not None:
            plotter.add_mesh(self.surface_mesh,
                             color="#4477dd",
                             opacity=surface_opacity,
                             style="surface",
                             smooth_shading=True)

        plotter.show_grid()
        plotter.show()

    @staticmethod
    def o3d_mesh_to_pyvista(mesh):
        # Конвертация open3d.geometry.TriangleMesh --> pyvista.PolyData
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        faces = np.hstack([np.full((triangles.shape[0], 1), 3), triangles]).ravel()
        return pv.PolyData(vertices, faces)
