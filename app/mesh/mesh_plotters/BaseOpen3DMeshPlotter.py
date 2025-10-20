import open3d as o3d

from CONFIG import DEFAULT_MESH_COLOR
from app.mesh.mesh_plotters.MeshPlotterABC import MeshPlotterABC


class BaseOpen3DMeshPlotter(MeshPlotterABC):

    def __init__(self, *args, **kwargs):
        pass

    def plot(self, mesh):
        wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh.mesh)
        wireframe.paint_uniform_color(DEFAULT_MESH_COLOR)
        o3d.visualization.draw_geometries([mesh.pcd, wireframe],
                                          mesh_show_wireframe=False)
