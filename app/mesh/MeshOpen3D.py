from abc import abstractmethod

import open3d as o3d

from CONFIG import DEFAULT_MESH_COLOR
from app.mesh.MeshABC import MeshABC
from app.scan.Scan import Scan


class MeshOpen3D(MeshABC):

    def __init__(self):
        self.scan = None
        self.pcd = None
        self. mesh = None

    def init_mesh_from_scan(self, scan: Scan):
        self.scan = scan
        points = scan.get_points_array()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        self.pcd = pcd
        self.mesh = self._calculate_mesh()
        return self

    def _estimate_normals(self, normals=None):
        # Если нормали не предоставлены, вычисляем их
        if normals is None:
            self.pcd.estimate_normals()
        else:
            self.pcd.normals = o3d.utility.Vector3dVector(normals)

    @abstractmethod
    def _calculate_mesh(self):
        pass

    def plot(self):
        wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(self.mesh)
        wireframe.paint_uniform_color(DEFAULT_MESH_COLOR)
        o3d.visualization.draw_geometries([self.pcd, wireframe],
                                          mesh_show_wireframe=False)
