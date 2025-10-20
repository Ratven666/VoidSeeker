import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

from app.mesh.BallPivotingAlgorithmMeshOpen3D import BallPivotingAlgorithmMeshOpen3D
from app.mesh.MeshOpen3DABC import MeshOpen3DABC


class PoissonMeshOpen3D(MeshOpen3DABC):

    def __init__(self, depth=9, width=0, scale=1.1):
        super().__init__()
        self.depth = depth
        self.width = width
        self.scale = scale

    def _calculate_mesh(self):
        self._estimate_normals()
        # Poisson reconstruction
        self.pcd.orient_normals_consistent_tangent_plane(10)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            self.pcd, depth=self.depth, width=self.width, scale=self.scale
        )
        # Убираем вершины с низкой плотностью (артефакты)
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        self.mesh = mesh
        return mesh


if __name__ == "__main__":
    from app.scan.Scan import Scan

    scan = Scan(scan_name="TestScan")
    scan.import_points_from_file(file_path="../../src/Камеры/data_2.dxf")

    mesh = PoissonMeshOpen3D(depth=12,
                             width=0,
                             scale=1.5).init_mesh_from_scan(scan)
    print(mesh)
    mesh.plot()
