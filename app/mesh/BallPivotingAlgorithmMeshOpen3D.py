import numpy as np
import open3d as o3d

from app.mesh.MeshOpen3D import MeshOpen3D


class BallPivotingAlgorithmMeshOpen3D(MeshOpen3D):

    def __init__(self, radii=None):
        super().__init__()
        self.radii = radii

    def calculate_radii_for_pcd(self):
        distances = self.pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radii = [avg_dist * 1.5, avg_dist * 2.0, avg_dist * 3.0]
        return radii

    def _calculate_mesh(self):
        self._estimate_normals()
        if self.radii is not None:
            self.radii = radii
        else:
            self.radii = self.calculate_radii_for_pcd()
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            self.pcd, o3d.utility.DoubleVector(self.radii)
        )
        return mesh

if __name__ == "__main__":
    from app.scan.Scan import Scan

    scan = Scan(scan_name="TestScan")
    scan.import_points_from_file(file_path="../../src/Камеры/data_3.dxf")

    radii = (0.1, 0.2, 0.4)
    radii = None
    mesh = BallPivotingAlgorithmMeshOpen3D(radii=radii).init_mesh_from_scan(scan)
    print(mesh)
    mesh.plot()
