import open3d as o3d

from app.mesh.MeshOpen3DABC import MeshOpen3DABC


class AlphaShapesMeshOpen3D(MeshOpen3DABC):

    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha


    def _calculate_mesh(self):
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(self.pcd,
                                                                             self.alpha)
        return mesh

if __name__ == "__main__":
    from app.scan.Scan import Scan

    scan = Scan(scan_name="TestScan")
    scan.import_points_from_file(file_path="../../src/Камеры/data_3.dxf")

    alpha = 0.5
    mesh = AlphaShapesMeshOpen3DABC(alpha=alpha).init_mesh_from_scan(scan)
    print(mesh)
    mesh.plot()
