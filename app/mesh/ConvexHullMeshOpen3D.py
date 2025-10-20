from app.mesh.MeshOpen3DABC import MeshOpen3DABC


class ConvexHullMeshOpen3D(MeshOpen3DABC):

    def _calculate_mesh(self):
        mesh, _ = self.pcd.compute_convex_hull()
        return mesh

if __name__ == "__main__":
    from app.scan.Scan import Scan

    scan = Scan(scan_name="TestScan")
    scan.import_points_from_file(file_path="../../src/Камеры/data_3.dxf")

    mesh = ConvexHullMeshOpen3D().init_mesh_from_scan(scan)
    print(mesh)
    mesh.plot()
