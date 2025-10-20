from app.mesh.BallPivotingAlgorithmMeshOpen3D import BallPivotingAlgorithmMeshOpen3D
from app.mesh.ConvexHullMeshOpen3D import ConvexHullMeshOpen3D
from app.mesh.MeshTrimmers import MeshTrimmers
from app.mesh.PoissonMeshOpen3D import PoissonMeshOpen3D
from app.scan.Scan import Scan

scan = Scan(scan_name="TestScan")
scan.import_points_from_file(file_path="src/Камеры/data_2.dxf")

poisson_mesh = PoissonMeshOpen3D(depth=12,
                                    width=0,
                                    scale=1.1).init_mesh_from_scan(scan)
poisson_mesh.plot()

conv_hull_mesh = ConvexHullMeshOpen3D().init_mesh_from_scan(scan)

# mesh.trim_by_bpa(scale=1)
conv_hull_mesh.plot()

# bpa_mesh = BallPivotingAlgorithmMeshOpen3D().init_mesh_from_scan(scan)

tr_mesh = MeshTrimmers(poisson_mesh).trim_by_border_mesh(conv_hull_mesh, scale=1, outside_only=True)
# tr_mesh = MeshTrimmers(poisson_mesh).trim_by_border_mesh(bpa_mesh, scale=1.2, outside_only=True)

tr_mesh.plot()
