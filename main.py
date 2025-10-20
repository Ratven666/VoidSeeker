from app.mesh.ConvexHullMeshOpen3D import ConvexHullMeshOpen3D
from app.mesh.mesh_trimmers.BorderMeshMeshTrimmer import BorderMeshMeshTrimmer
from app.mesh.PoissonMeshOpen3D import PoissonMeshOpen3D
from app.mesh.mesh_trimmers.ZLevelsMeshTrimmer import ZLevelsMeshTrimmer
from app.scan.Scan import Scan

scan = Scan(scan_name="TestScan")
scan.import_points_from_file(file_path="src/Камеры/data_3.dxf")
# scan.plot(point_size=8)

poisson_mesh = PoissonMeshOpen3D(depth=12,
                                 width=0.25,
                                 scale=2.5).init_mesh_from_scan(scan)
# poisson_mesh.plot()

# conv_hull_mesh = ConvexHullMeshOpen3D().init_mesh_from_scan(scan)

# mesh.trim_by_bpa(scale=1)
# conv_hull_mesh.plot()

# bpa_mesh = BallPivotingAlgorithmMeshOpen3D().init_mesh_from_scan(scan)

# tr_mesh = BorderMeshMeshTrimmer(poisson_mesh, conv_hull_mesh, scale=1).trim_mesh()
# tr_mesh = MeshTrimmers(poisson_mesh).trim_by_border_mesh(bpa_mesh, scale=1.2, outside_only=True)

z_level = scan.borders["z_max"]
print(z_level)
tr_mesh = ZLevelsMeshTrimmer(base_mesh=poisson_mesh, z_level=z_level).trim_mesh()
tr_mesh.plot()

